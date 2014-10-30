/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


/*
 * This sample demonstrates how 2D convolutions
 * with very large kernel sizes
 * can be efficiently implemented
 * using FFT transformations.
 */


#include <windows.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>
#include "convolutionFFT2D_common.h"
#include "fmvd_deconvolve_cuda.h"
#include "fmvd_deconvolve_common.h"


#define checkCudaErrors(ans) {gpuAssert((ans), __FILE__, __LINE__); }
#define checkCufftErrors(ans) {gpuAssert((ans), __FILE__, __LINE__); }

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
void gpuAssert(unsigned int code, const char *file, int line, bool abort=true)
{
	if(code != cudaSuccess) {
		const char *str = cudaGetErrorString((cudaError_t)code);
		fprintf(stderr, "GPUAssert: error %d %s %d\n", code, file, line);
		fprintf(stderr, "%s\n", str);
		if(abort)
			exit(code);
	}
}

static void
padDataClampToBorderGPU(const struct fmvd_plan_cuda *plan, int v, int stream)
{
	int fftOffset	   = stream * plan->fftW  * plan->fftH;
	int dataOffset	   = stream * plan->dataW * plan->dataH;
	padDataClampToBorder(
		plan->d_estimate + fftOffset,
		plan->d_PaddedData[v] + fftOffset,
		plan->d_Data[v] + dataOffset,
		plan->d_PaddedWeights[v],
		plan->fftH,
		plan->fftW,
		plan->dataH,
		plan->dataW,
		plan->kernelH,
		plan->kernelW,
		plan->streams[stream]
	);
}

static void
prepareKernelsGPU(const struct fmvd_plan_cuda *plan, float const* const* h_Kernel)
{
	int v, paddedsize, kernelsize;
	float *d_Kernel, *d_PaddedKernel, *d_PaddedKernelHat;

	paddedsize = plan->fftH * plan->fftW * sizeof(float);
	kernelsize = plan->kernelH * plan->kernelW * sizeof(float);

	checkCudaErrors(cudaMalloc((void **)&d_Kernel,          kernelsize));
	checkCudaErrors(cudaMalloc((void **)&d_PaddedKernel,    paddedsize));
	checkCudaErrors(cudaMalloc((void **)&d_PaddedKernelHat, paddedsize));
	for(v = 0; v < plan->nViews; v++) {
		checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel[v], kernelsize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemset(d_PaddedKernel, 0, paddedsize));
		checkCudaErrors(cudaMemset(d_PaddedKernelHat, 0, paddedsize));

		padKernel(
			d_PaddedKernel,
			d_PaddedKernelHat,
			d_Kernel,
			plan->fftH,
			plan->fftW,
			plan->kernelH,
			plan->kernelW,
			0 // default stream
		);

		checkCudaErrors(cufftExecR2C(
				plan->fftPlanFwd[0],
				(cufftReal *)d_PaddedKernel,
				(cufftComplex *)plan->d_KernelSpectrum[v]));
		checkCudaErrors(cufftExecR2C(
				plan->fftPlanFwd[0],
				(cufftReal *)d_PaddedKernelHat,
				(cufftComplex *)plan->d_KernelHatSpectrum[v]));

	}
	checkCudaErrors(cudaFree(d_Kernel));
	checkCudaErrors(cudaFree(d_PaddedKernel));
	checkCudaErrors(cudaFree(d_PaddedKernelHat));
}

static void
prepareWeightsGPU(const struct fmvd_plan_cuda *plan, data_t const* const* h_Weights)
{
	int v, paddedsize_data_t, paddedsize_float, datasize;
	data_t *d_Weights;
	float *d_PaddedWeightSums;

	paddedsize_data_t = plan->fftH * plan->fftW * sizeof(data_t);
	paddedsize_float = plan->fftH * plan->fftW * sizeof(float);
	datasize = plan->dataH * plan->dataW * sizeof(data_t);

	checkCudaErrors(cudaMalloc((void **)&d_Weights, datasize));
	checkCudaErrors(cudaMalloc((void **)&d_PaddedWeightSums, paddedsize_float));
	checkCudaErrors(cudaMemset(d_PaddedWeightSums, 0, paddedsize_float));
	for(v = 0; v < plan->nViews; v++) {
		checkCudaErrors(cudaMemcpy(d_Weights, h_Weights[v], datasize, cudaMemcpyHostToDevice));
		padWeights(
			plan->d_PaddedWeights[v],
			d_PaddedWeightSums,
			d_Weights,
			plan->fftH,
			plan->fftW,
			plan->dataH,
			plan->dataW,
			plan->kernelH,
			plan->kernelW,
			0 // default stream
		);
	}

	for(v = 0; v < plan->nViews; v++) {
		printf("Normalizing weights for view %d\n", v);
		normalizeWeights(plan->d_PaddedWeights[v], d_PaddedWeightSums, plan->fftH, plan->fftW, 0);
	}

	checkCudaErrors(cudaFree(d_Weights));
	checkCudaErrors(cudaFree(d_PaddedWeightSums));
}

struct fmvd_plan_cuda *
fmvd_initialize_cuda(
		int dataH,
		int dataW,
		data_t const* const* h_Weights,
		float const* const* h_Kernel,
		int kernelH,
		int kernelW,
		int nViews,
		int nstreams,
		datasource_t get_next_plane,
		datasink_t return_next_plane)
{
	int v, stream, fftH, fftW, fftsize, paddedsize_float, paddedsize_data_t, datasize;
	struct fmvd_plan_cuda *plan;

	fftH = snapTransformSize(dataH + kernelH - 1);
	fftW = snapTransformSize(dataW + kernelW - 1);
	fftsize = fftH * (fftW / 2 + 1) * sizeof(fComplex);
	paddedsize_float = fftH * fftW * sizeof(float);
	paddedsize_data_t = fftH * fftW * sizeof(data_t);
	datasize = dataH * dataW * sizeof(data_t);

	plan = (struct fmvd_plan_cuda *)malloc(sizeof(struct fmvd_plan_cuda));
	plan->dataH         = dataH;
	plan->dataW         = dataW;
	plan->fftH          = fftH;
	plan->fftW          = fftW;
	plan->kernelH       = kernelH;
	plan->kernelW       = kernelW;
	plan->nViews        = nViews;
	plan->nStreams      = nstreams;

	plan->get_next_plane = get_next_plane;
	plan->return_next_plane = return_next_plane;

	plan->h_Data              = (data_t **)  malloc(nViews * sizeof(data_t *));
	plan->d_Data              = (data_t **)  malloc(nViews * sizeof(data_t *));
	plan->d_PaddedData        = (data_t **)  malloc(nViews * sizeof(data_t *));
	plan->d_PaddedWeights     = (float **)   malloc(nViews * sizeof(float *));
	plan->d_KernelSpectrum    = (fComplex **)malloc(nViews * sizeof(fComplex *));
	plan->d_KernelHatSpectrum = (fComplex **)malloc(nViews * sizeof(fComplex *));

	for(v = 0; v < nViews; v++) {
		checkCudaErrors(cudaMallocHost(&plan->h_Data[v],                   nstreams * datasize));
		checkCudaErrors(cudaMalloc((void **)&plan->d_Data[v],              nstreams * datasize));
		checkCudaErrors(cudaMalloc((void **)&plan->d_PaddedData[v],        nstreams * paddedsize_data_t));
		checkCudaErrors(cudaMalloc((void **)&plan->d_PaddedWeights[v],     paddedsize_float));
		checkCudaErrors(cudaMalloc((void **)&plan->d_KernelSpectrum[v],    nstreams * fftsize));
		checkCudaErrors(cudaMalloc((void **)&plan->d_KernelHatSpectrum[v], nstreams * fftsize));
	}

	checkCudaErrors(cudaMalloc((void **)&plan->d_estimate,         nstreams * paddedsize_float));
	checkCudaErrors(cudaMalloc((void **)&plan->d_tmp,              nstreams * paddedsize_float));
	checkCudaErrors(cudaMalloc((void **)&plan->d_estimateSpectrum, nstreams * fftsize));


	plan->fftPlanFwd = (cufftHandle *)malloc(nstreams * sizeof(cufftHandle));
	plan->fftPlanInv = (cufftHandle *)malloc(nstreams * sizeof(cufftHandle));
	plan->streams    = (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));

	for(stream = 0; stream < nstreams; stream++) {
		cudaStreamCreate(&plan->streams[stream]);
		checkCudaErrors(cufftPlan2d(&plan->fftPlanFwd[stream], fftH, fftW, CUFFT_R2C));
		checkCudaErrors(cufftSetStream(plan->fftPlanFwd[stream], plan->streams[stream]));
		checkCudaErrors(cufftPlan2d(&plan->fftPlanInv[stream], fftH, fftW, CUFFT_C2R));
		checkCudaErrors(cufftSetStream(plan->fftPlanInv[stream], plan->streams[stream]));
	}

	prepareKernelsGPU(plan, h_Kernel);
	prepareWeightsGPU(plan, h_Weights);
	return plan;
}

void
fmvd_deconvolve_planes_cuda(const struct fmvd_plan_cuda *plan, int iterations)
{
	int z, v, stream_idx, it, fftH, fftW, paddedsize_float, paddedsize_data_t, datasize;
	long start, stop;

	fftH = plan->fftH;
	fftW = plan->fftW;
	paddedsize_float = fftH * fftW * sizeof(float);
	paddedsize_data_t = fftH * fftW * sizeof(data_t);
	datasize = plan->dataH * plan->dataW * sizeof(data_t);

	for(stream_idx = 0; stream_idx < plan->nStreams; stream_idx++) {
		int dataOffset = stream_idx * plan->dataH * plan->dataW;
		plan->get_next_plane(plan->h_Data, dataOffset);
	}

	start = GetTickCount();
	for(z = 0; ; z++) {
		stream_idx = z % plan->nStreams;
		cudaStream_t stream = plan->streams[stream_idx];
		int fftOffset	   = stream_idx * fftW * fftH;
		int spectrumOffset = stream_idx * (fftW / 2 + 1) * fftH;
		int dataOffset	   = stream_idx * plan->dataW * plan->dataH;

		float *d_estimate = plan->d_estimate + fftOffset;
		fComplex *d_estimateSpectrum = plan->d_estimateSpectrum + spectrumOffset;
		float *d_tmp = plan->d_tmp + fftOffset;

		cufftHandle fftPlanFwd = plan->fftPlanFwd[stream_idx];
		cufftHandle fftPlanInv = plan->fftPlanInv[stream_idx];

		// load from HDD
		if(z >= plan->nStreams) {
//			printf("z %d (stream %d): waiting for stream %d\n", z, stream_idx, stream_idx);
			checkCudaErrors(cudaStreamSynchronize(stream));
			plan->return_next_plane(plan->h_Data[0] + dataOffset);
			int has_more_data = plan->get_next_plane(plan->h_Data, dataOffset);
			if(!has_more_data)
				break;
		}

		// H2D
		for(v = 0; v < plan->nViews; v++) {
			// load data from file and upload it to GPU
			data_t *d_Data = plan->d_Data[v] + dataOffset;
			data_t *h_Data = plan->h_Data[v] + dataOffset;
			checkCudaErrors(cudaMemcpyAsync(d_Data, h_Data, datasize, cudaMemcpyHostToDevice, stream));
		}
		checkCudaErrors(cudaMemsetAsync(d_estimate, 0, paddedsize_float, plan->streams[stream_idx]));

		for(v = 0; v < plan->nViews; v++) {
			data_t *d_PaddedData = plan->d_PaddedData[v] + fftOffset;
			checkCudaErrors(cudaMemsetAsync(d_PaddedData, 0, paddedsize_data_t, stream));
			padDataClampToBorderGPU(plan, v, stream_idx);
		}

		for(it = 0; it < iterations; it++) {
			for(v = 0; v < plan->nViews; v++) {
				checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_estimate, (cufftComplex *)d_estimateSpectrum));
				modulateAndNormalize(d_estimateSpectrum, plan->d_KernelSpectrum[v], fftH, fftW, 1, stream);
				checkCudaErrors(cufftExecC2R(fftPlanInv, (cufftComplex *)d_estimateSpectrum, (cufftReal *)d_tmp));
				divide(plan->d_PaddedData[v] + fftOffset, d_tmp, d_tmp, fftH, fftW, stream);

				checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_tmp, (cufftComplex *)d_estimateSpectrum));
				modulateAndNormalize(d_estimateSpectrum, plan->d_KernelHatSpectrum[v], fftH, fftW, 1, stream);
				checkCudaErrors(cufftExecC2R(fftPlanInv, (cufftComplex *)d_estimateSpectrum, (cufftReal *)d_tmp));
				mul(d_estimate, d_tmp, plan->d_PaddedWeights[v], d_estimate, fftW, fftH, stream);
			}
		}

		unpadData(plan->d_Data[0] + dataOffset, d_estimate, fftH, fftW, plan->dataH, plan->dataW, stream);

		// D2H
		checkCudaErrors(cudaMemcpyAsync(
				plan->h_Data[0] + dataOffset,
				plan->d_Data[0] + dataOffset,
				datasize,
				cudaMemcpyDeviceToHost,
				stream));

	}

	// save the remaining nStreams planes
	stream_idx = stream_idx % plan->nStreams;
	for(z = 0; z < plan->nStreams; z++) {
		int dataOffset = stream_idx * plan->dataW * plan->dataH;
		checkCudaErrors(cudaStreamSynchronize(plan->streams[stream_idx]));
		plan->return_next_plane(plan->h_Data[0] + dataOffset);
		stream_idx = (stream_idx + 1) % plan->nStreams;
	}

	stop = GetTickCount();
	printf("Overall time: %d ms\n", (stop - start));
}

void
fmvd_destroy_cuda(struct fmvd_plan_cuda *plan)
{
	int v, stream;

	for(v = 0; v < plan->nViews; v++) {
		checkCudaErrors(cudaFreeHost(plan->h_Data[v]));
		checkCudaErrors(cudaFree(plan->d_Data[v]));
		checkCudaErrors(cudaFree(plan->d_PaddedData[v]));
		checkCudaErrors(cudaFree(plan->d_PaddedWeights[v]));
		checkCudaErrors(cudaFree(plan->d_KernelSpectrum[v]));
		checkCudaErrors(cudaFree(plan->d_KernelHatSpectrum[v]));
	}
	checkCudaErrors(cudaFree(plan->d_estimate));
	checkCudaErrors(cudaFree(plan->d_estimateSpectrum));
	checkCudaErrors(cudaFree(plan->d_tmp));

	free(plan->h_Data);
	free(plan->d_Data);
	free(plan->d_PaddedData);
	free(plan->d_PaddedWeights);
	free(plan->d_KernelSpectrum);
	free(plan->d_KernelHatSpectrum);

	for(stream = 0; stream < plan->nStreams; stream++) {
		checkCudaErrors(cufftDestroy(plan->fftPlanInv[stream]));
		checkCudaErrors(cufftDestroy(plan->fftPlanFwd[stream]));
		checkCudaErrors(cudaStreamDestroy(plan->streams[stream]));
	}
	free(plan->streams);
	free(plan->fftPlanFwd);
	free(plan->fftPlanInv);

	free(plan);
}

void *
fmvd_malloc(size_t size)
{
	void *p;
	checkCudaErrors(cudaMallocHost(&p, size));
	return p;
}

void
fmvd_free(void *p)
{
	cudaFreeHost(p);
}

