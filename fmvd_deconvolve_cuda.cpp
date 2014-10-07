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
		plan->fftH,
		plan->fftW,
		plan->dataH,
		plan->dataW,
		plan->kernelH,
		plan->kernelW,
		plan->nViews,
		plan->streams[stream]
	);
}

void prepareKernelsGPU(const struct fmvd_plan_cuda *plan, float const* const* h_Kernel)
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

struct fmvd_plan_cuda *
fmvd_initialize_cuda(
		int dataH,
		int dataW,
		float const* const* h_Kernel,
		int kernelH,
		int kernelW,
		int nViews,
		int nstreams,
		datasource_t get_next_plane,
		datasink_t return_next_plane)
{
	int v, stream, fftH, fftW, fftsize, paddedsize, datasize;
	struct fmvd_plan_cuda *plan;
	
	fftH = snapTransformSize(dataH + kernelH - 1);
	fftW = snapTransformSize(dataW + kernelW - 1);
	fftsize = fftH * (fftW / 2 + 1) * sizeof(fComplex);
	paddedsize = fftH * fftW * sizeof(float);
	datasize = dataH * dataW * sizeof(float);

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

	plan->h_Data              = (float **)   malloc(nViews * sizeof(float *));
	plan->d_Data              = (float **)   malloc(nViews * sizeof(float *));
	plan->d_PaddedData        = (float **)   malloc(nViews * sizeof(float *));
	plan->d_KernelSpectrum    = (fComplex **)malloc(nViews * sizeof(fComplex *));
	plan->d_KernelHatSpectrum = (fComplex **)malloc(nViews * sizeof(fComplex *));

	for(v = 0; v < nViews; v++) {
		checkCudaErrors(cudaMallocHost(&plan->h_Data[v],                   nstreams * datasize));
		checkCudaErrors(cudaMalloc((void **)&plan->d_Data[v],              nstreams * datasize));
		checkCudaErrors(cudaMalloc((void **)&plan->d_PaddedData[v],        nstreams * paddedsize));
		checkCudaErrors(cudaMalloc((void **)&plan->d_KernelSpectrum[v],    nstreams * fftsize));
		checkCudaErrors(cudaMalloc((void **)&plan->d_KernelHatSpectrum[v], nstreams * fftsize));
	}

	checkCudaErrors(cudaMalloc((void **)&plan->d_estimate,         nstreams * paddedsize));
	checkCudaErrors(cudaMalloc((void **)&plan->d_tmp,              nstreams * paddedsize));
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
	return plan;
}

void
fmvd_deconvolve_plane_cuda(const struct fmvd_plan_cuda *plan, int iterations, void *userdata) 
{
	int z, v, stream_idx, it, fftH, fftW, paddedsize, datasize;
	long start, stop;

	fftH = plan->fftH;
	fftW = plan->fftW;
	paddedsize = fftH * fftW * sizeof(float);
	datasize = plan->dataH * plan->dataW * sizeof(float);

	int has_more_data = 1;
	for(v = 0; v < plan->nViews; v++)
		for(stream_idx = 0; stream_idx < plan->nStreams; stream_idx++)
			has_more_data = has_more_data && plan->get_next_plane(v, plan->h_Data[v], userdata);

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
		if(z > 3) {
			printf("z %d (stream %d): waiting for stream %d\n", z, stream_idx, stream_idx);
			checkCudaErrors(cudaStreamSynchronize(stream));
			plan->return_next_plane(plan->h_Data[0] + dataOffset, userdata);
			has_more_data = 1;
			for(v = 0; v < plan->nViews; v++)
				has_more_data = has_more_data && plan->get_next_plane(v, plan->h_Data[v] + dataOffset, userdata);
			if(!has_more_data)
				break;
		}

		// H2D
		for(v = 0; v < plan->nViews; v++) {
			// load data from file and upload it to GPU
			float *d_Data = plan->d_Data[v] + dataOffset;
			float *h_Data = plan->h_Data[v] + dataOffset;
			checkCudaErrors(cudaMemcpyAsync(d_Data, h_Data, datasize, cudaMemcpyHostToDevice, stream));
		}
		checkCudaErrors(cudaMemsetAsync(d_estimate, 0, paddedsize, plan->streams[stream_idx]));

		for(v = 0; v < plan->nViews; v++) {
			float *d_PaddedData = plan->d_PaddedData[v] + fftOffset;
			checkCudaErrors(cudaMemsetAsync(d_PaddedData, 0, paddedsize, stream));
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
				mul(d_estimate, d_tmp, d_estimate, fftW, fftH, stream);
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

		printf("z %d (stream %d): waiting for stream %d\n", z, stream_idx, stream_idx);
		checkCudaErrors(cudaStreamSynchronize(plan->streams[stream_idx]));
		plan->return_next_plane(plan->h_Data[0] + dataOffset, userdata);
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
		checkCudaErrors(cudaFree(plan->d_KernelSpectrum[v]));
		checkCudaErrors(cudaFree(plan->d_KernelHatSpectrum[v]));
	}
	checkCudaErrors(cudaFree(plan->d_estimate));
	checkCudaErrors(cudaFree(plan->d_estimateSpectrum));
	checkCudaErrors(cudaFree(plan->d_tmp));

	free(plan->h_Data);
	free(plan->d_Data);
	free(plan->d_PaddedData);
	free(plan->d_KernelSpectrum);
	free(plan->d_KernelHatSpectrum);

	printf("...shutting down\n");

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

struct iodata {
	FILE **dataFiles;
	FILE *resultFile;
	int datasize;
	int plane;
	int n_planes;
};

int get_next_plane(int view, float *data, void *userdata)
{
	struct iodata *io = (struct iodata *)userdata;
	if(io->plane >= io->n_planes)
		return 0;
	fread(data, sizeof(float), io->datasize, io->dataFiles[view]);
	if(view == 0)
		io->plane++;
	return 1;
}

void return_next_plane(float *data, void *userdata)
{
	struct iodata *io = (struct iodata *)userdata;
	fwrite(data, sizeof(float), io->datasize, io->resultFile);
}

void deconvolveGPU(FILE **dataFiles, FILE *resultFile, int dataW, int dataH, int dataD, float **h_Kernel, int kernelH, int kernelW, int nViews, int iterations)
{
	int nStreams = 3;

	struct iodata *io = (struct iodata *)malloc(sizeof(struct iodata));
	io->dataFiles = dataFiles;
	io->resultFile = resultFile;
	io->datasize = dataH * dataW;
	io->plane = 0;
	io->n_planes = dataD;

	struct fmvd_plan_cuda *plan = fmvd_initialize_cuda(
			dataH, dataW,
			h_Kernel, kernelH, kernelW,
			nViews, nStreams,
			get_next_plane,
			return_next_plane);


	fmvd_deconvolve_plane_cuda(plan, iterations, io);

	fmvd_destroy_cuda(plan);

	printf("...shutting down\n");
}

void load(int w, int h, float *data, const char *path)
{
	FILE *f = fopen(path, "rb");
	fread(data, sizeof(float), w * h, f);
	fclose(f);
}

void write(int w, int h, const float *data, const char *path)
{
	FILE *f = fopen(path, "wb");
	fwrite(data, sizeof(float), w * h, f);
	fclose(f);
}

int main(int argc, char **argv)
{
	int it;
	printf("[%s] - Starting...\n", argv[0]);

	const int W = 600;
	const int H = 600;
	const int D = 100;

	const int KW = 83;
	const int KH = 83;

	float **kernel = (float **)malloc(2 * sizeof(float *));
	cudaMallocHost(&kernel[0], KW * KH * sizeof(float));
	cudaMallocHost(&kernel[1], KW * KH * sizeof(float));


	load(KW, KH, kernel[0], "E:\\SPIM5_Deconvolution\\m6\\cropped\\forplanewisedeconv\\psf3.raw");
	load(KW, KH, kernel[1], "E:\\SPIM5_Deconvolution\\m6\\cropped\\forplanewisedeconv\\psf5.raw");

	normalize(kernel[0], KW * KH);
	normalize(kernel[1], KW * KH);

	FILE **dataFiles = (FILE**)malloc(2 * sizeof(FILE *));
	char path[256];
	for(it = 3; it < 4; it++) {
		dataFiles[0] = fopen("E:\\SPIM5_Deconvolution\\m6\\cropped\\forplanewisedeconv\\v3.raw", "rb");
		dataFiles[1] = fopen("E:\\SPIM5_Deconvolution\\m6\\cropped\\forplanewisedeconv\\v5.raw", "rb");
		fseek(dataFiles[0], 300 * W * H * sizeof(float), SEEK_SET);
		fseek(dataFiles[1], 300 * W * H * sizeof(float), SEEK_SET);

		sprintf(path, "E:\\SPIM5_Deconvolution\\m6\\cropped\\forplanewisedeconv\\v35.deconvolved.gpu.raw");
		FILE *resultFile = fopen(path, "wb");

		deconvolveGPU(dataFiles, resultFile, W, H, D, kernel, KH, KW, 2, it);

		fclose(dataFiles[0]);
		fclose(dataFiles[1]);
		fclose(resultFile);
	}

	cudaFreeHost(kernel[0]);
	cudaFreeHost(kernel[1]);
	free(kernel);
	free(dataFiles);

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();

	exit(EXIT_SUCCESS);
}

