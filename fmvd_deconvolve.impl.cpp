#define PASTE(name, type) name ## _ ## type
#define EVAL(name, type) PASTE(name, type)
#define MAKE_NAME(name) EVAL(name, BITS_PER_SAMPLE)

#include "fmvd_deconvolve.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fmvd_cuda_utils.h"

#ifdef _WIN32
#include <windows.h>
#endif


static void
MAKE_NAME(padData)(const struct MAKE_NAME(fmvd_plan_cuda) *plan, int v, int stream)
{
	int fftOffset	   = stream * plan->fftW  * plan->fftH;
	int dataOffset	   = stream * plan->dataW * plan->dataH;
	MAKE_NAME(padDataClampToBorderAndInitialize)(
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
MAKE_NAME(prepareKernels)(
		const struct MAKE_NAME(fmvd_plan_cuda) *plan,
		float **h_Kernel,
		fmvd_psf_type iteration_type)
{
	int v, paddedsize, kernelsize, kernelW, kernelH;
	float *d_Kernel, *d_PaddedKernel, *d_PaddedKernelHat;

	kernelW = plan->kernelW;
	kernelH = plan->kernelH;

	paddedsize = plan->fftH * plan->fftW * sizeof(float);
	kernelsize = kernelH * kernelW * sizeof(float);

	checkCudaErrors(cudaMalloc((void **)&d_Kernel,          kernelsize));
	checkCudaErrors(cudaMalloc((void **)&d_PaddedKernel,    paddedsize));
	checkCudaErrors(cudaMalloc((void **)&d_PaddedKernelHat, paddedsize));

	float **h_Kernel2 = computeKernel2(h_Kernel, kernelW, kernelH, plan->nViews, iteration_type);

	for(v = 0; v < plan->nViews; v++)
		normalize(h_Kernel[v], kernelW * kernelH);

	for(v = 0; v < plan->nViews; v++) {
		checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel[v], kernelsize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemset(d_PaddedKernel, 0, paddedsize));
		padKernel(d_PaddedKernel, d_Kernel, plan->fftH, plan->fftW, kernelH, kernelW, 0); // default stream
		checkCudaErrors(cufftExecR2C(plan->fftPlanFwd[0], (cufftReal *)d_PaddedKernel, (cufftComplex *)plan->d_KernelSpectrum[v]));

		checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel2[v], kernelsize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemset(d_PaddedKernelHat, 0, paddedsize));
		padKernel(d_PaddedKernelHat, d_Kernel, plan->fftH, plan->fftW, kernelH, kernelW, 0); // default stream
		checkCudaErrors(cufftExecR2C(plan->fftPlanFwd[0], (cufftReal *)d_PaddedKernelHat, (cufftComplex *)plan->d_KernelHatSpectrum[v]));
	}

	for(int v = 0; v < plan->nViews; v++)
		free(h_Kernel2[v]);
	free(h_Kernel2);
	checkCudaErrors(cudaFree(d_Kernel));
	checkCudaErrors(cudaFree(d_PaddedKernel));
	checkCudaErrors(cudaFree(d_PaddedKernelHat));
}

static void
MAKE_NAME(prepareWeights)(
		const struct MAKE_NAME(fmvd_plan_cuda) *plan,
		float const* const* h_Weights)
{
	int v, paddedsize_float, datasize;
	float *d_Weights;
	float *d_PaddedWeightSums;

	paddedsize_float = plan->fftH * plan->fftW * sizeof(float);
	datasize = plan->dataH * plan->dataW * sizeof(float);

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


/*************************
 * Public API
 *************************/
struct MAKE_NAME(fmvd_plan_cuda) *
MAKE_NAME(fmvd_initialize_cuda)(
		int dataH,
		int dataW,
		float const* const* h_Weights,
		float **h_Kernel,
		int kernelH,
		int kernelW,
		fmvd_psf_type iteration_type,
		int nViews,
		int nstreams,
		MAKE_NAME(datasource_t) get_next_plane,
		MAKE_NAME(datasink_t) return_next_plane)
{
	int v, stream, fftH, fftW, fftsize, paddedsize_float, paddedsize_data_t, datasize;
	struct MAKE_NAME(fmvd_plan_cuda) *plan;

	fftH = snapTransformSize(dataH + kernelH - 1);
	fftW = snapTransformSize(dataW + kernelW - 1);
	fftsize = fftH * (fftW / 2 + 1) * sizeof(fComplex);
	paddedsize_float = fftH * fftW * sizeof(float);
	paddedsize_data_t = fftH * fftW * sizeof(SAMPLE);
	datasize = dataH * dataW * sizeof(SAMPLE);

	plan = (struct MAKE_NAME(fmvd_plan_cuda) *)malloc(sizeof(struct MAKE_NAME(fmvd_plan_cuda)));
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

	plan->h_Data              = (SAMPLE **)  malloc(nViews * sizeof(SAMPLE *));
	plan->d_Data              = (SAMPLE **)  malloc(nViews * sizeof(SAMPLE *));
	plan->d_PaddedData        = (SAMPLE **)  malloc(nViews * sizeof(SAMPLE *));
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

	MAKE_NAME(prepareKernels)(plan, h_Kernel, iteration_type);
	MAKE_NAME(prepareWeights)(plan, h_Weights);
	return plan;
}

void
MAKE_NAME(fmvd_deconvolve_planes_cuda)(const struct MAKE_NAME(fmvd_plan_cuda) *plan, int iterations)
{
	int z, v, stream_idx, it, fftH, fftW, paddedsize_float, paddedsize_data_t, datasize;

	fftH = plan->fftH;
	fftW = plan->fftW;
	paddedsize_float = fftH * fftW * sizeof(float);
	paddedsize_data_t = fftH * fftW * sizeof(SAMPLE);
	datasize = plan->dataH * plan->dataW * sizeof(SAMPLE);

	for(stream_idx = 0; stream_idx < plan->nStreams; stream_idx++) {
		int dataOffset = stream_idx * plan->dataH * plan->dataW;
		plan->get_next_plane(plan->h_Data, dataOffset);
	}

#ifdef _WIN32
	int start = GetTickCount();
#endif
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
			checkCudaErrors(cudaStreamSynchronize(stream));
			plan->return_next_plane(plan->h_Data[0] + dataOffset);
			int has_more_data = plan->get_next_plane(plan->h_Data, dataOffset);
			if(!has_more_data)
				break;
		}

		// H2D
		for(v = 0; v < plan->nViews; v++) {
			// load data from file and upload it to GPU
			SAMPLE *d_Data = plan->d_Data[v] + dataOffset;
			SAMPLE *h_Data = plan->h_Data[v] + dataOffset;
			checkCudaErrors(cudaMemcpyAsync(d_Data, h_Data, datasize, cudaMemcpyHostToDevice, stream));
		}
		checkCudaErrors(cudaMemsetAsync(d_estimate, 0, paddedsize_float, plan->streams[stream_idx]));

		for(v = 0; v < plan->nViews; v++) {
			SAMPLE *d_PaddedData = plan->d_PaddedData[v] + fftOffset;
			checkCudaErrors(cudaMemsetAsync(d_PaddedData, 0, paddedsize_data_t, stream));
			MAKE_NAME(padData)(plan, v, stream_idx);
		}

		for(it = 0; it < iterations; it++) {
			for(v = 0; v < plan->nViews; v++) {
				checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_estimate, (cufftComplex *)d_estimateSpectrum));
				modulateAndNormalize(d_estimateSpectrum, plan->d_KernelSpectrum[v], fftH, fftW, 1, stream);
				checkCudaErrors(cufftExecC2R(fftPlanInv, (cufftComplex *)d_estimateSpectrum, (cufftReal *)d_tmp));
				MAKE_NAME(divide)(plan->d_PaddedData[v] + fftOffset, d_tmp, d_tmp, fftH, fftW, stream);

				checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_tmp, (cufftComplex *)d_estimateSpectrum));
				modulateAndNormalize(d_estimateSpectrum, plan->d_KernelHatSpectrum[v], fftH, fftW, 1, stream);
				checkCudaErrors(cufftExecC2R(fftPlanInv, (cufftComplex *)d_estimateSpectrum, (cufftReal *)d_tmp));
				multiply32(d_estimate, d_tmp, plan->d_PaddedWeights[v], d_estimate, fftH, fftW, stream);
			}
		}

		MAKE_NAME(unpadData)(plan->d_Data[0] + dataOffset, d_estimate, fftH, fftW, plan->dataH, plan->dataW, stream);

		// D2H
		checkCudaErrors(cudaMemcpyAsync(
				plan->h_Data[0] + dataOffset,
				plan->d_Data[0] + dataOffset,
				datasize,
				cudaMemcpyDeviceToHost,
				stream));

	}

	// save the remaining nStreams planes
	for(z = 0; z < plan->nStreams - 1; z++) {
		stream_idx = (stream_idx + 1) % plan->nStreams;
		int dataOffset = stream_idx * plan->dataW * plan->dataH;
		checkCudaErrors(cudaStreamSynchronize(plan->streams[stream_idx]));
		plan->return_next_plane(plan->h_Data[0] + dataOffset);
	}

#ifdef _WIN32
	int stop = GetTickCount();
	printf("Overall time: %d ms\n", (stop - start));
#endif
}

void
MAKE_NAME(fmvd_destroy_cuda)(struct MAKE_NAME(fmvd_plan_cuda) *plan)
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

