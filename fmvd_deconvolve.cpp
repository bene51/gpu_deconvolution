#include "fmvd_deconvolve.h"

#include <windows.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fmvd_utils.h"


/*************************
 * Helper functions.
 *************************/

//Align a to nearest higher multiple of b
static int
iAlignUp(int a, int b)
{
	return (a % b != 0) ? (a - a % b + b) : a;
}

static int
snapTransformSize(int dataSize)
{
	int hiBit;
	unsigned int lowPOT, hiPOT;

	dataSize = iAlignUp(dataSize, 16);

	for (hiBit = 31; hiBit >= 0; hiBit--)
		if (dataSize & (1U << hiBit))
			break;

	lowPOT = 1U << hiBit;

	if (lowPOT == (unsigned int)dataSize)
		return dataSize;

	hiPOT = 1U << (hiBit + 1);

	if (hiPOT <= 1024)
		return hiPOT;
	else
		return iAlignUp(dataSize, 512);
}

static void
normalize(float *kernel, int len)
{
	int i;
	double sum = 0;
	float *k = kernel;
	for(i = 0; i < len; i++) {
		sum += *k;
		k++;
	}
	k = kernel;
	for(i = 0; i < len; i++) {
		*k /= (float)sum;
		k++;
	}
}

static void
computeInvertedKernel(const float *kernel, float *out, int kw, int kh)
{
	int x, y;
	for(y = 0; y < kh; y++) {
		for(x = 0; x < kw; x++) {
			out[y * kw + x] = kernel[(kh - y - 1) * kw + (kw - x - 1)];
		}
	}
}

static void
computeExponentialKernel(const float *kernel, float *out, int kw, int kh, int exponent)
{
	int i;
	int wh = kw * kh;
	for(i = 0; i < wh; i++)
		out[i] = (float)pow(kernel[i], exponent);
}



static void
padData(const struct fmvd_plan_cuda *plan, int v, int stream)
{
	int fftOffset	   = stream * plan->fftW  * plan->fftH;
	int dataOffset	   = stream * plan->dataW * plan->dataH;
	padDataClampToBorderAndInitialize16(
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
convolve_single_plane(float *h_Data, int dataW, int dataH, const float *h_Kernel, int kernelW, int kernelH)
{
	int fftH = snapTransformSize(dataH + kernelH - 1);
	int fftW = snapTransformSize(dataW + kernelW - 1);
	int fftsize = fftH * (fftW / 2 + 1) * sizeof(fComplex);
	int paddedsize = fftH * fftW * sizeof(float);
	int datasize = dataH * dataW * sizeof(float);
	int kernelsize = kernelH * kernelW * sizeof(float);
	cufftHandle fftPlanFwd, fftPlanInv;

	float *d_Data, *d_PaddedData, *d_Kernel, *d_PaddedKernel;
	fComplex *d_KernelSpectrum, *d_DataSpectrum;

	// allocate device memory
	checkCudaErrors(cudaMalloc((void **)&d_Data, datasize));
	checkCudaErrors(cudaMalloc((void **)&d_PaddedData, paddedsize));
	checkCudaErrors(cudaMalloc((void **)&d_DataSpectrum, fftsize));
	checkCudaErrors(cudaMalloc((void **)&d_Kernel, kernelsize));
	checkCudaErrors(cudaMalloc((void **)&d_PaddedKernel, paddedsize));
	checkCudaErrors(cudaMalloc((void **)&d_KernelSpectrum, fftsize));

	// create cufft plans
	checkCudaErrors(cufftPlan2d(&fftPlanFwd, fftH, fftW, CUFFT_R2C));
	checkCudaErrors(cufftPlan2d(&fftPlanInv, fftH, fftW, CUFFT_C2R));

	// copy kernel to device and pad
	checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel, kernelsize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(d_PaddedKernel, 0, paddedsize));
	padKernel(d_PaddedKernel, d_Kernel, fftH, fftW, kernelH, kernelW, 0); // default stream

	// copy the data and pad it
	checkCudaErrors(cudaMemset(d_PaddedData, 0, paddedsize));
	checkCudaErrors(cudaMemcpy(d_Data, h_Data, datasize, cudaMemcpyHostToDevice));
	padDataClampToBorder32(d_PaddedData, d_Data, fftH, fftW, dataH, dataW, kernelH, kernelW, 0);

	// forward FFT
	checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_PaddedKernel, (cufftComplex *)d_KernelSpectrum));
	checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_PaddedData,   (cufftComplex *)d_DataSpectrum));
	modulateAndNormalize(d_DataSpectrum, d_KernelSpectrum, fftH, fftW, 1, 0);
	checkCudaErrors(cufftExecC2R(fftPlanInv, (cufftComplex *)d_DataSpectrum, (cufftReal *)d_PaddedData));

	// copy result back to host
	unpadData32(d_Data, d_PaddedData, fftH, fftW, dataH, dataW, 0);

	// D2H
	checkCudaErrors(cudaMemcpy(h_Data, d_Data, datasize, cudaMemcpyDeviceToHost));
}

static float **
computeKernel2(float const * const *h_Kernel, int kernelW, int kernelH, int nViews, fmvd_psf_type iteration_type)
{
	int v, w, kernelsize;
	kernelsize = kernelH * kernelW * sizeof(float);
	float **kernel2 = (float **)malloc(nViews * sizeof(float *));
	float *tmp1 = (float *)malloc(kernelsize);
	float *tmp2 = (float *)malloc(kernelsize);
	for(v = 0; v < nViews; v++) {
		kernel2[v] = (float *)malloc(kernelsize);

		switch(iteration_type) {
			case independent:
				computeInvertedKernel(h_Kernel[v], kernel2[v], kernelW, kernelH);
				break;
			case efficient_bayesian:
				// compute the compound kernel P_v^compound of the efficient bayesian multi-view deconvolution
				// for the current view \phi_v(x_v)
				// P_v^compound = P_v^{*} prod{w \in W_v} P_v^{*} \ast P_w \ast P_w^{*}
				computeInvertedKernel(h_Kernel[v], kernel2[v], kernelW, kernelH);
				for(w = 0; w < nViews; w++) {
					if(w != v) {
						// convolve first P_v^{*} with P_w
						computeInvertedKernel(h_Kernel[v], tmp1, kernelW, kernelH);
						convolve_single_plane(tmp1, kernelW, kernelH, h_Kernel[w], kernelW, kernelH);
						// now convolve the result with P_w^{*}
						computeInvertedKernel(h_Kernel[w], tmp2, kernelW, kernelH);
						convolve_single_plane(tmp1, kernelW, kernelH, tmp2, kernelW, kernelH);
						// multiply with P_v^{*} yielding the compound kernel
						for(int i = 0; i < kernelW * kernelH; i++)
							kernel2[v][i] *= tmp1[i];
					}
				}
				break;
			case optimization_1:
				// compute the simplified compound kernel P_v^compound of the efficient bayesian multi-view deconvolution
				// for the current view \phi_v(x_v)
				// P_v^compound = P_v^{*} prod{w \in W_v} P_v^{*} \ast P_w
				// we first get P_v^{*} -> {*} refers to the inverted coordinates
				computeInvertedKernel(h_Kernel[v], kernel2[v], kernelW, kernelH);
				for(w = 0; w < nViews; w++) {
					if(w != v) {
						// convolve first P_v^{*} with P_w
						computeInvertedKernel(h_Kernel[v], tmp1, kernelW, kernelH);
						convolve_single_plane(tmp1, kernelW, kernelH, h_Kernel[w], kernelW, kernelH);
						// multiply with P_v^{*} yielding the compound kernel
						for(int i = 0; i < kernelW * kernelH; i++)
							kernel2[v][i] *= tmp1[i];
					}
				}
				break;
			case optimization_2:
				// compute the squared kernel and its inverse
				float *expKernel = (float *)malloc(kernelsize);
				computeExponentialKernel(h_Kernel[v], expKernel, kernelW, kernelH, nViews);
				computeInvertedKernel(expKernel, kernel2[v], kernelW, kernelH);
				free(expKernel);
				break;
		}
		normalize(kernel2[v], kernelW * kernelH);
	}
	free(tmp1);
	free(tmp2);

	return kernel2;
}

static void
prepareKernels(const struct fmvd_plan_cuda *plan, float **h_Kernel, fmvd_psf_type iteration_type)
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
prepareWeights(const struct fmvd_plan_cuda *plan, data_t const* const* h_Weights)
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


/*************************
 * Public API
 *************************/
struct fmvd_plan_cuda *
fmvd_initialize_cuda(
		int dataH,
		int dataW,
		data_t const* const* h_Weights,
		float **h_Kernel,
		int kernelH,
		int kernelW,
		fmvd_psf_type iteration_type,
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

	prepareKernels(plan, h_Kernel, iteration_type);
	prepareWeights(plan, h_Weights);
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
			padData(plan, v, stream_idx);
		}

		for(it = 0; it < iterations; it++) {
			for(v = 0; v < plan->nViews; v++) {
				checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_estimate, (cufftComplex *)d_estimateSpectrum));
				modulateAndNormalize(d_estimateSpectrum, plan->d_KernelSpectrum[v], fftH, fftW, 1, stream);
				checkCudaErrors(cufftExecC2R(fftPlanInv, (cufftComplex *)d_estimateSpectrum, (cufftReal *)d_tmp));
				divide16(plan->d_PaddedData[v] + fftOffset, d_tmp, d_tmp, fftH, fftW, stream);

				checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_tmp, (cufftComplex *)d_estimateSpectrum));
				modulateAndNormalize(d_estimateSpectrum, plan->d_KernelHatSpectrum[v], fftH, fftW, 1, stream);
				checkCudaErrors(cufftExecC2R(fftPlanInv, (cufftComplex *)d_estimateSpectrum, (cufftReal *)d_tmp));
				multiply32(d_estimate, d_tmp, plan->d_PaddedWeights[v], d_estimate, fftW, fftH, stream);
			}
		}

		unpadData16(plan->d_Data[0] + dataOffset, d_estimate, fftH, fftW, plan->dataH, plan->dataW, stream);

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

