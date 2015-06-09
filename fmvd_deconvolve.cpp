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

#define SAMPLE              unsigned short
#define BITS_PER_SAMPLE     16 
#include "fmvd_deconvolve.impl.cpp"

#define SAMPLE              unsigned char
#define BITS_PER_SAMPLE     8
#include "fmvd_deconvolve.impl.cpp"

