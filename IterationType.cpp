#include "IterationType.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include "CudaUtils.h"
#include "DeconvolveKernels.cuh"

void
IterationType::normalizeKernel(float *kernel, int len)
{
	double sum = 0;
	float *k = kernel;
	for(int i = 0; i < len; i++) {
		sum += *k;
		k++;
	}
	k = kernel;
	for(int i = 0; i < len; i++) {
		*k /= (float)sum;
		k++;
	}
}

void
IterationType::computeInvertedKernel(
		const float *kernel, int kw, int kh, float *out)
{
	for(int y = 0; y < kh; y++) {
		for(int x = 0; x < kw; x++) {
			int idx = (kh - y - 1) * kw + (kw - x - 1);
			out[y * kw + x] = kernel[idx];
		}
	}
}

void
IterationType::computeExponentialKernel(
		const float *kernel, int kw, int kh, int exponent, float *out)
{
	int wh = kw * kh;
	for(int i = 0; i < wh; i++)
		out[i] = (float)pow(kernel[i], exponent);
}

void
IterationType::convolveSinglePlane(
		float *data, int dataW, int dataH,
	       	const float *kernel, int kernelW, int kernelH)
{
	int fftH = snapTransformSize(dataH + kernelH - 1);
	int fftW = snapTransformSize(dataW + kernelW - 1);
	int fftsize = fftH * (fftW / 2 + 1) * sizeof(fComplex);
	int paddedsize = fftH * fftW * sizeof(float);
	int datasize = dataH * dataW * sizeof(float);
	int kernelsize = kernelH * kernelW * sizeof(float);
	cufftHandle fftPlanFwd, fftPlanInv;

	float *d_data, *d_paddedData, *d_kernel, *d_paddedKernel;
	fComplex *d_kernelSpectrum, *d_dataSpectrum;

	// allocate device memory
	checkCudaErrors(cudaMalloc((void **)&d_data, datasize));
	checkCudaErrors(cudaMalloc((void **)&d_paddedData, paddedsize));
	checkCudaErrors(cudaMalloc((void **)&d_dataSpectrum, fftsize));
	checkCudaErrors(cudaMalloc((void **)&d_kernel, kernelsize));
	checkCudaErrors(cudaMalloc((void **)&d_paddedKernel, paddedsize));
	checkCudaErrors(cudaMalloc((void **)&d_kernelSpectrum, fftsize));

	// create cufft plans
	checkCudaErrors(cufftPlan2d(&fftPlanFwd, fftH, fftW, CUFFT_R2C));
	checkCudaErrors(cufftPlan2d(&fftPlanInv, fftH, fftW, CUFFT_C2R));

	// copy kernel to device and pad
	checkCudaErrors(cudaMemcpy(d_kernel, kernel, kernelsize,
				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(d_paddedKernel, 0, paddedsize));
	padKernel(d_paddedKernel, d_kernel, fftH, fftW, kernelH, kernelW, 0);

	// copy the data and pad it
	checkCudaErrors(cudaMemset(d_paddedData, 0, paddedsize));
	checkCudaErrors(cudaMemcpy(d_data, data, datasize,
				cudaMemcpyHostToDevice));
	padDataClampToBorder32(d_paddedData, d_data,
			fftH, fftW, dataH, dataW,
			kernelH, kernelW, 0);

	// forward FFT
	checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_paddedKernel,
				(cufftComplex *)d_kernelSpectrum));
	checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_paddedData,
				(cufftComplex *)d_dataSpectrum));
	modulateAndNormalize(d_dataSpectrum, d_kernelSpectrum,
				fftH, fftW, 1, 0);
	checkCudaErrors(cufftExecC2R(fftPlanInv, (cufftComplex *)d_dataSpectrum,
				(cufftReal *)d_paddedData));

	// copy result back to host
	unpadData32(d_data, d_paddedData, fftH, fftW, dataH, dataW, 0);

	// D2H
	checkCudaErrors(cudaMemcpy(data, d_data, datasize, cudaMemcpyDeviceToHost));
}

float **
IterationType::createKernelHat(
		const float * const * kernel, int kernelW, int kernelH,
		int nViews, Type type)
{
	int kernelWH = kernelW * kernelH;
	float **kernel2 = new float*[nViews];
	// float **kernel2 = (float **)malloc(nViews * sizeof(float *));
	float *tmp1 = new float[kernelWH];
	float *tmp2 = new float[kernelWH];
	for(int v = 0; v < nViews; v++) {
		kernel2[v] = new float[kernelWH];
		switch(type) {
			case INDEPENDENT:
				computeInvertedKernel(kernel[v], kernelW, kernelH, kernel2[v]);
				break;
			case EFFICIENT_BAYESIAN:
				// compute the compound kernel P_v^compound of the efficient bayesian multi-view deconvolution
				// for the current view \phi_v(x_v)
				// P_v^compound = P_v^{*} prod{w \in W_v} P_v^{*} \ast P_w \ast P_w^{*}
				computeInvertedKernel(kernel[v], kernelW, kernelH, kernel2[v]);
				for(int w = 0; w < nViews; w++) {
					if(w != v) {
						// convolve first P_v^{*} with P_w
						computeInvertedKernel(kernel[v], kernelW, kernelH, tmp1);
						convolveSinglePlane(tmp1, kernelW, kernelH, kernel[w], kernelW, kernelH);
						// now convolve the result with P_w^{*}
						computeInvertedKernel(kernel[w], kernelW, kernelH, tmp2);
						convolveSinglePlane(tmp1, kernelW, kernelH, tmp2, kernelW, kernelH);
						// multiply with P_v^{*} yielding the compound kernel
						for(int i = 0; i < kernelW * kernelH; i++)
							kernel2[v][i] *= tmp1[i];
					}
				}
				break;
			case OPTIMIZATION_1:
				// compute the simplified compound kernel P_v^compound of the efficient bayesian multi-view deconvolution
				// for the current view \phi_v(x_v)
				// P_v^compound = P_v^{*} prod{w \in W_v} P_v^{*} \ast P_w
				// we first get P_v^{*} -> {*} refers to the inverted coordinates
				computeInvertedKernel(kernel[v], kernelW, kernelH, kernel2[v]);
				for(int w = 0; w < nViews; w++) {
					if(w != v) {
						// convolve first P_v^{*} with P_w
						computeInvertedKernel(kernel[v], kernelW, kernelH, tmp1);
						convolveSinglePlane(tmp1, kernelW, kernelH, kernel[w], kernelW, kernelH);
						// multiply with P_v^{*} yielding the compound kernel
						for(int i = 0; i < kernelW * kernelH; i++)
							kernel2[v][i] *= tmp1[i];
					}
				}
				break;
			case OPTIMIZATION_2:
				// compute the squared kernel and its inverse
				float *expKernel = new float[kernelWH];
				computeExponentialKernel(kernel[v], kernelW, kernelH, nViews, expKernel);
				computeInvertedKernel(expKernel, kernelW, kernelH, kernel2[v]);
				delete expKernel;
				break;
		}
		normalizeKernel(kernel2[v], kernelW * kernelH);
	}
	delete tmp1;
	delete tmp2;

	return kernel2;
}

