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

#define checkCudaErrors(ans) {gpuAssert((ans), __FILE__, __LINE__); }
#define checkCufftErrors(ans) {gpuAssert((ans), __FILE__, __LINE__); }

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
void gpuAssert(unsigned int code, const char *file, int line, bool abort=true)
{
	if(code != cudaSuccess) {
		fprintf(stderr, "GPUAssert: error %d %s %d\n", code, file, line);
	if(abort)
		exit(code);
	}
}

int snapTransformSize(int dataSize)
{
	int hiBit;
	unsigned int lowPOT, hiPOT;

	dataSize = iAlignUp(dataSize, 16);

	for (hiBit = 31; hiBit >= 0; hiBit--)
		if (dataSize & (1U << hiBit))
		{
			break;
		}

	lowPOT = 1U << hiBit;

	if (lowPOT == (unsigned int)dataSize)
	{
		return dataSize;
	}

	hiPOT = 1U << (hiBit + 1);

	if (hiPOT <= 1024)
	{
		return hiPOT;
	}
	else
	{
		return iAlignUp(dataSize, 512);
	}
}

float getRand(void)
{
	return (float)(rand() % 16);
}

void prepareKernels(
	float **h_Kernel,
	int kernelH,
	int kernelW,
	int kernelY,
	int kernelX,
	int fftH,
	int fftW,
	int nViews,
	fComplex **d_KernelSpectrum,
	fComplex **d_KernelHatSpectrum,
	cufftHandle fftPlanFwd
)
{
	int v;
	// printf("...uploading to GPU and padding convolution kernel\n");
	float *d_Kernel, *d_PaddedKernel, *d_PaddedKernelHat;
	checkCudaErrors(cudaMalloc((void **)&d_Kernel,		 kernelH * kernelW		* sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_PaddedKernel,   fftH	* fftW		   * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_PaddedKernelHat,fftH	* fftW		   * sizeof(float)));
	for(v = 0; v < nViews; v++) {
		checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel[v], kernelH * kernelW * sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(float)));
		checkCudaErrors(cudaMemset(d_PaddedKernelHat, 0, fftH * fftW * sizeof(float)));

		padKernel(
			d_PaddedKernel,
			d_PaddedKernelHat,
			d_Kernel,
			fftH,
			fftW,
			kernelH,
			kernelW,
			kernelY,
			kernelX,
			0 // default stream
		);

		checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_PaddedKernel, (cufftComplex *)d_KernelSpectrum[v]));
		checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_PaddedKernelHat, (cufftComplex *)d_KernelHatSpectrum[v]));

	}
	checkCudaErrors(cudaFree(d_Kernel));
	checkCudaErrors(cudaFree(d_PaddedKernel));
	checkCudaErrors(cudaFree(d_PaddedKernelHat));
}

void normalizeMinMax(float *data, int len, float min, float max)
{
	int i;
	float *k = data;
	for(i = 0; i < len; i++) {
		float v = *k;
		v = (v - min) / (max - min);

		if(v < 0) v = 0;
		if(v > 1) v = 1;
		(*k) = v;
		k++;
	}
}


void deconvolve(FILE **dataFiles, FILE *resultFile, int dataW, int dataH, int dataD, float **h_Kernel, int kernelH, int kernelW, int nViews, int iterations)
{
	long start, stop;
	int v, z, it, stream, nStreams = 3;
	float
	*h_ResultGPU;
	
	cudaStream_t *streams;

	cufftHandle
	*fftPlanFwd,
	*fftPlanInv;

	const int kernelY = kernelH / 2;
	const int kernelX = kernelW / 2;
	const int    fftH = snapTransformSize(dataH + kernelH - 1);
	const int    fftW = snapTransformSize(dataW + kernelW - 1);

	// printf("...creating R2C & C2R FFT plans for %i x %i\n", fftH, fftW);

	fftPlanFwd = (cufftHandle *)malloc(nStreams * sizeof(cufftHandle));
	fftPlanInv = (cufftHandle *)malloc(nStreams * sizeof(cufftHandle));
	streams	= (cudaStream_t *)malloc(nStreams * sizeof(cudaStream_t));

	for(stream = 0; stream < nStreams; stream++) {
		cudaStreamCreate(&streams[stream]);
		checkCudaErrors(cufftPlan2d(&fftPlanFwd[stream], fftH, fftW, CUFFT_R2C));
		checkCudaErrors(cufftSetStream(fftPlanFwd[stream], streams[stream]));
		checkCudaErrors(cufftPlan2d(&fftPlanInv[stream], fftH, fftW, CUFFT_C2R));
		checkCudaErrors(cufftSetStream(fftPlanInv[stream], streams[stream]));
	}

	float	**h_Data               = (float **)   malloc(nViews * sizeof(float *));
	float	**d_Data               = (float **)   malloc(nViews * sizeof(float *));
	float	**d_PaddedData         = (float **)   malloc(nViews * sizeof(float *));
	fComplex **d_KernelSpectrum    = (fComplex **)malloc(nViews * sizeof(fComplex *));
	fComplex **d_KernelHatSpectrum = (fComplex **)malloc(nViews * sizeof(fComplex *));

	float *d_estimate, *d_tmp;
	fComplex *d_estimateSpectrum;

	checkCudaErrors(cudaMalloc((void **)&d_estimate,         nStreams * fftH * fftW	          * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_tmp,              nStreams * fftH * fftW	          * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_estimateSpectrum, nStreams * fftH * (fftW / 2 + 1) * sizeof(fComplex)));

	for(v = 0; v < nViews; v++) {
		checkCudaErrors(cudaMalloc((void **)&d_Data[v],	             nStreams * dataH * dataW            * sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_PaddedData[v],        nStreams * fftH	* fftW           * sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_KernelSpectrum[v],    nStreams * fftH	* (fftW / 2 + 1) * sizeof(fComplex)));
		checkCudaErrors(cudaMalloc((void **)&d_KernelHatSpectrum[v], nStreams * fftH	* (fftW / 2 + 1) * sizeof(fComplex)));
		checkCudaErrors(cudaMallocHost(&h_Data[v], dataD * dataW * dataH * sizeof(float)));
	}

	prepareKernels(h_Kernel, kernelH, kernelW, kernelY, kernelX, fftH, fftW, nViews, d_KernelSpectrum, d_KernelHatSpectrum, fftPlanFwd[0]);


	start = GetTickCount();

	// read the first nStreams planes from file
	for(v = 0; v < nViews; v++) {
		size_t bytes = fread(h_Data[v], sizeof(float), nStreams * dataW * dataH, dataFiles[v]);
		normalizeMinMax(h_Data[v], nStreams * dataW * dataH, 40, 62000);
		printf("read %d bytes\n", bytes);
	}

	for(z = 0; z < dataD; z += nStreams) {
		int maxStreams = dataD - z < nStreams ? dataD - z : nStreams;
		for(stream = 0; stream < maxStreams; stream++) {
			int fftOffset	  = stream * fftW * fftH;
			int spectrumOffset = stream * (fftW / 2 + 1) * fftH;
			int dataOffset	 = stream * dataW * dataH;

			// load from HDD
			if(z > 0) {
				printf("z %d (stream %d): waiting for stream %d\n", z, stream, stream);
				checkCudaErrors(cudaStreamSynchronize(streams[stream]));
				fwrite(h_Data[0] + dataOffset, sizeof(float), dataW * dataH, resultFile);
				for(v = 0; v < nViews; v++) {
					fread(h_Data[v] + dataOffset, sizeof(float), dataW * dataH, dataFiles[v]);
					normalizeMinMax(h_Data[v] + dataOffset, dataW * dataH, 40, 62000);
				}
			}

			// H2D
			for(v = 0; v < nViews; v++) {
				// load data from file and upload it to GPU
				checkCudaErrors(cudaMemcpyAsync(
						d_Data[v] + dataOffset,
						h_Data[v] + dataOffset,
						dataH * dataW * sizeof(float),
						cudaMemcpyHostToDevice, streams[stream]));
			}
			checkCudaErrors(cudaMemsetAsync(d_estimate + fftOffset, 0, fftH * fftW * sizeof(float), streams[stream]));
			for(v = 0; v < nViews; v++) {
				checkCudaErrors(cudaMemsetAsync(
						d_PaddedData[v] + fftOffset,
						0,
						fftH * fftW * sizeof(float),
						streams[stream]));
				padDataClampToBorder(
					d_estimate + fftOffset,
					d_PaddedData[v] + fftOffset,
					d_Data[v] + dataOffset,
					fftH,
					fftW,
					dataH,
					dataW,
					kernelH,
					kernelW,
					kernelY,
					kernelX,
					nViews,
					streams[stream]
				);
			}
			for(it = 0; it < iterations; it++) {
				for(v = 0; v < nViews; v++) {
					checkCudaErrors(cufftExecR2C(fftPlanFwd[stream], (cufftReal *)d_estimate + fftOffset, (cufftComplex *)d_estimateSpectrum + spectrumOffset));
					modulateAndNormalize(d_estimateSpectrum + spectrumOffset, d_KernelSpectrum[v], fftH, fftW, 1, streams[stream]);
					checkCudaErrors(cufftExecC2R(fftPlanInv[stream], (cufftComplex *)d_estimateSpectrum + spectrumOffset, (cufftReal *)d_tmp + fftOffset));
					divide(d_PaddedData[v] + fftOffset, d_tmp + fftOffset, d_tmp + fftOffset, fftH, fftW, streams[stream]);
			
					checkCudaErrors(cufftExecR2C(fftPlanFwd[stream], (cufftReal *)d_tmp + fftOffset, (cufftComplex *)d_estimateSpectrum + spectrumOffset));
					modulateAndNormalize(d_estimateSpectrum + spectrumOffset, d_KernelHatSpectrum[v], fftH, fftW, 1, streams[stream]);
					checkCudaErrors(cufftExecC2R(fftPlanInv[stream], (cufftComplex *)d_estimateSpectrum + spectrumOffset, (cufftReal *)d_tmp + fftOffset));
					mul(d_estimate + fftOffset, d_tmp + fftOffset, d_estimate + fftOffset, fftW, fftH, streams[stream]);
				}
			}
			unpadData(d_Data[0] + dataOffset, d_estimate + fftOffset, fftH, fftW, dataH, dataW, streams[stream]);

			// D2H
			checkCudaErrors(cudaMemcpyAsync(
					h_Data[0] + dataOffset,
			d_Data[0] + dataOffset,
			dataH * dataW * sizeof(float),
			cudaMemcpyDeviceToHost,
			streams[stream]));

	}
	}
	
	stream = stream % nStreams;

	// save the remaining nStreams planes
	for(z = 0; z < nStreams; z++) {
		int fftOffset	  = stream * fftW * fftH;
		int spectrumOffset = stream * (fftW / 2 + 1) * fftH;
		int dataOffset	   = stream * dataW * dataH;

		printf("z %d (stream %d): waiting for stream %d\n", z, stream, stream);
		checkCudaErrors(cudaStreamSynchronize(streams[stream]));
		fwrite(h_Data[0] + dataOffset, sizeof(float), dataW * dataH, resultFile);
		stream = (stream + 1) % nStreams;
	}

	checkCudaErrors(cudaDeviceSynchronize());

	stop = GetTickCount();
	printf("Overall time: %d ms\n", (stop - start));

	for(v = 0; v < nViews; v++) {
		checkCudaErrors(cudaFreeHost(h_Data[v]));
		checkCudaErrors(cudaFree(d_Data[v]));
		checkCudaErrors(cudaFree(d_PaddedData[v]));
		checkCudaErrors(cudaFree(d_KernelSpectrum[v]));
		checkCudaErrors(cudaFree(d_KernelHatSpectrum[v]));
	}
	checkCudaErrors(cudaFree(d_estimate));
	checkCudaErrors(cudaFree(d_estimateSpectrum));
	checkCudaErrors(cudaFree(d_tmp));

	free(h_Data);
	free(d_Data);
	free(d_PaddedData);
	free(d_KernelSpectrum);
	free(d_KernelHatSpectrum);

	printf("...shutting down\n");

	for(stream = 0; stream < nStreams; stream++) {
		checkCudaErrors(cufftDestroy(fftPlanInv[stream]));
		checkCudaErrors(cufftDestroy(fftPlanFwd[stream]));
		checkCudaErrors(cudaStreamDestroy(streams[stream]));
	}
	free(streams);
	free(fftPlanFwd);
	free(fftPlanInv);
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

void normalize(float *kernel, int len)
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
		*k /= sum;
		k++;
	}
}

void normalizeRange(float *kernel, int len)
{
	int i;
	float *k = kernel;
	float min = kernel[0];
	float max = kernel[0];
	k++;
	for(i = 1; i < len; i++) {
		float v = *k;
		if(v < min) min = v;
		if(v > max) max = v;
		k++;
	}
	k = kernel;
	for(i = 0; i < len; i++) {
		float v = *k;
		v = (v - min) / (max - min);
		(*k) = v;
		k++;
	}
}

int main(int argc, char **argv)
{
	int it;
	printf("[%s] - Starting...\n", argv[0]);

	const int W = 600;
	const int H = 600;
	const int D = 600;

	const int KW = 83;
	const int KH = 83;

	float **data = (float **)malloc(2 * sizeof(float *));
	cudaMallocHost(&data[0], W * H * sizeof(float));
	cudaMallocHost(&data[1], W * H * sizeof(float));

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
		// fseek(dataFiles[0], 300 * W * H * sizeof(float), SEEK_SET);
		// fseek(dataFiles[1], 300 * W * H * sizeof(float), SEEK_SET);

		sprintf(path, "E:\\SPIM5_Deconvolution\\m6\\cropped\\forplanewisedeconv\\v35.deconvolved.gpu.raw");
		FILE *resultFile = fopen(path, "wb");

		deconvolve(dataFiles, resultFile, W, H, D, kernel, KH, KW, 2, it);

		fclose(dataFiles[0]);
		fclose(dataFiles[1]);
		fclose(resultFile);
	}

	cudaFreeHost(data[0]);
	cudaFreeHost(data[1]);
	free(data);
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

