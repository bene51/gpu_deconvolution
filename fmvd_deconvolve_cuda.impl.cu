#include "fmvd_deconvolve_cuda.cuh"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fmvd_cuda_utils.h"

#define PASTE(name, type) name ## _ ## type
#define EVAL(name, type) PASTE(name, type)
#define MAKE_NAME(name) EVAL(name, BITS_PER_SAMPLE)


__global__ void
MAKE_NAME(padDataClampToBorderAndInitialize_kernel)(
		float *d_estimate,
		SAMPLE *d_PaddedData,
		SAMPLE *d_Data,
		float *d_Weights,
		int fftH,
		int fftW,
		int dataH,
		int dataW,
		int kernelH,
		int kernelW,
		int kernelY,
		int kernelX
		)
{
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int borderH = dataH + kernelY;
	const int borderW = dataW + kernelX;

	if (y < fftH && x < fftW) {
		int dy, dx, idx;
		SAMPLE v;

		if (y < dataH)
			dy = y;

		if (x < dataW)
			dx = x;

		if (y >= dataH && y < borderH)
			dy = dataH - 1;

		if (x >= dataW && x < borderW)
			dx = dataW - 1;

		if (y >= borderH)
			dy = 0;

		if (x >= borderW)
			dx = 0;

		v = d_Data[dy * dataW + dx];
		idx = y * fftW + x;
		d_PaddedData[idx] = v;
		// float change = v * d_Weights[idx];
		// d_estimate[idx] += change;
		d_estimate[idx] += 10;
	}
}

void
MAKE_NAME(padDataClampToBorderAndInitialize)(
		float *d_estimate,
		SAMPLE *d_Dst,
		SAMPLE *d_Src,
		float *d_Weights,
		int fftH,
		int fftW,
		int dataH,
		int dataW,
		int kernelW,
		int kernelH,
		cudaStream_t stream
		)
{
	assert(d_Src != d_Dst);
	dim3 threads(32, 8);
	dim3 grid(
			iDivUp(fftW, threads.x),
			iDivUp(fftH, threads.y));

	const int kernelY = kernelH / 2;
	const int kernelX = kernelW / 2;

	MAKE_NAME(padDataClampToBorderAndInitialize_kernel)<<<grid, threads, 0, stream>>>(
			d_estimate,
			d_Dst,
			d_Src,
			d_Weights,
			fftH,
			fftW,
			dataH,
			dataW,
			kernelH,
			kernelW,
			kernelY,
			kernelX
			);
	getLastCudaError("padDataClampToBorderAndInitialize_kernel<<<>>> execution failed\n");
}

__global__ void
MAKE_NAME(unpadData_kernel)(
		SAMPLE *d_Data,
		float *d_PaddedData,
		int fftH,
		int fftW,
		int dataH,
		int dataW,
		SAMPLE max
		)
{
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;

	// TODO round
	if (y < dataH && x < dataW) {
		float v = d_PaddedData[y * fftW + x];
		if(v > max)
			v = max;
		d_Data[y * dataW + x] = (SAMPLE)v;
	}
}

extern "C" void
MAKE_NAME(unpadData)(
		SAMPLE *d_Data,
		float *d_PaddedData,
		int fftH,
		int fftW,
		int dataH,
		int dataW,
		cudaStream_t stream
		)
{
	dim3 threads(32, 8);
	dim3 grid(
			iDivUp(dataW, threads.x),
			iDivUp(dataH, threads.y));
	SAMPLE max = (1 << (sizeof(SAMPLE) * 8)) - 1;

	MAKE_NAME(unpadData_kernel)<<<grid, threads, 0, stream>>>(
			d_Data,
			d_PaddedData,
			fftH,
			fftW,
			dataH,
			dataW,
			max
			);
	getLastCudaError("unpadData_kernel<<<>>> execution failed\n");
}

__global__ void
MAKE_NAME(divide_kernel)(
		SAMPLE *d_a,
		float *d_b,
		float *d_dest,
		int dataSize
		)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	float q;

	if (i >= dataSize)
		return;

	q = d_b[i];
	if(q > 1)
		d_dest[i] = d_a[i] / q;
}

extern "C" void
MAKE_NAME(divide)(
		SAMPLE *d_a,
		float *d_b,
		float *d_dest,
		int fftH,
		int fftW,
		cudaStream_t stream
		)
{
	const int dataSize = fftH * fftW;

	MAKE_NAME(divide_kernel)<<<iDivUp(dataSize, 256), 256, 0, stream>>>(
			d_a,
			d_b,
			d_dest,
			dataSize
			);
	getLastCudaError("divide_kernel<<<>>> execution failed\n");
}

