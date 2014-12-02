#include "fmvd_deconvolve_cuda.cuh"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fmvd_utils.h"

/**
 * Resize the kernel to fftW and fftH, padding it with zeros and
 * positioning it such that its center is at (0, 0).
 */
__global__ void padKernel_kernel(
		float *d_PaddedKernel,
		float *d_Kernel,
		int fftH,
		int fftW,
		int kernelH,
		int kernelW,
		int kernelY,
		int kernelX
		)
{
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;

	if (y < kernelH && x < kernelW) {
		int ky = y - kernelY;
		if (ky < 0)
			ky += fftH;

		int kx = x - kernelX;
		if (kx < 0)
			kx += fftW;

		d_PaddedKernel[ky * fftW + kx] = d_Kernel[y * kernelW + x];
	}
}

extern "C" void padKernel(
		float *d_Dst,
		float *d_Src,
		int fftH,
		int fftW,
		int kernelH,
		int kernelW,
		cudaStream_t stream
		)
{
	assert(d_Src != d_Dst);
	dim3 threads(32, 8);
	dim3 grid(iDivUp(kernelW, threads.x), iDivUp(kernelH, threads.y));

	const int kernelY = kernelH / 2;
	const int kernelX = kernelW / 2;

	padKernel_kernel<<<grid, threads, 0, stream>>>(
			d_Dst,
			d_Src,
			fftH,
			fftW,
			kernelH,
			kernelW,
			kernelY,
			kernelX
			);
	getLastCudaError("padKernel_kernel<<<>>> execution failed\n");
}


__global__ void padWeights_kernel(
		float *d_PaddedWeights,
		float *d_PaddedWeightSums,
		data_t *d_Weights,
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

	if (y < fftH && x < fftW)
	{
		int dy, dx, idx;
		data_t v;

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

		v = d_Weights[dy * dataW + dx];
		idx = y * fftW + x;

		d_PaddedWeights[idx] = (float)v;
		d_PaddedWeightSums[idx] += (float)v;
	}
}

extern "C" void padWeights(
		float *d_PaddedWeights,
		float *d_PaddedWeightSums,
		data_t *d_Weights,
		int fftH,
		int fftW,
		int dataH,
		int dataW,
		int kernelH,
		int kernelW,
		cudaStream_t stream
		)
{
	dim3 threads(32, 8);
	dim3 grid(
			iDivUp(fftW, threads.x),
			iDivUp(fftH, threads.y));

	const int kernelY = kernelH / 2;
	const int kernelX = kernelW / 2;

	padWeights_kernel<<<grid, threads, 0, stream>>>(
			d_PaddedWeights,
			d_PaddedWeightSums,
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
	getLastCudaError("padWeights<<<>>> execution failed\n");
}


__global__ void normalizeWeights_kernel(
		float *d_PaddedWeights,
		float *d_PaddedWeightSums,
		int fftH,
		int fftW
		)
{
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;

	if (y < fftH && x < fftW)
	{
		int idx = y * fftW + x;
		float d = d_PaddedWeightSums[idx];
		if(d > 0)
			d_PaddedWeights[idx] /= d;
	}
}

extern "C" void normalizeWeights(
		float *d_PaddedWeights,
		float *d_PaddedWeightSums,
		int fftH,
		int fftW,
		cudaStream_t stream
		)
{
	dim3 threads(32, 8);
	dim3 grid(
			iDivUp(fftW, threads.x),
			iDivUp(fftH, threads.y));

	normalizeWeights_kernel<<<grid, threads, 0, stream>>>(
			d_PaddedWeights,
			d_PaddedWeightSums,
			fftH,
			fftW
			);
	getLastCudaError("normalizeWeights_kernel<<<>>> execution failed\n");
}

__global__ void padDataClampToBorderAndInitialize16_kernel(
		float *d_estimate,
		data_t *d_PaddedData,
		data_t *d_Data,
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
		data_t v;

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
		d_estimate[idx] += 100;
	}
}

void padDataClampToBorderAndInitialize16(
		float *d_estimate,
		data_t *d_Dst,
		data_t *d_Src,
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

	padDataClampToBorderAndInitialize16_kernel<<<grid, threads, 0, stream>>>(
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
	getLastCudaError("padDataClampToBorderAndInitialize16_kernel<<<>>> execution failed\n");
}

__global__ void padDataClampToBorder32_kernel(
		float *d_PaddedData,
		float *d_Data,
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
		int dy, dx;

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

		d_PaddedData[y * fftW + x] = d_Data[dy * dataW + dx];
	}
}

extern "C" void padDataClampToBorder32(
		float *d_PaddedData,
		float *d_Data,
		int fftH,
		int fftW,
		int dataH,
		int dataW,
		int kernelH,
		int kernelW,
		cudaStream_t stream
		)
{
	assert(d_PaddedData != d_Data);
	dim3 threads(32, 8);
	dim3 grid(
			iDivUp(fftW, threads.x),
			iDivUp(fftH, threads.y));

	const int kernelY = kernelH / 2;
	const int kernelX = kernelW / 2;

	padDataClampToBorder32_kernel<<<grid, threads, 0, stream>>>(
			d_PaddedData,
			d_Data,
			fftH,
			fftW,
			dataH,
			dataW,
			kernelH,
			kernelW,
			kernelY,
			kernelX
			);
	getLastCudaError("padDataClampToBorder32_kernel<<<>>> execution failed\n");
}


__global__ void unpadData32_kernel(
		float *d_Data,
		float *d_PaddedData,
		int fftH,
		int fftW,
		int dataH,
		int dataW
		)
{
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;

	if (y < dataH && x < dataW)
		d_Data[y * dataW + x] = d_PaddedData[y * fftW + x];
}

extern "C" void unpadData32(
		float *d_Dst,
		float *d_Src,
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

	unpadData32_kernel<<<grid, threads, 0, stream>>>(
			d_Dst,
			d_Src,
			fftH,
			fftW,
			dataH,
			dataW
			);
	getLastCudaError("unpadData_kernel<<<>>> execution failed\n");
}

__global__ void unpadData16_kernel(
		data_t *d_Data,
		float *d_PaddedData,
		int fftH,
		int fftW,
		int dataH,
		int dataW
		)
{
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;

	// TODO round
	if (y < dataH && x < dataW)
		d_Data[y * dataW + x] = (data_t)d_PaddedData[y * fftW + x];
}

extern "C" void unpadData16(
		data_t *d_Data,
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

	unpadData16_kernel<<<grid, threads, 0, stream>>>(
			d_Data,
			d_PaddedData,
			fftH,
			fftW,
			dataH,
			dataW
			);
	getLastCudaError("unpadData16_kernel<<<>>> execution failed\n");
}


/**
 * Modulate Fourier image of padded data by Fourier image of padded kernel
 * and normalize by FFT size
 */
inline __device__ void mulAndScale(fComplex &a, const fComplex &b, const float &c)
{
	fComplex t = {c *(a.x * b.x - a.y * b.y), c *(a.y * b.x + a.x * b.y)};
	a = t;
}

__global__ void modulateAndNormalize_kernel(
		fComplex *d_Dst,
		fComplex *d_Src,
		int dataSize,
		float c
		)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= dataSize)
		return;

	fComplex a = d_Src[i];
	fComplex b = d_Dst[i];

	mulAndScale(a, b, c);

	d_Dst[i] = a;
}

extern "C" void modulateAndNormalize(
		fComplex *d_Dst,
		fComplex *d_Src,
		int fftH,
		int fftW,
		int padding,
		cudaStream_t stream
		)
{
	assert(fftW % 2 == 0);
	const int dataSize = fftH * (fftW / 2 + padding);

	modulateAndNormalize_kernel<<<iDivUp(dataSize, 256), 256, 0, stream>>>(
			d_Dst,
			d_Src,
			dataSize,
			1.0f / (float)(fftW *fftH)
			);
	getLastCudaError("modulateAndNormalize() execution failed\n");
}


__global__ void divide16_kernel(
		data_t *d_a,
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

extern "C" void divide16(
		data_t *d_a,
		float *d_b,
		float *d_dest,
		int fftH,
		int fftW,
		cudaStream_t stream
		)
{
	const int dataSize = fftH * fftW;

	divide16_kernel<<<iDivUp(dataSize, 256), 256, 0, stream>>>(
			d_a,
			d_b,
			d_dest,
			dataSize
			);
	getLastCudaError("divide16_kernel<<<>>> execution failed\n");
}

__global__ void multiply32_kernel(
		float *d_a,
		float *d_b,
		float *weights,
		float *d_dest,
		int dataSize
		)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= dataSize)
		return;

	float target = d_a[i] * d_b[i];
	float change = target - d_dest[i];
	float weight = weights[i];
	change *= weight;
	d_dest[i] += change;
}

extern "C" void multiply32(
		float *d_a,
		float *d_b,
		float *d_weights,
		float *d_dest,
		int fftH,
		int fftW,
		cudaStream_t stream
		)
{
	const int dataSize = fftH * fftW;

	multiply32_kernel<<<iDivUp(dataSize, 256), 256, 0, stream>>>(
			d_a,
			d_b,
			d_weights,
			d_dest,
			dataSize
			);
	getLastCudaError("multiply32_kernel<<<>>> execution failed\n");
}

