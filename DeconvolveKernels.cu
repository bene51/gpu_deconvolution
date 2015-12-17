#include "DeconvolveKernels.cuh"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CudaUtils.h"

/**
 * Resize the kernel to fftW and fftH, padding it with zeros and
 * positioning it such that its center is at (0, 0).
 */
__global__ void padKernel_kernel(
		float *paddedKernel,
		float *kernel,
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

		paddedKernel[ky * fftW + kx] = kernel[y * kernelW + x];
	}
}

void padKernel(
		float *paddedKernel,
		float *kernel,
		int fftH,
		int fftW,
		int kernelH,
		int kernelW,
		cudaStream_t stream
		)
{
	assert(paddedKernel != kernel);
	dim3 threads(32, 8);
	dim3 grid(iDivUp(kernelW, threads.x), iDivUp(kernelH, threads.y));

	const int kernelY = kernelH / 2;
	const int kernelX = kernelW / 2;

	padKernel_kernel<<<grid, threads, 0, stream>>>(
			paddedKernel,
			kernel,
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
		float *paddedWeights,
		float *paddedWeightSums,
		float *weights,
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
		float v;

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

		v = weights[dy * dataW + dx];
		idx = y * fftW + x;

		paddedWeights[idx] = v;
		paddedWeightSums[idx] += v;
	}
}

void padWeights(
		float *paddedWeights,
		float *paddedWeightSums,
		float *weights,
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
			paddedWeights,
			paddedWeightSums,
			weights,
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
		float *paddedWeights,
		float *paddedWeightSums,
		int fftH,
		int fftW
		)
{
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;

	if (y < fftH && x < fftW)
	{
		int idx = y * fftW + x;
		float d = paddedWeightSums[idx];
		if(d > 0)
			paddedWeights[idx] /= d;
	}
}

void normalizeWeights(
		float *paddedWeights,
		float *paddedWeightSums,
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
			paddedWeights,
			paddedWeightSums,
			fftH,
			fftW
			);
	getLastCudaError("normalizeWeights_kernel<<<>>> execution failed\n");
}

__global__ void padDataClampToBorder32_kernel(
		float *paddedData,
		float *data,
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

		paddedData[y * fftW + x] = data[dy * dataW + dx];
	}
}

void padDataClampToBorder32(
		float *paddedData,
		float *data,
		int fftH,
		int fftW,
		int dataH,
		int dataW,
		int kernelH,
		int kernelW,
		cudaStream_t stream
		)
{
	assert(paddedData != data);
	dim3 threads(32, 8);
	dim3 grid(
			iDivUp(fftW, threads.x),
			iDivUp(fftH, threads.y));

	const int kernelY = kernelH / 2;
	const int kernelX = kernelW / 2;

	padDataClampToBorder32_kernel<<<grid, threads, 0, stream>>>(
			paddedData,
			data,
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
		float *data,
		float *paddedData,
		int fftH,
		int fftW,
		int dataH,
		int dataW
		)
{
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;

	if (y < dataH && x < dataW)
		data[y * dataW + x] = paddedData[y * fftW + x];
}

void unpadData32(
		float *data,
		float *paddedData,
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
			data,
			paddedData,
			fftH,
			fftW,
			dataH,
			dataW
			);
	getLastCudaError("unpadData_kernel<<<>>> execution failed\n");
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
		fComplex *dst,
		fComplex *src,
		int dataSize,
		float c
		)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= dataSize)
		return;

	fComplex a = src[i];
	fComplex b = dst[i];

	mulAndScale(a, b, c);

	dst[i] = a;
}

void modulateAndNormalize(
		fComplex *dst,
		fComplex *src,
		int fftH,
		int fftW,
		int padding,
		cudaStream_t stream
		)
{
	assert(fftW % 2 == 0);
	const int dataSize = fftH * (fftW / 2 + padding);

	modulateAndNormalize_kernel<<<iDivUp(dataSize, 256), 256, 0, stream>>>(
			dst,
			src,
			dataSize,
			1.0f / (float)(fftW *fftH)
			);
	getLastCudaError("modulateAndNormalize() execution failed\n");
}

__global__ void multiply32_kernel(
		float *a,
		float *b,
		float *weights,
		float *dest,
		int dataSize
		)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= dataSize)
		return;

	float target = a[i] * b[i];
	float change = target - dest[i];
	float weight = weights[i];
	change *= weight;
	dest[i] += change;
}

void multiply32(
		float *a,
		float *b,
		float *weights,
		float *dest,
		int fftH,
		int fftW,
		cudaStream_t stream
		)
{
	const int dataSize = fftH * fftW;

	multiply32_kernel<<<iDivUp(dataSize, 256), 256, 0, stream>>>(
			a,
			b,
			weights,
			dest,
			dataSize
			);
	getLastCudaError("multiply32_kernel<<<>>> execution failed\n");
}

template<typename T>
__global__ void
padDataClampToBorderAndInitialize_kernel(
		float *estimate,
		T *paddedData,
		T *data,
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
		T v;

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

		v = data[dy * dataW + dx];
		idx = y * fftW + x;
		paddedData[idx] = v;
		estimate[idx] += 10;
	}
}

template<typename T>
void
padDataClampToBorderAndInitialize(
		float *estimate,
		T *paddedData,
		T *data,
		int fftH,
		int fftW,
		int dataH,
		int dataW,
		int kernelW,
		int kernelH,
		cudaStream_t stream
		)
{
	assert(data != paddedData);
	dim3 threads(32, 8);
	dim3 grid(
			iDivUp(fftW, threads.x),
			iDivUp(fftH, threads.y));

	const int kernelY = kernelH / 2;
	const int kernelX = kernelW / 2;

	padDataClampToBorderAndInitialize_kernel<<<grid, threads, 0, stream>>>(
			estimate,
			paddedData,
			data,
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

template<typename T>
__global__ void
unpadData_kernel(
		T *data,
		float *paddedData,
		int fftH,
		int fftW,
		int dataH,
		int dataW,
		T max
		)
{
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;

	// TODO round
	if (y < dataH && x < dataW) {
		float v = paddedData[y * fftW + x];
		if(v > max)
			v = max;
		data[y * dataW + x] = (T)v;
	}
}

template<typename T>
void
unpadData(
		T *data,
		float *paddedData,
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
	T max = (1 << (sizeof(T) * 8)) - 1;

	unpadData_kernel<<<grid, threads, 0, stream>>>(
			data,
			paddedData,
			fftH,
			fftW,
			dataH,
			dataW,
			max
			);
	getLastCudaError("unpadData_kernel<<<>>> execution failed\n");
}

template<typename T>
__global__ void
divide_kernel(
		T *a,
		float *b,
		float *dest,
		int dataSize
		)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	float q;

	if (i >= dataSize)
		return;

	q = b[i];
	if(q > 1)
		dest[i] = a[i] / q;
}

template<typename T>
void
divide(
	T *a,
	float *b,
	float *dest,
	int fftH,
	int fftW,
	cudaStream_t stream
	)
{
	const int dataSize = fftH * fftW;

	divide_kernel<<<iDivUp(dataSize, 256), 256, 0, stream>>>(
			a,
			b,
			dest,
			dataSize
			);
	getLastCudaError("divide_kernel<<<>>> execution failed\n");
}

// explicit template instantiation
template void
padDataClampToBorderAndInitialize(
		float *, unsigned short *, unsigned short *,
		int, int, int, int, int, int, cudaStream_t);
template void
padDataClampToBorderAndInitialize(
		float *, unsigned char *, unsigned char *,
		int, int, int, int, int, int, cudaStream_t);

template void
unpadData(unsigned short *, float *, int, int, int, int, cudaStream_t);
template void
unpadData(unsigned char *, float *, int, int, int, int, cudaStream_t);

template void
divide(unsigned short *, float *, float *, int, int, cudaStream_t);
template void
divide(unsigned char *, float *, float *, int, int, cudaStream_t);

