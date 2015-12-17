#include "TransformKernels.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <math_constants.h>

#include "CudaUtils.h"


template<typename T>
__global__ void
transformPlaneKernel(
		cudaTextureObject_t texture,
		T *d_transformed,
		int bitsPerSample,
		int z,
		int targetWidth,
		int targetHeight,
		const float *inverseTransform)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	const float *m = inverseTransform;

	if(x < targetWidth && y < targetHeight) {
		// apply inverse transform
		float rx = m[0] * x + m[1] * y + m[2]  * z + m[3];
		float ry = m[4] * x + m[5] * y + m[6]  * z + m[7];
		float rz = m[8] * x + m[9] * y + m[10] * z + m[11];

		/*
		// mirror
		if(rx < 0) rx = -rx; if(rx > w - 1) rx = 2 * w - rx - 2;
		if(ry < 0) ry = -ry; if(ry > h - 1) ry = 2 * h - ry - 2;
		if(rz < 0) rz = -rz; if(rz > d - 1) rz = 2 * d - rz - 2;
		*/

		float v = tex3D<float>(texture, rx + 0.5, ry + 0.5, rz + 0.5);
		T iv = (T)(v * (1 << bitsPerSample) + 0.5);
		d_transformed[y * targetWidth + x] = iv;
	}
}

__global__ void
createWeightsKernel(
		float *d_transformed,
		int z,
		int dataWidth,
		int dataHeight,
		int dataDepth,
		float zspacing,
		int targetWidth,
		int targetHeight,
		float border,
		const float *inverseTransform)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	const float *m = inverseTransform;

	if(x < targetWidth && y < targetHeight) {
		int idx = y * targetWidth + x;
		float rx = m[0] * x + m[1] * y + m[2]  * z + m[3];
		float ry = m[4] * x + m[5] * y + m[6]  * z + m[7];
		float rz = m[8] * x + m[9] * y + m[10] * z + m[11];

		if(rx < 0 || rx >= dataWidth ||
				ry < 0 || ry >= dataHeight ||
				rz < 0 || rz >= dataDepth) {
			d_transformed[idx] = 0;
		} else {
			float pi = CUDART_PI_F;
			float v = 1;
			float dx = rx < dataWidth  / 2 ? rx : dataWidth  - rx;
			float dy = ry < dataHeight / 2 ? ry : dataHeight - ry;
			float dz = rz < dataDepth  / 2 ? rz : dataDepth  - rz;

			dz *= zspacing;

			if(dx < border)
				v = v * (0.5f * (1 - cos(dx / border * pi)));
			if(dy < border)
				v = v * (0.5f * (1 - cos(dy / border * pi)));
			if(dz < border)
				v = v * (0.5f * (1 - cos(dz / border * pi)));
			d_transformed[idx] = v;
		}
	}
}

void
createWeightsCuda(
		float *d_transformed,
		int targetZ,
		int dataWidth,
		int dataHeight,
		int dataDepth,
		float zspacing,
		int targetWidth,
		int targetHeight,
		float border,
		const float *d_inverseTransform,
		cudaStream_t stream)
{
	dim3 threads(32, 32);

	dim3 grid(iDivUp(targetWidth, threads.x),
	          iDivUp(targetHeight, threads.y));
	createWeightsKernel<<<grid, threads, 0, stream>>>(
			d_transformed, targetZ,
			dataWidth, dataHeight, dataDepth, zspacing,
			targetWidth, targetHeight, border, d_inverseTransform);
	getLastCudaError("transform_mask_kernel<<<>>> execution failed\n");
}

template<typename T>
void
transformPlaneCuda(
		cudaTextureObject_t texture,
		T *d_transformed,
		int targetZ,
		int targetWidth,
		int targetHeight,
		const float *d_inverseTransform,
		cudaStream_t stream)
{
	dim3 threads(32, 32);
	dim3 grid(iDivUp(targetWidth, threads.x),
	          iDivUp(targetHeight, threads.y));
	int bitsPerSample = 8 * sizeof(T);
	transformPlaneKernel<<<grid, threads, 0, stream>>>(
			texture, d_transformed, bitsPerSample, targetZ,
			targetWidth, targetHeight, d_inverseTransform);
	getLastCudaError("transform_data_kernel<<<>>> execution failed\n");
}

// explicit template instantiation
template void transformPlaneCuda(
		cudaTextureObject_t texture,
		unsigned char *d_transformed,
		int targetZ,
		int targetWidth,
		int targetHeight,
		const float *d_inverseTransform,
		cudaStream_t stream);

template void transformPlaneCuda(
		cudaTextureObject_t texture,
		unsigned short *d_transformed,
		int targetZ,
		int targetWidth,
		int targetHeight,
		const float *d_inverseTransform,
		cudaStream_t stream);

