#ifndef TRANSFORM_KERNELS_H
#define TRANSFORM_KERNELS_H

#include <cuda_runtime.h>

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
		cudaStream_t stream);

template<typename T>
void
transformPlaneCuda(
		cudaTextureObject_t texture,
		T *d_transformed,
		int targetZ,
		int targetWidth,
		int targetHeight,
		const float *d_inverseTransform,
		cudaStream_t stream);




#endif
