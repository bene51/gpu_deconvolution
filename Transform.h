#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <cuda_runtime.h>

#define nStreams 2

template<typename T>
class Transform
{

private:
	T *h_transformed_;

	cudaArray *d_data_;
	T *d_transformed_;

	float *d_inverseTransform_;

	cudaTextureObject_t texture_;

	const int dataWidth_, dataHeight_, dataDepth_;
	const int targetWidth_, targetHeight_, targetDepth_;

	const int planeSize_;

	cudaStream_t *streams_;

public:
	Transform(const T * const * const data,
			int dataWidth, int dataHeight, int dataDepth,
			int targetWidth, int targetHeight, int targetDepth,
			const float * const inverseTransform);
	~Transform();

	void transform(const char *outfile);
	void createWeights(int border, float zspacing, const char *maskfile);
};

#endif

