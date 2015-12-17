#ifndef DECONVOLVE_KERNELS_CUH
#define DECONVOLVE_KERNELS_CUH

#include "cuda_runtime.h"

typedef float2 fComplex;

void padKernel(
		float *paddedKernel,
		float *kernel,
		int fftH,
		int fftW,
		int kernelH,
		int kernelW,
		cudaStream_t stream
		);

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
		);

void normalizeWeights(
		float *paddedWeights,
		float *paddedWeightSums,
		int fftH,
		int fftW,
		cudaStream_t stream
		);

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
		);

void unpadData32(
		float *data,
		float *paddedData,
		int fftH,
		int fftW,
		int dataH,
		int dataW,
		cudaStream_t stream
		);

void modulateAndNormalize(
		fComplex *dst,
		fComplex *src,
		int fftH,
		int fftW,
		int padding,
		cudaStream_t stream
		);

void multiply32(
		float *a,
		float *b,
		float *weights,
		float *dest,
		int fftH,
		int fftW,
		cudaStream_t stream
		);

template<typename T>
void padDataClampToBorderAndInitialize(
		float *estimate,
		T *paddedData,
		T *data,
		int fftH,
		int fftW,
		int dataH,
		int dataW,
		int kernelH,
		int kernelW,
		cudaStream_t stream
		);

template<typename T>
void unpadData(
		T *data,
		float *paddedData,
		int fftH,
		int fftW,
		int dataH,
		int dataW,
		cudaStream_t stream
		);

template<typename T>
void divide(
		T *a,
		float *b,
		float *dest,
		int fftH,
		int fftW,
		cudaStream_t stream
		);


#endif // DECONVOLVE_KERNELS_CUH
