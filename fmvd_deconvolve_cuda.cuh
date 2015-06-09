#ifndef FMVD_DECONVOLVE_CUDA_CUH
#define FMVD_DECONVOLVE_CUDA_CUH

#include "cuda_runtime.h"


// typedef unsigned short data_t;
// typedef float data_t;

#ifdef __CUDACC__
typedef float2 fComplex;
#else
typedef struct
{
	float x;
	float y;
} fComplex;
#endif

extern "C" void padKernel(
		float *d_PaddedKernel,
		float *d_Kernel,
		int fftH,
		int fftW,
		int kernelH,
		int kernelW,
		cudaStream_t stream
		);

extern "C" void padWeights(
		float *d_PaddedWeights,
		float *d_PaddedWeightSums,
		float *d_Weights,
		int fftH,
		int fftW,
		int dataH,
		int dataW,
		int kernelH,
		int kernelW,
		cudaStream_t stream
		);

extern "C" void normalizeWeights(
		float *d_PaddedWeights,
		float *d_PaddedWeightSums,
		int fftH,
		int fftW,
		cudaStream_t stream
		);

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
		);

extern "C" void unpadData32(
		float *d_Dst,
		float *d_Src,
		int fftH,
		int fftW,
		int dataH,
		int dataW,
		cudaStream_t stream
		);

extern "C" void modulateAndNormalize(
		fComplex *d_Dst,
		fComplex *d_Src,
		int fftH,
		int fftW,
		int padding,
		cudaStream_t stream
		);

extern "C" void multiply32(
		float *d_a,
		float *d_b,
		float *d_weights,
		float *d_dest,
		int fftH,
		int fftW,
		cudaStream_t stream
		);


#define SAMPLE              unsigned short
#define BITS_PER_SAMPLE     16 
#include "fmvd_deconvolve_cuda.impl.cuh"

#define SAMPLE              unsigned char
#define BITS_PER_SAMPLE     8
#include "fmvd_deconvolve_cuda.impl.cuh"


#endif // FMVD_DECONVOLVE_CUDA_CUH
