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



#ifndef CONVOLUTIONFFT2D_COMMON_H
#define CONVOLUTIONFFT2D_COMMON_H

#include "cuda_runtime.h"


typedef unsigned short data_t;
// typedef float data_t;

typedef unsigned int uint;

#ifdef __CUDACC__
typedef float2 fComplex;
#else
typedef struct
{
	float x;
	float y;
} fComplex;
#endif


////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value

extern "C" void convolutionClampToBorderCPU(
		float *h_Result,
		float *h_Data,
		float *h_Kernel,
		int dataH,
		int dataW,
		int kernelH,
		int kernelW,
		int kernelY,
		int kernelX
		);

extern "C" void padKernel(
		float *d_PaddedKernel,
		float *d_PaddedKernelHat,
		float *d_Kernel,
		int fftH,
		int fftW,
		int kernelH,
		int kernelW,
		cudaStream_t stream
		);

extern "C" void padDataClampToBorder(
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
		cudaStream_t stream
		);

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
		);

extern "C" void normalizeWeights(
		float *d_PaddedWeights,
		float *d_PaddedWeightSums,
		int fftH,
		int fftW,
		cudaStream_t stream
		);

extern "C" void unpadData(
		data_t *d_Dst,
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

extern "C" void divide(
		data_t *d_a,
		float *d_b,
		float *d_dest,
		int fftH,
		int fftW,
		cudaStream_t stream
		);

extern "C" void mul(
		float *d_a,
		float *d_b,
		float *d_weights,
		float *d_dest,
		int fftH,
		int fftW,
		cudaStream_t stream
		);


#endif //CONVOLUTIONFFT2D_COMMON_H
