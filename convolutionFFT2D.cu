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


#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fmvd_deconvolve_common.h"
#include "convolutionFFT2D_common.h"
#include "convolutionFFT2D.cuh"

#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
				file, line, errorMessage, (int)err, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}


////////////////////////////////////////////////////////////////////////////////
/// Position convolution kernel center at (0, 0) in the image
////////////////////////////////////////////////////////////////////////////////
extern "C" void padKernel(
		float *d_Dst,
		float *d_DstHat,
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
			d_DstHat,
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



////////////////////////////////////////////////////////////////////////////////
// Prepare data for "pad to border" addressing mode
////////////////////////////////////////////////////////////////////////////////
extern "C" void padDataClampToBorder(
		float *d_estimate,
		float *d_Dst,
		float *d_Src,
		int fftH,
		int fftW,
		int dataH,
		int dataW,
		int kernelW,
		int kernelH,
		int nViews,
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

	SET_FLOAT_BASE;
	padDataClampToBorder_kernel<<<grid, threads, 0, stream>>>(
			d_estimate,
			d_Dst,
			d_Src,
			fftH,
			fftW,
			dataH,
			dataW,
			kernelH,
			kernelW,
			kernelY,
			kernelX,
			nViews
			);
	getLastCudaError("padDataClampToBorder_kernel<<<>>> execution failed\n");
}

extern "C" void unpadData(
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

	SET_FLOAT_BASE;
	unpadData_kernel<<<grid, threads, 0, stream>>>(
			d_Dst,
			d_Src,
			fftH,
			fftW,
			dataH,
			dataW
			);
	getLastCudaError("unpadData_kernel<<<>>> execution failed\n");
}



////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded kernel
// and normalize by FFT size
////////////////////////////////////////////////////////////////////////////////
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

extern "C" void divide(
		float *d_a,
		float *d_b,
		float *d_dest,
		int fftH,
		int fftW,
		cudaStream_t stream
		)
{
	const int dataSize = fftH * fftW;

	divide_kernel<<<iDivUp(dataSize, 256), 256, 0, stream>>>(
			d_a,
			d_b,
			d_dest,
			dataSize
			);
	getLastCudaError("modulateAndNormalize() execution failed\n");
}

extern "C" void mul(
		float *d_a,
		float *d_b,
		float *d_dest,
		int fftH,
		int fftW,
		cudaStream_t stream
		)
{
	const int dataSize = fftH * fftW;

	multiply_kernel<<<iDivUp(dataSize, 256), 256, 0, stream>>>(
			d_a,
			d_b,
			d_dest,
			dataSize
			);
	getLastCudaError("modulateAndNormalize() execution failed\n");
}

