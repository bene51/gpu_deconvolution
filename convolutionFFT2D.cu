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
#include "fmvd_utils.h"
#include "fmvd_deconvolve_common.h"
#include "convolutionFFT2D_common.h"
#include "convolutionFFT2D.cuh"


////////////////////////////////////////////////////////////////////////////////
/// Position convolution kernel center at (0, 0) in the image
////////////////////////////////////////////////////////////////////////////////
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



////////////////////////////////////////////////////////////////////////////////
// Prepare data for "pad to border" addressing mode
////////////////////////////////////////////////////////////////////////////////
extern "C" void padDataClampToBorderFloat(
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

	padDataClampToBorderFloat_kernel<<<grid, threads, 0, stream>>>(
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
	getLastCudaError("padDataClampToBorder_kernel<<<>>> execution failed\n");
}

extern "C" void padDataClampToBorderAndInitialize(
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

	padDataClampToBorderAndInitialize_kernel<<<grid, threads, 0, stream>>>(
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

extern "C" void unpadData(
		data_t *d_Dst,
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

extern "C" void unpadDataFloat(
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

	unpadDataFloat_kernel<<<grid, threads, 0, stream>>>(
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
		data_t *d_a,
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
		float *d_weights,
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
			d_weights,
			d_dest,
			dataSize
			);
	getLastCudaError("modulateAndNormalize() execution failed\n");
}

