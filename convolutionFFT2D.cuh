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

////////////////////////////////////////////////////////////////////////////////
/// Position convolution kernel center at (0, 0) in the image
////////////////////////////////////////////////////////////////////////////////
__global__ void padKernel_kernel(
		float *d_Dst,
		float *d_Src,
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

	if (y < kernelH && x < kernelW)
	{
		int ky = y - kernelY;

		if (ky < 0)
		{
			ky += fftH;
		}

		int kx = x - kernelX;

		if (kx < 0)
		{
			kx += fftW;
		}

		d_Dst[ky * fftW + kx] = d_Src[y * kernelW + x];
	}
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
		{
			dy = y;
		}

		if (x < dataW)
		{
			dx = x;
		}

		if (y >= dataH && y < borderH)
		{
			dy = dataH - 1;
		}

		if (x >= dataW && x < borderW)
		{
			dx = dataW - 1;
		}

		if (y >= borderH)
		{
			dy = 0;
		}

		if (x >= borderW)
		{
			dx = 0;
		}

		v = d_Weights[dy * dataW + dx];
		idx = y * fftW + x;

		d_PaddedWeights[idx] = (float)v;
		d_PaddedWeightSums[idx] += (float)v;
	}
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

////////////////////////////////////////////////////////////////////////////////
// Prepare data for "pad to border" addressing mode
////////////////////////////////////////////////////////////////////////////////
__global__ void padDataClampToBorderAndInitialize_kernel(
		float *d_estimate,
		data_t *d_Dst,
		data_t *d_Src,
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

	if (y < fftH && x < fftW)
	{
		int dy, dx, idx;
		data_t v;

		if (y < dataH)
		{
			dy = y;
		}

		if (x < dataW)
		{
			dx = x;
		}

		if (y >= dataH && y < borderH)
		{
			dy = dataH - 1;
		}

		if (x >= dataW && x < borderW)
		{
			dx = dataW - 1;
		}

		if (y >= borderH)
		{
			dy = 0;
		}

		if (x >= borderW)
		{
			dx = 0;
		}

		v = d_Src[dy * dataW + dx];
		idx = y * fftW + x;
		d_Dst[idx] = v;
		// float change = v * d_Weights[idx];
		// d_estimate[idx] += change;
		d_estimate[idx] += 100;
	}
}

__global__ void padDataClampToBorderFloat_kernel(
		float *d_Dst,
		float *d_Src,
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
		{
			dy = y;
		}

		if (x < dataW)
		{
			dx = x;
		}

		if (y >= dataH && y < borderH)
		{
			dy = dataH - 1;
		}

		if (x >= dataW && x < borderW)
		{
			dx = dataW - 1;
		}

		if (y >= borderH)
		{
			dy = 0;
		}

		if (x >= borderW)
		{
			dx = 0;
		}

		v = d_Src[dy * dataW + dx];
		idx = y * fftW + x;
		d_Dst[idx] = v;
	}
}

__global__ void unpadDataFloat_kernel(
		float *d_Dst,
		float *d_Src,
		int fftH,
		int fftW,
		int dataH,
		int dataW
		)
{
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;

	if (y < dataH && x < dataW)
	{
		d_Dst[y * dataW + x] = d_Src[y * fftW + x];
	}
}

__global__ void unpadData_kernel(
		data_t *d_Dst,
		float *d_Src,
		int fftH,
		int fftW,
		int dataH,
		int dataW
		)
{
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;

	if (y < dataH && x < dataW)
	{
		d_Dst[y * dataW + x] = (data_t)d_Src[y * fftW + x]; // TODO round
	}
}


////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded kernel
// and normalize by FFT size
////////////////////////////////////////////////////////////////////////////////
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
	{
		return;
	}

	fComplex a = d_Src[i];
	fComplex b = d_Dst[i];

	mulAndScale(a, b, c);

	d_Dst[i] = a;
}

__global__ void divide_kernel(
		data_t *d_a,
		float *d_b,
		float *d_dest,
		int dataSize
		)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	float q;

	if (i >= dataSize)
	{
		return;
	}
	q = d_b[i];
	if(q > 1)
		d_dest[i] = d_a[i] / q;
}

__global__ void multiply_kernel(
		float *d_a,
		float *d_b,
		float *weights,
		float *d_dest,
		int dataSize
		)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= dataSize)
	{
		return;
	}

	// d_dest[i] = d_a[i] * d_b[i];
	float target = d_a[i] * d_b[i];
	float change = target - d_dest[i];
	float weight = weights[i];
	change *= weight;
	d_dest[i] += change;
}


