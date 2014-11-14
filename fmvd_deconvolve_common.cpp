#include "fmvd_deconvolve_common.h"
#include <math.h>

//Align a to nearest higher multiple of b
int
iAlignUp(int a, int b)
{
	return (a % b != 0) ? (a - a % b + b) : a;
}

int
snapTransformSize(int dataSize)
{
	int hiBit;
	unsigned int lowPOT, hiPOT;

	dataSize = iAlignUp(dataSize, 16);

	for (hiBit = 31; hiBit >= 0; hiBit--)
		if (dataSize & (1U << hiBit))
			break;

	lowPOT = 1U << hiBit;

	if (lowPOT == (unsigned int)dataSize)
		return dataSize;

	hiPOT = 1U << (hiBit + 1);

	if (hiPOT <= 1024)
		return hiPOT;
	else
		return iAlignUp(dataSize, 512);
}

void
normalize(float *kernel, int len)
{
	int i;
	double sum = 0;
	float *k = kernel;
	for(i = 0; i < len; i++) {
		sum += *k;
		k++;
	}
	k = kernel;
	for(i = 0; i < len; i++) {
		*k /= (float)sum;
		k++;
	}
}

void
computeInvertedKernel(const float *kernel, float *out, int kw, int kh)
{
	int x, y;
	for(y = 0; y < kh; y++) {
		for(x = 0; x < kw; x++) {
			out[y * kw + x] = kernel[(kh - y - 1) * kw + (kw - x - 1)];
		}
	}
}

void
computeExponentialKernel(const float *kernel, float *out, int kw, int kh, int exponent)
{
	int i;
	int wh = kw * kh;
	for(i = 0; i < wh; i++)
		out[i] = (float)pow(kernel[i], exponent);
}


