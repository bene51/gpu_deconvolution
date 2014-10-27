#include "fmvd_deconvolve_common.h"

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
normalizeMinMax(float *data, int len, float min, float max)
{
	int i;
	float *k = data;
	for(i = 0; i < len; i++) {
		float v = *k;
		v = (v - min) / (max - min);

		if(v < 0) v = 0;
		if(v > 1) v = 1;
		(*k) = v;
		k++;
	}
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
		*k /= sum;
		k++;
	}
}

void
normalizeRange(float *kernel, int len)
{
	int i;
	float *k = kernel;
	float min = kernel[0];
	float max = kernel[0];
	k++;
	for(i = 1; i < len; i++) {
		float v = *k;
		if(v < min) min = v;
		if(v > max) max = v;
		k++;
	}
	k = kernel;
	for(i = 0; i < len; i++) {
		float v = *k;
		v = (v - min) / (max - min);
		(*k) = v;
		k++;
	}
}

