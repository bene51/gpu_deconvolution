#ifndef __FMVD_DECONVOLVE_COMMON__
#define __FMVD_DECONVOLVE_COMMON__

int
iDivUp(int a, int b);

//Align a to nearest higher multiple of b
int
iAlignUp(int a, int b);

int
snapTransformSize(int dataSize);

void
normalizeMinMax(float *data, int len, float min, float max);

void
normalize(float *kernel, int len);

void
normalizeRange(float *kernel, int len);

#endif // __FMVD_DECONVOLVE_COMMON__

