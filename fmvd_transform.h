#ifndef FMVD_TRANSFORM_H
#define FMVD_TRANSFORM_H

void transform_cuda_8(
		unsigned char **h_data,
		int w, int h, int d,
		int tw, int th, int td,
		float *h_inverse,
		const char *outfile,
		int createTransformedMasks,
		int border,
		float zspacing,
		const char *maskfile);

void transform_cuda_16(
		unsigned short **h_data,
		int w, int h, int d,
		int tw, int th, int td,
		float *h_inverse,
		const char *outfile,
		int createTransformedMasks,
		int border,
		float zspacing,
		const char *maskfile);

#endif
