#ifndef FMVD_TRANSFORM_H
#define FMVD_TRANSFORM_H

typedef void (*return_plane_16)(unsigned short *plane);
typedef void (*return_plane_8)(unsigned char *plane);

void transform_cuda_8(
		unsigned char **h_data,
		int w, int h, int d,
		int tw, int th, int td,
		float *h_inverse,
		return_plane_8 callback,
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
		return_plane_16 callback,
		const char *outfile,
		int createTransformedMasks,
		int border,
		float zspacing,
		const char *maskfile);

#endif
