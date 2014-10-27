#ifndef __FMVD_TRANSFORM_CUDA_H__
#define __FMVD_TRANSFORM_CUDA_H__

void transform_cuda(
		unsigned short *h_data,
		int w, int h, int d, int tw, int th, int td, float *h_inverse, const char *outfile,
		int createTransformedMasks,
		int border,
		float zspacing,
		const char *maskfile);


#endif
