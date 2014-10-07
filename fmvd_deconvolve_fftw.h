#ifndef __FMVD_DECONVOLVE_FFTW__
#define __FMVD_DECONVOLVE_FFTW__

#include "fftw-3.3.4-dll64/fftw3.h"

struct fmvd_plan {
	int dataH, dataW;
	int fftH, fftW;
	int kernelH, kernelW;
	int nViews;

	fftwf_complex **h_KernelSpectrum;
	fftwf_complex **h_KernelHatSpectrum;

	float **h_PaddedData;

	float *h_estimate;
	fftwf_complex *h_estimateSpectrum;

	float *h_tmp;

	fftwf_plan fftPlanFwd, fftPlanInv;
};

struct fmvd_plan *
fmvd_initializeCPU(int dataH, int dataW, float const* const* h_Kernel, int kernelH, int kernelW, int nViews, int nthreads);

void
fmvd_deconvolvePlaneCPU(const struct fmvd_plan *plan, float **h_Data, int iterations);

void
fmvd_destroy(struct fmvd_plan *plan);

void
fmvd_deconvolveCPU(
		FILE **dataFiles,
		FILE *resultFile,
		int dataW,
		int dataH,
		int dataD,
		float const* const* h_Kernel,
		int kernelH,
		int kernelW,
		int nViews,
		int iterations,
		int nthreads);

#endif // __FMVD_DECONVOLVE_FFTW__
