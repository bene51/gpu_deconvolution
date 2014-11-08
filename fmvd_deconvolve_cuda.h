#ifndef __FMVD_DECONVOLVE_CUDA__
#define __FMVD_DECONVOLVE_CUDA__

#include "convolutionFFT2D_common.h"
#include <cufft.h>
#include <stdio.h>

// returns 0 if no more data is available
typedef int (*datasource_t)(data_t **buffer, int offset);

typedef void (*datasink_t)(data_t *buffer);

enum fmvd_psf_type { independent, efficient_bayesian, optimization_1, optimization_2 };

struct fmvd_plan_cuda {
	int dataH, dataW;
	int fftH, fftW;
	int kernelH, kernelW;
	int nViews;
	int nStreams;

	fComplex **d_KernelSpectrum;
	fComplex **d_KernelHatSpectrum;

	data_t **h_Data;
	data_t **d_Data;
	data_t **d_PaddedData;
	float **d_PaddedWeights;

	float *d_estimate;
	fComplex *d_estimateSpectrum;

	float *d_tmp;

	cudaStream_t *streams;

	cufftHandle *fftPlanFwd, *fftPlanInv;

	datasource_t get_next_plane;
	datasink_t return_next_plane;

};

struct fmvd_plan_cuda *
fmvd_initialize_cuda(int dataH, int dataW, data_t const* const* h_Weights, float const* const* h_Kernel, int kernelH, int kernelW, fmvd_psf_type iteration_type, int nViews, int nstreams, datasource_t get_next_plane, datasink_t return_next_plane);


void
fmvd_deconvolve_planes_cuda(const struct fmvd_plan_cuda *plan, int iterations);

void
fmvd_destroy_cuda(struct fmvd_plan_cuda *plan);

void
fmvd_deconvolve_files_cuda(FILE **dataFiles, FILE *resultFile, int dataW, int dataH, int dataD, data_t **h_Weights, float **h_Kernel, int kernelH, int kernelW, fmvd_psf_type iteration_type, int nViews, int iterations);

void *
fmvd_malloc(size_t size);

void
fmvd_free(void *p);

#endif // __FMVD_DECONVOLVE_CUDA__
