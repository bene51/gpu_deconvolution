#ifdef _WIN32
#define _CRT_SECURE_NO_DEPRECATE
#endif

#define PASTE(name, type) name ## _ ## type
#define EVAL(name, type) PASTE(name, type)
#define MAKE_NAME(name) EVAL(name, BITS_PER_SAMPLE)

#include "fmvd_deconvolve.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static int
MAKE_NAME(get_next_plane)(SAMPLE **data, int offset)
{
	int v;
	if(io->plane >= io->n_planes)
		return 0;
	printf("Reading plane %d\n", io->plane);
	for(v = 0; v < io->n_views; v++)
		fread(data[v] + offset, sizeof(SAMPLE), io->datasize, io->dataFiles[v]);
	io->plane++;
	return 1;
}

static void
MAKE_NAME(return_next_plane)(SAMPLE *data)
{
	// printf("writing plane %d\n", plane);
	fwrite(data, sizeof(SAMPLE), io->datasize, io->resultFile);
}

void
MAKE_NAME(fmvd_deconvolve_files_cuda)(
		FILE **dataFiles,
		FILE *resultFile,
		int dataW,
		int dataH,
		int dataD,
		float **h_Weights,
		float **h_Kernel,
		int kernelH,
		int kernelW,
		fmvd_psf_type iteration_type,
		int nViews,
		int iterations)
{
	int nStreams = 3;
	int datasize = dataH * dataW;

	io = (struct iodata *)malloc(sizeof(struct iodata));
	io->dataFiles = dataFiles;
	io->resultFile = resultFile;
	io->datasize = datasize;
	io->plane = 0;
	io->n_planes = dataD;
	io->n_views = nViews;

	MAKE_NAME(fmvd_plan_cuda) *plan = MAKE_NAME(fmvd_initialize_cuda)(
		dataH, dataW,
		h_Weights,
		h_Kernel, kernelH, kernelW, iteration_type,
		nViews, nStreams,
		MAKE_NAME(get_next_plane),
		MAKE_NAME(return_next_plane));

	MAKE_NAME(fmvd_deconvolve_planes_cuda)(plan, iterations);

	MAKE_NAME(fmvd_destroy_cuda)(plan);
	free(io);
	io = NULL;
	printf("...shutting down\n");
}


