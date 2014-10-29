#include "fmvd_deconvolve_cuda.h"
#include "fmvd_deconvolve_common.h"
#include <stdlib.h>
#include <stdio.h>

struct iodata {
	FILE **dataFiles;
	FILE *resultFile;
	int datasize;
	int plane;
	int n_planes;
	int n_views;
};

static iodata *io = NULL;

static int
get_next_plane(data_t **data, int offset)
{
	int v;
	if(io->plane >= io->n_planes)
		return 0;
	printf("Reading plane %d\n", io->plane);
	for(v = 0; v < io->n_views; v++)
		fread(data[v] + offset, sizeof(data_t), io->datasize, io->dataFiles[v]);
	io->plane++;
	return 1;
}

static void
return_next_plane(data_t *data)
{
	// printf("writing plane %d\n", plane);
	fwrite(data, sizeof(data_t), io->datasize, io->resultFile);
}

void
fmvd_deconvolve_files_cuda(
		FILE **dataFiles,
		FILE *resultFile,
		int dataW,
		int dataH,
		int dataD,
		data_t **h_Weights,
		float **h_Kernel,
		int kernelH,
		int kernelW,
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

	fmvd_plan_cuda *plan = fmvd_initialize_cuda(
		dataH, dataW,
		h_Weights,
		h_Kernel, kernelH, kernelW,
		nViews, nStreams,
		get_next_plane,
		return_next_plane);

	fmvd_deconvolve_plane_cuda(plan, iterations);

	fmvd_destroy_cuda(plan);
	free(io);
	io = NULL;
	printf("...shutting down\n");
}

static void
load(int w, int h, float *data, const char *path)
{
	FILE *f = fopen(path, "rb");
	fread(data, sizeof(float), w * h, f);
	fclose(f);
}

/*
int
main(int argc, char **argv)
{
	int it;
	printf("[%s] - Starting...\n", argv[0]);

	const int W = 600;
	const int H = 600;
	const int D = 600;
	const int SEEK = 0;

	const int KW = 83;
	const int KH = 83;

	float **kernel = (float **)malloc(2 * sizeof(float *));
	kernel[0] = (float *)fmvd_malloc(KW * KH * sizeof(float));
	kernel[1] = (float *)fmvd_malloc(KW * KH * sizeof(float));

	load(KW, KH, kernel[0], "E:\\SPIM5_Deconvolution\\m6\\cropped\\forplanewisedeconv\\psf3.raw");
	load(KW, KH, kernel[1], "E:\\SPIM5_Deconvolution\\m6\\cropped\\forplanewisedeconv\\psf5.raw");

	normalize(kernel[0], KW * KH);
	normalize(kernel[1], KW * KH);

	FILE **dataFiles = (FILE**)malloc(2 * sizeof(FILE *));
	char path[256];
	for(it = 3; it < 4; it++) {
		dataFiles[0] = fopen("E:\\SPIM5_Deconvolution\\m6\\cropped\\forplanewisedeconv\\v3.raw", "rb");
		dataFiles[1] = fopen("E:\\SPIM5_Deconvolution\\m6\\cropped\\forplanewisedeconv\\v5.raw", "rb");
		fseek(dataFiles[0], SEEK * W * H * sizeof(float), SEEK_SET);
		fseek(dataFiles[1], SEEK * W * H * sizeof(float), SEEK_SET);

		sprintf(path, "E:\\SPIM5_Deconvolution\\m6\\cropped\\forplanewisedeconv\\v35.deconvolved.gpu.raw");
		FILE *resultFile = fopen(path, "wb");

		fmvd_deconvolve_files_cuda(dataFiles, resultFile, W, H, D, kernel, KH, KW, 2, it);

		fclose(dataFiles[0]);
		fclose(dataFiles[1]);
		fclose(resultFile);
	}

	fmvd_free(kernel[0]);
	fmvd_free(kernel[1]);
	free(kernel);
	free(dataFiles);

	exit(EXIT_SUCCESS);
}

*/
