#ifdef _WIN32
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include "fmvd_deconvolve.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

	fmvd_plan_cuda *plan = fmvd_initialize_cuda(
		dataH, dataW,
		h_Weights,
		h_Kernel, kernelH, kernelW, iteration_type,
		nViews, nStreams,
		get_next_plane,
		return_next_plane);

	fmvd_deconvolve_planes_cuda(plan, iterations);

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

static void print_usage()
{
	printf("Usage:\n");
	printf("fmvd_deconvolve -v <nViews>\n");
	printf("                -i <nIterations>\n");
	printf("                -t <iteration type>\n");
	printf("                -o <output file>\n");
	printf("                -d <datafile view1> <datafile view2> ...\n");
	printf("                -w <weightfile view1> <weightfile view2> ...\n");
	printf("                -k <kernelfile view1> <kernelfile view2> ...\n");
	printf("                -dw <data width>\n");
	printf("                -dh <data height>\n");
	printf("                -dd <data depth>\n");
	printf("                -kw <kernel width>\n");
	printf("                -kh <kernel height>\n");
}

int
main(int argc, char **argv)
{

	int i = 1;
	int nViews, nIterations, w, h, d, kw, kh, iterationType;

	printf("argc = %d\n", argc);

	if (argc == 1 || strcmp(argv[i++], "-v")) {
		print_usage();
		return -1;
	}

	nViews = atoi(argv[i++]);

	char **datafiles = (char **)malloc(nViews * sizeof(char *));
	char **weightfiles = (char **)malloc(nViews * sizeof(char *));
	char **kernelfiles = (char **)malloc(nViews * sizeof(char *));
	char *outputfile = NULL;
	int tmp;

	for(; i < argc; i++) {
		if(!strcmp(argv[i], "-i"))
			nIterations = atoi(argv[++i]);
		else if(!strcmp(argv[i], "-t"))
			iterationType = atoi(argv[++i]);
		else if(!strcmp(argv[i], "-dw"))
			w = atoi(argv[++i]);
		else if(!strcmp(argv[i], "-dh"))
			h = atoi(argv[++i]);
		else if(!strcmp(argv[i], "-dd"))
			d = atoi(argv[++i]);
		else if(!strcmp(argv[i], "-kw"))
			kw = atoi(argv[++i]);
		else if(!strcmp(argv[i], "-kh"))
			kh = atoi(argv[++i]);
		else if(!strcmp(argv[i], "-o"))
			outputfile = argv[++i];
		else if(!strcmp(argv[i], "-d")) {
			for(tmp = 0; tmp < nViews; tmp++)
				datafiles[tmp] = argv[++i];
		}
		else if(!strcmp(argv[i], "-w")) {
			for(tmp = 0; tmp < nViews; tmp++)
				weightfiles[tmp] = argv[++i];
		}
		else if(!strcmp(argv[i], "-k")) {
			for(tmp = 0; tmp < nViews; tmp++)
				kernelfiles[tmp] = argv[++i];
		}
	}

	printf("Starting plane-wise deconvolution\n");
	printf("	nViews:        %d\n", nViews);
	printf("	nIterations:   %d\n", nIterations);
	printf("	iterationType: %d\n", iterationType);
	printf("	dimensions:    %d x %d x %d\n", w, h, d);
	printf("        kernel dims:   %d x %d\n", kw, kh);
	printf("        data files:\n");
	for(tmp = 0; tmp < nViews; tmp++)
		printf("              %s\n", datafiles[tmp]);
	printf("        weight files:\n");
	for(tmp = 0; tmp < nViews; tmp++)
		printf("              %s\n", weightfiles[tmp]);
	printf("        kernel files:\n");
	for(tmp = 0; tmp < nViews; tmp++)
		printf("              %s\n", kernelfiles[tmp]);


	// Read kernels
	float **kernel = (float **)malloc(nViews * sizeof(float *));
	for(int v = 0; v < nViews; v++) {
		kernel[v] = (float *)fmvd_malloc(kw * kh * sizeof(float));
		FILE *f = fopen(kernelfiles[v], "rb");
		if(!f) {
			printf("Could not open %s for reading\n", kernelfiles[v]);
			exit(-1);
		}
		fread(kernel[v], sizeof(float), kw * kh, f);
		fclose(f);
	}

	// Read weights
	data_t **weights = (data_t **)malloc(nViews * sizeof(data_t *));
	int datasize = w * h;
	for(int v = 0; v < nViews; v++) {
		weights[v] = (data_t *)malloc(datasize * sizeof(data_t));
		FILE *f = fopen(weightfiles[v], "rb");
		if(!f) {
			printf("Could not open %s for reading\n", weightfiles[v]);
			exit(-1);
		}
		fread(weights[v], sizeof(data_t), datasize, f);
		fclose(f);
	}

	// Open input files
	FILE **dataFiles = (FILE**)malloc(nViews * sizeof(FILE *));
	for(int v = 0; v < nViews; v++) {
		dataFiles[v] = fopen(datafiles[v], "rb");
		if(!dataFiles[v]) {
			printf("Could not open %s for reading\n", datafiles[v]);
			exit(-1);
		}
	}

	// Open output file
	FILE *resultFile = fopen(outputfile, "wb");
	if(!resultFile) {
		printf("Could not open %s for reading\n", outputfile);
		exit(-1);
	}

	// Do the deconvolution
	fmvd_psf_type iteration_type = (fmvd_psf_type)iterationType;
	fmvd_deconvolve_files_cuda(dataFiles, resultFile, w, h, d, weights, kernel, kh, kw, iteration_type, nViews, nIterations);

	// Close input and output files
	for(int v = 0; v < nViews; v++)
		fclose(dataFiles[v]);
	fclose(resultFile);

	// Cleanup
	for(int v = 0; v < nViews; v++) {
		fmvd_free(kernel[v]);
		free(weights[v]);
	}
	free(kernel);
	free(weights);
	free(dataFiles);

	exit(EXIT_SUCCESS);
}

