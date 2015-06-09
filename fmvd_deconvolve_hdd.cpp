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
	printf("                [-8bit]\n");
}

int
main(int argc, char **argv)
{

	int i = 1;
	int nViews, nIterations, w, h, d, kw, kh, iterationType;
	int eightbit = 0;

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
		else if(!strcmp(argv[i], "-8bit"))
			eightbit = 1;
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
	float **weights = (float **)malloc(nViews * sizeof(float *));
	int datasize = w * h;
	for(int v = 0; v < nViews; v++) {
		weights[v] = (float *)malloc(datasize * sizeof(float));
		FILE *f = fopen(weightfiles[v], "rb");
		if(!f) {
			printf("Could not open %s for reading\n", weightfiles[v]);
			exit(-1);
		}
		fread(weights[v], sizeof(float), datasize, f);
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
	int bytes = eightbit ? 1 : 2;
	if(eightbit)
		fmvd_deconvolve_files_cuda_8(dataFiles, resultFile, w, h, d, weights, kernel, kh, kw, iteration_type, nViews, nIterations);
	else
		fmvd_deconvolve_files_cuda_16(dataFiles, resultFile, w, h, d, weights, kernel, kh, kw, iteration_type, nViews, nIterations);

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


#define SAMPLE              unsigned short
#define BITS_PER_SAMPLE     16 
#include "fmvd_deconvolve_hdd.impl.cpp"

#define SAMPLE              unsigned char
#define BITS_PER_SAMPLE     8
#include "fmvd_deconvolve_hdd.impl.cpp"


