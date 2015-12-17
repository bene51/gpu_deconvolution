#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "DeconvolveFiles.h"

static void
printUsage()
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
		printUsage();
		return -1;
	}

	nViews = atoi(argv[i++]);

	char **datafiles = new char*[nViews];
	char **weightfiles = new char*[nViews];
	char **kernelfiles = new char*[nViews];
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

	IterationType::Type type = (IterationType::Type)iterationType;

	if(eightbit) {
		DeconvolveFiles<unsigned char> dec(
				datafiles,
				outputfile,
				kernelfiles,
				weightfiles,
				w, h, d,
				kw, kh, nViews, type);
		dec.process(nIterations);
	}
	else {
		DeconvolveFiles<unsigned short> dec(
				datafiles,
				outputfile,
				kernelfiles,
				weightfiles,
				w, h, d,
				kw, kh, nViews, type);
		dec.process(nIterations);
	}
	delete[] datafiles;
	delete[] weightfiles;
	delete[] kernelfiles;
	exit(EXIT_SUCCESS);
}


