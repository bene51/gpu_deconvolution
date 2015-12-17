#include "DeconvolveFiles.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "IterationType.h"

template<typename T>
bool
DeconvolveFiles<T>::getNextPlane(T **data, int offset)
{
	if(plane_ >= nPlanes_)
		return false;
	printf("Reading plane %d\n", plane_);
	for(int v = 0; v < nViews_; v++)
		fread(data[v] + offset, sizeof(T), datasize_, dataFiles_[v]);
	++plane_;
	return true;
}

template<typename T>
void
DeconvolveFiles<T>::returnNextPlane(T *data)
{
	// printf("writing plane %d\n", plane);
	fwrite(data, sizeof(T), datasize_, resultFile_);
}

template<typename T>
DeconvolveFiles<T>::DeconvolveFiles(
			const char *const *const dataPaths,
		       	const char *const resultPath,
			const char *const *const kernelPaths,
			const char *const *const weightPaths,
			int dataW, int dataH, int dataD,
			int kernelW, int kernelH, int nViews,
			IterationType::Type type) :
	datasize_(dataH * dataW), 
	plane_(0),
	nPlanes_(dataD),
	nViews_(nViews)
{
	// Read kernels
	kernel_ = new float*[nViews];
	for(int v = 0; v < nViews; v++) {
		kernel_[v] = (float *)fmvd_malloc(
				kernelW * kernelH * sizeof(float));
		FILE *f = fopen(kernelPaths[v], "rb");
		if(!f) {
			// TODO throw
			printf("Could not open %s for reading\n", kernelPaths[v]);
			exit(-1);
		}
		fread(kernel_[v], sizeof(float), kernelW * kernelH, f);
		fclose(f);
	}

	// Read weights
	weights_ = new float*[nViews];
	for(int v = 0; v < nViews; v++) {
		weights_[v] = new float[datasize_];
		FILE *f = fopen(weightPaths[v], "rb");
		if(!f) {
			// TODO throw
			printf("Could not open %s for reading\n", weightPaths[v]);
			exit(-1);
		}
		fread(weights_[v], sizeof(float), datasize_, f);
		fclose(f);
	}

	// Open input files
	dataFiles_ = new FILE*[nViews];
	for(int v = 0; v < nViews; v++) {
		dataFiles_[v] = fopen(dataPaths[v], "rb");
		if(!dataFiles_[v]) {
			// TODO throw
			printf("Could not open %s for reading\n", dataPaths[v]);
			exit(-1);
		}
	}

	// Open output file
	resultFile_ = fopen(resultPath, "wb");
	if(!resultFile_) {
		printf("Could not open %s for reading\n", resultPath);
		exit(-1);
	}

	int nStreams = 3;
	deconvolution_ = new Deconvolve<T>(
			dataH, dataW,
			weights_,
			kernel_, kernelH, kernelW, type,
			nViews_, nStreams,
			this,
			this);
}

template<typename T>
DeconvolveFiles<T>::~DeconvolveFiles()
{
	for(int v = 0; v < nViews_; v++) {
		fmvd_free(kernel_[v]);
		delete weights_[v];
		fclose(dataFiles_[v]);
	}
	delete kernel_;
	delete weights_;
	delete dataFiles_;

	fclose(resultFile_);

	delete deconvolution_;
}

template<typename T>
void
DeconvolveFiles<T>::process(int iterations)
{
	deconvolution_->deconvolvePlanes(iterations);
	printf("...shutting down\n");
}

// explicit template instantiation
template class DeconvolveFiles<unsigned char>;
template class DeconvolveFiles<unsigned short>;

