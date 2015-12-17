#ifndef DECONVOLVE_FILES_H
#define DECONVOLVE_FILES_H

#include <stdio.h>

#include "DataRetrieverInterface.h"
#include "DataReceiverInterface.h"
#include "Deconvolve.h"

template<typename T>
class DeconvolveFiles :
	public DataRetrieverInterface<T>,
       	public DataReceiverInterface<T>
{
private:
	FILE **dataFiles_;
	FILE *resultFile_;
	const int datasize_;
	int plane_;
	const int nPlanes_;
	const int nViews_;
	Deconvolve<T> *deconvolution_;
	float **kernel_;
	float **weights_;

public:
	DeconvolveFiles(const char *const *const dataPaths,
		       	const char *const resultPath,
			const char *const *const kernelPaths,
			const char *const *const weightPaths,
			int dataW, int dataH, int dataD,
			int kernelW, int kernelH, int nViews,
			IterationType::Type type);
	~DeconvolveFiles();
	virtual bool getNextPlane(T **data, int offset);
	virtual void returnNextPlane(T *data);
	void process(int iterations);
};

#endif

