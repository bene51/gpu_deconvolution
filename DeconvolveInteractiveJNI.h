#ifndef DECONVOLVE_INTERACTIVE_JNI_H
#define DECONVOLVE_INTERACTIVE_JNI_H

#include <jni.h>

#include "DataRetrieverInterface.h"
#include "DataReceiverInterface.h"
#include "Deconvolve.h"

template<typename T>
class DeconvolveInteractiveJNI :
	public DataRetrieverInterface<T>,
	public DataReceiverInterface<T>
{
private:
	JavaVM *jvm_;
	jmethodID getNextPlaneMethodID_;
	jmethodID returnNextPlaneMethodID_;
	jclass callingClass_;
	int planesize_;
	int nViews_;
	Deconvolve<T> *deconvolution_;
	float **kernel_;
	float **weights_;

public:
	DeconvolveInteractiveJNI(
		JNIEnv *env,
		jclass clazz,
		const char *const *const kernelPaths,
		const char *const *const weightPaths,
		int dataW, int dataH, int dataD,
		int kernelH, int kernelW, int nViews,
		IterationType::Type iterationType);
	~DeconvolveInteractiveJNI();
	virtual bool getNextPlane(T **data, int offset);
	virtual void returnNextPlane(T *data);
	void process(int iterations);
};

#endif

