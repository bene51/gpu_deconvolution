#include "DeconvolveInteractiveJNI.h"

#include <stdlib.h>
#include <string.h>

#include "CudaUtils.h"

template<typename T>
bool
DeconvolveInteractiveJNI<T>::getNextPlane(T **data, int offset)
{
	JNIEnv *env;
	jvm_->AttachCurrentThread((void **)&env, NULL);
	jobjectArray jarr = (jobjectArray)env->CallStaticObjectMethod(
			callingClass_, getNextPlaneMethodID_);
	if(jarr == NULL) {
		return false;
	}
	for(int v = 0; v < nViews_; v++) {
		jshortArray view = (jshortArray)env->GetObjectArrayElement(jarr, v);
		T *tgt = (T *)env->GetPrimitiveArrayCritical(view, NULL);
		memcpy(data[v] + offset, tgt, planesize_ * sizeof(T));
		env->ReleasePrimitiveArrayCritical(view, tgt, JNI_ABORT);
	}
	return true;
}

template<typename T>
void
DeconvolveInteractiveJNI<T>::returnNextPlane(T *data)
{
	JNIEnv *env;
	jvm_->AttachCurrentThread((void **)&env, NULL);
	jshortArray param = env->NewShortArray(planesize_);
	env->SetShortArrayRegion(param, 0, planesize_, (jshort *)data);
	env->CallStaticVoidMethod(callingClass_, returnNextPlaneMethodID_, param);
}

template<typename T>
DeconvolveInteractiveJNI<T>::DeconvolveInteractiveJNI(
		JNIEnv *env,
		jclass clazz,
		const char *const *const kernelPaths,
		const char *const *const weightPaths,
		int dataW, int dataH, int dataD,
		int kernelH, int kernelW, int nViews,
		IterationType::Type type) :
	callingClass_(clazz),
	planesize_(dataH * dataW),
	nViews_(nViews)
{
	env->GetJavaVM(&jvm_);
	getNextPlaneMethodID_ =
		env->GetStaticMethodID(clazz, "getNextPlane", "()[Ljava/lang/Object;");
	returnNextPlaneMethodID_ =
		env->GetStaticMethodID(clazz, "returnNextPlane", "(Ljava/lang/Object;)V");
	if(getNextPlaneMethodID_ == NULL || returnNextPlaneMethodID_ == NULL) {
		printf("error in callback: Method not found\n");
		return;
	}

	// Read kernels
	kernel_ = new float*[nViews];
	for(int v = 0; v < nViews; v++) {
		kernel_[v] = (float *)fmvd_malloc(kernelW * kernelH * sizeof(float));
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
	// weights_ = (float **)malloc(nViews * sizeof(float *));
	for(int v = 0; v < nViews; v++) {
		weights_[v] = new float[planesize_];
		FILE *f = fopen(weightPaths[v], "rb");
		if(!f) {
			// TODO throw
			printf("Could not open %s for reading\n", weightPaths[v]);
			exit(-1);
		}
		fread(weights_[v], sizeof(float), planesize_, f);
		fclose(f);
	}


	// Do the deconvolution
	int nStreams = dataD < 3 ? dataD : 3;
	deconvolution_ = new Deconvolve<T>(
			dataH, dataW,
			weights_,
			kernel_, kernelH, kernelW, type,
			nViews, nStreams,
			this,
			this);
}

template<typename T>
DeconvolveInteractiveJNI<T>::~DeconvolveInteractiveJNI()
{
	for(int v = 0; v < nViews_; v++) {
		fmvd_free(kernel_[v]);
		delete[] weights_[v];
	}
	delete kernel_;
	delete weights_;

	delete deconvolution_;
}

template<typename T>
void
DeconvolveInteractiveJNI<T>::process(int iterations)
{
	deconvolution_->deconvolvePlanes(iterations);
}

// explicit template instantiation
template class DeconvolveInteractiveJNI<unsigned char>;
template class DeconvolveInteractiveJNI<unsigned short>;

