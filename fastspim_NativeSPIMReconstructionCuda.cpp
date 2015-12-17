#include "fastspim_NativeSPIMReconstructionCuda.h"

#include <string.h>
#include <stdlib.h>

#include "TransformJNI.h"
#include "CudaUtils.h"
#include "DeconvolveFiles.h"
#include "DeconvolveInteractiveJNI.h"

static void
ThrowException(void *env_ptr, const char *message)
{
	JNIEnv *env = (JNIEnv *)env_ptr;
	jclass cl;
	cl = env->FindClass("java/lang/RuntimeException");
	env->ThrowNew(cl, message);
}

static void
setCudaExceptionHandler(JNIEnv *env)
{
	setErrorHandler(ThrowException, env);
}

JNIEXPORT jint JNICALL
Java_fastspim_NativeSPIMReconstructionCuda_getNumCudaDevices(
		JNIEnv *env,
		jclass)
{
	setCudaExceptionHandler(env);
	return getNumCUDADevices();
}

JNIEXPORT jstring JNICALL
Java_fastspim_NativeSPIMReconstructionCuda_getCudaDeviceName(
		JNIEnv *env,
		jclass,
		jint deviceIdx)
{
	char name[256];
	setCudaExceptionHandler(env);
	getCudaDeviceName(deviceIdx, name);
	jstring result = env->NewStringUTF(name);
	return result;
}

JNIEXPORT void JNICALL
Java_fastspim_NativeSPIMReconstructionCuda_setCudaDevice(
		JNIEnv *env,
		jclass,
		jint deviceIdx)
{
	setCudaExceptionHandler(env);
	setCudaDevice(deviceIdx);
}

JNIEXPORT void JNICALL
Java_fastspim_NativeSPIMReconstructionCuda_transform8(
		JNIEnv *env,
		jclass,
		jobjectArray data,
		jint w,
		jint h,
		jint d,
		jfloatArray invMatrix,
		jint targetW,
		jint targetH,
		jint targetD,
		jstring outfile,
		jboolean createTransformedMasks,
		jint border,
		jfloat zspacing,
		jstring maskfile)

{
	setCudaExceptionHandler(env);
	TransformJNI<unsigned char> transform(env, data, w, h, d);
	transform.transform(env, invMatrix,
			targetW, targetH, targetD,
			outfile,
			createTransformedMasks,
			border,
			zspacing,
			maskfile);
}

JNIEXPORT void JNICALL
Java_fastspim_NativeSPIMReconstructionCuda_transform16(
		JNIEnv *env,
		jclass,
		jobjectArray data,
		jint w,
		jint h,
		jint d,
		jfloatArray invMatrix,
		jint targetW,
		jint targetH,
		jint targetD,
		jstring outfile,
		jboolean createTransformedMasks,
		jint border,
		jfloat zspacing,
		jstring maskfile)

{
	TransformJNI<unsigned short> transform(env, data, w, h, d);
	transform.transform(env, invMatrix,
			targetW, targetH, targetD,
			outfile,
			createTransformedMasks,
			border,
			zspacing,
			maskfile);
}

JNIEXPORT void JNICALL
Java_fastspim_NativeSPIMReconstructionCuda_deconvolve(
		JNIEnv *env,
		jclass,
		jobjectArray inputfiles,
		jstring outputfile,
		jint dataW,
		jint dataH,
		jint dataD,
		jobjectArray weightfiles,
		jobjectArray kernelfiles,
		jint kernelH,
		jint kernelW,
		jint iterationType,
		jint nViews,
		jint iterations,
		jint bitDepth)
{
	// get kernels files
	setCudaDevice(0);
	const char **kernels = (const char **)malloc(nViews * sizeof(char *));
	jstring *jkernels = (jstring *)malloc(nViews * sizeof(jstring));
	for(int v = 0; v < nViews; v++) {
		jkernels[v] = (jstring)env->GetObjectArrayElement(kernelfiles, v);
		kernels[v] = env->GetStringUTFChars(jkernels[v], NULL);
	}

	// get weight files
	const char **weights = (const char **)malloc(nViews * sizeof(char *));
	jstring *jweights = (jstring *)malloc(nViews * sizeof(jstring));
	for(int v = 0; v < nViews; v++) {
		jweights[v] = (jstring)env->GetObjectArrayElement(weightfiles, v);
		weights[v] = env->GetStringUTFChars(jweights[v], NULL);
	}

	// get input files
	const char **data = (const char **)malloc(nViews * sizeof(char *));
	jstring *jdata = (jstring *)malloc(nViews * sizeof(jstring));
	for(int v = 0; v < nViews; v++) {
		jdata[v] = (jstring)env->GetObjectArrayElement(inputfiles, v);
		data[v] = env->GetStringUTFChars(jdata[v], NULL);
	}

	// get output file
	const char *path = env->GetStringUTFChars(outputfile, NULL);

	// Do the deconvolution
	IterationType::Type type = (IterationType::Type)iterationType;

	setCudaExceptionHandler(env);
	if(bitDepth == 8) {
		DeconvolveFiles<unsigned char> dec(
				data,
				path,
				kernels,
				weights,
				dataW, dataH, dataD,
				kernelW, kernelH, nViews, type);
		dec.process(iterations);
	}
	else if(bitDepth == 16){
		DeconvolveFiles<unsigned short> dec(
				data,
				path,
				kernels,
				weights,
				dataW, dataH, dataD,
				kernelW, kernelH, nViews, type);
		dec.process(iterations);
	}
	else {
		ThrowException((void *)env, "Unsupported bit depth");
		return;
	}
	for(int v = 0; v < nViews; v++) {
		env->ReleaseStringUTFChars(jkernels[v], kernels[v]);
		env->ReleaseStringUTFChars(jweights[v], weights[v]);
		env->ReleaseStringUTFChars(jdata[v], data[v]);
	}
	free(kernels);
	free(weights);
	free(data);

	env->ReleaseStringUTFChars(outputfile, path);
}

static DeconvolveInteractiveJNI<unsigned char> *interactive8 = NULL;
static DeconvolveInteractiveJNI<unsigned short> *interactive16 = NULL;

JNIEXPORT void JNICALL Java_fastspim_NativeSPIMReconstructionCuda_deconvolve_1init(
		JNIEnv *env,
		jclass clazz,
		jint dataW,
		jint dataH,
		jint dataD,
		jobjectArray weightfiles,
		jobjectArray kernelfiles,
		jint kernelH,
		jint kernelW,
		jint iterationType,
		jint nViews,
		jint bitDepth)
{
	if(interactive8 != NULL || interactive16 != NULL) {
		ThrowException(env, "deconvolution already initialized");
		return;
	}

	// get kernel files
	const char **kernels = (const char **)malloc(nViews * sizeof(char *));
	jstring *jkernels = (jstring *)malloc(nViews * sizeof(jstring));
	for(int v = 0; v < nViews; v++) {
		jkernels[v] = (jstring)env->GetObjectArrayElement(kernelfiles, v);
		kernels[v] = env->GetStringUTFChars(jkernels[v], NULL);
	}

	// get weight files
	const char **weights = (const char **)malloc(nViews * sizeof(char *));
	jstring *jweights = (jstring *)malloc(nViews * sizeof(jstring));
	for(int v = 0; v < nViews; v++) {
		jweights[v] = (jstring)env->GetObjectArrayElement(weightfiles, v);
		weights[v] = env->GetStringUTFChars(jweights[v], NULL);
	}

	IterationType::Type type = (IterationType::Type)iterationType;

	setCudaExceptionHandler(env);

	if(bitDepth == 8) {
		interactive8 = new DeconvolveInteractiveJNI<unsigned char>(
			env, clazz,
			kernels,
			weights,
			dataW, dataH, dataD,
			kernelH, kernelW, nViews,
			type);
	}
	else if(bitDepth == 16) {
		interactive16 = new DeconvolveInteractiveJNI<unsigned short>(
			env, clazz,
			kernels,
			weights,
			dataW, dataH, dataD,
			kernelH, kernelW, nViews,
			type);
	}
	else {
		ThrowException((void *)env, "Unsupported bit depth");
		return;
	}
	for(int v = 0; v < nViews; v++) {
		env->ReleaseStringUTFChars(jkernels[v], kernels[v]);
		env->ReleaseStringUTFChars(jweights[v], weights[v]);
	}
	free(kernels);
	free(weights);
}

JNIEXPORT void JNICALL Java_fastspim_NativeSPIMReconstructionCuda_deconvolve_1interactive16(
		JNIEnv *env,
		jclass clazz,
		jint iterations)
{
	interactive16->process(iterations);
}

JNIEXPORT void JNICALL Java_fastspim_NativeSPIMReconstructionCuda_deconvolve_1interactive8(
		JNIEnv *env,
		jclass clazz,
		jint iterations)
{
	interactive8->process(iterations);
}

JNIEXPORT void JNICALL Java_fastspim_NativeSPIMReconstructionCuda_deconvolve_1quit16(
		JNIEnv *env,
		jclass clazz)
{
	if(interactive16 == NULL) {
		ThrowException(env, "deconvolution already quit");
		return;
	}
	delete interactive16;
	interactive16 = NULL;
}

JNIEXPORT void JNICALL Java_fastspim_NativeSPIMReconstructionCuda_deconvolve_1quit8(
		JNIEnv *env,
		jclass clazz)
{
	if(interactive8 == NULL) {
		ThrowException(env, "deconvolution already quit");
		return;
	}
	delete interactive8;
	interactive8 = NULL;
}

