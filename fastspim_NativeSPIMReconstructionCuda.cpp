#ifdef _WIN32
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include "fastspim_NativeSPIMReconstructionCuda.h"

#include <string.h>
#include <stdlib.h>
#include "fmvd_transform.h"
#include "fmvd_deconvolve.h"
#include "fmvd_cuda_utils.h"

static void ThrowException(void *env_ptr, const char *message)
{
	JNIEnv *env = (JNIEnv *)env_ptr;
	jclass cl;
	cl = env->FindClass("java/lang/RuntimeException");
	env->ThrowNew(cl, message);
}

static void setCudaExceptionHandler(JNIEnv *env)
{
	setErrorHandler(ThrowException, env);
}

JNIEXPORT jint JNICALL Java_fastspim_NativeSPIMReconstructionCuda_getNumCudaDevices(
		JNIEnv *env,
		jclass)
{
	setCudaExceptionHandler(env);
	return getNumCUDADevices();
}

JNIEXPORT jstring JNICALL Java_fastspim_NativeSPIMReconstructionCuda_getCudaDeviceName(
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

JNIEXPORT void JNICALL Java_fastspim_NativeSPIMReconstructionCuda_setCudaDevice(
		JNIEnv *env,
		jclass,
		jint deviceIdx)
{
	setCudaExceptionHandler(env);
	setCudaDevice(deviceIdx);
}

struct interactive_transform {
	JavaVM *jvm;
	jmethodID receivePlaneMethodID;
	jobject callback;
	int planesize;
};

struct interactive_transform *ia_transform = NULL;

void
transform_return_next_plane_16(unsigned short *data)
{
	JNIEnv *env;
	ia_transform->jvm->AttachCurrentThread((void **)&env, NULL);
	jshortArray param = env->NewShortArray(ia_transform->planesize);
	env->SetShortArrayRegion(param, 0, ia_transform->planesize, (jshort *)data);
	env->CallVoidMethod(ia_transform->callback, ia_transform->receivePlaneMethodID, param);
}

void
transform_return_next_plane_8(unsigned char *data)
{
	JNIEnv *env;
	ia_transform->jvm->AttachCurrentThread((void **)&env, NULL);
	jbyteArray param = env->NewByteArray(ia_transform->planesize);
	env->SetByteArrayRegion(param, 0, ia_transform->planesize, (jbyte *)data);
	env->CallVoidMethod(ia_transform->callback, ia_transform->receivePlaneMethodID, param);
}

JNIEXPORT void JNICALL Java_fastspim_NativeSPIMReconstructionCuda_transform16Interactive(
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
		jobject callback)
{
	int z;
	int planesize = w * h * sizeof(unsigned short);
	unsigned short **cdata = (unsigned short **)malloc(d * sizeof(unsigned short *));
	jshortArray *jdata = (jshortArray *)malloc(d * sizeof(jshortArray));
	if(!cdata) {
		printf("not enough memory\n");
		return;
	}
	for(z = 0; z < d; z++) {
		jdata[z] = (jshortArray)env->GetObjectArrayElement(data, z);
		cdata[z] = (unsigned short *)env->GetShortArrayElements(jdata[z], NULL);
		if(!cdata[z]) {
			printf("not enough memory\n");
			return;
		}
	}

	float *mat = (float *)env->GetFloatArrayElements(invMatrix, NULL);

	setCudaExceptionHandler(env);

	// setup callback
	JavaVM *jvm = NULL;
	env->GetJavaVM(&jvm);
	jclass cls = env->GetObjectClass(callback);
	jmethodID mid = env->GetMethodID(cls, "receivePlane", "(Ljava/lang/Object;)V");
	if(mid == 0) {
		printf("Could not find method\n");
		return;
	}

	ia_transform = (struct interactive_transform *)malloc(sizeof(struct interactive_transform));
	ia_transform->jvm = jvm;
	ia_transform->receivePlaneMethodID = mid;
	ia_transform->callback = callback;
	ia_transform->planesize = targetW * targetH;

	transform_cuda_16(cdata, w, h, d, targetW, targetH, targetD, mat, transform_return_next_plane_16,
		NULL, false, 0, 0, NULL);

	for(z = 0; z < d; z++)
		env->ReleaseShortArrayElements(jdata[z], (jshort *)cdata[z], JNI_ABORT);

	env->ReleaseFloatArrayElements(invMatrix, mat, JNI_ABORT);

	free(cdata);
	free(jdata);
}

JNIEXPORT void JNICALL Java_fastspim_NativeSPIMReconstructionCuda_transform8Interactive(
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
		jobject callback)
{
	int z;
	int planesize = w * h * sizeof(unsigned char);
	unsigned char **cdata = (unsigned char **)malloc(d * sizeof(unsigned char *));
	jbyteArray *jdata = (jbyteArray *)malloc(d * sizeof(jbyteArray));
	if(!cdata) {
		printf("not enough memory\n");
		return;
	}
	for(z = 0; z < d; z++) {
		jdata[z] = (jbyteArray)env->GetObjectArrayElement(data, z);
		cdata[z] = (unsigned char *)env->GetByteArrayElements(jdata[z], NULL);
		if(!cdata[z]) {
			printf("not enough memory\n");
			return;
		}
	}

	float *mat = (float *)env->GetFloatArrayElements(invMatrix, NULL);

	setCudaExceptionHandler(env);

	// setup callback
	JavaVM *jvm = NULL;
	env->GetJavaVM(&jvm);
	jclass cls = env->GetObjectClass(callback);
	jmethodID mid = env->GetMethodID(cls, "receivePlane", "()Ljava/lang/Object;");
	if(mid == 0)
		printf("Could not find method\n");

	ia_transform = (struct interactive_transform *)malloc(sizeof(struct interactive_transform));
	ia_transform->jvm = jvm;
	ia_transform->receivePlaneMethodID = mid;
	ia_transform->callback = callback;
	ia_transform->planesize = targetW * targetH;

	transform_cuda_8(cdata, w, h, d, targetW, targetH, targetD, mat, transform_return_next_plane_8,
		NULL, false, 0, 0, NULL);

	for(z = 0; z < d; z++)
		env->ReleaseByteArrayElements(jdata[z], (jbyte *)cdata[z], JNI_ABORT);

	env->ReleaseFloatArrayElements(invMatrix, mat, JNI_ABORT);

	free(cdata);
	free(jdata);
}

JNIEXPORT void JNICALL Java_fastspim_NativeSPIMReconstructionCuda_transform16(
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
	int z;
	int planesize = w * h * sizeof(unsigned short);
	unsigned short **cdata = (unsigned short **)malloc(d * sizeof(unsigned short *));
	jshortArray *jdata = (jshortArray *)malloc(d * sizeof(jshortArray));
	if(!cdata) {
		printf("not enough memory\n");
		return;
	}
	for(z = 0; z < d; z++) {
		jdata[z] = (jshortArray)env->GetObjectArrayElement(data, z);
		cdata[z] = (unsigned short *)env->GetShortArrayElements(jdata[z], NULL);
		if(!cdata[z]) {
			printf("not enough memory\n");
			return;
		}
	}

	float *mat = (float *)env->GetFloatArrayElements(invMatrix, NULL);

	const char *outpath = env->GetStringUTFChars(outfile, NULL);
	const char *maskpath = NULL;
	if(createTransformedMasks)
		maskpath = env->GetStringUTFChars(maskfile, NULL);

	setCudaExceptionHandler(env);
	transform_cuda_16(cdata, w, h, d, targetW, targetH, targetD, mat, NULL, outpath,
		createTransformedMasks, border, zspacing, maskpath);

	for(z = 0; z < d; z++)
		env->ReleaseShortArrayElements(jdata[z], (jshort *)cdata[z], JNI_ABORT);

	env->ReleaseFloatArrayElements(invMatrix, mat, JNI_ABORT);
	env->ReleaseStringUTFChars(outfile, outpath);
	if(createTransformedMasks)
		env->ReleaseStringUTFChars(maskfile, maskpath);

	free(cdata);
	free(jdata);
}

JNIEXPORT void JNICALL Java_fastspim_NativeSPIMReconstructionCuda_transform8(
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
	int z;
	int planesize = w * h * sizeof(unsigned char);
	unsigned char **cdata = (unsigned char **)malloc(d * sizeof(unsigned char *));
	jbyteArray *jdata = (jbyteArray *)malloc(d * sizeof(jbyteArray));
	if(!cdata) {
		printf("not enough memory\n");
		return;
	}
	for(z = 0; z < d; z++) {
		jdata[z] = (jbyteArray)env->GetObjectArrayElement(data, z);
		cdata[z] = (unsigned char *)env->GetByteArrayElements(jdata[z], NULL);
		if(!cdata[z]) {
			printf("not enough memory\n");
			return;
		}
	}

	float *mat = (float *)env->GetFloatArrayElements(invMatrix, NULL);

	const char *outpath = env->GetStringUTFChars(outfile, NULL);
	const char *maskpath = NULL;
	if(createTransformedMasks)
		maskpath = env->GetStringUTFChars(maskfile, NULL);

	setCudaExceptionHandler(env);
	transform_cuda_8(cdata, w, h, d, targetW, targetH, targetD, mat, NULL, outpath,
		createTransformedMasks, border, zspacing, maskpath);

	for(z = 0; z < d; z++)
		env->ReleaseByteArrayElements(jdata[z], (jbyte *)cdata[z], JNI_ABORT);

	env->ReleaseFloatArrayElements(invMatrix, mat, JNI_ABORT);
	env->ReleaseStringUTFChars(outfile, outpath);
	if(createTransformedMasks)
		env->ReleaseStringUTFChars(maskfile, maskpath);

	free(cdata);
	free(jdata);
}

JNIEXPORT void JNICALL Java_fastspim_NativeSPIMReconstructionCuda_deconvolve(
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
	// Read kernels
	setCudaDevice(0);
	float **kernel = (float **)malloc(nViews * sizeof(float *));
	for(int v = 0; v < nViews; v++) {
		kernel[v] = (float *)fmvd_malloc(kernelW * kernelH * sizeof(float));
		jstring jpath = (jstring)env->GetObjectArrayElement(kernelfiles, v);
		const char *path = env->GetStringUTFChars(jpath, NULL);
		FILE *f = fopen(path, "rb");
		fread(kernel[v], sizeof(float), kernelW * kernelH, f);
		fclose(f);
		env->ReleaseStringUTFChars(jpath, path);
	}

	float **h_Weights = (float **)malloc(nViews * sizeof(float *));
	int datasize = dataW * dataH;
	for(int v = 0; v < nViews; v++) {
		h_Weights[v] = (float *)malloc(datasize * sizeof(float));
		jstring path = (jstring)env->GetObjectArrayElement(weightfiles, v);
		const char *wFile = env->GetStringUTFChars(path, NULL);
		FILE *f = fopen(wFile, "rb");
		fread(h_Weights[v], sizeof(float), datasize, f);
		fclose(f);
		env->ReleaseStringUTFChars(path, wFile);
	}


	// Open input files
	FILE **dataFiles = (FILE**)malloc(nViews * sizeof(FILE *));
	for(int v = 0; v < nViews; v++) {
		jstring jpath = (jstring)env->GetObjectArrayElement(inputfiles, v);
		const char *path = env->GetStringUTFChars(jpath, NULL);
		dataFiles[v] = fopen(path, "rb");
		env->ReleaseStringUTFChars(jpath, path);
	}

	// Open output file
	const char *path = env->GetStringUTFChars(outputfile, NULL);
	FILE *resultFile = fopen(path, "wb");
	env->ReleaseStringUTFChars(outputfile, path);

	// Do the deconvolution
	setCudaExceptionHandler(env);
	fmvd_psf_type iteration_type = (fmvd_psf_type)iterationType;

	if(bitDepth == 8)
		fmvd_deconvolve_files_cuda_8(dataFiles, resultFile, dataW, dataH, dataD, h_Weights, kernel, kernelH, kernelW, iteration_type, nViews, iterations);
	else if(bitDepth == 16)
		fmvd_deconvolve_files_cuda_16(dataFiles, resultFile, dataW, dataH, dataD, h_Weights, kernel, kernelH, kernelW, iteration_type, nViews, iterations);
	else {
		ThrowException((void *)env, "Unsupported bit depth");
		return;
	}


	// Close input and output files
	for(int v = 0; v < nViews; v++)
		fclose(dataFiles[v]);
	fclose(resultFile);

	// Cleanup
	for(int v = 0; v < nViews; v++) {
		fmvd_free(kernel[v]);
		free(h_Weights[v]);
	}
	free(kernel);
	free(h_Weights);
	free(dataFiles);
}

struct interactive_8 {
	fmvd_plan_cuda_8 *plan;
	JavaVM *jvm;
	JNIEnv *env;
	jmethodID getNextPlaneMethodID;
	jmethodID returnNextPlaneMethodID;
	jclass callingClass;
	int planesize;
	int nViews;
};

struct interactive_16 {
	fmvd_plan_cuda_16 *plan;
	JavaVM *jvm;
	JNIEnv *env;
	jmethodID getNextPlaneMethodID;
	jmethodID returnNextPlaneMethodID;
	jclass callingClass;
	int planesize;
	int nViews;
};


struct interactive_8 *ia_8 = NULL;
struct interactive_16 *ia_16 = NULL;

static struct interactive_8 *
init_ia_8(fmvd_plan_cuda_8 *plan, JavaVM *jvm, jmethodID get, jmethodID ret, jclass caller, int planesize, int nViews)
{
	struct interactive_8 *ia = (struct interactive_8 *)malloc(sizeof(struct interactive_8));
	ia->plan = plan;
	ia->jvm = jvm;
	ia->getNextPlaneMethodID = get;
	ia->returnNextPlaneMethodID = ret;
	ia->callingClass = caller;
	ia->planesize = planesize;
	ia->nViews = nViews;
	return ia;
}

static struct interactive_16 *
init_ia_16(fmvd_plan_cuda_16 *plan, JavaVM *jvm, jmethodID get, jmethodID ret, jclass caller, int planesize, int nViews)
{
	struct interactive_16 *ia = (struct interactive_16 *)malloc(sizeof(struct interactive_16));
	ia->plan = plan;
	ia->jvm = jvm;
	ia->getNextPlaneMethodID = get;
	ia->returnNextPlaneMethodID = ret;
	ia->callingClass = caller;
	ia->planesize = planesize;
	ia->nViews = nViews;
	return ia;
}

static void
free_ia_8(struct interactive_8 *ia)
{
	free(ia);
}

static void
free_ia_16(struct interactive_16 *ia)
{
	free(ia);
}

static int
get_next_plane_8(unsigned char **data, int offset)
{
	JNIEnv *env;
	ia_8->jvm->AttachCurrentThread((void **)&env, NULL);
	jobjectArray jarr = (jobjectArray)env->CallStaticObjectMethod(ia_8->callingClass, ia_8->getNextPlaneMethodID);
	if(jarr == NULL) {
		return 0;
	}
	for(int v = 0; v < ia_8->nViews; v++) {
		jbyteArray view = (jbyteArray)env->GetObjectArrayElement(jarr, v);
		unsigned char *tgt = (unsigned char *)env->GetPrimitiveArrayCritical(view, NULL);
		memcpy(data[v] + offset, tgt, ia_8->planesize * sizeof(unsigned char));
		env->ReleasePrimitiveArrayCritical(view, tgt, JNI_ABORT);
	}
	return 1;
}

static int
get_next_plane_16(unsigned short **data, int offset)
{
	JNIEnv *env;
	ia_16->jvm->AttachCurrentThread((void **)&env, NULL);
	jobjectArray jarr = (jobjectArray)env->CallStaticObjectMethod(ia_16->callingClass, ia_16->getNextPlaneMethodID);
	if(jarr == NULL) {
		return 0;
	}
	for(int v = 0; v < ia_16->nViews; v++) {
		jshortArray view = (jshortArray)env->GetObjectArrayElement(jarr, v);
		unsigned short *tgt = (unsigned short *)env->GetPrimitiveArrayCritical(view, NULL);
		memcpy(data[v] + offset, tgt, ia_16->planesize * sizeof(unsigned short));
		env->ReleasePrimitiveArrayCritical(view, tgt, JNI_ABORT);
	}
	return 1;
}

static void
return_next_plane_8(unsigned char *data)
{
	JNIEnv *env;
	ia_8->jvm->AttachCurrentThread((void **)&env, NULL);
	jbyteArray param = env->NewByteArray(ia_8->planesize);
	env->SetByteArrayRegion(param, 0, ia_8->planesize, (jbyte *)data);
	env->CallStaticVoidMethod(ia_8->callingClass, ia_8->returnNextPlaneMethodID, param);
}

static void
return_next_plane_16(unsigned short *data)
{
	JNIEnv *env;
	ia_16->jvm->AttachCurrentThread((void **)&env, NULL);
	jshortArray param = env->NewShortArray(ia_16->planesize);
	env->SetShortArrayRegion(param, 0, ia_16->planesize, (jshort *)data);
	env->CallStaticVoidMethod(ia_16->callingClass, ia_16->returnNextPlaneMethodID, param);
}

JNIEXPORT void JNICALL Java_fastspim_NativeSPIMReconstructionCuda_deconvolve_1quit8(
		JNIEnv *env,
		jclass clazz)
{
	if(ia_8 == NULL) {
		ThrowException(env, "deconvolution already quit");
		return;
	}
	fmvd_destroy_cuda_8(ia_8->plan);
	free_ia_8(ia_8);
	ia_8 = NULL;
}

JNIEXPORT void JNICALL Java_fastspim_NativeSPIMReconstructionCuda_deconvolve_1quit16(
		JNIEnv *env,
		jclass clazz)
{
	if(ia_16 == NULL) {
		ThrowException(env, "deconvolution already quit");
		return;
	}
	fmvd_destroy_cuda_16(ia_16->plan);
	free_ia_16(ia_16);
	ia_16 = NULL;
}

JNIEXPORT void JNICALL Java_fastspim_NativeSPIMReconstructionCuda_deconvolve_1interactive8(
		JNIEnv *env,
		jclass clazz,
		jint iterations)
{
	fmvd_deconvolve_planes_cuda_8(ia_8->plan, iterations);
}

JNIEXPORT void JNICALL Java_fastspim_NativeSPIMReconstructionCuda_deconvolve_1interactive16(
		JNIEnv *env,
		jclass clazz,
		jint iterations)
{
	fmvd_deconvolve_planes_cuda_16(ia_16->plan, iterations);
}

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
	switch(bitDepth) {
	case 8:
		if(ia_8 != NULL) {
			ThrowException(env, "deconvolution already initialized");
			return;
		}
		break;
	case 16:
		if(ia_16 != NULL) {
			ThrowException(env, "deconvolution already initialized");
			return;
		}
		break;
	}



	// Read kernels
	float **kernel = (float **)malloc(nViews * sizeof(float *));
	for(int v = 0; v < nViews; v++) {
		kernel[v] = (float *)fmvd_malloc(kernelW * kernelH * sizeof(float));
		jstring jpath = (jstring)env->GetObjectArrayElement(kernelfiles, v);
		const char *path = env->GetStringUTFChars(jpath, NULL);
		FILE *f = fopen(path, "rb");
		fread(kernel[v], sizeof(float), kernelW * kernelH, f);
		fclose(f);
		env->ReleaseStringUTFChars(jpath, path);
	}

	// Read weights
	float **h_Weights = (float **)malloc(nViews * sizeof(float *));
	int datasize = dataW * dataH;
	for(int v = 0; v < nViews; v++) {
		h_Weights[v] = (float *)malloc(datasize * sizeof(float));
		jstring path = (jstring)env->GetObjectArrayElement(weightfiles, v);
		const char *wFile = env->GetStringUTFChars(path, NULL);
		FILE *f = fopen(wFile, "rb");
		fread(h_Weights[v], sizeof(float), datasize, f);
		fclose(f);
		env->ReleaseStringUTFChars(path, wFile);
	}


	// Do the deconvolution
	setCudaExceptionHandler(env);
	fmvd_psf_type iteration_type = (fmvd_psf_type)iterationType;

	int nStreams = dataD < 3 ? dataD : 3;

	void *plan = NULL;
	jmethodID get, ret;
	switch(bitDepth) {
	case 8: plan = (struct fmvd_plan_cuda_8 *)fmvd_initialize_cuda_8(
			dataH, dataW,
			h_Weights,
			kernel, kernelH, kernelW, iteration_type,
			nViews, nStreams,
			get_next_plane_8,
			return_next_plane_8);
			get = env->GetStaticMethodID(clazz, "getNextPlane", "()[Ljava/lang/Object;");
			ret = env->GetStaticMethodID(clazz, "returnNextPlane", "(Ljava/lang/Object;)V");
		break;
	case 16: plan = (struct fmvd_plan_cuda_16 *)fmvd_initialize_cuda_16(
			dataH, dataW,
			h_Weights,
			kernel, kernelH, kernelW, iteration_type,
			nViews, nStreams,
			get_next_plane_16,
			return_next_plane_16);
			get = env->GetStaticMethodID(clazz, "getNextPlane", "()[Ljava/lang/Object;");
			ret = env->GetStaticMethodID(clazz, "returnNextPlane", "(Ljava/lang/Object;)V");
		break;
	}

	if(get == NULL) {
		printf("error in callback: Method not found\n");
		return;
	}

	if(ret == NULL) {
		printf("error in callback: Method not found\n");
		return;
	}

	JavaVM *jvm = NULL;
	env->GetJavaVM(&jvm);

	switch(bitDepth) {
	case 8:
		ia_8 = (struct interactive_8 *)init_ia_8((struct fmvd_plan_cuda_8 *)plan, jvm, get, ret, clazz, dataW * dataH, nViews);
		break;
	case 16:
		ia_16 = (struct interactive_16 *)init_ia_16((struct fmvd_plan_cuda_16 *)plan, jvm, get, ret, clazz, dataW * dataH, nViews);
		break;
	}

	// Cleanup
	for(int v = 0; v < nViews; v++) {
		fmvd_free(kernel[v]);
		free(h_Weights[v]);
	}
	free(kernel);
	free(h_Weights);
}
