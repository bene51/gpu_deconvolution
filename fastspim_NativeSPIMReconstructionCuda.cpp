#include "fastspim_NativeSPIMReconstructionCuda.h"
#include "fmvd_transform_cuda.h"
#include "fmvd_deconvolve_cuda.h"
#include "fmvd_deconvolve_common.h"
#include <string.h>
#include <stdlib.h>

JNIEXPORT void JNICALL Java_fastspim_NativeSPIMReconstructionCuda_transform(
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
	unsigned short *cdata = (unsigned short *)malloc(d * planesize);
	const char *outpath = env->GetStringUTFChars(outfile, NULL);
	const char *maskpath = NULL;
	if(createTransformedMasks)
		maskpath = env->GetStringUTFChars(maskfile, NULL);
	for(z = 0; z < d; z++) {
		jshortArray plane = (jshortArray)env->GetObjectArrayElement(data, z);
		jshort *elements = env->GetShortArrayElements(plane, NULL);
		memcpy(cdata + z * w * h, elements, planesize);
		env->ReleaseShortArrayElements(plane, elements, JNI_ABORT);
	}
	float *mat = (float *)env->GetFloatArrayElements(invMatrix, NULL);
	
	transform_cuda(cdata, w, h, d, targetW, targetH, targetD, mat, outpath,
		createTransformedMasks, border, zspacing, maskpath);
	
	env->ReleaseFloatArrayElements(invMatrix, mat, JNI_ABORT);
	env->ReleaseStringUTFChars(outfile, outpath);
	if(createTransformedMasks)
		env->ReleaseStringUTFChars(maskfile, maskpath);

	free(cdata);
}

JNIEXPORT void JNICALL Java_fastspim_NativeSPIMReconstructionCuda_deconvolve(
		JNIEnv *env,
		jclass,
		jobjectArray inputfiles,
		jstring outputfile,
		jint dataW,
		jint dataH,
		jint dataD,
		jobjectArray kernelfiles,
		jint kernelH,
		jint kernelW,
		jint nViews,
		jint iterations)
{
	int it;

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
		normalize(kernel[v], kernelW * kernelH);
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
	fmvd_deconvolve_files_cuda(dataFiles, resultFile, dataW, dataH, dataD, kernel, kernelH, kernelW, nViews, iterations);

	// Close input and output files
	for(int v = 0; v < nViews; v++)
		fclose(dataFiles[v]);
	fclose(resultFile);

	// Cleanup
	for(int v = 0; v < nViews; v++)
		fmvd_free(kernel[v]);
	free(kernel);
	free(dataFiles);
}

