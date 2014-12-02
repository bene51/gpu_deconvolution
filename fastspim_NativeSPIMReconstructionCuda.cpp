#ifdef _WIN32
#define _CRT_SECURE_NO_DEPRECATE
#endif
#include <string.h>
#include <stdlib.h>
#include "fastspim_NativeSPIMReconstructionCuda.h"
#include "fmvd_transform.h"
#include "fmvd_deconvolve_cuda.h"
#include "fmvd_deconvolve_common.h"

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
	printf("Should transform now, but just testing at the moment\n");
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
	
	transform_cuda(cdata, w, h, d, targetW, targetH, targetD, mat, outpath,
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
		jint iterations)
{
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

	data_t **h_Weights = (data_t **)malloc(nViews * sizeof(data_t *));
	int datasize = dataW * dataH;
	for(int v = 0; v < nViews; v++) {
		h_Weights[v] = (data_t *)malloc(datasize * sizeof(data_t));
		jstring path = (jstring)env->GetObjectArrayElement(weightfiles, v);
		const char *wFile = env->GetStringUTFChars(path, NULL);
		FILE *f = fopen(wFile, "rb");
		fread(h_Weights[v], sizeof(data_t), datasize, f);
		fclose(f);
		env->ReleaseStringUTFChars(path, wFile);
	}


	// Open input files
	FILE **dataFiles = (FILE**)malloc(nViews * sizeof(FILE *));
	for(int v = 0; v < nViews; v++) {
		jstring jpath = (jstring)env->GetObjectArrayElement(inputfiles, v);
		const char *path = env->GetStringUTFChars(jpath, NULL);
		dataFiles[v] = fopen(path, "rb");
		// fseek(dataFiles[v], 300 * dataW * dataH * sizeof(data_t), SEEK_SET); // TODO remove
		env->ReleaseStringUTFChars(jpath, path);
	}

	// Open output file
	const char *path = env->GetStringUTFChars(outputfile, NULL);
	FILE *resultFile = fopen(path, "wb");
	env->ReleaseStringUTFChars(outputfile, path);

	// Do the deconvolution
	fmvd_psf_type iteration_type = (fmvd_psf_type)iterationType;
	fmvd_deconvolve_files_cuda(dataFiles, resultFile, dataW, dataH, dataD, h_Weights, kernel, kernelH, kernelW, iteration_type, nViews, iterations);

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

