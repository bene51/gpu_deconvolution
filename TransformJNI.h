#ifndef TRANSFORM_JNI_H
#define TRANSFORM_JNI_H

#include <jni.h>

template<typename T>
class TransformJNI
{
private:
	jobject *javaArray_;
	T **cArray_;

	JNIEnv *env_;
	jint width_, height_, depth_;

	void initPlane(JNIEnv *env, jobjectArray data, int z);
	void deletePlane(JNIEnv *env, int z);
public:
	TransformJNI(JNIEnv *env, jobjectArray data,
			jint width, jint height, jint depth);
	~TransformJNI();
	void transform(JNIEnv *env,
		jfloatArray inverseTransform,
		jint targetWidth,
		jint targetHeight,
		jint targetDepth,
		jstring outfile,
		jboolean createTransformedMasks,
		jint border,
		jfloat zspacing,
		jstring maskfile);
};


#endif

