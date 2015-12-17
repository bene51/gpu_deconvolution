#include "TransformJNI.h"
#include "Transform.h"

template<typename T>
TransformJNI<T>::TransformJNI(
		JNIEnv *env,
		jobjectArray data,
		jint width,
		jint height,
		jint depth) :
	env_(env), width_(width), height_(height), depth_(depth)
{
	javaArray_ = new jobject[depth];
	cArray_ = new T*[depth];
	for(int z = 0; z < depth; z++)
		initPlane(env, data, z);
}

// template version of initPlane
template<typename T>
void
TransformJNI<T>::initPlane(
		JNIEnv *env,
		jobjectArray data,
		int z)
{
	printf("error\n");
}

// explicit specification of initPlane for 8-bit data
void
TransformJNI<unsigned char>::initPlane(
		JNIEnv *env,
		jobjectArray data,
		int z)
{
	javaArray_[z] = env->GetObjectArrayElement(data, z);
	cArray_[z] = (unsigned char *)env->GetByteArrayElements(
			(jbyteArray)javaArray_[z], NULL);
	if(!cArray_[z]) {
		printf("not enough memory\n");
		return; // TODO throw
	}
}

// explicit specification of initPlane for 16-bit data
void
TransformJNI<unsigned short>::initPlane(
		JNIEnv *env,
		jobjectArray data,
		int z)
{
	javaArray_[z] = env->GetObjectArrayElement(data, z);
	cArray_[z] = (unsigned short *)env->GetShortArrayElements(
			(jshortArray)javaArray_[z], NULL);
	if(!cArray_[z]) {
		printf("not enough memory\n");
		return; // TODO throw
	}
}

// template version of deletePlane
template<typename T>
void
TransformJNI<T>::deletePlane(JNIEnv *env, int z)
{
	printf("error\n");
}

// explicit specification of deletePlane for 8-bit data
void
TransformJNI<unsigned char>::deletePlane(JNIEnv *env, int z)
{
	env->ReleaseByteArrayElements(
			(jbyteArray)javaArray_[z],
			(jbyte *)cArray_[z],
			JNI_ABORT);
}

// explicit specification of deletePlane for 16-bit data
void
TransformJNI<unsigned short>::deletePlane(JNIEnv *env, int z)
{
	env->ReleaseShortArrayElements(
			(jshortArray)javaArray_[z],
			(jshort *)cArray_[z],
			JNI_ABORT);
}

template<typename T>
void TransformJNI<T>::transform(
		JNIEnv *env,
		jfloatArray inverseTransform,
		jint targetWidth,
		jint targetHeight,
		jint targetDepth,
		jstring outfile,
		jboolean createTransformedMasks,
		jint border,
		jfloat zspacing,
		jstring maskfile)
{
	float *mat = (float *)env->GetFloatArrayElements(inverseTransform, NULL);
	const char *outpath = env->GetStringUTFChars(outfile, NULL);

	Transform<T> transform(cArray_, width_, height_, depth_,
			targetWidth, targetHeight, targetDepth, mat);
	transform.transform(outpath);
	if(createTransformedMasks) {
		const char *maskpath = env->GetStringUTFChars(maskfile, NULL);
		transform.createWeights(border, zspacing, maskpath);
		env->ReleaseStringUTFChars(maskfile, maskpath);
	}

	env->ReleaseFloatArrayElements(inverseTransform, mat, JNI_ABORT);
	env->ReleaseStringUTFChars(outfile, outpath);
}

template<typename T>
TransformJNI<T>::~TransformJNI()
{
	for(int z = 0; z < depth_; z++)
		deletePlane(env_, z);
	delete[] javaArray_;
	delete[] cArray_;
}

// explicit template instantiation
template class TransformJNI<unsigned char>;
template class TransformJNI<unsigned short>;

