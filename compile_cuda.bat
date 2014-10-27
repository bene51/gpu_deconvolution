nvcc -O2 --cl-version 2010 --use-local-env -lcufft fmvd_transform_cuda.cu convolutionFFT2D.cu fmvd_deconvolve_cuda.cpp fmvd_deconvolve_common.cpp fmvd_deconvolve_cuda_hdd.cpp
