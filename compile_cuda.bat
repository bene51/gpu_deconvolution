nvcc -O2 --cl-version 2010 --use-local-env -lcufft convolutionFFT2D.cu fmvd_deconvolve_cuda.cpp fmvd_deconvolve_common.cpp multithreading.cpp
