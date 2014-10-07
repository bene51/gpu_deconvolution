nvcc -O2 --cl-version 2010 --use-local-env -Xcompiler /EHsc,/Zi,/MT -L fftw-3.3.4-dll64 -llibfftw3f-3 fmvd_deconvolve_fftw.cpp fmvd_deconvolve_common.cpp
