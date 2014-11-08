:: nvcc -O2 --cl-version 2010 --use-local-env fmvd_transform_cuda.cu

nvcc --shared -O2 -Xcompiler "/EHsc /W3 /nologo /O2 /Zi /MT" --cl-version 2012 --use-local-env -lcufft --include-path "c:/Program Files/Java/jdk1.7.0_51/include" --include-path "c:/Program Files/Java/jdk1.7.0_51/include/win32" -o NativeSPIMReconstructionCuda.dll fmvd_transform_cuda.cu fastspim_NativeSPIMReconstructionCuda.cpp convolutionFFT2D.cu fmvd_deconvolve_cuda.cpp fmvd_deconvolve_common.cpp fmvd_deconvolve_cuda_hdd.cpp

:: nvcc --shared -O2 -lcufft --include-path "c:/Program Files/Java/jdk1.7.0_51/include" --include-path "c:/Program Files/Java/jdk1.7.0_51/include/win32" -o NativeSPIMReconstructionCuda.dll fmvd_transform_cuda.cu fastspim_NativeSPIMReconstructionCuda.cpp convolutionFFT2D.cu fmvd_deconvolve_cuda.cpp fmvd_deconvolve_common.cpp fmvd_deconvolve_cuda_hdd.cpp

copy /Y NativeSPIMReconstructionCuda.dll ..\..\Fiji.app
