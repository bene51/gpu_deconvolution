NVCC := nvcc
INCL := --include-path "$(JAVA_HOME)/include" --include-path "$(JAVA_HOME)/include/linux"
NVCC_COMPILE_FLAGS = -c -O2 -m64 -Xcompiler -fPIC
NVCC_LINK_FLAGS = --shared -m64 -lcufft

all: libNativeSPIMReconstructionCuda.so

CPP_SOURCES = fastspim_NativeSPIMReconstructionCuda.cpp \
	      fmvd_deconvolve_cuda.cpp \
	      fmvd_deconvolve_cuda_hdd.cpp \
	      fmvd_deconvolve_common.cpp

CU_SOURCES  = fmvd_transform_cuda.cu \
	      convolutionFFT2D.cu

CPP_OBJS = fastspim_NativeSPIMReconstructionCuda.obj fmvd_deconvolve_cuda.obj fmvd_deconvolve_cuda_hdd.obj fmvd_deconvolve_common.obj
CU_OBJS = fmvd_transform_cuda.obj convolutionFFT2D.obj 

REBUILDABLES = $(CPP_OBJS) $(CU_OBJS) libNativeSPIMReconstructionCuda.so

clean :
	rm -f $(REBUILDABLES)

libNativeSPIMReconstructionCuda.so : $(CPP_OBJS) $(CU_OBJS)
	$(NVCC) $(NVCC_LINK_FLAGS) $^ -o $@

$(CPP_OBJS) : %.obj : %.cpp
	$(NVCC) $(NVCC_COMPILE_FLAGS) $(INCL) -o $@ $<

$(CU_OBJS) : %.obj : %.cu
	$(NVCC) $(NVCC_COMPILE_FLAGS) $(INCL) -o $@ $<


