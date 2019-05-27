CUDA_HOME   = /Soft/cuda/8.0.61

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include -arch=sm_20 --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -Wno-deprecated-gpu-targets -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
PROG_FLAGS  = -DSIZE=32

EXE01	        = kernel01.exe
OBJ01	        = main01.o

default: $(EXE01)

main01.o: bnFilter.cu
	$(NVCC) -c -o $@ bnFilter.cu $(NVCC_FLAGS) $(PROG_FLAGS)

$(EXE01): $(OBJ01)
	$(NVCC) $(OBJ01) -o $(EXE01) $(LD_FLAGS)


all:	$(EXE01) $(EXE0)

clean:
	rm -rf *.o* *.e*