#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#include <math.h>
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <time.h>
#include <dirent.h>

#define SIZE 32

struct Frame {
	uint8_t* data;
	int width;
	int height;
	int bpp;
};

struct Frame read_frame() {
    int width, height, bpp;
	struct Frame result;
    uint8_t* rgb_image = stbi_load("pics/thumb1.jpg", &width, &height, &bpp, 3);
	result.data = rgb_image;
	result.width = width;
	result.height = height;
	result.bpp = bpp;
    //stbi_image_free(rgb_image);
    return result;
}

void process_frame_bn(struct Frame* frame, char* filename) {
	for (int i = 0; i < (*frame).width*(*frame).height*3; i+=3)
	{
		int R = (*frame).data[i];
		int G = (*frame).data[i+1];
		int B = (*frame).data[i+2];
		int gray = (R*0.299 + G*0.587 + B*0.114);
		(*frame).data[i]=gray;
		(*frame).data[i+1] = gray;
		(*frame).data[i+2] = gray;
	}
	char ruta [300];
	sprintf(ruta, "pics2/%s",filename);
	stbi_write_jpg(ruta, frame->width, frame->height, 3, frame->data, frame->width*3);
}

int max(int n1) {
	return n1>255 ? 255 : n1;
}


////////////////////  |
///CODIGO CUDA//////  |
///////////////////   v

__global__ void KernelByN (int N, int M, uint8_t *A) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + 3*threadIdx.x;
  printf("%dX%d\n", N, M);

  if(row < M && col < N){
		printf("%d\n", A[row*N+col+1]);
		A[row*N+col] = A[row*N+col+1] = A[row*N+col+2] = (A[row*N+col] + A[row*N+col+1] + A[row*N+col+2])/3;
  }
}

void CheckCudaError(char sms[], int line);


int main(int argc, char** argv)
{
	unsigned int Nfil, Ncol;
  	unsigned int numBytes;
	unsigned int nThreads;

	float TiempoTotal, TiempoKernel;
  	cudaEvent_t E0, E1, E2, E3;

  	uint8_t *h_A;
  	uint8_t *h_B;
  	uint8_t *d_A;

  	///Leer jpeg de la carpeta////
  	struct Frame frame = read_frame();
  	Nfil = frame.width;
  	Ncol = frame.height;
  	printf("%dX%d\n", Nfil, Ncol);
  	//////////////////////////////

  	numBytes = Nfil * Ncol * 3 * sizeof(uint8_t);

  	cudaEventCreate(&E0);	cudaEventCreate(&E1);
	cudaEventCreate(&E2);	cudaEventCreate(&E3);

	// Obtener Memoria en el host
    h_A = (uint8_t*) malloc(numBytes);
    h_B = (uint8_t*) malloc(numBytes);
    // Inicializa con los datos de la imagen leída
    h_A = frame.data;
    //printf("prova= %d\n", h_A[1256]);

    // Obtener Memoria en el device
  	cudaMalloc((uint8_t**)&d_A, numBytes);
  	// Copiar datos desde el host en el device 
	cudaMemcpy(d_A, h_A, numBytes, cudaMemcpyHostToDevice);
	CheckCudaError((char *) "Copiar Datos Host --> Device", __LINE__);
  	
	//
  	// KERNEL ELEMENTO a ELEMENTO
  	//

	// numero de Threads en cada dimension 
  	nThreads = SIZE;
  	printf("%d\n", nThreads);

	// numero de Blocks en cada dimension
  	int nBlocksFil = (Nfil+nThreads-1)/nThreads; //tener en cuenta 3componentes RGB??
  	int nBlocksCol = (Ncol+nThreads-1)/nThreads;

  	dim3 dimGridE(nBlocksCol, nBlocksFil, 1);
  	dim3 dimBlockE(nThreads, nThreads, 1);


	cudaEventRecord(E0, 0);
	cudaEventSynchronize(E0);

	// Ejecutar el kernel elemento a elemento
	KernelByN<<<dimGridE, dimBlockE>>>(Nfil, Ncol, d_A);
	CheckCudaError((char *) "Invocar Kernel", __LINE__);

	cudaEventRecord(E1, 0);
	cudaEventSynchronize(E1);

	// Obtener el resultado desde el host
	cudaMemcpy(h_B, d_A, numBytes, cudaMemcpyDeviceToHost);
  	CheckCudaError((char *) "Copiar Datos Device --> Host", __LINE__);

  	printf("%d\n", h_B[0]);

	// Liberar Memoria del device 
	cudaFree(d_A);

	cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);

	//Guardar fotograma
	//stbi_write_jpg("edited2.jpg", N, M, 3, frame.data, N*3);
}

void CheckCudaError(char sms[], int line) {
  cudaError_t error;

  error = cudaGetLastError();
  if (error) {
    printf("(ERROR) %s - %s in %s at line %d\n", sms, cudaGetErrorString(error), __FILE__, line);
    exit(EXIT_FAILURE);
  }
  //else printf("(OK) %s \n", sms);
}