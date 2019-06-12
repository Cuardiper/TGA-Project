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

// struct Frame {
// 	uint8_t* data;
// 	uint8_t width;
// 	uint8_t height;
// 	uint8_t bpp;
// };
// 
// struct Frame read_frame() {
//     int width, height, bpp;
// 	struct Frame result;
//     uint8_t* rgb_image = stbi_load("pics/thumb1.jpg", &width, &height, &bpp, 3);
// 	result.data = rgb_image;
// 	result.width = width;
// 	result.height = height;
// 	result.bpp = bpp;
//     //stbi_image_free(rgb_image);
//     return result;
// }

void read_frames(uint8_t* frame, int size, int sizeFrame) {
	for (int i = 0; i < size; ++i) {
		char filename[300];
		sprintf(filename, "pics/thumb%d.jpg",i+1);
		int width, height, bpp;
		uint8_t* rgb_image = stbi_load(filename, &width, &height, &bpp, 3);
        frame[i*sizeFrame] = height;
        frame[i*sizeFrame+1] = width;
        frame[i*sizeFrame+2] = bpp;
        for(int j = 0; j < height*width*3; ++i)
            frame[i*sizeFrame+3+j] = rgb_image[j];
	}
}


// void process_frame_bn(struct Frame* frame, char* filename) {
// 	for (int i = 0; i < (*frame).width*(*frame).height*3; i+=3)
// 	{
// 		int R = (*frame).data[i];
// 		int G = (*frame).data[i+1];
// 		int B = (*frame).data[i+2];
// 		int gray = (R*0.299 + G*0.587 + B*0.114);
// 		(*frame).data[i]=gray;
// 		(*frame).data[i+1] = gray;
// 		(*frame).data[i+2] = gray;
// 	}
// 	char ruta [300];
// 	sprintf(ruta, "pics2/%s",filename);
// 	stbi_write_jpg(ruta, frame->width, frame->height, 3, frame->data, frame->width*3);
// }

int max(int n1) {
	return n1>255 ? 255 : n1;
}


////////////////////  |
///CODIGO CUDA//////  |
///////////////////   v

__global__ void KernelByN (int Nfil, int Ncol, uint8_t *A, int Nframes, int SzFrame) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    
    for (int i = 0; i < Nframes; ++i) {
        int ind = row * Ncol + col + i*SzFrame + 3;
        if(row < Nfil && col < Ncol){
            A[ind] = A[ind+1] = A[ind+2] = (A[ind] + A[ind+1] + A[ind+2])/3;
        }
    }
}

void CheckCudaError(char sms[], int line);




int main(int argc, char** argv)
{
    
    if (argc < 2) {
		printf("Necesito la ruta del video en mp4!\n");
		return -1;
	}
	unsigned int Nfil, Ncol;
  	unsigned int numBytes;
	unsigned int nThreads;

	float TiempoTotal, TiempoKernel;
  	cudaEvent_t E0, E1, E2, E3;

  	uint8_t *Host_I;
  	uint8_t *Host_O;
  	uint8_t *Dev_I;

  	///Leer jpeg de la carpeta////
	//Sacar los fotogramas del video usando FFMPEG
	char *filename = argv[1];
	system("mkdir pics");
	system("mkdir pics2");
	char *auxCommand = "pics/thumb%d.jpg -hide_banner";
	char comando[300];
	sprintf(comando, "ffmpeg -i %s.mp4 %s",filename,auxCommand);
	system(comando);
	sprintf(comando,"ffmpeg -i %s.mp4 -vn -acodec copy audio.aac",filename);
	system(comando);

	//Contar el numero de fotogramas obtenidos
	DIR *d;
	struct dirent *dir;
	d = opendir("pics/");
	int frames = 0;
	if (d) {
		while ((dir = readdir(d)) != NULL) {
			frames++;
		}
		closedir(d);
	}
	printf("Leyendo %d fotogramas...\n",frames-2);

    Host_I = (uint8_t*) malloc(numBytes);
    read_frames(Host_I, frames-1, 3 * Nfil * Ncol);
  	Ncol = Host_I[0];
  	Nfil = Host_I[1] * 3;
  	printf("%dX%d\n", Nfil, Ncol);
  	//////////////////////////////

//   	numBytes = Nfil * Ncol * sizeof(uint8_t);
  	numBytes = (frames-2) * (3 + Nfil * Ncol) * sizeof(uint8_t); //Guardamos 3 uint8_t (height, width i bpp) + un uint8_t por cada color (3*width*height)
    //Podemos cargarnos la struct y considerar que los 3 primeros valores son height, width y bpp, y los (3*width*height) siguientes el data, todo eso por cada frame.
    //Cada frame ocupa 3*Nfil*Ncol uint8_t.
    
  	cudaEventCreate(&E0);	cudaEventCreate(&E1);
	cudaEventCreate(&E2);	cudaEventCreate(&E3);

	// Obtener Memoria en el host
//     output = (uint8_t*) malloc(numBytes);
    // Inicializa con los datos de la imagen leída

    // Obtener Memoria en el device
  	cudaMalloc((uint8_t**)&Dev_I, numBytes);
  	// Copiar datos desde el host en el device 
	cudaMemcpy(Dev_I, Host_I, numBytes, cudaMemcpyHostToDevice);
	CheckCudaError((char *) "Copiar Datos Host --> Device", __LINE__);
  	
	//
  	// KERNEL ELEMENTO a ELEMENTO
  	//

	// numero de Threads en cada dimension 
  	nThreads = SIZE;

	// numero de Blocks en cada dimension
  	int nBlocksFil = (Nfil+nThreads-1)/nThreads; //tener en cuenta 3componentes RGB??
  	int nBlocksCol = (Ncol+nThreads-1)/nThreads;

  	dim3 dimGridE(nBlocksCol, nBlocksFil, 1);
  	dim3 dimBlockE(nThreads, nThreads, 1);

	cudaEventRecord(E0, 0);
	cudaEventSynchronize(E0);

	// Ejecutar el kernel elemento a elemento
	KernelByN<<<dimGridE, dimBlockE>>>(Nfil, Ncol, Dev_I, frames-2, 3 + Nfil * Ncol);
	CheckCudaError((char *) "Invocar Kernel", __LINE__);

	cudaEventRecord(E1, 0);
	cudaEventSynchronize(E1);

	// Obtener el resultado desde el host
	cudaMemcpy(Host_O, Dev_I, numBytes, cudaMemcpyDeviceToHost);
  	CheckCudaError((char *) "Copiar Datos Device --> Host", __LINE__);
    printf("Writing...\n");
    for (int i = 0; i < frames-2; ++i) {
        printf("\rIn progress %d", i*100/(SIZE-1)); ///'size' no definido (solución: lo pongo en mayusculas, no se si es la variable a la que te querias referir)
		sprintf(filename, "thumb%d.jpg",i+1);
        char ruta [300];
        sprintf(ruta, "pics2/%s",filename);
        stbi_write_jpg(ruta, Host_O[i+1], Host_O[i], 3, &Host_O[i+3], Host_O[i+1]*3);   //He cambiado out[] por Host_O[]
    }

	// Liberar Memoria del device 
	cudaFree(Dev_I);

	cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);

	//Guardar fotograma
    
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
