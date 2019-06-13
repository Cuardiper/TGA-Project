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


void read_frames(uint8_t* frame, int size, int sizeFrame) {
	for (int i = 0; i < size; ++i) {
		char filename[300];
		sprintf(filename, "pics/thumb%d.jpg",i+1);
		int width, height, bpp;
		uint8_t* rgb_image = stbi_load(filename, &width, &height, &bpp, 3);
        uint8_t he = height;
        uint8_t wi = width;
        frame[i*sizeFrame] = he;
        frame[i*sizeFrame+1] = wi;
        frame[i*sizeFrame+2] = bpp;
        for(int j = 0; j < height*width*3; ++j)
            frame[i*sizeFrame+3+j] = rgb_image[j];
    }
}


////////////////////  |
///CODIGO CUDA//////  |
///////////////////   v

__global__ void KernelByN (int Nfil, int Ncol, uint8_t *A, int Nframes, int SzFrame) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    
    if(row < Nfil && col < Ncol){
        for (int i = 0; i < Nframes; ++i) {
            int ind = (row * Ncol + col)*3 + i*SzFrame + 3;
            A[ind] = A[ind+1] = A[ind+2] = (A[ind] + A[ind+1] + A[ind+2])/3 > 127 ? (uint8_t) 255 : (uint8_t) 0;
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
	int Nfil, Ncol;
	unsigned long numBytes;
	unsigned int nThreads;

	float TiempoTotal, TiempoKernel;
	cudaEvent_t E0, E1, E2, E3;

	uint8_t *Host_I;
	uint8_t *Host_O;
	uint8_t *Dev_I;

	//Sacar los fotogramas del video usando FFMPEG
    char *filename = argv[1];
//     system("mkdir pics");
    system("mkdir pics2");
    char *auxCommand = "pics/thumb%d.jpg -hide_banner";
    char comando[300];
    sprintf(comando, "ffmpeg -i %s.mp4 %s",filename,auxCommand);
//     system(comando);
    sprintf(comando,"ffmpeg -i %s.mp4 -vn -acodec copy audio.aac",filename);
    system(comando);

	//Contar el numero de fotogramas obtenidos
	DIR *d;
	struct dirent *dir;
	d = opendir("pics/");
	unsigned long frames = 0;
	if (d) {
		while ((dir = readdir(d)) != NULL) {
			frames++;
		}
		closedir(d);
	}

    int bpp;
    stbi_load("pics/thumb1.jpg", &Nfil, &Ncol, &bpp, 3);
	printf("Leyendo %d fotogramas de %d x %d resolucion...\n",frames-2, Nfil, Ncol);
    Nfil = Nfil * 3;

    numBytes = (frames-2) * (3 + Nfil * Ncol) * sizeof(uint8_t); //Guardamos 3 uint8_t (height, width i bpp) + un uint8_t por cada color (3*width*height)
    //Podemos cargarnos la struct y considerar que los 3 primeros valores son height, width y bpp, y los (3*width*height) siguientes el data, todo eso por cada frame.
    //Cada frame ocupa 3*Nfil*Ncol uint8_t.

    // Obtener Memoria en el host
    printf("Numero de bytes: %lu\n", numBytes);
    Host_I = (uint8_t*) malloc(numBytes);
    if(Host_I == NULL)
    {
        printf("Memory allocation failed\n");
        return;
    }
    Host_O = (uint8_t*) malloc(numBytes);
    if(Host_O == NULL)
    {
        printf("Memory allocation failed\n");
        return;
    }
    read_frames(Host_I, frames-2, 3 + Nfil * Ncol);

	cudaEventCreate(&E0);	cudaEventCreate(&E1);
    cudaEventCreate(&E2);	cudaEventCreate(&E3);
    printf("Aplicando filtro...\n");
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
    // Obtener Memoria en el device
    cudaMalloc((uint8_t**)&Dev_I, numBytes);
    // Copiar datos desde el host en el device 
    cudaMemcpy(Dev_I, Host_I, numBytes, cudaMemcpyHostToDevice);
    CheckCudaError((char *) "Copiar Datos Host --> Device", __LINE__);
        
    cudaEventRecord(E1, 0);
    cudaEventSynchronize(E1);
	// Ejecutar el kernel elemento a elemento
	KernelByN<<<dimGridE, dimBlockE>>>(Nfil/3, Ncol, Dev_I, frames-2, 3 + Nfil * Ncol);
	CheckCudaError((char *) "Invocar Kernel", __LINE__);

	cudaEventRecord(E2, 0);
	cudaEventSynchronize(E2);

	// Obtener el resultado desde el host
	cudaMemcpy(Host_I, Dev_I, numBytes, cudaMemcpyDeviceToHost);
	CheckCudaError((char *) "Copiar Datos Device --> Host", __LINE__);

	// Liberar Memoria del device 
	cudaFree(Dev_I);
    cudaEventRecord(E3, 0);
    cudaEventSynchronize(E3);

    cudaEventElapsedTime(&TiempoTotal,  E0, E3);
    cudaEventElapsedTime(&TiempoKernel, E1, E2);
    printf("Tiempo Global: %4.6f milseg\n", TiempoTotal);
    printf("Tiempo Kernel: %4.6f milseg\n", TiempoKernel);
    printf("Bandwidth: %4.6f GB/s\n", (float)(((float)(numBytes/TiempoKernel))/1000000));
    printf("Rendimiento Global: %4.2f GFLOPS\n", (3.0 * (float) Nfil/3 * (float) Ncol * (float) (frames-2)) / (1000000.0 * TiempoTotal));
    printf("Rendimiento Kernel: %4.2f GFLOPS\n", (3.0 * (float) Nfil/3 * (float) Ncol * (float) (frames-2)) / (1000000.0 * TiempoKernel));
	cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);
    printf("Writing...\n");
    char picname[300];
    for (int i = 0; i < frames-2; ++i) {
        printf("\rIn progress %d", i*100/(frames-2)); ///'size' no definido (solución: lo pongo en mayusculas, no se si es la variable a la que te querias referir)
        sprintf(picname, "thumb%d.jpg",i+1);
        char ruta [300];
        sprintf(ruta, "pics2/%s",picname);
        stbi_write_jpg(ruta, Nfil/3, Ncol, 3, &Host_I[i*(3 + Nfil * Ncol)+3], Nfil);   //He cambiado out[] por Host_O[]
    }
    auxCommand = "ffmpeg -framerate 25 -i pics2/thumb%d.jpg";
	sprintf(comando, "%s -pattern_type glob -c:v libx264 -pix_fmt yuv420p %s_out_provisional.mp4",auxCommand, filename);
	system(comando);
	sprintf(comando,"ffmpeg -i %s_out_provisional.mp4 -i audio.aac -c:v copy -c:a aac -strict experimental %s_out.mp4",filename,filename);
	system(comando);
	sprintf(comando,"rm %s_out_provisional.mp4",filename);
	system(comando);
	system("rm audio.aac");
	system("rm -rf pics2");
    return 0;
    
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
