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



void read_frames(uint8_t* frame, int size, int sizeFrame, int iteracion) {
	for (int i = 0; i < size; ++i) {
		char filename[300];
		sprintf(filename, "pics/thumb%d.jpg",i+1+iteracion*size);
		int width, height, bpp;
		uint8_t* rgb_image = stbi_load(filename, &width, &height, &bpp, 3);
        for(int j = 0; j < sizeFrame; ++j)
            frame[i*sizeFrame+j] = rgb_image[j];
    }
}


////////////////////  |
///CODIGO CUDA//////  |
///////////////////   v

__global__ void KernelByN (int Ncol, int Nfil, uint8_t *A, int Nframes, int SzFrame) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < Nfil && col < Ncol){
        for (int i = 0; i < Nframes; ++i) {
            int ind = (row * Ncol + col)*3 + i*SzFrame;
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
	int Nfil, Ncol;
	unsigned long numBytes;
	unsigned int nThreads;

    uint8_t *Host_I;
	float TiempoTotal, TiempoKernel;
    TiempoKernel = TiempoTotal = 0;
	cudaEvent_t E0, E3;

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
	unsigned long frames = 0;
	if (d) {
		while ((dir = readdir(d)) != NULL) {
			frames++;
		}
		closedir(d);
	}

    int bpp;
    stbi_load("pics/thumb1.jpg", &Ncol, &Nfil, &bpp, 3);
	printf("Leyendo %d fotogramas de %d x %d resolucion...\n",frames-2, Ncol, Nfil);

    numBytes = (frames-2)/4 * (3 * Nfil * Ncol) * sizeof(uint8_t);

    //Obtener Memoria en el host
    printf("Numero de bytes: %lu\n", numBytes);
    Host_I = (uint8_t*) malloc(numBytes);
    if(Host_I == NULL)
    {
        printf("Memory allocation failed\n");
        return;
    }

    for (int i = 0; i < 4; ++i)
    {
        printf("Leyendo %d fotogramas de %d\n",(frames-2)/4,frames-2);

        uint8_t *Dev_I;
        cudaEvent_t E1, E2;

        read_frames(Host_I, (frames-2)/4, 3 * Nfil * Ncol, i);

        cudaEventCreate(&E0);   cudaEventCreate(&E1);
        cudaEventCreate(&E2);   cudaEventCreate(&E3);

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
        CheckCudaError((char *) "malloc device", __LINE__);
        // Copiar datos desde el host en el device 
        cudaMemcpy(Dev_I, Host_I, numBytes, cudaMemcpyHostToDevice);
        CheckCudaError((char *) "Copiar Datos Host --> Device", __LINE__);

        cudaEventRecord(E1, 0);
        cudaEventSynchronize(E1);
        // Ejecutar el kernel elemento a elemento
        KernelByN<<<dimGridE, dimBlockE>>>(Ncol, Nfil, Dev_I, (frames-2)/4, Nfil * Ncol * 3);
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

        float auxTotal, auxKernel;
        cudaEventElapsedTime(&auxTotal,  E0, E3);
        cudaEventElapsedTime(&auxKernel, E1, E2);
        TiempoTotal += auxTotal;
        TiempoKernel += auxKernel;
        cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);

        printf("Writing...\n");
        char picname[300];
        for (int k = 0; k < (frames-2)/4; ++k) {
            printf("\n%d\n", k);
            printf("\rIn progress %d %", k*100/((frames-2)/4)); ///'size' no definido (solución: lo pongo en mayusculas, no se si es la variable a la que te querias referir)
            sprintf(picname, "thumb%d.jpg",k+1+(i*(frames-2)/4));
            char ruta [300];
            sprintf(ruta, "pics2/%s",picname);
            stbi_write_jpg(ruta, Ncol, Nfil, 3, &Host_I[k * 3 * Nfil * Ncol], Ncol);
        }
    }

    
    printf("Tiempo Global: %4.6f milseg\n", TiempoTotal);
    printf("Tiempo Kernel: %4.6f milseg\n", TiempoKernel);
    printf("Bandwidth: %4.6f GB/s\n", (float)(((float)(numBytes/TiempoKernel))/1000000));
    printf("Rendimiento Global: %4.2f GFLOPS\n", (9.0 * (float) Nfil * (float) Ncol * (float) (frames-2)) / (1000000.0 * TiempoTotal));
    printf("Rendimiento Kernel: %4.2f GFLOPS\n", (9.0 * (float) Nfil * (float) Ncol * (float) (frames-2)) / (1000000.0 * TiempoKernel));
    
    printf("\nRemoving residuals...\n");
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
