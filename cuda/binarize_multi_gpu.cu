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


void read_frames(uint8_t* frame, int ini, int size, int sizeFrame) {
	for (int i = ini; i < size; ++i) {
		char filename[300];
		sprintf(filename, "pics/thumb%d.jpg",i+1);
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
    cudaEvent_t X1, X2, X3;
	uint8_t *Host_I1;
    uint8_t *Host_I2;
    uint8_t *Host_I3;
    uint8_t *Host_I4;
	uint8_t *Dev_I1;
	uint8_t *Dev_I2;
	uint8_t *Dev_I3;
	uint8_t *Dev_I4;

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
    stbi_load("pics/thumb1.jpg", &Ncol, &Nfil, &bpp, 3);
	printf("Leyendo %d fotogramas de %d x %d resolucion...\n",frames-2, Ncol, Nfil);

    numBytes = (frames-2) * (3 * Nfil * Ncol) * sizeof(uint8_t); //Guardamos 3 uint8_t (height, width i bpp) + un uint8_t por cada color (3*width*height)
    //Podemos cargarnos la struct y considerar que los 3 primeros valores son height, width y bpp, y los (3*width*height) siguientes el data, todo eso por cada frame.
    //Cada frame ocupa 3*Nfil*Ncol uint8_t.

    // Obtener Memoria en el host
    printf("Numero de bytes: %lu\n", numBytes);
    cudaMallocHost((float**)&Host_I1,  numBytes/4); 
    cudaMallocHost((float**)&Host_I2,  numBytes/4); 
    cudaMallocHost((float**)&Host_I3,  numBytes/4); 
    cudaMallocHost((float**)&Host_I4, numBytes/4); 
    read_frames(Host_I1, 0, (frames-2)/4, 3 * Nfil * Ncol);
    read_frames(Host_I2, (frames-2)/4, (frames-2)/2, 3 * Nfil * Ncol);
    read_frames(Host_I3, (frames-2)/2, 3*(frames-2)/4, 3 * Nfil * Ncol);
    read_frames(Host_I4, 3*(frames-2)/4, (frames-2), 3 * Nfil * Ncol);

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
    

	dim3 dimGridE(nBlocksFil, nBlocksCol, 1);
	dim3 dimBlockE(nThreads, nThreads, 1);
    
    cudaEventRecord(E0, 0);
    cudaEventSynchronize(E0);
    // Obtener Memoria en el devicecudaMallocHost((float**)&hA0,  numBytesA); 
    cudaSetDevice(0);
    cudaMallocHost((float**)&Dev_I1,  numBytes/4); 
    cudaSetDevice(1);
    cudaMallocHost((float**)&Dev_I2,  numBytes/4); 
    cudaEventCreate(&X1);
    cudaSetDevice(2);
    cudaMallocHost((float**)&Dev_I3,  numBytes/4);
    cudaEventCreate(&X2);
    cudaSetDevice(3); 
    cudaMallocHost((float**)&Dev_I4, numBytes/4); 
    cudaEventCreate(&X3);
    // Copiar datos desde el host en el device 
    cudaSetDevice(0);
    cudaMemcpy(Dev_I1, Host_I1, numBytes/4, cudaMemcpyHostToDevice);
    CheckCudaError((char *) "Copiar Datos Host --> Device", __LINE__);
	// Ejecutar el kernel elemento a elemento	
	KernelByN<<<dimGridE, dimBlockE>>>(Ncol, Nfil, Dev_I1, (frames-2)/4, 3 * Nfil * Ncol);
	CheckCudaError((char *) "Invocar Kernel", __LINE__);

    cudaSetDevice(1);
    cudaMemcpy(Dev_I2, Host_I2, numBytes/4, cudaMemcpyHostToDevice);
    CheckCudaError((char *) "Copiar Datos Host --> Device", __LINE__);
	// Ejecutar el kernel elemento a elemento
	KernelByN<<<dimGridE, dimBlockE>>>(Ncol, Nfil, Dev_I2, (frames-2)/4, 3 * Nfil * Ncol);
	CheckCudaError((char *) "Invocar Kernel", __LINE__);
    cudaSetDevice(2);
    cudaMemcpy(Dev_I3, Host_I3, numBytes/4, cudaMemcpyHostToDevice);
    CheckCudaError((char *) "Copiar Datos Host --> Device", __LINE__);
	// Ejecutar el kernel elemento a elemento
	KernelByN<<<dimGridE, dimBlockE>>>(Ncol, Nfil, Dev_I3, (frames-2)/4, 3 * Nfil * Ncol);
	CheckCudaError((char *) "Invocar Kernel", __LINE__);
    cudaSetDevice(3);
    cudaMemcpy(Dev_I4, Host_I4, numBytes/4, cudaMemcpyHostToDevice);
    CheckCudaError((char *) "Copiar Datos Host --> Device", __LINE__);
	// Ejecutar el kernel elemento a elemento
	KernelByN<<<dimGridE, dimBlockE>>>(Ncol, Nfil, Dev_I4, (frames-2)/4, 3 * Nfil * Ncol);
	CheckCudaError((char *) "Invocar Kernel", __LINE__);
	cudaEventRecord(E2, 0);
	cudaEventSynchronize(E2);

	// Obtener el resultado desde el host
	
    cudaSetDevice(0);
    // Obtener el resultado desde el host 
    cudaMemcpyAsync(Host_I1, Dev_I1, numBytes/4, cudaMemcpyDeviceToHost);
	CheckCudaError((char *) "Copiar Datos Device --> Host", __LINE__); 

    cudaSetDevice(1);
    // Obtener el resultado desde el host 
    cudaMemcpyAsync(Host_I2, Dev_I2, numBytes/4, cudaMemcpyDeviceToHost);
	CheckCudaError((char *) "Copiar Datos Device --> Host", __LINE__); 
    cudaEventRecord(X1, 0);

    cudaSetDevice(2);
    // Obtener el resultado desde el host 
    cudaMemcpyAsync(Host_I3, Dev_I3, numBytes/4, cudaMemcpyDeviceToHost); 
	CheckCudaError((char *) "Copiar Datos Device --> Host", __LINE__);
    cudaEventRecord(X2, 0);

    cudaSetDevice(3);
    // Obtener el resultado desde el host 
    cudaMemcpyAsync(Host_I4, Dev_I4, numBytes/4, cudaMemcpyDeviceToHost); 
	CheckCudaError((char *) "Copiar Datos Device --> Host", __LINE__);
    cudaEventRecord(X3, 0);

	// Liberar Memoria del device 
    cudaEventRecord(E3, 0);
    cudaEventSynchronize(E3);

    cudaSetDevice(0);
    cudaEventSynchronize(X1);
    cudaEventSynchronize(X2);
    cudaEventSynchronize(X3);
    cudaSetDevice(0); cudaFree(Dev_I1); 
    cudaSetDevice(1); cudaFree(Dev_I2);
    cudaSetDevice(2); cudaFree(Dev_I3);
    cudaSetDevice(3); cudaFree(Dev_I4);
    cudaEventElapsedTime(&TiempoTotal,  E0, E3);
    printf("Tiempo Global: %4.6f milseg\n", TiempoTotal);
    printf("Bandwidth: %4.6f GB/s\n", (float)(((float)(numBytes/TiempoTotal))/1000000));
    printf("Rendimiento Global: %4.2f GFLOPS\n", (4.0 * (float) Nfil/3 * (float) Ncol * (float) (frames-2)) / (1000000.0 * TiempoTotal));
	cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);
    printf("Writing...\n");
    char picname[300];
    for (int i = 0; i < frames-2; ++i) {
        printf("\rIn progress %d %", i*100/(frames-2)); ///'size' no definido (solución: lo pongo en mayusculas, no se si es la variable a la que te querias referir)
        sprintf(picname, "thumb%d.jpg",i+1);
        char ruta [300];
        sprintf(ruta, "pics2/%s",picname);
        if (i < (frames-2)/4)
            stbi_write_jpg(ruta, Nfil/3, Ncol, 3, &Host_I1[i*Nfil * Ncol], Nfil);
        if (i >= (frames-2)/4 && i < (frames-2)/2)
            stbi_write_jpg(ruta, Nfil/3, Ncol, 3, &Host_I2[(i-(frames-2)/4)*Nfil * Ncol], Nfil);
        if (i >= (frames-2)/2 && i < 3*(frames-2)/4)
            stbi_write_jpg(ruta, Nfil/3, Ncol, 3, &Host_I3[(i-(frames-2)/2)*Nfil * Ncol], Nfil);
        if (i >= 3*(frames-2)/4 && i < (frames-2))
            stbi_write_jpg(ruta, Nfil/3, Ncol, 3, &Host_I4[(i-3*(frames-2)/4)*Nfil * Ncol], Nfil);
    }
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
