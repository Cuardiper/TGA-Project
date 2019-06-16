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

int kRows = 3;
int kCols = 3;
// double kernel[3][3] = {{(double)1/9,(double)1/9,(double)1/9}, {(double)1/9,(double)1/9,(double)1/9}, {(double)1/9,(double)1/9,(double)1/9}}; //Blur normal
 double kernel[3][3] = {{-1,-1,-1}, {-1,8,-1}, {-1,-1,-1}}; //Bordes
//double kernel[3][3] = {{0,-1,0}, {-1,5,-1}, {0,-1,0}}; //Sharpening
// double kernel[3][3] = {{(double)1/16,(double)1/8,(double)1/16}, {(double)1/8,(double)1/4,(double)1/8}, {(double)1/16,(double)1/8,(double)1/16}}; //Gaussian blur


struct Frame {
	uint8_t* data;
	int width;
	int height;
	int bpp;
};


void read_frames(uint8_t* data_in, int size, int sizeFrame) {
	for (int i = 0; i < size; ++i) {
		char filename[300];
		sprintf(filename, "pics/thumb%d.jpg",i+1);
		int width, height, bpp;
		uint8_t* rgb_image = stbi_load(filename, &width, &height, &bpp, 3);
        for(int j = 0; j < sizeFrame; ++j)
            data_in[i*sizeFrame+j] = rgb_image[j];
	}
}

void process_convolution(uint8_t* data, uint8_t* data_out, int intensity, int height, int width, int current){
    int rows = height;
    int cols = width;
    int padding = width * height * 3 * current;
	for (int k = 0; k < 1; k++) {
//         printf("Intensity %d\n",i);
		//find center position of kernel
		int kCenterX = kCols / 2;
		int kCenterY = kRows / 2;
		for (int i = 0; i < rows; ++i)					// rows
		{
			for (int j = 0; j < cols; ++j)				// columns
			{
                data_out[padding + (i*cols+j)*3] = 0;
                data_out[padding + (i*cols+j)*3+1] = 0;
				data_out[padding + (i*cols+j)*3+2] = 0;
				for (int m = 0; m < kRows; ++m)			// kernel rows
				{

					for (int n = 0; n < kCols; ++n)		// kernel columns	
					{

						// index of input signal used for checking boundary
						int ii = i + (kCenterY - m);
						int jj = j + (kCenterX - n);

						//ignore input samples which are out of bound
						if(ii >= 0 && ii < rows && jj >= 0 && jj < cols){

							data_out[padding + (i*cols+j)*3] += data[width * height *3 * current + (ii*cols + jj)*3] * kernel[m][n];
							data_out[padding + ((i*cols+j)*3)+1] += data[width * height *3 * current + ((ii*cols + jj)*3)+1] * kernel[m][n];
							data_out[padding + ((i*cols+j)*3)+2] += data[width * height *3 * current + ((ii*cols + jj)*3)+2] * kernel[m][n];
						}
					}
				}
			}
		}
		//*frame = *out;
	}
}

void applyFilter(int size, int width, int height, uint8_t* data_in, uint8_t* data_out) {
	printf("Aplicando filtro....\n");
	char filename[300];
	clock_t start, end;
	start = clock();
	int intensidad = 1;
	for (int i = 0; i < size-1; ++i) {
		process_convolution(data_in, data_out, intensidad, height, width, i);
	}
	end = clock();
	printf("Tiempo total kernel secuencial: %f\n",((double) (end-start)/CLOCKS_PER_SEC));
    printf("Writing...\n");
	for (int i = 0; i < size-1; ++i) {
        printf("\rIn progress %d", i*100/(size-1));
		sprintf(filename, "thumb%d.jpg",i+1);
        char ruta [300];
        sprintf(ruta, "pics2/%s",filename);
        stbi_write_jpg(ruta, width, height, 3, &data_out[i*width * height * 3], width*3);
    }
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
		printf("Necesito la ruta del video en mp4!\n");
		return -1;
	}
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

	int width, height, bpp;
	stbi_load("pics/thumb1.jpg", &width, &height, &bpp, 3);
	uint8_t* data_in = malloc(width * height *3*sizeof(uint8_t) * frames-2);
	if(data_in == NULL)
    {
        printf("Memory allocation failed\n");
        return -1;
    }
	read_frames(data_in, frames-2, 3 * width * height);

	int intensidad = 6;
    uint8_t* data_out = malloc(width * height *3*sizeof(uint8_t) * frames-2);
    if(data_out == NULL)
    {
        printf("Memory allocation failed\n");
        return -1;
    }
    applyFilter(frames-1, width, height, data_in, data_out);
	//applyFilter(frames-1, &fotogramas[0], data_out);
	auxCommand = "ffmpeg -framerate 25 -i pics2/thumb%d.jpg";
	sprintf(comando, "%s -pattern_type glob -c:v libx264 -pix_fmt yuv420p %s_out_provisional.mp4",auxCommand, filename);
	system(comando);
	sprintf(comando,"ffmpeg -i %s_out_provisional.mp4 -i audio.aac -c:v copy -c:a aac -strict experimental %s_out.mp4",filename,filename);
	system(comando);
	sprintf(comando,"rm %s_out_provisional.mp4",filename);
	system(comando);
	system("rm audio.aac");
    return 0;
}
