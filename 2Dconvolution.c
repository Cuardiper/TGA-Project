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
//double kernel[3][3] = {{(double)1/9,(double)1/9,(double)1/9}, {(double)1/9,(double)1/9,(double)1/9}, {(double)1/9,(double)1/9,(double)1/9}};
//double kernel[3][3] = {{-1,-1,-1}, {-1,8,-1}, {-1,-1,-1}};
double kernel[3][3] = {{(double)1/16,(double)1/8,(double)1/16}, {(double)1/8,(double)1/4,(double)1/8}, {(double)1/16,(double)1/8,(double)1/16}};

struct Frame {
	uint8_t* data;
	int width;
	int height;
	int bpp;
};


void read_frames(struct Frame* frame, int size) {
	for (int i = 0; i < size; ++i) {
		char filename[300];
		sprintf(filename, "pics/thumb%d.jpg",i+1);
		int width, height, bpp;
		uint8_t* rgb_image = stbi_load(filename, &width, &height, &bpp, 3);
		frame[i].data = rgb_image;
		frame[i].width = width;
		frame[i].height = height;
		frame[i].bpp = bpp;
		//stbi_image_free(rgb_image);
	}
}

void inicializa(struct Frame* frame, int size){
	(*frame).data = malloc(size * sizeof(uint8_t));
	for (int i = 0; i < size; i++)
	{
		(*frame).data[i] = 0;
	}
}

void process_convolution(struct Frame* fotograma, int intensity, char* filename){
	
	struct Frame frame[1];
	*frame = *fotograma;
	for (int i = 0; i < intensity*4; i++) {
		struct Frame out[1];
		int rows = frame->height;
		int cols = frame->width;
		(*out).height = rows;
		(*out).width = cols;
		inicializa(out, rows*cols*3);
		//find center position of kernel
		int kCenterX = kCols / 2;
		int kCenterY = kRows / 2;
		for (int i = 0; i < rows; ++i)					// rows
		{
			for (int j = 0; j < cols; ++j)				// columns
			{
				for (int m = 0; m < kRows; ++m)			// kernel rows
				{

					for (int n = 0; n < kCols; ++n)		// kernel columns	
					{

						// index of input signal used for checking boundary
						int ii = i + (kCenterY - m);
						int jj = j + (kCenterX - n);

						//ignore input samples which are out of bound
						if(ii >= 0 && ii < rows && jj >= 0 && jj < cols){

							(*out).data[(i*cols+j)*3] += (*frame).data[(ii*cols + jj)*3] * kernel[m][n];
							(*out).data[((i*cols+j)*3)+1] += (*frame).data[((ii*cols + jj)*3)+1] * kernel[m][n];
							(*out).data[((i*cols+j)*3)+2] += (*frame).data[((ii*cols + jj)*3)+2] * kernel[m][n];
						}
					}
				}
			}
		}
		*frame = *out;
	}
	char ruta [300];
	sprintf(ruta, "pics2/%s",filename);
	stbi_write_jpg(ruta, frame->width, frame->height, 3, frame->data, frame->width*3);
}

void applyFilter(int size, struct Frame* frames) {
	printf("Aplicando filtro....\n");
	char filename[300];
	clock_t start, end;
	start = clock();
	int intensidad = 6;
	for (int i = 0; i < size-1; ++i) {
		printf("%d\n", i);
		sprintf(filename, "thumb%d.jpg",i+1);
		process_convolution(&frames[i], intensidad, filename);
	}
	end = clock();
	printf("Tiempo total: %f\n",((double) (end-start)/CLOCKS_PER_SEC));
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
		printf("Necesito la ruta del video en mp4!\n");
		return -1;
	}
	//Sacar los fotogramas del video usando FFMPEG
	char *filename = argv[1];
	//system("mkdir pics");
	system("mkdir pics2");
	char *auxCommand = "pics/thumb%d.jpg -hide_banner";
	char comando[300];
	sprintf(comando, "ffmpeg -i %s.mp4 %s",filename,auxCommand);
	//system(comando);
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

	struct Frame fotogramas[frames-2];
	read_frames(&fotogramas[0],frames-1);
	int intensidad = 6;
	applyFilter(frames-1, &fotogramas[0]);
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
