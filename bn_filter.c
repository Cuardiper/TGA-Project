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
	}
}

void process_frame_bn(struct Frame* frame) {
	for (int i = 0; i < (*frame).width*(*frame).height*3; i+=3)
	{
		int R = (*frame).data[i];
		int G = (*frame).data[i+1];
		int B = (*frame).data[i+2];
		int gray = (R+G+B)/3;
		(*frame).data[i]=gray;
		(*frame).data[i+1] = gray;
		(*frame).data[i+2] = gray;
	}
}

int max(int n1) {
	return n1>255 ? 255 : n1;
}


void applyFilter(int size, struct Frame* frames) {
	printf("Aplicando filtro....\n");
	char filename[300];
	clock_t start, end;
	start = clock();
	int intensidad = 1;
	for (int i = 0; i < size-1; ++i) {
		process_frame_bn(&frames[i]);
	}
	end = clock();
	printf("Tiempo total kernel secuencial: %f\n",((double) (end-start)/CLOCKS_PER_SEC));
    printf("Writing...\n");
	for (int i = 0; i < size-1; ++i) {
        printf("\rIn progress %d", i*100/(size-1));
		sprintf(filename, "thumb%d.jpg",i+1);
        char ruta [300];
        sprintf(ruta, "pics2/%s",filename);
        stbi_write_jpg(ruta, frames[i].width, frames[i].height, 3, frames[i].data, frames[i].width*3);
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
//  	system("mkdir pics");
	system("mkdir pics2");
	char *auxCommand = "pics/thumb%d.jpg -hide_banner";
	char comando[300];
 	sprintf(comando, "ffmpeg -i %s.mp4 %s",filename,auxCommand);
// 	system(comando);
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
