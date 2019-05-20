#include <stdio.h>
#include <stdint.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#include <math.h>
#include "stb_image.h"
#include <dirent.h>

struct Frame {
	uint8_t* data;
	int width;
	int height;
	int bpp;
};

struct Frame read_frame() {
    int width, height, bpp;
	struct Frame result;
    uint8_t* rgb_image = stbi_load("image.png", &width, &height, &bpp, 3);
	result.data = rgb_image;
	result.width = width;
	result.height = height;
	result.bpp = bpp;
    //stbi_image_free(rgb_image);
    return result;
}


struct Frame* bnFilter(int elems) {
	struct Frame imgBn[1];
	return &imgBn[0];
}


int main(int argc, char* argv[])
{
    if (argc < 2) {
		printf("Necesito la ruta del video en mp4!\n");
	}
	char* filename = argv[1];
	system("mkdir pics");
	char* auxCommand = "pics/thumb%d.jpg -hide_banner";
	char comando[300];
	sprintf(comando, "ffmpeg -i %s.mp4 %s",filename,auxCommand);
	system(comando);
	sprintf(comando,"ffmpeg -i %s.mp4 -vn -acodec copy audio.aac",filename);
	system(comando);
	char* frames = system("ls -l pics | wc -l");
	int numberFrames = atoi(frames)-1;
	printf("%d",numberFrames);
	int filter;
	printf("Que filtro quieres aplicar al video?\n1 - Grayscale\n2 - Sepia Filter\n3 - Binarize Filter\n");
	scanf("%d",&filter);
	if (filter == 1 || filter == 3) {
		//BN filter
		if (filter == 3) {
			//Binarizar
		}
	}
	else if (filter == 2) {
	}
	auxCommand = "ffmpeg -framerate 25 -i pics/thumb%d.jpg";
	sprintf(comando, "%s -pattern_type glob -c:v libx264 -pix_fmt yuv420p %s_out.mp4",auxCommand, filename);
	system(comando);
	sprintf(comando,"ffmpeg -i video.mp4 -i audio.aac -c:v copy -c:a aac -strict experimental %s_out.mp4",filename);
	system(comando);
	system("rm -rf pics audio.aac");
    return 0;
}
