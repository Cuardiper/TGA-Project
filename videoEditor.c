#include <stdio.h>
#include <stdint.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#include <math.h>
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


struct Frame {
	uint8_t* data;
	int width;
	int height;
	int bpp;
};

struct Frame read_frame(char* filename) {
    int width, height, bpp;
	struct Frame result;
    uint8_t* rgb_image = stbi_load(filename, &width, &height, &bpp, 3);
	result.data = rgb_image;
	result.width = width;
	result.height = height;
	result.bpp = bpp;
    //stbi_image_free(rgb_image);
    return result;
}


void process_frame_bn(char* filename) {
	struct Frame frame = read_frame(filename);
	for (int i = 0; i < frame.width*frame.height*3; i+=3)
	{
		int R = frame.data[i];
		int G = frame.data[i+1];
		int B = frame.data[i+2];
		int gray = (R*0.299 + G*0.587 + B*0.114);
		frame.data[i]=gray;
		frame.data[i+1] = gray;
		frame.data[i+2] = gray;
	}
	stbi_write_jpg(filename, frame.width, frame.height, 3, frame.data, frame.width*3);
}

void bnFilter() {
	printf("Reading....\n");
	char filename[300] = "pics/thumb1.jpg";
	for (int i = 1; i < 2586; ++i) {
		sprintf(filename, "pics/thumb%d.jpg",i);
		process_frame_bn(filename);
	}
}


int main(int argc, char* argv[])
{
    if (argc < 2) {
		printf("Necesito la ruta del video en mp4!\n");
		return -1;
	}
	char* filename = argv[1];
	system("mkdir pics");
	char* auxCommand = "pics/thumb%d.jpg -hide_banner";
	char comando[300];
	sprintf(comando, "ffmpeg -i %s.mp4 %s",filename,auxCommand);
	system(comando);
	sprintf(comando,"ffmpeg -i %s.mp4 -vn -acodec copy audio.aac",filename);
	system(comando);
	int filter;
	printf("Que filtro quieres aplicar al video?\n1 - Grayscale\n2 - Sepia Filter\n3 - Binarize Filter\n");
	scanf("%d",&filter);
	if (filter == 1 || filter == 3) {
		//BN filter
		bnFilter();
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
