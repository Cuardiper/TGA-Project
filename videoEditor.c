#include <stdio.h>
#include <stdint.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#include <math.h>
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <time.h>


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
	clock_t start,end;
	start = clock();
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
	end = clock();
	//printf("%f\n",((double) (end-start)/CLOCKS_PER_SEC));
	//stbi_write_jpg(filename, frame.width, frame.height, 3, frame.data, frame.width*3);
}

void process_frame_binarize(char* filename) {
	struct Frame frame = read_frame(filename);
	for (int i = 0; i < frame.width*frame.height*3; i+=3)
	{
		int R = frame.data[i];
		int G = frame.data[i+1];
		int B = frame.data[i+2];
		int gray = (R*0.299 + G*0.587 + B*0.114);
		gray = gray > 127 ? 255 : 0;
		frame.data[i]=gray;
		frame.data[i+1] = gray;
		frame.data[i+2] = gray;
	}
	stbi_write_jpg(filename, frame.width, frame.height, 3, frame.data, frame.width*3);
}

int max(int n1) {
	return n1>255 ? 255 : n1;
}


void process_frame_sepia(char* filename) {
	struct Frame frame = read_frame(filename);
	for (int i = 0; i < frame.width*frame.height*3; i+=3)
	{
		int R = frame.data[i];
		int G = frame.data[i+1];
		int B = frame.data[i+2];
		int R1 = max(R*0.383 + G*0.769 + B*0.189);
		int G1 = max(R*0.349 + G*0.686 + B*0.168);
		int B1 = max(R*0.272 + G*0.534 + B*0.131);
		frame.data[i]=R1;
		frame.data[i+1] = G1;
		frame.data[i+2] = B1;
	}
	stbi_write_jpg(filename, frame.width, frame.height, 3, frame.data, frame.width*3);
}

void applyFilter(int filter) {
	printf("Reading....\n");
	char filename[300] = "pics/thumb1.jpg";
	clock_t start, end;
	start = clock();
	for (int i = 1; i < 2586; ++i) {
		sprintf(filename, "pics/thumb%d.jpg",i);
		if (filter == 1)
			process_frame_bn(filename);
		else if (filter == 2)
			process_frame_binarize(filename);
		else if (filter == 3)
			process_frame_sepia(filename);
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
	applyFilter(filter);
	auxCommand = "ffmpeg -framerate 25 -i pics/thumb%d.jpg";
	sprintf(comando, "%s -pattern_type glob -c:v libx264 -pix_fmt yuv420p %s_out_provisional.mp4",auxCommand, filename);
	system(comando);
	sprintf(comando,"ffmpeg -i %s_out_provisional.mp4 -i audio.aac -c:v copy -c:a aac -strict experimental %s_out.mp4",filename,filename);
	system(comando);
	sprintf(comando,"rm %s_out_provisional.mp4",filename);
	system(comando);
	system("rm -rf pics audio.aac");
    return 0;
}
