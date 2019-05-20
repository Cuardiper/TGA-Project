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

struct Frame read_frame() {
    int width, height, bpp;
	struct Frame result;
    uint8_t* rgb_image = stbi_load("pics/thumb1.jpg", &width, &height, &bpp, 3);
	result.data = rgb_image;
	result.width = width;
	result.height = height;
	result.bpp = bpp;
    //stbi_image_free(rgb_image);
    return result;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
		printf("Necesito la ruta del video en mp4!\n");
	}
	char* filename = argv[1];
	char* auxCommand = "pics/thumb%d.jpg -hide_banner";
	char comando[300];
	sprintf(comando, "ffmpeg -i %s.mp4 %s",filename,auxCommand);
	//system(comando);
	struct Frame frame = read_frame();


	unsigned bytePerPixel = 3;
	/*for (int i = 0; i < frame.width; ++i) {
		for (int j = 0; j < frame.height; ++j) {
			int offset = i + (j * frame.height);
			printf("R=%d, G=%d, B=%d\n", frame.data[offset], frame.data[offset+1], frame.data[offset+2]);	
		}
	}*/

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

	stbi_write_jpg("edited2.jpg", frame.width, frame.height, 3, frame.data, frame.width*3);

	/*
	unsigned char* pixelOffset = data + (i + y * j) * bytePerPixel;
	unsigned char r = pixelOffset[0];
	unsigned char g = pixelOffset[1];
	unsigned char b = pixelOffset[2];
	unsigned char a = channelCount >= 4 ? pixelOffset[3] : 0xff;
	*/

	//auxCommand = "ffmpeg -framerate 25 -i pics/thumb%d.jpg" 
	//sprintf(comando, "%s -pattern_type glob -c:v libx264 -pix_fmt yuv420p %s_out.mp4",auxCommand, filename);
	//system(comando);
    return 0;
}
