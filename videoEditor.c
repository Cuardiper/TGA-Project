#include <stdio.h>
#include <stdint.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#include <math.h>
#include "stb_image.h"

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

int main(int argc, char* argv[])
{
    if (argc < 2) {
		printf("Necesito la ruta del video en mp4!\n");
	}
	char* filename = argv[1];
	char* auxCommand = "pics/thumb%d.jpg -hide_banner";
	char comando[300];
	sprintf(comando, "ffmpeg -i %s.mp4 %s",filename,auxCommand);
	system(comando);
	//auxCommand = "ffmpeg -framerate 25 -i pics/thumb%d.jpg" 
	//sprintf(comando, "%s -pattern_type glob -c:v libx264 -pix_fmt yuv420p %s_out.mp4",auxCommand, filename);
	//system(comando);
    return 0;
}
