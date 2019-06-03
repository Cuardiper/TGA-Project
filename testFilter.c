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
double kernel[3][3] = {{(double)1/16,(double)1/8,(double)1/16}, {(double)1/8,(double)1/4,(double)1/8}, {(double)1/16,(double)1/8,(double)1/16}};
//double kernel[3][3] = {{-1,-1,-1}, {-1,8,-1}, {-1,-1,-1}};
//double kernel[3][3] = {{0,-1,0}, {-1,5,-1}, {0,-1,0}};

struct Frame {
	uint8_t* data;
	int width;
	int height;
	int bpp;
};


void inicializa(struct Frame* frame, int size){
	(*frame).data = malloc(size * sizeof(uint8_t));
	for (int i = 0; i < size; i++)
	{
		(*frame).data[i] = 0;
	}
}

void test_filter(struct Frame* frame, int frames, int intensity){
	for (int i = 0; i < intensity; i++) {
		struct Frame out[frames];
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
	sprintf(ruta, "multimedia/test_blur.jpg");
	stbi_write_jpg(ruta, frame->width, frame->height, 3, frame->data, frame->width*3);
}

void read_frame(struct Frame* frame ) {
		char filename[300];
		sprintf(filename, "multimedia/test.jpg");
		int width, height, bpp;
		uint8_t* rgb_image = stbi_load(filename, &width, &height, &bpp, 3);
		frame[0].data = rgb_image;
		frame[0].width = width;
		frame[0].height = height;
		frame[0].bpp = bpp;
		//stbi_image_free(rgb_image);
}

int main(int argc, char* argv[])
{
	struct Frame fotogramas[1];
	read_frame(&fotogramas[0]);
	struct Frame *fots = &fotogramas[0];
	int intensidad = 6;
	test_filter(fots, 1, 6);
}
