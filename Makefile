all: sepia_filter.c bn_filter.c binarize.c convolution3x3.c convolution5x5.c
	gcc convolution5x5.c -o convolution5x5.exe -lm
	gcc sepia_filter.c -o sepia_filter.exe -lm
	gcc bn_filter.c -o bn_filter.exe -lm
	gcc binarize.c -o binarize.exe -lm
	gcc convolution3x3.c -o convolution3x3.exe -lm

convolution3x3: convolution3x3.c
	gcc convolution3x3.c -o convolution3x3.exe -lm

convolution5x5: convolution5x5.c
	gcc convolution5x5.c -o convolution5x5.exe -lm

binarize: binarize.c
	gcc binarize.c -o binarize.exe -lm

sepia_filter: sepia_filter.c
	gcc sepia_filter.c -o sepia_filter.exe -lm

bn_filter: bn_filter.c
	gcc bn_filter.c -o bn_filter.exe -lm

clean:
	rm sepia_filter.exe
	rm bn_filter.exe
	rm binarize.exe
	rm convolution3x3.exe
	rm convolution5x5.exe
