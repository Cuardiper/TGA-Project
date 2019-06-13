all: videoEditor.c 2Dconvolution.c testFilter.c
	gcc videoEditor.c -o videoEditor.exe -lm
	gcc 2Dconvolution.c -o 2Dconvolution.exe -lm

videoEditor: videoEditor.c
	gcc videoEditor.c -o videoEditor.exe -lm

2Dconvolution: 2Dconvolution.c
	gcc 2Dconvolution.c -o 2Dconvolution.exe -lm

testFilter: testFilter.c
	gcc testFilter.c -o testFilter.exe -lm

clean:
	rm videoEditor.exe
	rm 2Dconvolution.exe
	rm testFilter.exe
