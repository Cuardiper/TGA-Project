all: videoEditor.c
	gcc videoEditor.c -o videoEditor.exe -lm

clean:
	rm videoEditor.exe