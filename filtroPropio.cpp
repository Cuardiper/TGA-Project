//
//  test.cpp
//  parserAudio.cc
//
//  Created by Alvaro on 10/04/2019.
//

#include <cstdio>
#include <sndfile.hh>
#include <stdio.h>
#include <stdlib.h>

void leerAudio(int *&buffer, SF_INFO &info, char nombre[]){
    SNDFILE *sf;
    FILE *out;
    info.format=0;
    
    sf = sf_open(nombre, SFM_READ, &info);
    if (sf == NULL)
    {
        printf("Failed to open the file.\n");
        exit(-1);
    }
    
    /* Print some of the info, and figure out how much data to read. */
    int channels = info.channels;
    int frames = info.frames;
    int sampleRate = info.samplerate;
    
    int num_items =  frames * channels;
    
    printf("channels=%d\n",channels);
    printf("frames=%d\n",frames);
    printf("samplerate=%d\n",sampleRate);
    printf("num_items=%d\n",num_items);
    
    /* Allocate space for the data to be read, then read it. */
    buffer = (int *) malloc(num_items*sizeof(int));
    int num_read = sf_read_int(sf,buffer,num_items);
    sf_close(sf);
    
    printf("Read %d items\n",num_read);
    
    //Escribe los datos en filedata.out.
    /*out = fopen("filedata.out","w");
     for (int i = 0; i < num_read; i += channels)
     {
     for (int j = 0; j < channels; ++j){
     fprintf(out,"%d ",buffer[i+j]);
     fprintf(out,"\n");
     }
     }
     fclose(out);*/
}




#define FILTER_TAP_NUM 25

static double filter_taps[FILTER_TAP_NUM] = {
    0.037391727827352866,
    -0.03299884552335965,
    0.04423058396732152,
    0.002305097083362817,
    -0.067680871959501,
    -0.04634710540912475,
    -0.011717387509232492,
    -0.0707342284185183,
    -0.04976651728299963,
    0.1608641354383635,
    0.21561058688743145,
    -0.10159456907827968,
    0.6638637561392534,
    -0.10159456907827968,
    0.21561058688743145,
    0.1608641354383635,
    -0.04976651728299963,
    -0.0707342284185183,
    -0.011717387509232492,
    -0.04634710540912475,
    -0.067680871959501,
    0.002305097083362817,
    0.04423058396732152,
    -0.03299884552335965,
    0.037391727827352866
};



void filtro (int datalen, int *&d_data, int *&output){
    for (int i=0; i<datalen; i++){
        for (int j = 0; j < FILTER_TAP_NUM; j++)
        {
            // The first (numeratorLength-1) elements contain the filter state
            if (i + FILTER_TAP_NUM - j -1 < datalen) {
                output[i] += filter_taps[j] * d_data[i + FILTER_TAP_NUM - j - 1];
            }
        }
    }
    
}



int main (void)
{
    SF_INFO info1, info2;
    int *buffer1, *buffer2;
    
    char nombre1[] = "./audios/musica.wav";
    leerAudio(buffer1, info1, nombre1);
    
    int channels = info1.channels;
    int frames = info1.frames;
    int sampleRate = info1.samplerate;
    int num_items =  frames * channels;
    
    int *result = buffer1;
    filtro(num_items, buffer1, result);
    
    SNDFILE *Newsf;
    SF_INFO Newinfo;
    
    Newinfo.samplerate = sampleRate;
    Newinfo.channels = channels;
    Newinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    Newsf = sf_open("filtrado.wav", SFM_WRITE, &Newinfo);
    int num_write = sf_write_int(Newsf, result, num_items);
    sf_close(Newsf);
    printf("Writed %d items\n",num_write);
    
    return 0 ;
}
