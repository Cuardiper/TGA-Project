//
//  test.cpp
//  parserAudio.cc
//
//  Created by Alvaro on 29/04/2019.
//

#include <cstdio>
#include <sndfile.hh>

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

//suma audios, numero de canales igual!
int* sumarAudios(int *bufferShort, SF_INFO Shortinfo, int *bufferLong)
{
    int channels = Shortinfo.channels;
    int num_items =  Shortinfo.frames * channels;
    
    int *Newbuffer = bufferLong;
    
    for (int i = 0; i < num_items; i += channels)
    {
        for (int j = 0; j < channels; ++j){
            Newbuffer[i+j] += bufferShort[i+j];
        }
    }
    return Newbuffer;
}

int main (void)
{
    SF_INFO info1, info2;
    int *buffer1, *buffer2;
    char nombre1[] = "./audios/discurso.wav";
    leerAudio(buffer1, info1, nombre1);
    
    char nombre2[] = "./audios/musica.wav";
    leerAudio(buffer2, info2, nombre2);
    
    
    int channels = info1.channels;
    int frames = info1.frames;
    int sampleRate = info1.samplerate;
    int num_items =  frames * channels;
    
    for (int i = 0; i < num_items; i += channels)
    {
        for (int j = 0; j < channels; ++j){
            buffer1[i+j] *= 2;
        }
    }
    
    SNDFILE *Newsf;
    SF_INFO Newinfo;
    
    Newinfo.samplerate = sampleRate / 2;
    Newinfo.channels = channels;
    Newinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    
    Newsf = sf_open("new1.wav", SFM_WRITE, &Newinfo);
    int num_write = sf_write_int(Newsf, buffer1, num_items);
    sf_close(Newsf);
    printf("Writed %d items\n",num_write);
    
    /*
     ///////////////////
     //Crear un nuevo .wav con lo datos leidos
     SNDFILE *Newsf;
     SF_INFO Newinfo;
     
     //comparar size de buffer para llamar correctamente
     int *Newbuffer = sumarAudios(buffer2, info2, buffer1);
     
     //escribir audio
     int channels = info1.channels;
     int frames = info1.frames;
     int sampleRate = info1.samplerate;
     int num_items =  frames * channels;
     
     Newinfo.samplerate = sampleRate;
     Newinfo.channels = channels;
     Newinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
     
     Newsf = sf_open("new.wav", SFM_WRITE, &Newinfo);
     int num_write = sf_write_int(Newsf, Newbuffer, num_items);
     sf_close(Newsf);
     printf("Writed %d items\n",num_write);
     */
    return 0 ;
}
