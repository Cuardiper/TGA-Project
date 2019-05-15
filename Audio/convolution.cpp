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
}




/*
 
 FIR filter designed with
 http://t-filter.appspot.com
 
 sampling frequency: 2000 Hz
 
 * 0 Hz - 400 Hz
 gain = 1
 desired ripple = 5 dB
 actual ripple = 4.139389496607156 dB
 
 * 500 Hz - 1000 Hz
 gain = 0
 desired attenuation = -40 dB
 actual attenuation = -40.07355419274887 dB
 
 */

#define FILTER_TAP_NUM 21

static double filter_taps[FILTER_TAP_NUM] = {
    -0.020104118828857275,
    -0.058427980043525084,
    -0.061178403647822,
    -0.010939393385339008,
    0.05125096443534968,
    0.03322086767894787,
    -0.05655276971833924,
    -0.08565500737264509,
    0.06337959966054495,
    0.31085440365663597,
    0.4344309124179415,
    0.31085440365663597,
    0.06337959966054495,
    -0.08565500737264509,
    -0.05655276971833924,
    0.03322086767894787,
    0.05125096443534968,
    -0.010939393385339008,
    -0.061178403647822,
    -0.058427980043525084,
    -0.020104118828857275
};



void convolve (double *p_coeffs, int p_coeffs_n,
               double *p_in, int *p_out, int n)
{
    int i, j, k;
    double tmp;
    
    for (k = 0; k < n; k++)  //  position in output
    {
        tmp = 0;
        
        for (i = 0; i < p_coeffs_n; i++)  //  position in coefficients array
        {
            j = k - i;  //  position in input
            
            if (j >= 0)  //  bounds check for input buffer
            {
                tmp += p_coeffs [k] * p_in [j];
            }
        }
        //printf("tmp: %f\n",tmp);
        if ( tmp > 32767.0 ) {
            tmp = 32767.0;
        } else if ( tmp < -32768.0 ) {
            tmp = -32768.0;
        }
        // convert
        p_out[i] = (int16_t)tmp;
        //printf("out: %d\n",p_out[i]);
    }
}

void intToFloat( int *input, double *output, int length )
{
    for (int i = 0; i < length; i++ ) {
        output[i] = (double)input[i];
    }
}

int main (void)
{
    SF_INFO info1;
    int *buffer1;
    
    char nombre1[] = "./audios/epic.wav";
    leerAudio(buffer1, info1, nombre1);
    
    int channels = info1.channels;
    int frames = info1.frames;
    int sampleRate = info1.samplerate;
    int num_items =  frames * channels;
    
    FILE   *out_fid;
    // open the output waveform file
    out_fid = fopen( "outputFloat.pcm", "wb" );
    if ( out_fid == 0 ) {
        printf("couldn't open outputFloat.pcm");
        return -1;
    }
    
    double *coeffs = filter_taps;
    int coeffs_n = FILTER_TAP_NUM;
    
    int size=80;
    int aux[size];
    double floatInput[size];
    int Output[size];
    int current=0;
    do {
        // read samples from file
        for(int iter=current; iter< std::min(current+size, num_items); iter++){
            aux[iter-current] = buffer1[iter];
        }
        current+=size;
        // convert to doubles
        intToFloat( aux, floatInput, size );
        // perform the filtering
        convolve(coeffs, coeffs_n, floatInput, Output, size);
        //for (int co = 0; co<size; co++) {
            //printf("%d --> %d\n", co,Output[co]);
        //}
        // convert to ints
        //floatToInt( floatOutput, output, size );
        // write samples to file
        fwrite( Output, sizeof(int16_t), size, out_fid );
    } while (current < info1.frames*info1.channels);
    
    fclose( out_fid );

    return 0 ;
}
