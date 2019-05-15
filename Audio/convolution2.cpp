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
 
 sampling frequency: 44100 Hz
 
 * 0 Hz - 800 Hz
 gain = 0
 desired attenuation = -40 dB
 actual attenuation = -42.35492877058113 dB
 
 * 2000 Hz - 6000 Hz
 gain = 1
 desired ripple = 5 dB
 actual ripple = 3.1586736834970783 dB
 
 */

#define FILTER_TAP_NUM 15

static double filter_taps[FILTER_TAP_NUM] = {
    17830.208106445272,
    -200528.65753843088,
    1102888.4155863246,
    -3867783.5731915575,
    9617827.131572342,
    -17912564.257079516,
    25749913.681529522,
    -29015165.89034492,
    25749913.681529522,
    -17912564.257079516,
    9617827.131572342,
    -3867783.5731915575,
    1102888.4155863246,
    -200528.65753843088,
    17830.208106445272
};





void convolve (double *p_coeffs, int p_coeffs_n,
               double *p_in, double *p_out, int n)
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
        printf("tmp:%f\n",tmp);
        p_out [i] = tmp;
    }
}

void conv(const double v1[], size_t n1, const double v2[], size_t n2, double r[])
{
    for (size_t n = 0; n < n1 + n2 - 1; n++)
        for (size_t k = 0; k < std::max(n1, n2); k++)
            r[n] += (k < n1 ? v1[k] : 0) * (n - k < n2 ? v2[n - k] : 0);
}

void intToFloat( int *input, double *output, int length )
{
    for (int i = 0; i < length; i++ ) {
        output[i] = (double)input[i];
    }
}

void floatToInt( double input[], int16_t output[], int length )
{
    int i;
    
    for ( i = 0; i < length; i++ ) {
        if ( input[i] > 32767.0 ) {
            input[i] = 32767.0;
        } else if ( input[i] < -32768.0 ) {
            input[i] = -32768.0;
        }
        // convert
        output[i] = (int16_t)input[i];
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
    double floatOutput[size];
    int16_t intOutput[size];
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
        conv(filter_taps, FILTER_TAP_NUM, floatInput, size, floatOutput);
        //for (int co = 0; co<size; co++) {
          //  printf("%f --&-- %f\n", floatInput[co], floatOutput[co]);
        //}
        // convert to ints
        floatToInt( floatOutput, intOutput, size );
        //for (int co = 0; co<size; co++) {
          //printf("%f --&-- %d\n", floatOutput[co], intOutput[co]);
        //}
        // write samples to file
        fwrite( intOutput, sizeof(int16_t), size, out_fid );
    } while (current < info1.frames*info1.channels);
    
    fclose( out_fid );

    return 0 ;
}
