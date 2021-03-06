#include <stdio.h>
#include <stdint.h>
#include <sndfile.hh>

//funcion leer fichero wav y guardar en buffer
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
    
    printf("formato=%d\n",info.format);
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


//////////////////////////////////////////////////////////////
//  Filter Code Definitions
//////////////////////////////////////////////////////////////

// maximum number of inputs that can be handled
// in one function call
#define MAX_INPUT_LEN   80
// maximum length of filter than can be handled
#define MAX_FLT_LEN     63
// buffer to hold all of the input samples
#define BUFFER_LEN      (MAX_FLT_LEN - 1 + MAX_INPUT_LEN)

// array to hold input samples
double insamp[ BUFFER_LEN ];

// FIR init
void firFloatInit( void )
{
    memset( insamp, 0, sizeof( insamp ) );
}

// the FIR filter function
void firFloat( double *coeffs, double *input, double *output,
              int length, int filterLength )
{
    double acc;     // accumulator for MACs
    double *coeffp; // pointer to coefficients
    double *inputp; // pointer to input samples
    int n;
    int k;
    
    // put the new samples at the high end of the buffer
    memcpy( &insamp[filterLength - 1], input,
           length * sizeof(double) );
    
    // apply the filter to each input sample
    for ( n = 0; n < length; n++ ) {
        // calculate output n
        coeffp = coeffs;
        inputp = &insamp[filterLength - 1 + n];
        acc = 0;
        for ( k = 0; k < filterLength; k++ ) {
            acc += (*coeffp++) * (*inputp--);
        }
        output[n] = acc;
    }
    // shift input samples back in time for next time
    memmove( &insamp[0], &insamp[length],
            (filterLength - 1) * sizeof(double) );
    
}

//////////////////////////////////////////////////////////////
//  Test program
//////////////////////////////////////////////////////////////

/*
 sampling frequency: 44100 Hz
 
 * 60 Hz - 450 Hz
 gain = 1
 desired ripple = 5 dB
 actual ripple = 3.7630218400563242 dB
 
 * 1000 Hz - 4000 Hz
 gain = 0
 desired attenuation = -40 dB
 actual attenuation = -40.87368601079213 dB
 
 */

#define FILTER_LEN  25
double coeffs[ FILTER_LEN ] =
{
    308248792.5006799,
    -791103354.6051474,
    634248454.09906,
    1430227023.1418133,
    -7583108178.48914,
    19630697094.246506,
    -36271579183.17313,
    51590799242.98169,
    -57896859780.24451,
    51590799242.98169,
    -36271579183.17313,
    19630697094.246506,
    -7583108178.48914,
    1430227023.1418133,
    634248454.09906,
    -791103354.6051474,
    308248792.5006799
};






void intToFloat( int16_t *input, double *output, int length )
{
    int i;
    
    for ( i = 0; i < length; i++ ) {
        output[i] = (double)input[i];
    }
}

void floatToInt( double *input, int16_t *output, int length )
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

// number of samples to read per loop
#define SAMPLES   80

int main( void )
{
    // open the input waveform file
    SF_INFO info1;
    int *buffer1;
    char nombre1[] = "./audios/1epic.wav";
    leerAudio(buffer1, info1, nombre1);
    
    int size;
    int16_t input[SAMPLES];
    int16_t output[SAMPLES];
    double floatInput[SAMPLES];
    double floatOutput[SAMPLES];
    FILE   *out_fid;
    
    // open the output waveform file
    out_fid = fopen( "outputFloat.pcm", "wb" );
    if ( out_fid == 0 ) {
        printf("couldn't open outputFloat.pcm");
        return -1;
    }
    
    // initialize the filter
    firFloatInit();
    // process all of the samples
    size=80;
    int current=0;
    do {
        // read samples from file
        for(int iter=current; iter<current+size; iter++){
            input[iter-current] = buffer1[iter];
        }
        current+=size;
        // convert to doubles
        intToFloat( input, floatInput, size );
        // perform the filtering
        firFloat( coeffs, floatInput, floatOutput, size,
                 FILTER_LEN );
        // convert to ints
        floatToInt( floatOutput, output, size );
        // write samples to file
        fwrite( output, sizeof(int16_t), size, out_fid );
    } while (current < info1.frames*info1.channels);
    
    fclose( out_fid );
    
    return 0;
}
