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
int16_t insamp[ BUFFER_LEN ];

// FIR init
void firFixedInit( void )
{
    memset( insamp, 0, sizeof( insamp ) );
}

// the FIR filter function
void firFixed( int16_t *coeffs, int16_t *input, int16_t *output,
              int length, int filterLength )
{
    int32_t acc;     // accumulator for MACs
    int16_t *coeffp; // pointer to coefficients
    int16_t *inputp; // pointer to input samples
    int n;
    int k;
    
    // put the new samples at the high end of the buffer
    memcpy( &insamp[filterLength - 1], input,
           length * sizeof(int16_t) );
    
    // apply the filter to each input sample
    for ( n = 0; n < length; n++ ) {
        // calculate output n
        coeffp = coeffs;
        inputp = &insamp[filterLength - 1 + n];
        // load rounding constant
        acc = 1 << 14;
        // perform the multiply-accumulate
        for ( k = 0; k < filterLength; k++ ) {
            acc += (int32_t)(*coeffp++) * (int32_t)(*inputp--);
        }
        // saturate the result
        if ( acc > 0x3fffffff ) {
            acc = 0x3fffffff;
        } else if ( acc < -0x40000000 ) {
            acc = -0x40000000;
        }
        // convert from Q30 to Q15
        output[n] = (int16_t)(acc >> 15);
    }
    
    // shift input samples back in time for next time
    memmove( &insamp[0], &insamp[length],
            (filterLength - 1) * sizeof(int16_t) );
    
}

//////////////////////////////////////////////////////////////
//  Test program
//////////////////////////////////////////////////////////////

// bandpass filter centred around 1000 Hz
// sampling rate = 8000 Hz
// gain at 1000 Hz is about 1.13

#define FILTER_LEN  63
int16_t coeffs[ FILTER_LEN ] =
{
    -1468, 1058,   594,   287,    186,  284,   485,   613,
    495,   90,  -435,  -762,   -615,   21,   821,  1269,
    982,    9, -1132, -1721,  -1296,    1,  1445,  2136,
    1570,    0, -1666, -2413,  -1735,   -2,  1770,  2512,
    1770,   -2, -1735, -2413,  -1666,    0,  1570,  2136,
    1445,    1, -1296, -1721,  -1132,    9,   982,  1269,
    821,   21,  -615,  -762,   -435,   90,   495,   613,
    485,  284,   186,   287,    594, 1058, -1468
};

// number of samples to read per loop
#define SAMPLES   80

int main( void )
{
    // open the input waveform file
    SF_INFO info1;
    int *buffer1;
    char nombre1[] = "./audios/test.wav";
    leerAudio(buffer1, info1, nombre1);
    
    int size;
    int16_t input[SAMPLES];
    int16_t output[SAMPLES];
    FILE   *in_fid;
    FILE   *out_fid;
    
    // open the output waveform file
    out_fid = fopen( "outputFixed.pcm", "wb" );
    if ( out_fid == 0 ) {
        printf("couldn't open outputFixed.pcm");
        exit(EXIT_FAILURE);
    }
    
    // initialize the filter
    firFixedInit();
    
    // process all of the samples
    size=80;
    int start=0;
    do {
        // read samples from file
        size = 80;
        for(int iter=start; iter<size; iter++){
            input[iter] = buffer1[iter];
        }
        start+=80;
        // perform the filtering
        firFixed( coeffs, input, output, size, FILTER_LEN );
        // write samples to file
        fwrite( output, sizeof(int16_t), size, out_fid );
    } while (start < info1.frames*info1.channels);
    
    fclose( in_fid );
    fclose( out_fid );
    
    return 0;
}
