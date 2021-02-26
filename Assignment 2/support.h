/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/
/*
 * Modified by Moumita Dey
 * */

#ifndef __FILEH__
#define __FILEH__

#include <sys/time.h>

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

// Matrix Structure declaration
typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    float* elements;
} Matrix;

#define FILTER_SIZE 3

Matrix allocateMatrix(unsigned height, unsigned width);
void initImage(Matrix R, Matrix G, Matrix B);
void initFilter(Matrix F);
void CPUConvolution(Matrix M, Matrix N, Matrix P);
void verify(Matrix M, Matrix F, Matrix P);
void output(char *outputFileName, Matrix R, Matrix G, Matrix B);
void freeMatrix(Matrix mat);
void freeDeviceMatrix(Matrix mat);
void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);

void initGPUInput(Matrix in, Matrix R, Matrix G, Matrix B);
void initGPUOutput(Matrix out, Matrix R, Matrix G, Matrix B);

#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

#endif

