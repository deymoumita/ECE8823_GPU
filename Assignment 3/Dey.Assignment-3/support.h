/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

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

#define IMAGE_SIZE 28

Matrix allocateMatrix(unsigned height, unsigned width);
void input(char *inputImageFile, Matrix **conv_weight, Matrix *conv_bias, Matrix *fc1_weight, Matrix *fc1_bias, Matrix *fc2_weight, Matrix *fc2_bias, Matrix *test_image);
void output(char *outputFileName, Matrix R);
void freeMatrix(Matrix mat);
void freeDeviceMatrix(Matrix mat);
void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);
void verify(int result, char *labelFile);

void initGPUInput(Matrix in, Matrix out);

#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

#endif

