/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

/*
Modified by Moumita Dey
*/

#include <stdlib.h>
#include <stdio.h>

#include "support.h"

Matrix allocateMatrix(unsigned height, unsigned width)
{
	Matrix mat;
	mat.height = height;
	mat.width = mat.pitch = width;
	mat.elements = (float*)malloc(height*width*sizeof(float));
	if(mat.elements == NULL) FATAL("Unable to allocate host");

	return mat;
}

void initImage(Matrix R, Matrix G, Matrix B)
{
    for (unsigned int i=0; i < R.height*R.width; i++) {
		// printf("%d\t", i);
        if (scanf("%f%f%f", &(R.elements[i]), &(G.elements[i]), &(B.elements[i])) == EOF) {
            printf("\nincorrect input\n");
            exit(0);
        }
    }
}

void initFilter(Matrix F)
{
    for (unsigned int i=0; i < F.height*F.width; i++) {
        if (scanf("%f", &(F.elements[i])) == EOF) {
            printf("\nincorrect input\n");
            exit(0);
        }
    }
}

void output(char *outputFileName, Matrix R, Matrix G, Matrix B) {
    FILE *fp;
    fp=fopen(outputFileName, "wb");
    fprintf(fp, "%d %d ", R.height, R.width);
    for (unsigned int i=0; i < R.height*R.width; i++) {
        fprintf(fp, "%f %f %f ", R.elements[i], G.elements[i], B.elements[i]);
    }
    fclose(fp);
}

void CPUConvolution(Matrix M, Matrix  N, Matrix P) {

int kCols = M.height;
int kRows = M.width;
int cols = N.height;
int rows = N.width;
int kCenterX = kCols/2;
int kCenterY = kRows/2;

for (int i = 0; i < rows; ++i)
{
	for (int j = 0; j <  cols; ++j)
	{
		for (int m = 0; m < kRows; ++m)
		{
			int mm = kRows-1-m;
			for (int n = 0; n < kCols; ++n)
			{
				int nn = kCols-1-n;

				int ii = i + (m-kCenterY);
				int jj = j + (n-kCenterX);

				if(ii >= 0 && ii < rows && jj >= 0 && jj < cols)
				{
					P.elements[i*cols+j] += N.elements[ii*rows+jj]*M.elements[mm*kRows+nn];
					// printf("out[%d][%d] \t in[%d][%d] * kernel[%d][%d]\n", i, j, ii, jj, mm, nn);
				}
			}

		}
	}
}

/*
  for(int row = 0; row < N.height; ++row) {
    for(int col = 0; col < N.width; ++col) {
      float sum = 0.0f;
      for(int i = 0; i < M.height; ++i) {
        for(int j = 0; j < M.width; ++j) {
            int iN = row - M.height/2 + i;
            int jN = col - M.width/2 + j;
            if(iN >= 0 && iN < N.height && jN >= 0 && jN < N.width) {
                sum += M.elements[i*M.width + j]*N.elements[iN*N.width + jN];
			}
        }
      }
      // P.elements[row*P.width + col] = sum;
    } 
  }

*/

}

void verify(Matrix M, Matrix  N, Matrix P) {

  const float relativeTolerance = 1e-6;

  for(int row = 0; row < N.height; ++row) {
    for(int col = 0; col < N.width; ++col) {
      float sum = 0.0f;
      for(int i = 0; i < M.height; ++i) {
        for(int j = 0; j < M.width; ++j) {
            int iN = row - M.height/2 + i;
            int jN = col - M.width/2 + j;
            if(iN >= 0 && iN < N.height && jN >= 0 && jN < N.width) {
                sum += M.elements[i*M.width + j]*N.elements[iN*N.width + jN];
            }
        }
      }
      float relativeError = (sum - P.elements[row*P.width + col])/sum;
      if (relativeError > relativeTolerance
        || relativeError < -relativeTolerance) {
		 // printf("\n%kernel: %f\texpected: %f\t out[%d][%d]\n", P.elements[row*P.width + col], sum, row, col);
        printf("\nTEST FAILED\n\n");
        exit(0);
	   }
//	printf("%f\n", sum);
	
    }
  }
  printf("\nTEST PASSED\n\n");
}

void freeMatrix(Matrix mat)
{
	free(mat.elements);
	mat.elements = NULL;
}

void freeDeviceMatrix(Matrix mat)
{
	cudaFree(mat.elements);
	mat.elements = NULL;
}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

// This copies RGB into a single array along with the apron
void initGPUInput(Matrix in, Matrix R, Matrix G, Matrix B)
{	
	// initialize the input matrix. this will automatically load all 0 in apron
	in.height = R.height+2;
	in.width = R.width+2;
	for(unsigned int i=0; i < in.height*in.width; ++i)
		in.elements[i] = 0;

	for (unsigned int i=0; i < R.height; ++i)
	{
		for(unsigned int j=0; j < R.width; ++j)
		{
			in.elements[(i+1)*in.height+(j+1)] = R.elements[i*R.height+j];
			in.elements[in.height*in.width+(i+1)*in.height+(j+1)] = G.elements[i*R.height+j];
			in.elements[in.height*in.width*2+(i+1)*in.height+(j+1)] = B.elements[i*R.height+j];
        }
    }
}

// copies to RGB separately
void initGPUOutput(Matrix out, Matrix R, Matrix G, Matrix B)
{
	for(unsigned int i=0; i < R.height*R.width; ++i)
	{
		R.elements[i] = out.elements[i];
		G.elements[i] = out.elements[R.height*R.height+i];
		B.elements[i] = out.elements[R.height*R.height*2+i];
	}
}

