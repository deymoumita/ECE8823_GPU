/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/
/*
Made by Moumita Dey
*/

#include <stdio.h>
#include "support.h"

// constant memory
__constant__ float conv_weight_c[10*10*8];
__constant__ float test_image_c[(28+9)*(28+9)];
__constant__ float conv_bias_c[8];
__constant__ float fc1_bias_c[512];
__constant__ float fc2_bias_c[10];


// convolution kernel
__global__ void convolution(float* A)
{
    //INSERT KERNEL CODE HERE

	// common parameters
	int imageDim = 28; // image height and width
	int imageDim_mod = 28+9; // image dimensions with apron
	int conv_filterDim = 10;

	// output index calculation - output stationary approach
	int A_x = blockIdx.x * blockDim.x + threadIdx.x;
	int A_y = blockIdx.y * blockDim.y + threadIdx.y;
	int A_z = blockIdx.z * blockDim.z + threadIdx.z;
	int ind = A_z*imageDim*imageDim + A_y*imageDim + A_x;

	// each dimension of thread block will take image and corresponding weight matrix
	float A_result = 0.000000;

	int i, j;

	// convolution
	#pragma unroll
	for(i=0; i<conv_filterDim; ++i)
	{
		for(j=0; j<conv_filterDim; ++j)
		{
			//if(ind == 0)
				//printf("\n%d %d %d %d %d", i, j, A_x, A_y, A_z);
				//printf("\n%d(%f) %d(%f)", (i+A_y)*imageDim_mod+(j+A_x), test_image_c[(i+A_y)*imageDim_mod+(j+A_x)], A_z*conv_filterDim*conv_filterDim + i*conv_filterDim + j, conv_weight_c[A_z*conv_filterDim*conv_filterDim + i*conv_filterDim + j]);
			A_result += test_image_c[(i+A_y)*imageDim_mod+(j+A_x)] * conv_weight_c[A_z*conv_filterDim*conv_filterDim + i*conv_filterDim + j];
		}
	}

	// bias addition
	A[ind] = A_result + conv_bias_c[A_z];

	// ReLu	
	if(A[ind] < 0.000000000)
		A[ind] = 0.000000000;
}


// Fully connected layer 1 kernel
__global__ void fc1(float* D, float* fc1_weight, float* K)
{
	// output stationary approach

	// index calculation
	int K_x = blockIdx.x * blockDim.x + threadIdx.x;

	// matrix multiplication
	float K_result = 0.000000;
	for(int i=0; i<6272; ++i)
	{
		K_result += D[i] * fc1_weight[i*512+K_x];
	}

	// bias addition
	K[K_x] = K_result + fc1_bias_c[K_x];

	// ReLu
	if(K[K_x] < 0)
		K[K_x] = 0.000000;
}

// Fully connected layer 2 kernel
__global__ void fc2(float* K, float* fc2_weight, float* result)
{
	// output stationary approach

	// index calculation
	int result_x = blockIdx.x * blockDim.x + threadIdx.x;

	// matrix multiplication
	float result_result = 0.000000;
	for(int i=0; i<512; ++i)
	{
		result_result += K[i] * fc2_weight[i*10+result_x];
	}

	// bias addition
	result[result_x] = result_result + fc2_bias_c[result_x];
}

// debugging information ; ignore

/*	if(ind == 784)
		printf("\n\n\n%f\n\n\n", A[ind]);

	__syncthreads();
/*	if(ind==784)
	{
		printf("\n\n");
		for(int i=0; i<8; ++i)
		{
			for(int j=0; j<28; ++j)
			{
				for(int k=0; k<28; ++k)
				{
					printf("%f ", A[i*28*28+j*28+k]);
				}
				printf("\n");
			}
			printf("\n\n");
		}
	}
/*	
*/	

/*	if(ind == 8)
	{
		printf("\n%f", A_result);
		printf("\n%f", A[ind]);
	}
*/		
/*	if(ind == 0)
	{
		printf("\n");
		for(int i=0; i<imageDim; ++i)
		{
			for(int j=0; j<imageDim; ++j)
				printf("%f ", A[(i)*imageDim+j]);
			printf("\n");
		}
		printf("\n\n%f\n\n", A[ind]);
	}
		
*/

	
/*	if(ind == 0)
	{
		printf("\n\n\n");
		for(i=0; i<28; ++i)
		{
			for(j=0; j<28; ++j)
			{
				for(k=0; k<28; ++k)
				{
					printf("%f ", A[i*imageDim*imageDim+j*imageDim+k]);
				}
				printf("\n");
			}
			printf("\n\n\n");
		}
	}
	__syncthreads();
*/





