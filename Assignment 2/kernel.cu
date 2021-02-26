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
__constant__ float F_c[FILTER_SIZE * FILTER_SIZE];

// kernel
__global__ void convolution(float* in, float* out, int img_dim, int conv_dim)
{

    //INSERT KERNEL CODE HERE

	// define shared memory being used by the kernel
	__shared__ float in_s[34*34];	// 32*32 + apron

	// define indices
	int img_x = blockIdx.x*blockDim.x+threadIdx.x+1;	// input image indexing
	int img_y = blockIdx.y*blockDim.y+threadIdx.y+1;
	int img_z = blockIdx.z;

	int conv_x = blockIdx.x*blockDim.x+threadIdx.x;		// output image indexing
	int conv_y = blockIdx.y*blockDim.y+threadIdx.y;
	int conv_z = blockIdx.z;

	int s_x = threadIdx.x+1;	// shared memory indexing
	int s_y = threadIdx.y+1;

	int s_dim = 34;	// shared memory dimension

	// load center pixel to shared memory
	in_s[s_y*s_dim+s_x] = in[img_z*img_dim*img_dim + img_y*img_dim + img_x];
	
	// load left apron
	if(s_x == 1)
		in_s[s_y*s_dim] = in[img_z*img_dim*img_dim + img_y*img_dim + img_x-1];
	

	// load right apron
	if(s_x == 32)
		in_s[s_y*s_dim+s_dim-1] = in[img_z*img_dim*img_dim + img_y*img_dim + img_x+1];


	// load top apron
	if(s_y == 1)
		in_s[s_x] = in[img_z*img_dim*img_dim + (img_y-1)*img_dim + img_x];

	// load bottom apron
	if(s_y == 32)
		in_s[(s_y+1)*s_dim+s_x] = in[img_z*img_dim*img_dim + (img_y+1)*img_dim + img_x];

	// load top left corner
	if(s_x == 1 && s_y == 1)
		in_s[0] = in[img_z*img_dim*img_dim + (img_y-1)*img_dim + img_x-1];

	// load top right corner
	if(s_x == 32 && s_y == 1)
		in_s[33] = in[img_z*img_dim*img_dim + (img_y-1)*img_dim + img_x+1];
	
	// load bottom left corner
	if(s_x == 1 && s_y == 32)
		in_s[33*s_dim] =  in[img_z*img_dim*img_dim + (img_y+1)*img_dim + img_x - 1];
	
	// load bottom right corner
	if(s_x == 32 && s_y == 32)
		in_s[33*s_dim+33] = in[img_z*img_dim*img_dim + (img_y+1)*img_dim + img_x + 1];
	
	__syncthreads();

	// compute convolution and store to output
	out[conv_z*conv_dim*conv_dim + conv_y*conv_dim + conv_x] = F_c[0]*in_s[(s_y-1)*s_dim+(s_x-1)] + F_c[1]*in_s[(s_y-1)*s_dim+(s_x)] + F_c[2]*in_s[(s_y-1)*s_dim+(s_x+1)] + F_c[3]*in_s[(s_y)*s_dim+(s_x-1)] + F_c[4]*in_s[(s_y)*s_dim+(s_x)] + F_c[5]*in_s[(s_y)*s_dim+(s_x+1)] + F_c[6]*in_s[(s_y+1)*s_dim+(s_x-1)] + F_c[7]*in_s[(s_y+1)*s_dim+(s_x)] + F_c[8]*in_s[(s_y+1)*s_dim+(s_x+1)];



///////////////////////////////////////////////////////////////////
/*	// debugging 
	if(img_x == 256 && img_y == 256 && img_z == 1)
	{
		printf("\n\n%d\t%d\t%d\n\n", conv_z, conv_y, conv_dim);
		printf("\n%d %d %d \n%d %d %d \n%d %d %d\n", (s_y-1)*s_dim+(s_x-1), (s_y-1)*s_dim+(s_x), (s_y-1)*s_dim+(s_x+1) , (s_y)*s_dim+(s_x-1) , (s_y)*s_dim+(s_x) , (s_y)*s_dim+(s_x+1) , (s_y+1)*s_dim+(s_x-1) , (s_y+1)*s_dim+(s_x) , (s_y+1)*s_dim+(s_x+1));

		printf("\n%f %f %f \n%f %f %f \n%f %f %f\n", in_s[(s_y-1)*s_dim+(s_x-1)], in_s[(s_y-1)*s_dim+(s_x)], in_s[(s_y-1)*s_dim+(s_x+1)] , in_s[(s_y)*s_dim+(s_x-1)] , in_s[(s_y)*s_dim+(s_x)] , in_s[(s_y)*s_dim+(s_x+1)] , in_s[(s_y+1)*s_dim+(s_x-1)] , in_s[(s_y+1)*s_dim+(s_x)] , in_s[(s_y+1)*s_dim+(s_x+1)]);
	printf("\n%d\n", img_z*img_dim*img_dim + img_y*img_dim + img_x);
	}
*/


}


