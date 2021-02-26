/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <iostream>
#include "support.h"
#include "kernel.cu"

/*
Made by Moumita Dey
*/

int main(int argc, char* argv[])
{

	// absorbs all contexts
	cudaFree(0);	

	// get device info
	int device;
	cudaGetDevice(&device);	
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);	
	std::cout << "Device " << device << ": " << prop.name << "\n";
	std::cout << "GPU Cores: " << prop.multiProcessorCount << "\n";
	std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
	std::cout << "Shared memory per block " << prop.sharedMemPerBlock << "\n";	
	std::cout << "Maximum threads per block: " << prop.maxThreadsPerBlock << "\n";
	std::cout << "Maximum Grid dimensions: " << prop.maxGridSize[0] << "x" << prop.maxGridSize[1] << "x" << prop.maxGridSize[2] << "\n";
	std::cout << "Total constant memory: " << prop.totalConstMem << "\n";
	
	Timer timer;
	if (argc != 3) {
	    printf("\nInput files not specified");
	    exit(0);
	}
        char *inputImageFile = argv[1];
        char *labelFile = argv[2];

	//-------------------------------------------------------------------------
    // CONVOLUTION LAYER
    //-------------------------------------------------------------------------

	// Initialize host variables ----------------------------------------------
	printf("\nSetting up the problem..."); fflush(stdout);
	startTime(&timer);

	// Allocate and initialize host variables ----------------------------------------------
	int result;
	Matrix *conv_weight, conv_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, test_image;
    input(inputImageFile, &conv_weight, &conv_bias, &fc1_weight, &fc1_bias, &fc2_weight, &fc2_bias, &test_image);

    // Append apron to the image itself
	Matrix test_image_mod;
	test_image_mod = allocateMatrix(test_image.height + conv_weight[0].height - 1, test_image.width + conv_weight[0].width - 1);
	initGPUInput(test_image, test_image_mod);

	float *result_h;
	result_h = (float*) malloc (sizeof(float)*10);
	float *A_h, *D_h, *K_h;
	A_h = (float*) malloc (sizeof(float)*28*28*8);
	D_h = (float*) malloc (sizeof(float)*28*28*8);
	K_h = (float*) malloc (sizeof(float)*512);

	// converting convolution weight matrix to a single dimension array
	float conv_weight_transposed[10*10*8];
	for(int i=0; i<8; ++i)
	{
		for(int j=0; j<10*10; ++j)
		{
			conv_weight_transposed[i*100+j] = conv_weight[i].elements[j];//100-j-1];
		}
	}

	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));

	// Allocate device variables ----------------------------------------------
	printf("Allocating device variables..."); fflush(stdout);
	startTime(&timer);

	//INSERT DEVICE ALLOCATION CODE HERE
	float* A_d;
	cudaMalloc(&A_d, sizeof(float)*28*28*8);

	cudaDeviceSynchronize();
	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));
	// Copy host variables to device ------------------------------------------

	printf("Copying data from host to device..."); fflush(stdout);
	startTime(&timer);
	
	//INSERT HOST TO DEVICE COPY CODE HERE
	cudaMemcpyToSymbol(conv_weight_c, conv_weight_transposed, sizeof(float)*conv_weight[0].height*conv_weight[0].width*8);
	cudaMemcpyToSymbol(conv_bias_c, conv_bias.elements, sizeof(float)*conv_bias.height*conv_bias.width);
	cudaMemcpyToSymbol(test_image_c, test_image_mod.elements, sizeof(float)*test_image_mod.height*test_image_mod.width);
	cudaMemset(A_d, 0, sizeof(float)*28*28*8);

	cudaDeviceSynchronize();
	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));

	// Launch kernel ----------------------------------------------------------
	printf("Launching kernel..."); fflush(stdout);
	startTime(&timer);

	//INSERT KERNEL LAUNCH CODE HERE
	int BLOCK_SIZE_X = 7;
	int BLOCK_SIZE_Y = 7;
	int BLOCK_SIZE_Z = 8;
	int GRID_SIZE_X = 4;
	int GRID_SIZE_Y = 4;
	int GRID_SIZE_Z = 1; 

	dim3 dim_grid(GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z);
	dim3 dim_block(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
	convolution<<<dim_grid, dim_block>>>(A_d);

	cudaError_t cuda_ret;
	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

	cudaDeviceSynchronize();
	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));

	// Copy device variables from host ----------------------------------------

	printf("Copying data from device to host..."); fflush(stdout);
	startTime(&timer);

	//INSERT DEVICE TO HOST COPY CODE HERE
	cudaMemcpy(A_h, A_d, sizeof(float)*28*28*8, cudaMemcpyDeviceToHost); 

	cudaDeviceSynchronize();
	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));
	
/*	printf("\n\n");
	for(int i=0; i<8; ++i)
	{
		for(int j=0; j<28; ++j)
		{
			for(int k=0; k<28; ++k)
			{
				printf("%f ", A_h[i*28*28+j*28+k]);
			}
			printf("\n");
		}
		printf("\n\n");
	}
*/
	//-------------------------------------------------------------------------
    // FULLY CONNECTED LAYER 1
    //-------------------------------------------------------------------------

	// Reshaping output of convolution kernel to feed to fully connected layer
	//printf("\n");
	int ind = 0;
	for(int i=0; i<28; ++i)
	{
		for(int j=0; j<28; ++j)
		{
			for(int k=0; k<8; ++k)
			{
				D_h[ind] = A_h[k*28*28 + i*28 + j];
				ind++;
			}
		}
	}
 
/*	printf("\n\n");
	for(int i=0; i<6272; ++i)
	{
		if (i%4 == 0)
			printf("\n");
		if (i%(8*28) == 0)
			printf("\n");
		printf("%f ", D_h[i]);
	}
*/
	// Allocate device variables ----------------------------------------------
	printf("Allocating device variables..."); fflush(stdout);
	startTime(&timer);

	//INSERT DEVICE ALLOCATION CODE HERE
	float *D_d, *K_d, *fc1_weight_d;
	cudaMalloc(&D_d, sizeof(float)*28*28*8);
	cudaMalloc(&K_d, sizeof(float)*512);
	cudaMalloc(&fc1_weight_d, sizeof(float)*6272*512);

	cudaDeviceSynchronize();
	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));
	// Copy host variables to device ------------------------------------------

	printf("Copying data from host to device..."); fflush(stdout);
	startTime(&timer);
	
	//INSERT HOST TO DEVICE COPY CODE HERE
	cudaMemcpyToSymbol(fc1_bias_c, fc1_bias.elements, sizeof(float)*fc1_bias.height*fc1_bias.width);
	cudaMemcpy(D_d, D_h, sizeof(float)*28*28*8, cudaMemcpyHostToDevice);
	cudaMemset(K_d, 0, sizeof(float)*512);
	cudaMemcpy(fc1_weight_d, fc1_weight.elements, sizeof(float)*fc1_weight.height*fc1_weight.width, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));

	// Launch kernel ----------------------------------------------------------
	printf("Launching kernel..."); fflush(stdout);
	startTime(&timer);

	//INSERT KERNEL LAUNCH CODE HERE
	BLOCK_SIZE_X = 32; // chosen to keep warp sized accesses
	BLOCK_SIZE_Y = 1;
	BLOCK_SIZE_Z = 1;
	GRID_SIZE_X = 512/32;
	GRID_SIZE_Y = 1;
	GRID_SIZE_Z = 1; 

	dim3 dim_grid_fc1(GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z);
	dim3 dim_block_fc1(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
	fc1<<<dim_grid_fc1, dim_block_fc1>>>(D_d, fc1_weight_d, K_d);

	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

	cudaDeviceSynchronize();
	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));

	// Copy device variables from host ----------------------------------------

	printf("Copying data from device to host..."); fflush(stdout);
	startTime(&timer);

	//INSERT DEVICE TO HOST COPY CODE HERE
	cudaMemcpy(K_h, K_d, sizeof(float)*512, cudaMemcpyDeviceToHost);

/*	printf("\n\n");
	for(int i=0; i<512; ++i)
	{
		if (i%6 == 0)
			printf("\n");
		printf("%f ", K_h[i]);
	}	
*/
	cudaDeviceSynchronize();
	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));

	//-------------------------------------------------------------------------
    // FULLY CONNECTED LAYER 2
    //-------------------------------------------------------------------------

	// Allocate device variables ----------------------------------------------
	printf("Allocating device variables..."); fflush(stdout);
	startTime(&timer);

	//INSERT DEVICE ALLOCATION CODE HERE
	float *fc2_weight_d;
	float *result_d;
	cudaMalloc(&fc2_weight_d, sizeof(float)*512*10);
	cudaMalloc(&result_d, sizeof(float)*10);

	cudaDeviceSynchronize();
	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));
	// Copy host variables to device ------------------------------------------

	printf("Copying data from host to device..."); fflush(stdout);
	startTime(&timer);
	
	//INSERT HOST TO DEVICE COPY CODE HERE
	cudaMemcpyToSymbol(fc2_bias_c, fc2_bias.elements, sizeof(float)*fc2_bias.height*fc2_bias.width);
	cudaMemcpy(K_d, K_h, sizeof(float)*512, cudaMemcpyHostToDevice);
	cudaMemset(result_d, 0, sizeof(float)*10);
	cudaMemcpy(fc2_weight_d, fc2_weight.elements, sizeof(float)*fc2_weight.height*fc2_weight.width, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));

	// Launch kernel ----------------------------------------------------------
	printf("Launching kernel..."); fflush(stdout);
	startTime(&timer);

	//INSERT KERNEL LAUNCH CODE HERE
	BLOCK_SIZE_X = 10;
	BLOCK_SIZE_Y = 1;
	BLOCK_SIZE_Z = 1;
	GRID_SIZE_X = 1;
	GRID_SIZE_Y = 1;
	GRID_SIZE_Z = 1; 

	dim3 dim_grid_fc2(GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z);
	dim3 dim_block_fc2(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
	fc2<<<dim_grid_fc2, dim_block_fc2>>>(K_d, fc2_weight_d, result_d);

	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

	cudaDeviceSynchronize();
	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));

	// Copy device variables from host ----------------------------------------

	printf("Copying data from device to host..."); fflush(stdout);
	startTime(&timer);

	//INSERT DEVICE TO HOST COPY CODE HERE
	cudaMemcpy(result_h, result_d, sizeof(float)*10, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));


	//-------------------------------------------------------------------------
    // INTERPRETING RESULT
    //-------------------------------------------------------------------------
	
/*	for(int i=0; i<10; ++i)
	{
		printf("\n%f", result_h[i]);
	}
*/
	// Find the index of the maximum of the result array from output of fully connected layer 2
	float max = result_h[0];
	for(int i=0; i<10; ++i)
	{	
		if(result_h[i] > max)
		{
			max = result_h[i];
			result = i;
		}
	}
//	printf("\n\n\t\t%d\n\n", result);


	// Verify correctness -----------------------------------------------------
    verify(result, labelFile);

	// Free host and device memory ------------------------------------------------------------
	freeMatrix(*conv_weight);
	freeMatrix(conv_bias);
	freeMatrix(fc1_weight);
	freeMatrix(fc1_bias);
	freeMatrix(fc2_weight);
	freeMatrix(fc2_bias);
	freeMatrix(test_image);
	freeMatrix(test_image_mod);

	return 0;
}

// debugging information; ignore

/*	printf("\n\n");
	for(int i=0; i<8; ++i)
	{
		for(int j=0; j<28; ++j)
		{
			for(int k=0; k<28; ++k)
			{
				printf("%f ", A_h[i*28*28+j*28+k]);
			}
			printf("\n");
		}
		printf("\n\n");
	}
*/

	/*	printf("\n\n");
	for(int i=0; i<8; ++i)
	{
		for(int j=0; j<10; ++j)
		{
			for(int k=0; k<10; ++k)
			{
				printf("%f ", conv_weight_transposed[i][j*10+k]);
			}
			printf("\n");
		}
		printf("\n\n");
	}
*/

	/*	printf("\n\n");
	for(int i=0; i<28; ++i)
	{
		for(int j=0; j<28; ++j)
			printf("%f ", test_image.elements[i*28+j]);
		printf("\n");
	}
	printf("\n\n");
	for(int i=0; i<37; ++i)
	{
		for(int j=0; j<37; ++j)
			printf("%f ", test_image_mod.elements[i*37+j]);
		printf("\n");
	}
*/