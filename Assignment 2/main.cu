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


#include <stdio.h>
#include <iostream>
#include "support.h"
#include "kernel.cu"


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

	// set input output files
	Timer timer;
	if (argc != 2) {
	    printf("\nOutput file not specified");
	    exit(0);
	}
	char *outputFileName = argv[1];

	// Initialize host variables ----------------------------------------------
	printf("\nSetting up the problem..."); fflush(stdout);
	startTime(&timer);

	Matrix Rin_h, Gin_h, Bin_h, Rout_h, Gout_h, Bout_h, F_h, F_transposed_h, in_h, out_h; 
	// Matrix Rin_d, Gin_d, Bin_d, Rout_d, Gout_d, Bout_d, F_d; 
	Matrix in_d, out_d;
	unsigned imageHeight, imageWidth;
	cudaError_t cuda_ret;
	//dim3 dim_grid, dim_block;

        if (scanf("%d%d", &imageHeight, &imageWidth) == 0) {
            printf("invalid image dimensions");
            exit(0);
        }

	int BLOCK_SIZE_X = 32;
	int BLOCK_SIZE_Y = 32;
	int BLOCK_SIZE_Z = 1;
	int GRID_SIZE_X = (imageWidth/BLOCK_SIZE_X);
	int GRID_SIZE_Y = (imageHeight/BLOCK_SIZE_Y);
	int GRID_SIZE_Z = 3; // each z dimension for each of R, G, B
	
	
	/* Allocate host memory */
	F_h = allocateMatrix(FILTER_SIZE, FILTER_SIZE);
	Rin_h = allocateMatrix(imageHeight, imageWidth);
	Gin_h = allocateMatrix(imageHeight, imageWidth);
	Bin_h = allocateMatrix(imageHeight, imageWidth);
	Rout_h = allocateMatrix(imageHeight, imageWidth);
	Gout_h = allocateMatrix(imageHeight, imageWidth);
	Bout_h = allocateMatrix(imageHeight, imageWidth);
	in_h = allocateMatrix(1, (imageWidth+2)*(imageHeight+2)*3);
	out_h = allocateMatrix(1, imageWidth*imageHeight*3);
	F_transposed_h = allocateMatrix(FILTER_SIZE, FILTER_SIZE);

	/* Initialize filter and images */
	initImage(Rin_h, Gin_h, Bin_h);
	initFilter(F_h);
	initGPUInput(in_h, Rin_h, Gin_h, Bin_h);
/*	printf("\n");
	for(unsigned int i=0; i < imageWidth*imageHeight*3; ++i)
		printf("%f\t", in_h.elements[i]);
*/
	for(unsigned int i=0; i < FILTER_SIZE * FILTER_SIZE; ++i)
		F_transposed_h.elements[i] = F_h.elements[FILTER_SIZE * FILTER_SIZE - i - 1];
	
	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));
	printf("    Image: %u x %u\n", imageHeight, imageWidth);
	printf("    Mask: %u x %u\n", FILTER_SIZE, FILTER_SIZE);

	// Allocate device variables ----------------------------------------------

	printf("Allocating device variables..."); fflush(stdout);
	startTime(&timer);

	//INSERT DEVICE ALLOCATION CODE HERE
	cudaMalloc(&in_d.elements, sizeof(float) * (imageHeight+2)*(imageWidth+2)*3);
	cudaMalloc(&out_d.elements, sizeof(float) * imageHeight*imageWidth*3);

	cudaDeviceSynchronize();
	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));

	// Copy host variables to device ------------------------------------------

	printf("Copying data from host to device..."); fflush(stdout);
	startTime(&timer);

	//INSERT HOST TO DEVICE COPY CODE HERE
	cudaMemcpy(in_d.elements, in_h.elements, sizeof(float) * (imageHeight+2)*(imageWidth+2)*3, cudaMemcpyHostToDevice);
	cudaMemset(out_d.elements, 0, sizeof(float) * imageHeight*imageWidth*3);
	cudaMemcpyToSymbol(F_c, F_transposed_h.elements, sizeof(float) * FILTER_SIZE * FILTER_SIZE);

	cudaDeviceSynchronize();
	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));

	// Launch kernel ----------------------------------------------------------
	printf("Launching kernel..."); fflush(stdout);
	startTime(&timer);

	//INSERT KERNEL LAUNCH CODE HERE
	dim3 dim_grid(GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z);
	dim3 dim_block(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
	convolution<<<dim_grid, dim_block>>>(in_d.elements, out_d.elements, imageWidth+2, imageWidth);

	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

	cudaDeviceSynchronize();
	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));

	// Copy device variables from host ----------------------------------------

	printf("Copying data from device to host..."); fflush(stdout);
	startTime(&timer);

	//INSERT DEVICE TO HOST COPY CODE HERE
	cudaMemcpy(out_h.elements, out_d.elements, sizeof(float) * imageHeight*imageWidth*3, cudaMemcpyDeviceToHost); 

	cudaDeviceSynchronize();
	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));

	// CPUConvolution(F_h, Rin_h, Rout_h);
	// CPUConvolution(F_h, Gin_h, Gout_h);
	// CPUConvolution(F_h, Bin_h, Bout_h);

	initGPUOutput(out_h, Rout_h, Gout_h, Bout_h);
	output(outputFileName, Rout_h, Gout_h, Bout_h);

	// Verify correctness -----------------------------------------------------

	printf("Verifying results..."); fflush(stdout);
	verify(F_h, Rin_h, Rout_h);
	verify(F_h, Gin_h, Gout_h);
	verify(F_h, Bin_h, Bout_h);

	// Free memory ------------------------------------------------------------

	freeMatrix(Rin_h);
	freeMatrix(Gin_h);
	freeMatrix(Bin_h);
	freeMatrix(F_h);
	freeMatrix(Rout_h);
	freeMatrix(Gout_h);
	freeMatrix(Bout_h);
	freeMatrix(in_h);
	freeMatrix(out_h);
	freeMatrix(F_transposed_h);
	freeDeviceMatrix(in_d);
	freeDeviceMatrix(out_d);
	

	return 0;
}


