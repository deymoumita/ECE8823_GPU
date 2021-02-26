/*
	Name: Moumita Dey
	Class: ECE 8823 GPU Architectures, Spring 2018
	Assignment 1: 2D Matrix Addition
	GTID: 903099216
	Email: mdey6@gatech.edu

	Date: 28 Jan, 2018

*/

#include <iostream>
#include <stdio.h>
#include <math.h>

#include "common.cuh"

// Kernel for matrix addition
__global__ void MatAdd(int * a, int * b, int * c, int size, int width) 
{
	// computes x and y indices of a and b 
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx < size && idy < size)		// takes care of edge case -> gridsize*blocksize > matrixdim 
	{
		c[idy * width + idx] = a[idy * width + idx] + b[idy * width + idx];
	}
}

// main
int main() {
	
	// do device enquiry
	int device;

	cudaFree(0);
	checkCudaError(cudaGetDevice(&device));
	cudaDeviceProp prop;
	checkCudaError(cudaGetDeviceProperties(&prop, device));
/*	std::cout << "Device " << device << ": " << prop.name << "\n";
	std::cout << "GPU Cores: " << prop.multiProcessorCount << "\n";
	std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
	std::cout << "Maximum Grid Size: " << prop.maxGridSize[0] << "x" << prop.maxGridSize[1] << "x" <<prop.maxGridSize[2] << "\n";	
	std::cout << "Maximum Block Size: " << prop.maxThreadsDim[0] << "x" << prop.maxThreadsDim[1] << "x" <<prop.maxThreadsDim[2] << "\n";
*/	
	int GRID_SIZE_X = 0;
	int GRID_SIZE_Y = 0;
	int GRID_SIZE_Z = 0;
	int BLOCK_SIZE_X = 0;
	int BLOCK_SIZE_Y = 0;
	int BLOCK_SIZE_Z = 0;
	int MATRIX_DIM_X = 0;
	int MATRIX_DIM_Y = 0;
	
	int * a, * b, * c;
	int * dev_a, * dev_b, * dev_c;
	
	// parse dimensions of kernel from input file
	std::cin >> GRID_SIZE_X >> GRID_SIZE_Y >> GRID_SIZE_Z;
	std::cin >> BLOCK_SIZE_X >> BLOCK_SIZE_Y >> BLOCK_SIZE_Z;
	std::cin >> MATRIX_DIM_X >> MATRIX_DIM_Y;

	// Checking if dimensions are equal and square, and if hardware is supported for the matrix provided
	if (GRID_SIZE_X != GRID_SIZE_Y)
	{
		std::cout << "\nERROR!! Grid Dimensions are not equal, ABORT...\n";
		exit(1);
	}
	if (BLOCK_SIZE_X != BLOCK_SIZE_Y)
	{
		std::cout << "\nERROR!! Block Dimensions are not equal, ABORT...\n";
		exit(1);
	}
	if (MATRIX_DIM_X != MATRIX_DIM_Y)
	{
		std::cout << "\nERROR!! Matrix Dimensions are not equal, ABORT...\n";
		exit(1);
	}
	if (BLOCK_SIZE_X*BLOCK_SIZE_Y > prop.maxThreadsPerBlock)
	{
		std::cout << "\nERROR! Matrix Dimensions exceed maximum threads supported, ABORT...\n";
		exit(1);
	}

/*	// Debugging: Parsed dimensions
	std::cout << "Grid Dimensions: " << GRID_SIZE_X << "x" << GRID_SIZE_Y << "x" << GRID_SIZE_Z << "\n";
	std::cout << "Block Dimensions: " << BLOCK_SIZE_X << "x" << BLOCK_SIZE_Y << "x" << BLOCK_SIZE_Z << "\n";
	std::cout << "Matrix Dimensions: " << MATRIX_DIM_X << "x" << MATRIX_DIM_Y << "\n";
*/	
	// allocate input and output matrices
	const int size = MATRIX_DIM_X * MATRIX_DIM_Y;
	const int width = MATRIX_DIM_X;
	a = (int *) malloc (sizeof(int) * size);
	b = (int *) malloc (sizeof(int) * size);
	c = (int *) malloc (sizeof(int) * size);

	if(!a || !b || !c) {
		std::cout << "Error: out of memory\n";
		exit(-1);
	}
	
	// parse input matrices from input file
	int i = 0;
	for(i = 0; i < size; i++)
		std::cin >> a[i];
	for(i = 0; i < size; i++)
		std::cin >> b[i];
	
/*	// Debugging: Parsed input matrices
	for(i = 0; i < size; i++)
	{
		if (i % MATRIX_DIM_X == 0)
			std::cout << "\n";
		std::cout << a[i] <<" ";
	}
	for(i = 0; i < size; i++)
	{
		if (i % MATRIX_DIM_X == 0)
			std::cout << "\n";
		std::cout << b[i] <<" ";
	}
*/	

	// invoke kernel
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	memset(c, 0, sizeof(int) * size);

	checkCudaError(cudaMalloc(&dev_a, sizeof(int) * size));
	checkCudaError(cudaMalloc(&dev_b, sizeof(int) * size));	
	checkCudaError(cudaMalloc(&dev_c, sizeof(int) * size));	
	
	checkCudaError(cudaMemcpy(dev_a, a, sizeof(int) * size, cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(dev_b, b, sizeof(int) * size, cudaMemcpyHostToDevice));
	checkCudaError(cudaMemset(dev_c, 0, sizeof(int) * size));

	dim3 DimGrid (GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z);
	dim3 DimBlock (BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
	MatAdd<<<DimGrid, DimBlock>>>(dev_a, dev_b, dev_c, size, width);

	checkCudaError(cudaDeviceSynchronize());
	checkCudaError(cudaMemcpy(c, dev_c, sizeof(int) * size, cudaMemcpyDeviceToHost));

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	// measure elapsed time
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Elapsed time = "<< milliseconds << " ms\n";
	
	// output computed matrix
	for(i = 0; i < size; i++) 
	{
		if (i % width == 0)
			std::cout << "\n";
		std::cout << c[i] << " ";
	}
	std::cout << "\n\n";

	checkCudaError(cudaFree(dev_a));
	checkCudaError(cudaFree(dev_b));
	checkCudaError(cudaFree(dev_c));

	free(a);
	free(b);
	free(c);

}

