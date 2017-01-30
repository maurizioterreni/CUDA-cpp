
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <fstream>
#include <string>
#include <iostream>

#include <cuda_runtime_api.h>


#include "getColor.h"
#include "getName140.h"
#include "getArrayColor.h"
#include "getArrayColor140.h"

static void CheckCudaErrorAux(const char *, unsigned,const char*, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)



static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err){
	if(err == cudaSuccess) return;
	std::cerr<< statement << " returned " << cudaGetErrorString(err) << "("
			<< err << ") at " << file << ":" << line << std::endl;
	exit(1);
}

unsigned long *generate_data(int size){
	unsigned long *data = (unsigned long *) malloc(sizeof(unsigned long) * size);
	std::string line;
	std::ifstream file ("colors.txt");
	int i = 0;
	if(file.is_open()){
		while(std::getline(file,line)  && i < size){
			data[i] = std::strtoul(line.c_str(),NULL, 16);
			i++;
		}
	}
	return data;
}
void writeOut(unsigned long *data , unsigned int dim){
	for(int i = 0; i < dim; i++){

		printf("Color %d: %s=> %lu\n",i, getName140(data[i*2]).c_str() , data[i*2+1]);
	}

}
unsigned long *prepareOutput(int size){
	unsigned long *data = (unsigned long *) malloc(size * size * sizeof(unsigned long));
	for(int i = 0; i<size * size; i++){
		data[i] = 0;
	}

	return data;
}

void sortOut(unsigned long *data , unsigned int dim){
	for (unsigned int i = 0; i < dim; i++) {
		unsigned int posMax = i;
		for (unsigned int k = i; k < dim; k++) {
			if(data[k*2+1] > data[posMax*2+1]){
				posMax = k;
			}
		}

		if (posMax != i) {
			unsigned long tempHex = data[i*2];
			unsigned long tempCount = data[i*2+1];

			data[i*2] = data[posMax*2];
			data[i*2+1] = data[posMax*2+1];

			data[posMax*2] = tempHex;
			data[posMax*2+1] = tempCount;
		}
	}
}





/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 *
 * See cuda.h for error code descriptions.
 */

__global__ void colorCount(unsigned long* vectArrayColor, unsigned long* vectColor,unsigned long* vectRisu, unsigned int size , unsigned int sizeColor){

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < sizeColor) {
		long sum = -1;
		for (int ii = 0; ii < size; ii++) {
			if(vectArrayColor[index] == vectArrayColor[ii]){
				sum = sum + 1;
			}
		}
		vectRisu[index * 2] = vectArrayColor[index];
		vectRisu[index * 2 + 1] = sum;
	}

}

int main(int argc, char **argv)
{
	unsigned int size = 276;
	unsigned int sizeColor = 140;
	unsigned long *hostArrayColor = getArrayColor140();
	unsigned long *hostColor;
	unsigned long *hostRisu;
	unsigned long *deviceArrayColor;
	unsigned long *deviceColor;
	unsigned long *deviceRisu;

	hostColor = generate_data(size);
	hostRisu = prepareOutput(sizeColor);

	CUDA_CHECK_RETURN(
			cudaMalloc((void ** )&deviceArrayColor,
					sizeof(unsigned long) * sizeColor));
	CUDA_CHECK_RETURN(
			cudaMalloc((void ** )&deviceColor,
					sizeof(unsigned long) * size));
	CUDA_CHECK_RETURN(
			cudaMalloc((void ** )&deviceRisu,
					sizeof(unsigned long) * sizeColor * sizeColor));

	//copy dataHost to datatDevice
	CUDA_CHECK_RETURN(
			cudaMemcpy(deviceArrayColor,hostArrayColor, sizeColor * sizeof(unsigned long),
					cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(
			cudaMemcpy(deviceColor,hostColor, size * sizeof(unsigned long),
					cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(
			cudaMemcpy(deviceRisu,hostRisu, sizeColor * sizeColor * sizeof(unsigned long),
					cudaMemcpyHostToDevice));


	colorCount<<<1,sizeColor>>>(deviceArrayColor,deviceColor , deviceRisu , size, sizeColor);

	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(
			cudaMemcpy(hostRisu,deviceRisu, sizeColor * sizeColor * sizeof(unsigned long),
					cudaMemcpyDeviceToHost));


	//Free GPU
	cudaFree(deviceArrayColor);
	cudaFree(deviceColor);
	cudaFree(deviceRisu);
	sortOut(hostRisu,sizeColor);
	//writeOut(hostRisu,*hostRisuDim);

	writeOut(hostRisu,sizeColor);
	//Free host memory
	free(hostArrayColor);
	free(hostColor);
	free(hostRisu);

	return 0;
}
