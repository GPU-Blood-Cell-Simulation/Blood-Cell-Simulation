#pragma once

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include "../utilities/cuda_vec3.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


// A handy function for easy error checking
static void HandleError(cudaError_t err, const char* file, int line);

static void HandleError(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
            file, line);
        cudaDeviceSynchronize();
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ));


static void char_bin(unsigned char v, bool newLine = true, bool printNormalNumber = false)
{
	if (printNormalNumber)
		printf("(%d) ", v);
	for (int i = 7; i >= 0; --i) {
		printf("%d", (v >> i) & 1);
	}
	if (newLine)
		printf("\n");
}
static void int_bin(unsigned int v, bool newLine = true, bool printNormalNumber = false)
{
	if (printNormalNumber)
		printf("(%d) ", v);
	for (int i = 31; i >= 0; --i) {
		printf("%d", (v >> i) & 1);
		if (!(i % 8))
			printf(" ");
	}
	if (newLine)
		printf("\n");
}

static void printFirstBytesFromGpu(unsigned char* gpuMem, unsigned int count)
{
	unsigned char* cpuMem = new unsigned char[count];
	HANDLE_ERROR(cudaMemcpy(cpuMem, gpuMem, count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaPeekAtLastError());
	for (int i = 0; i < count; ++i)
		char_bin(cpuMem[i]);
	delete[] cpuMem;
}


static void printGpuArray(unsigned int* gpuMem, unsigned int count)
{
	unsigned int* cpuMem = new unsigned int[count];
	HANDLE_ERROR(cudaMemcpy(cpuMem, gpuMem, count * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaPeekAtLastError());
	for (int i = 0; i < count; ++i) {
		printf("[%d] ", i); int_bin(cpuMem[i], true, true);
	}
	delete[] cpuMem;
}

static void printGpuArray(unsigned char* gpuMem, unsigned int count)
{
	unsigned char* cpuMem = new unsigned char[count];
	HANDLE_ERROR(cudaMemcpy(cpuMem, gpuMem, count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaPeekAtLastError());
	for (int i = 0; i < count; ++i) {
		printf("[%d] ", i); char_bin(cpuMem[i], true, true);
	}
	delete[] cpuMem;
}

static void printGpuVec3(cudaVec3& vec, unsigned int count, const char* message = "") {
	float* cpuMemX = new float[count];
	float* cpuMemY = new float[count];
	float* cpuMemZ = new float[count];
	HANDLE_ERROR(cudaMemcpy(cpuMemX, vec.x, count * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(cpuMemY, vec.y, count * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(cpuMemZ, vec.z, count * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaPeekAtLastError());

	printf("%s\n", message);
	for (int i = 0; i < count; ++i) {
		printf("[%d] x=%.5f ; y=%.5f ; z=%.5f \n", i, cpuMemX[i], cpuMemY[i], cpuMemZ[i]);
	}
	delete[] cpuMemX, cpuMemY, cpuMemZ;
}

static void printGpuKeyValueArray(unsigned int* gpuMemKey, unsigned int* gpuMemValue, unsigned int count)
{
	unsigned int* cpuMemKey = new unsigned int[count];
	unsigned int* cpuMemValue = new unsigned int[count];
	HANDLE_ERROR(cudaMemcpy(cpuMemKey, gpuMemKey, count * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(cpuMemValue, gpuMemValue, count * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaPeekAtLastError());
	for (int i = 0; i < count; ++i) {
		printf("[%d] key = ", i); int_bin(cpuMemKey[i], false, true); printf(", value = "); int_bin(cpuMemValue[i], true, true);
	}
	delete[] cpuMemKey;
	delete[] cpuMemValue;
}

static void SaveFirstBytesFromGpuToFile(unsigned char* gpuMem, unsigned int count)
{
	unsigned char* cpuMem = new unsigned char[count];
	HANDLE_ERROR(cudaMemcpy(cpuMem, gpuMem, count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaPeekAtLastError());
	std::ofstream fout;
	fout.open("C:\\Users\\hlaw\\Desktop\\file_bcs_debug.bin", std::ios_base::binary | std::ios_base::out);
	for (int i = 0; i < count; ++i)
		fout.write((const char*)(gpuMem + i * sizeof(unsigned char)), sizeof(unsigned char));
	fout.close();

}