#include "objects.cuh"
#include "defines.cuh"
#include "utilities.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// cudaVec3
void cudaVec3::createVec(int n)
{
    size = n;
	// allocate 
	HANDLE_ERROR(cudaMalloc((void**)&x, n * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&y, n * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&z, n * sizeof(float)));
}

void cudaVec3::freeVec()
{
	cudaFree(x);
	cudaFree(y);
	cudaFree(z);
}


__global__ void calculateCenters(cudaVec3 v1, cudaVec3 v2, cudaVec3 v3, cudaVec3 centers, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= size)
		return;
	float3 vv1 = v1.get(id);
	float3 vv2 = v2.get(id);
	float3 vv3 = v3.get(id);

	float x = (vv1.x + vv2.x + vv3.x) / 3;
	float y = (vv1.y + vv2.y + vv3.y) / 3;
	float z = (vv1.z + vv2.z + vv3.z) / 3;
	centers.set(id, make_float3(x, y, z));
}

Triangles::Triangles(Mesh m)
{
	size = m.indices.size() / 3;
	v1.createVec(size);
	v2.createVec(size);
	v3.createVec(size);
	centers.createVec(size);
	float* cpuBuffer = new float[9 * size];
	for (int i = 0; i < m.indices.size(); ++i)
	{
		int ind = clamp((int)m.indices[i], 0, (int)m.vertices.size() - 1);
		cpuBuffer[((3 * i) % 9) * size + i / 3] = m.vertices[ind].Position.x;
		cpuBuffer[((3 * i + 1) % 9) * size + i / 3] = m.vertices[ind].Position.y;
		cpuBuffer[((3 * i + 2) % 9) * size + i / 3] = m.vertices[ind].Position.z;
	}

	cudaMemcpy(v1.x, cpuBuffer, size, cudaMemcpyHostToDevice);
	cudaMemcpy(v1.y, cpuBuffer + size, size, cudaMemcpyHostToDevice);
	cudaMemcpy(v1.z, cpuBuffer + 2 * size, size, cudaMemcpyHostToDevice);
	cudaMemcpy(v2.x, cpuBuffer + 3 * size, size, cudaMemcpyHostToDevice);
	cudaMemcpy(v2.y, cpuBuffer + 4 * size, size, cudaMemcpyHostToDevice);
	cudaMemcpy(v2.z, cpuBuffer + 5 * size, size, cudaMemcpyHostToDevice);
	cudaMemcpy(v3.x, cpuBuffer + 6 * size, size, cudaMemcpyHostToDevice);
	cudaMemcpy(v3.y, cpuBuffer + 7 * size, size, cudaMemcpyHostToDevice);
	cudaMemcpy(v3.z, cpuBuffer + 8 * size, size, cudaMemcpyHostToDevice);


	// translate our CUDA positions into Vertex offsets
	int threadsPerBlock = size > 1024 ? 1024 : size;
	int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
	calculateCenters << <blocks, threadsPerBlock >> > (v1, v2, v3, centers, size);
	cudaDeviceSynchronize();

	delete[] cpuBuffer;
}
