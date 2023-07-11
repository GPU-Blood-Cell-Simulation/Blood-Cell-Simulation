#include "objects.cuh"
#include "defines.cuh"
#include "utilities.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>

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


__global__ void calculateCenters(cudaVec3 vertices, int* indices, cudaVec3 centers, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= size)
		return;
	float3 vv1 = vertices.get(indices[3 * id]);
	float3 vv2 = vertices.get(indices[3 * id + 1]);
	float3 vv3 = vertices.get(indices[3 * id + 2]);

	float x = (vv1.x + vv2.x + vv3.x) / 3;
	float y = (vv1.y + vv2.y + vv3.y) / 3;
	float z = (vv1.z + vv2.z + vv3.z) / 3;
	centers.set(id, make_float3(x, y, z));
}

DeviceTriangles::DeviceTriangles(Mesh m)
{
	// sizes
	trianglesCount = m.indices.size() / 3;
	verticesCount = m.vertices.size();

	// allocate
	centers.createVec(trianglesCount);
	vertices.createVec(verticesCount);
	HANDLE_ERROR(cudaMalloc((void**)&indices, 3 * trianglesCount * sizeof(int)));

	int* indiceMem = new int[3 * trianglesCount];
	std::copy(m.indices.begin(), m.indices.end(), indiceMem);

	float* vx = new float[verticesCount];
	float* vy = new float[verticesCount];
	float* vz = new float[verticesCount];

	int iter = 0;
	std::for_each(m.vertices.begin(), m.vertices.end(), [&](auto& v) {
		vx[iter] = v.Position.x;
		vy[iter] = v.Position.y;
		vz[iter++] = v.Position.z;
		});

	// copy
	HANDLE_ERROR(cudaMemcpy(indices, indiceMem, 3 * trianglesCount * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(vertices.x, vx, verticesCount * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(vertices.y, vy, verticesCount * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(vertices.z, vz, verticesCount * sizeof(float), cudaMemcpyHostToDevice));

	// centers
	int threadsPerBlock = trianglesCount > 1024 ? 1024 : trianglesCount;
	int blocks = (trianglesCount + threadsPerBlock - 1) / threadsPerBlock;
	calculateCenters << <blocks, threadsPerBlock >> > (vertices, indices, centers, trianglesCount);
	cudaDeviceSynchronize();
}
