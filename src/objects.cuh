#pragma once
#include <vector>
#include "defines.cuh"
#include "utilities.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "graphics/mesh.hpp"


struct cudaVec3
{
	float* x = 0;
	float* y = 0;
	float* z = 0;
	int size = 0;

	// allocated on host
	cudaVec3() {}
	void createVec(int n);
	void freeVec();

	// to use only on device
	__device__ inline float3 get(int index)
	{
		return make_float3(x[index], y[index], z[index]);
	}
	__device__ inline void set(int index, float3 v)
	{
		x[index] = v.x;
		y[index] = v.y;
		z[index] = v.z;
	}
	__device__ inline void add(int index, float3 v)
	{
		x[index] += v.x;
		y[index] += v.y;
		z[index] += v.z;
	}
};


// Global structure of particles
struct Particles
{
	cudaVec3 position;
	cudaVec3 velocity;
	cudaVec3 force;

	Particles(int n) {
		position.createVec(n);
		velocity.createVec(n);
		force.createVec(n);
	}

	void freeParticles() {
		position.freeVec();
		velocity.freeVec();
		force.freeVec();
	}

};


struct DeviceTriangles
{
	int trianglesCount;
	int verticesCount;

	cudaVec3 vertices;
	int* indices;
	cudaVec3 centers;


	DeviceTriangles(Mesh m);
	
	/// <param name="vertexIndex">0, 1 or 2 as triangle vertices</param>
	/// <returns></returns>
	__device__ float3 get(int triangleIndex, int vertexIndex)
	{
		int index = indices[3 * triangleIndex + vertexIndex];
		return vertices.get(index);
	}

	/// <param name="vertexIndex">0, 1 or 2 as triangle vertices</param>
	/// <returns></returns>
	__device__ void set(int triangleIndex, int vertexIndex, float3 value)
	{
		int index = indices[3 * triangleIndex + vertexIndex];
		vertices.set(index, value);
	}

	/// <param name="vertexIndex">0, 1 or 2 as triangle vertices</param>
	/// <returns></returns>
	__device__ void add(int triangleIndex, int vertexIndex, float3 value)
	{
		int index = indices[3 * triangleIndex + vertexIndex];
		vertices.add(index, value);
	}
};

struct ray
{
	float3 origin;
	float3 direction;
	float t = 1e10f;

	// rays may be used to determine intersection with objects
	// so its easy to store object index inside ray
	unsigned int objectIndex = -1;

	__device__ ray(float3 origin, float3 direction) : origin(origin), direction(direction) {}
};