#pragma once
#include <vector>
#include "defines.cuh"
#include "utilities.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


struct cudaVec3
{
	float* x;
	float* y;
	float* z;
	int size;

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


class Corpuscles
{
	float L0;

public:
	Corpuscles(int initialLength = 0.5f){
		L0 = initialLength;
	}

	__device__ inline void propagateForces(Particles particles, int particleIndex)
	{
		int secondParticle = particleIndex % 2 == 0 ? particleIndex + 1 : particleIndex - 1;
		float3 p1 = particles.position.get(particleIndex);
		float3 p2 = particles.position.get(secondParticle);
		float3 v1 = particles.velocity.get(particleIndex);
		float3 v2 = particles.velocity.get(secondParticle);

		float Fr = (length(p1 - p2) - L0) * k_sniff + dot(normalize(p1 - p2), (v1 - v2)) * d_fact;

		particles.force.add(particleIndex, Fr * normalize(p2 - p1));
		particles.force.add(secondParticle, Fr * normalize(p1 - p2));
	}

	__device__ inline void setCorpuscle(int index, float3 center, Particles particles, int particleCount)
	{
		//printf("index: %d\n", index);
		if (2 * index < particleCount)
		{
			particles.position.set(2 * index, make_float3(0, -L0 / 2, 0) + center);
			particles.position.set(2 * index + 1, make_float3(0, L0 / 2, 0) + center);

			particles.velocity.set(2 * index, make_float3(v0, 0, 0));
			particles.velocity.set(2 * index + 1, make_float3(v0, 0, 0));

			particles.force.set(2 * index, make_float3(0, 0, 0));
			particles.force.set(2 * index + 1, make_float3(0, 0, 0));
		}
	}
};