#pragma once
#include "../utilities/cuda_vec3.cuh"

// Global structure of particles
struct Particles
{
	cudaVec3 position;
	cudaVec3 velocity;
	cudaVec3 force;

	Particles(int n) : position(n), velocity(n), force(n) {}
};