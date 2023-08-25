#pragma once
#include "../utilities/cuda_vec3.cuh"

/// <summary>
/// A structure containing the position, velocity and force vectors of all particles
/// </summary>
struct Particles
{
	cudaVec3 position;
	cudaVec3 velocity;
	cudaVec3 force;

	Particles(int n) : position(n), velocity(n), force(n) {}
};