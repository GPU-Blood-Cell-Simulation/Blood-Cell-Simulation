#pragma once

#include "../utilities/cuda_vec3.cuh"

/// <summary>
/// A structure containing the position, velocity and force vectors of all particles
/// </summary>
struct Particles
{
	cudaVec3 positions;
	cudaVec3 velocities;
	cudaVec3 forces;

	Particles(int n) : positions(n), velocities(n), forces(n) {}
};