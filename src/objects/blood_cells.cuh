#ifndef BLOOD_CELLS_H
#define BLOOD_CELLS_H

#include "particles.cuh"
#include "../utilities/math.cuh"
#include "../defines.hpp"

constexpr auto NO_SPRING = 0;

/// <summary>
/// Contains all the particles and the springs graph
/// </summary>
class BloodCells
{
	bool isCopy = false;

public:
	Particles particles;
	int particlesInCell;
	int particleCount;
	float* springsGraph;

	BloodCells(int cellsCount, int particlesInCell, const float* graphDesc);
	BloodCells(const BloodCells& other);

	~BloodCells();

	void gatherForcesFromNeighbors(unsigned int blocks, unsigned int threadsPerBlock);

	__device__ inline float calculateParticleSpringForce(float3 p1, float3 p2, float3 v1, float3 v2, float springLength)
	{
		return (length(p1 - p2) - springLength)* particle_k_sniff + dot(normalize(p1 - p2), (v1 - v2)) * particle_d_fact;
	}
};

#endif