#ifndef BLOOD_CELLS_H
#define BLOOD_CELLS_H

#include "particles.cuh"

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

	void propagateForces();
};

#endif