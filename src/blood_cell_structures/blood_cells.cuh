#ifndef BLOOD_CELLS_H
#define BLOOD_CELLS_H

#include "particles.cuh"

constexpr auto NO_SPRING = 0;


class BloodCells
{
	bool isCopy = false;

public:
	Particles particles;
	int particlesInCell;
	int particleCount;
	float* springsGraph;

	BloodCells(int cellsCnt, int particlesInCell, float* graphDesc);
	BloodCells(const BloodCells& other);

	~BloodCells();

	void PropagateForces();
};

#endif