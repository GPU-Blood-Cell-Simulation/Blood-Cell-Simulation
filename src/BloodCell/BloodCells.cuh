#ifndef BLOOD_CELLS_H
#define BLOOD_CELLS_H

#include "../objects.cuh"

constexpr auto NO_SPRING = 0;


class BloodCells
{
public:
	Particles particles;
	int particlesInCell;
	int particlesCnt;
	float* springsGraph;

	BloodCells(int cellsCnt, int particlsInCells, float* graphDesc);

	void Deallocate();

	void PropagateForces();
};

#endif