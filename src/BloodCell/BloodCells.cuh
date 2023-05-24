#ifndef BLOOD_CELLS_H
#define BLOOD_CELLS_H

#include "../objects.cuh"

#define NO_SPRING 0


class BloodCells
{
public:
	Particles particles;
	int particlesInCell;
	int particlesCnt;
	float* springsGrah;

	BloodCells(int cellsCnt, int particlsInCells, float* graphDesc);
	void Deallocate();

	void PropagateForces();
};

#endif