#pragma once
#include "objects.cuh"

class UniformGrid
{
public:
	unsigned int* cellIds = 0;
	unsigned int* particleIds = 0;
	unsigned int* cellStarts = 0;
	unsigned int* cellEnds = 0;

	UniformGrid();
	~UniformGrid();

	void calculateGrid(const Particles& particles);
	void calculateGrid(const float* positionX, const float* positionY, const float* positionZ, unsigned int particleCount);
};
