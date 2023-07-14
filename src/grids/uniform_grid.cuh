#pragma once

#include "../blood_cell_structures/particles.cuh"

class UniformGrid
{
	bool isCopy = false;
public:
	unsigned int* cellIds = 0;
	unsigned int* particleIds = 0;
	unsigned int* cellStarts = 0;
	unsigned int* cellEnds = 0;

	UniformGrid(const unsigned int particleCount);
	UniformGrid(const UniformGrid& other);
	~UniformGrid();

	void calculateGrid(const Particles& particles);
	void calculateGrid(const float* positionX, const float* positionY, const float* positionZ, unsigned int particleCount);
};
