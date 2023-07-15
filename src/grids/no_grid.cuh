#pragma once

#include "base_grid.cuh"
#include "../blood_cell_structures/particles.cuh"

class NoGrid : public BaseGrid<NoGrid>
{
public:
	inline void calculateGrid(const Particles& particles, unsigned int particleCount) {}

	inline void calculateGrid(const float* positionX, const float* positionY, const float* positionZ, unsigned int particleCount) {}
};