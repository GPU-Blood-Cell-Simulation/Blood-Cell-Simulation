#ifndef BLOOD_CELLS_FACTORY_H
#define BLOOD_CELLS_FACTORY_H

#include "blood_cells.cuh"

#include <vector>

class BloodCellsFactory
{
public:
	BloodCellsFactory(int cellCount, int particlesInSingleCell);

	void AddSpring(int cellId1, int cellId2, float springLength);

	BloodCells CreateBloodCells();

private:
	std::vector<float> springGraph;
	int particlesInCell;
	int cellCount;
};

#endif