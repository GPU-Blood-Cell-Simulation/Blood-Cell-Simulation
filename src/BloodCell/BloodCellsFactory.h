#ifndef BLOOD_CELLS_FACTORY_H
#define BLOOD_CELLS_FACTORY_H

#include "BloodCells.cuh"

#include <vector>

class BloodCellsFactory
{
public:
	BloodCellsFactory(int cellsCnt, int particlesInSingleCell);

	void AddSpring(int cellID1, int cellID2, float springLen);

	BloodCells CreateBloodCells();

private:
	std::vector<float> springGraph;
	int particlesInCell;
	int cellsCnt;
};

#endif