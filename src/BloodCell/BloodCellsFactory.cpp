#include "BloodCellsFactory.h"

#include <stdexcept>


BloodCellsFactory::BloodCellsFactory(int cellsCnt, int particlesInSingleCell):
	springGraph(particlesInSingleCell* particlesInSingleCell)
{
	particlesInCell = particlesInSingleCell;
	this->cellsCnt = cellsCnt;

	for (int i = 0; i < particlesInSingleCell * particlesInSingleCell; i++) {
		springGraph[i] = NO_SPRING;
	}
}


void BloodCellsFactory::AddSpring(int cellID1, int cellID2, float springLen)
{
	if (cellID1 == cellID2 || springLen <= 0)
		throw std::runtime_error("Invalid spring addition.");

	springGraph[cellID1 * particlesInCell + cellID2] = springLen;
	springGraph[cellID2 * particlesInCell + cellID1] = springLen;
}


BloodCells BloodCellsFactory::CreateBloodCells()
{
	return BloodCells(cellsCnt, particlesInCell, springGraph.data());
}
