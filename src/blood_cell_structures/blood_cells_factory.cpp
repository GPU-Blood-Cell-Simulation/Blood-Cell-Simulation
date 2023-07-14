#include "blood_cells_factory.hpp"

#include <stdexcept>


BloodCellsFactory::BloodCellsFactory(int cellCount, int particlesInSingleCell) :
	springGraph(particlesInSingleCell * particlesInSingleCell),  cellCount(cellCount), particlesInCell(particlesInSingleCell)
{
	for (int i = 0; i < particlesInCell * particlesInCell; i++) {
		springGraph[i] = NO_SPRING;
	}
}


void BloodCellsFactory::AddSpring(int cellId1, int cellId2, float springLength)
{
	if (cellId1 == cellId2 || springLength <= 0)
		throw std::runtime_error("Invalid spring addition.");

	springGraph[cellId1 * particlesInCell + cellId2] = springLength;
	springGraph[cellId2 * particlesInCell + cellId1] = springLength;
}


BloodCells BloodCellsFactory::CreateBloodCells() const
{
	return BloodCells(cellCount, particlesInCell, springGraph.data());
}