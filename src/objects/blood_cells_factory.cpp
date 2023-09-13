#include "blood_cells_factory.hpp"

#include <stdexcept>


BloodCellsFactory::BloodCellsFactory() : springGraph(particlesInBloodCell * particlesInBloodCell)
{
	for (int i = 0; i < particlesInBloodCell * particlesInBloodCell; i++) {
		springGraph[i] = NO_SPRING;
	}
}


void BloodCellsFactory::addSpring(int cellId1, int cellId2, float springLength)
{
	if (cellId1 == cellId2 || springLength <= 0)
		throw std::runtime_error("Invalid spring addition.");

	springGraph[cellId1 * particlesInBloodCell + cellId2] = springLength;
	springGraph[cellId2 * particlesInBloodCell + cellId1] = springLength;
}


BloodCells BloodCellsFactory::getBloodCells() const
{
	return BloodCells(particleCount / particlesInBloodCell, particlesInBloodCell, springGraph.data());
}