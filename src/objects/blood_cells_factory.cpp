#include "blood_cells_factory.hpp"

#include <stdexcept>


BloodCellsFactory::BloodCellsFactory() : springGraph(particlesInCell * particlesInCell)
{
	for (int i = 0; i < particlesInCell * particlesInCell; i++) {
		springGraph[i] = NO_SPRING;
	}
}


void BloodCellsFactory::addSpring(int cellId1, int cellId2, float springLength)
{
	if (cellId1 == cellId2 || springLength <= 0)
		throw std::runtime_error("Invalid spring addition.");

	springGraph[cellId1 * particlesInCell + cellId2] = springLength;
	springGraph[cellId2 * particlesInCell + cellId1] = springLength;

	springIndices.push_back(static_cast<unsigned int>(cellId1));
	springIndices.push_back(static_cast<unsigned int>(cellId2));
}


BloodCells BloodCellsFactory::getBloodCells() const
{
	return BloodCells(particleCount / particlesInCell, particlesInCell, springGraph.data());
}
