#include "blood_cells_factory.hpp"

#include <stdexcept>


BloodCellsFactory::BloodCellsFactory(int cellCount, int particlesInSingleCell) :
	springGraph(particlesInSingleCell * particlesInSingleCell),  cellCount(cellCount), particlesInCell(particlesInSingleCell)
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
}


BloodCells BloodCellsFactory::createBloodCells() const
{
	return BloodCells(cellCount, particlesInCell, springGraph.data());
}


BloodCells BloodCellsFactory::createMonopols(int cellCount)
{
	BloodCellsFactory factory(cellCount, 1);

	return factory.createBloodCells();
}


BloodCells BloodCellsFactory::createDipols(int cellCount, float springLength)
{
	BloodCellsFactory factory(cellCount, 2);

	factory.addSpring(0, 1, springLength);

	return factory.createBloodCells();
}


BloodCells BloodCellsFactory::createQuadrupole(int cellCount, float springLength)
{
	BloodCellsFactory factory(cellCount, 4);

	factory.addSpring(0, 1, springLength);
	factory.addSpring(1, 2, springLength);
	factory.addSpring(2, 3, springLength);
	factory.addSpring(3, 0, springLength);

	return factory.createBloodCells();
}


BloodCells BloodCellsFactory::createOctuple(int cellCount, float springLength)
{
	BloodCellsFactory factory(cellCount, 8);

	factory.addSpring(0, 1, springLength);
	factory.addSpring(1, 2, springLength);
	factory.addSpring(2, 3, springLength);
	factory.addSpring(3, 0, springLength);

	factory.addSpring(4, 5, springLength);
	factory.addSpring(5, 6, springLength);
	factory.addSpring(6, 7, springLength);
	factory.addSpring(7, 4, springLength);

	factory.addSpring(0, 4, springLength);
	factory.addSpring(1, 5, springLength);
	factory.addSpring(2, 6, springLength);
	factory.addSpring(3, 7, springLength);

	return factory.createBloodCells();
}
