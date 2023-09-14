#ifndef BLOOD_CELLS_FACTORY_H
#define BLOOD_CELLS_FACTORY_H

#include "blood_cells.cuh"

#include <vector>

class BloodCellsFactory
{
public:
	BloodCellsFactory();

	void addSpring(int cellId1, int cellId2, float springsInCellLength);

	BloodCells getBloodCells() const;

	template<int particlesInCell>
	BloodCells createBloodCells() = delete;

	template<>
	BloodCells createBloodCells<1>()
	{
		return getBloodCells();
	}

	template<>
	BloodCells createBloodCells<2>()
	{
		addSpring(0, 1, springsInCellsLength);

		return getBloodCells();
	}

	template<>
	BloodCells createBloodCells<4>()
	{
		addSpring(0, 1, springsInCellsLength);
		addSpring(1, 2, springsInCellsLength);
		addSpring(2, 3, springsInCellsLength);
		addSpring(3, 0, springsInCellsLength);

		return getBloodCells();
	}

	template<>
	BloodCells createBloodCells<8>()
	{
		addSpring(0, 1, springsInCellsLength);
		addSpring(1, 2, springsInCellsLength);
		addSpring(2, 3, springsInCellsLength);
		addSpring(3, 0, springsInCellsLength);

		addSpring(4, 5, springsInCellsLength);
		addSpring(5, 6, springsInCellsLength);
		addSpring(6, 7, springsInCellsLength);
		addSpring(7, 4, springsInCellsLength);

		addSpring(0, 4, springsInCellsLength);
		addSpring(1, 5, springsInCellsLength);
		addSpring(2, 6, springsInCellsLength);
		addSpring(3, 7, springsInCellsLength);

		return getBloodCells();
	}

	inline std::vector<unsigned int> getSpringIndices() const
	{
		return springIndices;
	}

private:
	std::vector<float> springGraph;
	std::vector<unsigned int> springIndices;
};

#endif