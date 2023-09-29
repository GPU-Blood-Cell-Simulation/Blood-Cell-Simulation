#pragma once

#include "blood_cell_presets.hpp"
#include "../objects/blood_cells_def_type.hpp"

#include <boost/mp11/list.hpp>


using namespace boost::mp11;

// BLOOD CELL PARAMETERS

inline constexpr float particleRadius = 5;
inline constexpr float springLengthCoefficient = 1.0f;

// Please always double check your commas!
using BloodCellList = mp_list<

	BloodCellDef<10, 2,
	preset::Dipole
	>,

	BloodCellDef<2, 4,
	preset::Quadrupole
	>,

	BloodCellDef<10, 3,
	mp_list<
	Spring<0, 1, 10>,
	Spring<1, 2, 10>,
	Spring<2, 0, 10>
	>
	>

>;