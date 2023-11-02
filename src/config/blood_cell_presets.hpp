#pragma once

#include "../objects/blood_cells_def_type.hpp"

#include <boost/mp11/list.hpp>
#include <boost/mp11/mpl_tuple.hpp>
#include <glm/vec3.hpp>


namespace preset
{
	using namespace boost::mp11;

	inline constexpr int springLength = 30;
	inline constexpr int springLengthDiagonal = 1.7 * 30;
	inline constexpr int springLengthDiagonalSmall = 1.41 * 30;

	template<int x, int y, int z>
	struct mpInt3
	{
		inline static constexpr int x;
		inline static constexpr int y;
		inline static constexpr int z;
	};

	template <int x, int y, int z, int decimalPrecision = 1>
	struct mpFloat3
	{
		inline static constexpr float x = float(x)/decimalPrecision;
		inline static constexpr float y = float(y)/decimalPrecision;
		inline static constexpr float z = float(z)/decimalPrecision;
	};

	// TODO: add a full-graph preset

	using NoSprings =
		mp_list<>;

	using Dipole =
		mp_list<
		Spring<0, 1, springLength>
		>;

	using Quadrupole =
		mp_list<
		Spring<0, 1, springLength>,
		Spring<1, 2, springLength>,
		Spring<2, 3, springLength>,
		Spring<3, 0, springLength>
		>;

	using Octupole =
		mp_list<
		Spring<0, 1, springLength>,
		Spring<1, 2, springLength>,
		Spring<2, 3, springLength>,
		Spring<3, 0, springLength>,

		Spring<4, 5, springLength>,
		Spring<5, 6, springLength>,
		Spring<6, 7, springLength>,
		Spring<7, 4, springLength>,

		Spring<0, 4, springLength>,
		Spring<1, 5, springLength>,
		Spring<2, 6, springLength>,
		Spring<3, 7, springLength>
		>;

	using Cube =
		mp_list<
		Spring<0,1,springLength>,
		Spring<1,2,springLength>,
		Spring<2,3,springLength>,
		Spring<3,0,springLength>,
		Spring<0,2,springLengthDiagonalSmall>,
		Spring<3,1,springLengthDiagonalSmall>,
		Spring<0,4,springLength>,
		Spring<1,5,springLength>,
		Spring<0,5,springLengthDiagonalSmall>,
		Spring<1,4,springLengthDiagonalSmall>,
		Spring<1,6,springLengthDiagonalSmall>,
		Spring<2,5,springLengthDiagonalSmall>,
		Spring<2,6,springLength>,
		Spring<3,7,springLength>,
		Spring<2,7,springLengthDiagonalSmall>,
		Spring<3,6,springLengthDiagonalSmall>,
		Spring<3,4,springLengthDiagonalSmall>,
		Spring<0,7,springLengthDiagonalSmall>,
		Spring<4,5,springLength>,
		Spring<5,6,springLength>,
		Spring<6,7,springLength>,
		Spring<7,4,springLength>,
		Spring<4,6,springLengthDiagonalSmall>,
		Spring<7,5,springLengthDiagonalSmall>,
		Spring<0,6,springLengthDiagonal>,
		Spring<1,7,springLengthDiagonal>,
		Spring<2,4,springLengthDiagonal>,
		Spring<3,5,springLengthDiagonal>
		>;

	using CubeVertices =
		mp_list<
		mpFloat3<0, 0, 0, 1>,
		mpFloat3<1, 0, 0, 1>,
		mpFloat3<1, 1, 0, 1>,
		mpFloat3<0, 1, 0, 1>,
		mpFloat3<0, 0, 1, 1>,
		mpFloat3<1, 0, 1, 1>,
		mpFloat3<1, 1, 1, 1>,
		mpFloat3<0, 1, 1, 1>,
		>;
}