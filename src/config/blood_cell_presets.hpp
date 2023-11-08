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

	template<int X, int Y, int Z>
	struct mpInt3
	{
		inline static constexpr int x = X;
		inline static constexpr int y = Y;
		inline static constexpr int z = Z;
	};

	template <int X, int Y, int Z, int decimalPrecision = 1>
	struct mpFloat3
	{
		inline static constexpr float x = float(X) / decimalPrecision;
		inline static constexpr float y = float(Y) / decimalPrecision;
		inline static constexpr float z = float(Z) / decimalPrecision;
	};

	template<int I>
	struct mpIndex
	{
		inline static constexpr int index = I;
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
		mp_list <
		Spring<0, 1, springLength>,
		Spring<1, 2, springLength>,
		Spring<2, 3, springLength>,
		Spring<3, 0, springLength>,
		Spring<0, 2, springLengthDiagonalSmall>,
		Spring<3, 1, springLengthDiagonalSmall>,
		Spring<0, 4, springLength>,
		Spring<1, 5, springLength>,
		Spring<0, 5, springLengthDiagonalSmall>,
		Spring<1, 4, springLengthDiagonalSmall>,
		Spring<1, 6, springLengthDiagonalSmall>,
		Spring<2, 5, springLengthDiagonalSmall>,
		Spring<2, 6, springLength>,
		Spring<3, 7, springLength>,
		Spring<2, 7, springLengthDiagonalSmall>,
		Spring<3, 6, springLengthDiagonalSmall>,
		Spring<3, 4, springLengthDiagonalSmall>,
		Spring<0, 7, springLengthDiagonalSmall>,
		Spring<4, 5, springLength>,
		Spring<5, 6, springLength>,
		Spring<6, 7, springLength>,
		Spring<7, 4, springLength>,
		Spring<4, 6, springLengthDiagonalSmall>,
		Spring<7, 5, springLengthDiagonalSmall>,
		Spring<0, 6, springLengthDiagonal>,
		Spring<1, 7, springLengthDiagonal>,
		Spring<2, 4, springLengthDiagonal>,
		Spring<3, 5, springLengthDiagonal>
		> ;

	using CubeVertices =
		mp_list<
		mpFloat3<0, 0, 0, 1>,
		mpFloat3<10, 0, 0, 1>,
		mpFloat3<10, 10, 0, 1>,
		mpFloat3<0, 10, 0, 1>,
		mpFloat3<0, 0, 10, 1>,
		mpFloat3<10, 0, 10, 1>,
		mpFloat3<10, 10, 10, 1>,
		mpFloat3<0, 10, 10, 1>
		>;

	using CubeIndices =
		mp_list<
		
		mp_int<0>, mp_int<2>, mp_int<3>,
		mp_int<1>, mp_int<2>, mp_int<0>,
		
		mp_int<4>, mp_int<0>, mp_int<3>,
		mp_int<7>, mp_int<4>, mp_int<3>,

		mp_int<7>, mp_int<3>, mp_int<2>,
		mp_int<6>, mp_int<7>, mp_int<2>,

		mp_int<6>, mp_int<2>, mp_int<1>,
		mp_int<5>, mp_int<6>, mp_int<1>,

		mp_int<4>, mp_int<1>, mp_int<0>,
		mp_int<5>, mp_int<1>, mp_int<4>,

		mp_int<5>, mp_int<4>, mp_int<7>,
		mp_int<5>, mp_int<7>, mp_int<6>
		>;

	using CubeNormals =
		mp_list<
		mpFloat3<-57735, -57735, -57735, 5>, // 0
		mpFloat3<57735, -57735, -57735, 5>, // 1
		mpFloat3<57735, 57735, -57735, 5>, // 2
		mpFloat3<-57735, 57735, -57735, 5>, // 3
		mpFloat3<-57735, -57735, 57735, 5>, // 4
		mpFloat3<57735, -57735, 57735, 5>, // 5
		mpFloat3<57735, 57735, 57735, 5>, // 6
		mpFloat3<-57735, 57735, 57735, 5> // 7
		>;
}