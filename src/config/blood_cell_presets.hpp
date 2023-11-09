#pragma once

#include "../objects/blood_cells_def_type.hpp"

#include <boost/mp11/list.hpp>
#include <boost/mp11/mpl_tuple.hpp>
#include <glm/vec3.hpp>


namespace preset
{
	using namespace boost::mp11;

	/// constexpr power helper
	template <int A, int B>
	struct get_power
	{
		static const int value = A * get_power<A, B - 1>::value;
	};
	template <int A>
	struct get_power<A, 0>
	{
		static const int value = 1;
	};

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
		inline static constexpr float x = float(X) / get_power<10, decimalPrecision-1>::value;
		inline static constexpr float y = float(Y) / get_power<10, decimalPrecision-1>::value;
		inline static constexpr float z = float(Z) / get_power<10, decimalPrecision-1>::value;
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

	/// <summary>
	/// Blood cell model vertices
	/// </summary>
	using BloodCellVertices =
		mp_list <
		mpFloat3<-30901, -19079, 0, 5>,
		mpFloat3<-25000, 19079, -18163, 5>,
		mpFloat3<-47552, 24515, -34549, 5>,
		mpFloat3<-65450, 25736, -47552, 5>,
		mpFloat3<-76942, 17869, -55901, 5>,
		mpFloat3<-80901, 0, -58778, 5>,
		mpFloat3<-76942, -17869, -55901, 5>,
		mpFloat3<-65450, -25736, -47552, 5>,
		mpFloat3<-47552, -24515, -34549, 5>,
		mpFloat3<-25000, -19079, -18163, 5>,
		mpFloat3<-9549, 19079, -29389, 5>,
		mpFloat3<-18163, 24515, -55901, 5>,
		mpFloat3<-25000, 25736, -76942, 5>,
		mpFloat3<-29389, 17869, -90450, 5>,
		mpFloat3<-30901, 0, -95105, 5>,
		mpFloat3<-29389, -17869, -90450, 5>,
		mpFloat3<-25000, -25736, -76942, 5>,
		mpFloat3<-18163, -24515, -55901, 5>,
		mpFloat3<-9549, -19079, -29389, 5>,
		mpFloat3<9549, 19079, -29389, 5>,
		mpFloat3<18163, 24515, -55901, 5>,
		mpFloat3<25000, 25736, -76942, 5>,
		mpFloat3<29389, 17869, -90450, 5>,
		mpFloat3<30901, 0, -95105, 5>,
		mpFloat3<29389, -17869, -90450, 5>,
		mpFloat3<25000, -25736, -76942, 5>,
		mpFloat3<18163, -24515, -55901, 5>,
		mpFloat3<9549, -19079, -29389, 5>,
		mpFloat3<25000, 19079, -18163, 5>,
		mpFloat3<47552, 24515, -34549, 5>,
		mpFloat3<65450, 25736, -47552, 5>,
		mpFloat3<76942, 17869, -55901, 5>,
		mpFloat3<80901, 0, -58778, 5>,
		mpFloat3<76942, -17869, -55901, 5>,
		mpFloat3<65450, -25736, -47552, 5>,
		mpFloat3<47552, -24515, -34549, 5>,
		mpFloat3<25000, -19079, -18163, 5>,
		mpFloat3<30901, 19079, 0, 5>,
		mpFloat3<58778, 24515, 0, 5>,
		mpFloat3<80901, 25736, 0, 5>,
		mpFloat3<95105, 17869, 0, 5>,
		mpFloat3<100000, 0, 0, 5>,
		mpFloat3<95105, -17869, 0, 5>,
		mpFloat3<80901, -25736, 0, 5>,
		mpFloat3<58778, -24515, 0, 5>,
		mpFloat3<30901, -19079, 0, 5>,
		mpFloat3<25000, 19079, 18163, 5>,
		mpFloat3<47552, 24515, 34549, 5>,
		mpFloat3<65450, 25736, 47552, 5>,
		mpFloat3<76942, 17869, 55901, 5>,
		mpFloat3<80901, 0, 58778, 5>,
		mpFloat3<76942, -17869, 55901, 5>,
		mpFloat3<65450, -25736, 47552, 5>,
		mpFloat3<47552, -24515, 34549, 5>,
		mpFloat3<25000, -19079, 18163, 5>,
		mpFloat3<9549, 19079, 29389, 5>,
		mpFloat3<18163, 24515, 55901, 5>,
		mpFloat3<25000, 25736, 76942, 5>,
		mpFloat3<29389, 17869, 90450, 5>,
		mpFloat3<30901, 0, 95105, 5>,
		mpFloat3<29389, -17869, 90450, 5>,
		mpFloat3<25000, -25736, 76942, 5>,
		mpFloat3<18163, -24515, 55901, 5>,
		mpFloat3<9549, -19079, 29389, 5>,
		mpFloat3<0, -16018, 0, 5>,
		mpFloat3<-9549, 19079, 29389, 5>,
		mpFloat3<-18163, 24515, 55901, 5>,
		mpFloat3<-25000, 25736, 76942, 5>,
		mpFloat3<-29389, 17869, 90450, 5>,
		mpFloat3<-30901, 0, 95105, 5>,
		mpFloat3<-29389, -17869, 90450, 5>,
		mpFloat3<-25000, -25736, 76942, 5>,
		mpFloat3<-18163, -24515, 55901, 5>,
		mpFloat3<-9549, -19079, 29389, 5>,
		mpFloat3<0, 16018, 0, 5>,
		mpFloat3<-25000, 19079, 18163, 5>,
		mpFloat3<-47552, 24515, 34549, 5>,
		mpFloat3<-65450, 25736, 47552, 5>,
		mpFloat3<-76942, 17869, 55901, 5>,
		mpFloat3<-80901, 0, 58778, 5>,
		mpFloat3<-76942, -17869, 55901, 5>,
		mpFloat3<-65450, -25736, 47552, 5>,
		mpFloat3<-47552, -24515, 34549, 5>,
		mpFloat3<-25000, -19079, 18163, 5>,
		mpFloat3<-30901, 19079, 0, 5>,
		mpFloat3<-58778, 24515, 0, 5>,
		mpFloat3<-80901, 25736, 0, 5>,
		mpFloat3<-95105, 17869, 0, 5>,
		mpFloat3<-100000, 0, 0, 5>,
		mpFloat3<-95105, -17869, 0, 5>,
		mpFloat3<-80901, -25736, 0, 5>,
		mpFloat3<-58778, -24515, 0, 5>
		> ;

	using BloodCellIndices =
		mp_list <
		mp_int<4>, mp_int<14>, mp_int<5>,
		mp_int<9>, mp_int<17>, mp_int<18>,
		mp_int<3>, mp_int<13>, mp_int<4>,
		mp_int<7>, mp_int<17>, mp_int<8>,
		mp_int<2>, mp_int<12>, mp_int<3>,
		mp_int<6>, mp_int<16>, mp_int<7>,
		mp_int<2>, mp_int<10>, mp_int<11>,
		mp_int<6>, mp_int<14>, mp_int<15>,
		mp_int<18>, mp_int<26>, mp_int<27>,
		mp_int<12>, mp_int<22>, mp_int<13>,
		mp_int<17>, mp_int<25>, mp_int<26>,
		mp_int<11>, mp_int<21>, mp_int<12>,
		mp_int<15>, mp_int<25>, mp_int<16>,
		mp_int<11>, mp_int<19>, mp_int<20>,
		mp_int<14>, mp_int<24>, mp_int<15>,
		mp_int<13>, mp_int<23>, mp_int<14>,
		mp_int<26>, mp_int<34>, mp_int<35>,
		mp_int<20>, mp_int<30>, mp_int<21>,
		mp_int<24>, mp_int<34>, mp_int<25>,
		mp_int<20>, mp_int<28>, mp_int<29>,
		mp_int<23>, mp_int<33>, mp_int<24>,
		mp_int<23>, mp_int<31>, mp_int<32>,
		mp_int<26>, mp_int<36>, mp_int<27>,
		mp_int<22>, mp_int<30>, mp_int<31>,
		mp_int<33>, mp_int<43>, mp_int<34>,
		mp_int<28>, mp_int<38>, mp_int<29>,
		mp_int<32>, mp_int<42>, mp_int<33>,
		mp_int<31>, mp_int<41>, mp_int<32>,
		mp_int<36>, mp_int<44>, mp_int<45>,
		mp_int<31>, mp_int<39>, mp_int<40>,
		mp_int<35>, mp_int<43>, mp_int<44>,
		mp_int<29>, mp_int<39>, mp_int<30>,
		mp_int<41>, mp_int<51>, mp_int<42>,
		mp_int<41>, mp_int<49>, mp_int<50>,
		mp_int<45>, mp_int<53>, mp_int<54>,
		mp_int<39>, mp_int<49>, mp_int<40>,
		mp_int<44>, mp_int<52>, mp_int<53>,
		mp_int<38>, mp_int<48>, mp_int<39>,
		mp_int<42>, mp_int<52>, mp_int<43>,
		mp_int<37>, mp_int<47>, mp_int<38>,
		mp_int<50>, mp_int<58>, mp_int<59>,
		mp_int<53>, mp_int<63>, mp_int<54>,
		mp_int<49>, mp_int<57>, mp_int<58>,
		mp_int<53>, mp_int<61>, mp_int<62>,
		mp_int<47>, mp_int<57>, mp_int<48>,
		mp_int<52>, mp_int<60>, mp_int<61>,
		mp_int<47>, mp_int<55>, mp_int<56>,
		mp_int<50>, mp_int<60>, mp_int<51>,
		mp_int<62>, mp_int<73>, mp_int<63>,
		mp_int<58>, mp_int<67>, mp_int<68>,
		mp_int<62>, mp_int<71>, mp_int<72>,
		mp_int<56>, mp_int<67>, mp_int<57>,
		mp_int<61>, mp_int<70>, mp_int<71>,
		mp_int<55>, mp_int<66>, mp_int<56>,
		mp_int<59>, mp_int<70>, mp_int<60>,
		mp_int<59>, mp_int<68>, mp_int<69>,
		mp_int<72>, mp_int<81>, mp_int<82>,
		mp_int<66>, mp_int<77>, mp_int<67>,
		mp_int<71>, mp_int<80>, mp_int<81>,
		mp_int<65>, mp_int<76>, mp_int<66>,
		mp_int<69>, mp_int<80>, mp_int<70>,
		mp_int<68>, mp_int<79>, mp_int<69>,
		mp_int<72>, mp_int<83>, mp_int<73>,
		mp_int<67>, mp_int<78>, mp_int<68>,
		mp_int<80>, mp_int<90>, mp_int<81>,
		mp_int<75>, mp_int<85>, mp_int<76>,
		mp_int<79>, mp_int<89>, mp_int<80>,
		mp_int<79>, mp_int<87>, mp_int<88>,
		mp_int<77>, mp_int<87>, mp_int<78>,
		mp_int<82>, mp_int<90>, mp_int<91>,
		mp_int<76>, mp_int<86>, mp_int<77>,
		mp_int<64>, mp_int<0>, mp_int<9>,
		mp_int<88>, mp_int<4>, mp_int<5>,
		mp_int<0>, mp_int<8>, mp_int<9>,
		mp_int<86>, mp_int<4>, mp_int<87>,
		mp_int<91>, mp_int<7>, mp_int<8>,
		mp_int<85>, mp_int<3>, mp_int<86>,
		mp_int<89>, mp_int<7>, mp_int<90>,
		mp_int<85>, mp_int<1>, mp_int<2>,
		mp_int<89>, mp_int<5>, mp_int<6>,
		mp_int<84>, mp_int<74>, mp_int<1>,
		mp_int<64>, mp_int<9>, mp_int<18>,
		mp_int<1>, mp_int<74>, mp_int<10>,
		mp_int<10>, mp_int<74>, mp_int<19>,
		mp_int<64>, mp_int<18>, mp_int<27>,
		mp_int<19>, mp_int<74>, mp_int<28>,
		mp_int<64>, mp_int<27>, mp_int<36>,
		mp_int<28>, mp_int<74>, mp_int<37>,
		mp_int<64>, mp_int<36>, mp_int<45>,
		mp_int<37>, mp_int<74>, mp_int<46>,
		mp_int<64>, mp_int<45>, mp_int<54>,
		mp_int<64>, mp_int<54>, mp_int<63>,
		mp_int<46>, mp_int<74>, mp_int<55>,
		mp_int<55>, mp_int<74>, mp_int<65>,
		mp_int<64>, mp_int<63>, mp_int<73>,
		mp_int<65>, mp_int<74>, mp_int<75>,
		mp_int<64>, mp_int<73>, mp_int<83>,
		mp_int<75>, mp_int<74>, mp_int<84>,
		mp_int<64>, mp_int<83>, mp_int<0>,
		mp_int<82>, mp_int<0>, mp_int<83>,
		mp_int<4>, mp_int<13>, mp_int<14>,
		mp_int<9>, mp_int<8>, mp_int<17>,
		mp_int<3>, mp_int<12>, mp_int<13>,
		mp_int<7>, mp_int<16>, mp_int<17>,
		mp_int<2>, mp_int<11>, mp_int<12>,
		mp_int<6>, mp_int<15>, mp_int<16>,
		mp_int<2>, mp_int<1>, mp_int<10>,
		mp_int<6>, mp_int<5>, mp_int<14>,
		mp_int<18>, mp_int<17>, mp_int<26>,
		mp_int<12>, mp_int<21>, mp_int<22>,
		mp_int<17>, mp_int<16>, mp_int<25>,
		mp_int<11>, mp_int<20>, mp_int<21>,
		mp_int<15>, mp_int<24>, mp_int<25>,
		mp_int<11>, mp_int<10>, mp_int<19>,
		mp_int<14>, mp_int<23>, mp_int<24>,
		mp_int<13>, mp_int<22>, mp_int<23>,
		mp_int<26>, mp_int<25>, mp_int<34>,
		mp_int<20>, mp_int<29>, mp_int<30>,
		mp_int<24>, mp_int<33>, mp_int<34>,
		mp_int<20>, mp_int<19>, mp_int<28>,
		mp_int<23>, mp_int<32>, mp_int<33>,
		mp_int<23>, mp_int<22>, mp_int<31>,
		mp_int<26>, mp_int<35>, mp_int<36>,
		mp_int<22>, mp_int<21>, mp_int<30>,
		mp_int<33>, mp_int<42>, mp_int<43>,
		mp_int<28>, mp_int<37>, mp_int<38>,
		mp_int<32>, mp_int<41>, mp_int<42>,
		mp_int<31>, mp_int<40>, mp_int<41>,
		mp_int<36>, mp_int<35>, mp_int<44>,
		mp_int<31>, mp_int<30>, mp_int<39>,
		mp_int<35>, mp_int<34>, mp_int<43>,
		mp_int<29>, mp_int<38>, mp_int<39>,
		mp_int<41>, mp_int<50>, mp_int<51>,
		mp_int<41>, mp_int<40>, mp_int<49>,
		mp_int<45>, mp_int<44>, mp_int<53>,
		mp_int<39>, mp_int<48>, mp_int<49>,
		mp_int<44>, mp_int<43>, mp_int<52>,
		mp_int<38>, mp_int<47>, mp_int<48>,
		mp_int<42>, mp_int<51>, mp_int<52>,
		mp_int<37>, mp_int<46>, mp_int<47>,
		mp_int<50>, mp_int<49>, mp_int<58>,
		mp_int<53>, mp_int<62>, mp_int<63>,
		mp_int<49>, mp_int<48>, mp_int<57>,
		mp_int<53>, mp_int<52>, mp_int<61>,
		mp_int<47>, mp_int<56>, mp_int<57>,
		mp_int<52>, mp_int<51>, mp_int<60>,
		mp_int<47>, mp_int<46>, mp_int<55>,
		mp_int<50>, mp_int<59>, mp_int<60>,
		mp_int<62>, mp_int<72>, mp_int<73>,
		mp_int<58>, mp_int<57>, mp_int<67>,
		mp_int<62>, mp_int<61>, mp_int<71>,
		mp_int<56>, mp_int<66>, mp_int<67>,
		mp_int<61>, mp_int<60>, mp_int<70>,
		mp_int<55>, mp_int<65>, mp_int<66>,
		mp_int<59>, mp_int<69>, mp_int<70>,
		mp_int<59>, mp_int<58>, mp_int<68>,
		mp_int<72>, mp_int<71>, mp_int<81>,
		mp_int<66>, mp_int<76>, mp_int<77>,
		mp_int<71>, mp_int<70>, mp_int<80>,
		mp_int<65>, mp_int<75>, mp_int<76>,
		mp_int<69>, mp_int<79>, mp_int<80>,
		mp_int<68>, mp_int<78>, mp_int<79>,
		mp_int<72>, mp_int<82>, mp_int<83>,
		mp_int<67>, mp_int<77>, mp_int<78>,
		mp_int<80>, mp_int<89>, mp_int<90>,
		mp_int<75>, mp_int<84>, mp_int<85>,
		mp_int<79>, mp_int<88>, mp_int<89>,
		mp_int<79>, mp_int<78>, mp_int<87>,
		mp_int<77>, mp_int<86>, mp_int<87>,
		mp_int<82>, mp_int<81>, mp_int<90>,
		mp_int<76>, mp_int<85>, mp_int<86>,
		mp_int<88>, mp_int<87>, mp_int<4>,
		mp_int<0>, mp_int<91>, mp_int<8>,
		mp_int<86>, mp_int<3>, mp_int<4>,
		mp_int<91>, mp_int<90>, mp_int<7>,
		mp_int<85>, mp_int<2>, mp_int<3>,
		mp_int<89>, mp_int<6>, mp_int<7>,
		mp_int<85>, mp_int<84>, mp_int<1>,
		mp_int<89>, mp_int<88>, mp_int<5>,
		mp_int<82>, mp_int<91>, mp_int<0>
		> ;

	using BloodCellNormals =
		mp_list <
		mpFloat3<-56879, 25209, -78289, 5>,
		mpFloat3<11810, -97960, 16249, 5>,
		mpFloat3<-29580, 86409, -40709, 5>,
		mpFloat3<3409, -99830, 4690, 5>,
		mpFloat3<3409, 99830, 4690, 5>,
		mpFloat3<-29580, -86409, -40709, 5>,
		mpFloat3<11810, 97960, 16249, 5>,
		mpFloat3<-56879, -25209, -78289, 5>,
		mpFloat3<0, -97960, 20080, 5>,
		mpFloat3<0, 86409, -50330, 5>,
		mpFloat3<0, -99830, 5799, 5>,
		mpFloat3<0, 99830, 5799, 5>,
		mpFloat3<0, -86409, -50330, 5>,
		mpFloat3<0, 97960, 20080, 5>,
		mpFloat3<0, -25209, -96770, 5>,
		mpFloat3<0, 25209, -96770, 5>,
		mpFloat3<-3409, -99830, 4690, 5>,
		mpFloat3<-3409, 99830, 4690, 5>,
		mpFloat3<29580, -86409, -40709, 5>,
		mpFloat3<-11810, 97960, 16249, 5>,
		mpFloat3<56879, -25209, -78289, 5>,
		mpFloat3<56879, 25209, -78289, 5>,
		mpFloat3<-11810, -97960, 16249, 5>,
		mpFloat3<29580, 86409, -40709, 5>,
		mpFloat3<47859, -86409, -15549, 5>,
		mpFloat3<-19099, 97960, 6210, 5>,
		mpFloat3<92030, -25209, -29899, 5>,
		mpFloat3<92030, 25209, -29899, 5>,
		mpFloat3<-19099, -97960, 6210, 5>,
		mpFloat3<47859, 86409, -15549, 5>,
		mpFloat3<-5510, -99830, 1789, 5>,
		mpFloat3<-5510, 99830, 1789, 5>,
		mpFloat3<92030, -25209, 29899, 5>,
		mpFloat3<92030, 25209, 29899, 5>,
		mpFloat3<-19099, -97960, -6210, 5>,
		mpFloat3<47859, 86409, 15549, 5>,
		mpFloat3<-5510, -99830, -1789, 5>,
		mpFloat3<-5510, 99830, -1789, 5>,
		mpFloat3<47859, -86409, 15549, 5>,
		mpFloat3<-19099, 97960, -6210, 5>,
		mpFloat3<56879, 25209, 78289, 5>,
		mpFloat3<-11810, -97960, -16249, 5>,
		mpFloat3<29580, 86409, 40709, 5>,
		mpFloat3<-3409, -99830, -4690, 5>,
		mpFloat3<-3409, 99830, -4690, 5>,
		mpFloat3<29580, -86409, 40709, 5>,
		mpFloat3<-11810, 97960, -16249, 5>,
		mpFloat3<56879, -25209, 78289, 5>,
		mpFloat3<0, -97960, -20080, 5>,
		mpFloat3<0, 86409, 50330, 5>,
		mpFloat3<0, -99830, -5799, 5>,
		mpFloat3<0, 99830, -5799, 5>,
		mpFloat3<0, -86409, 50330, 5>,
		mpFloat3<0, 97960, -20080, 5>,
		mpFloat3<0, -25209, 96770, 5>,
		mpFloat3<0, 25209, 96770, 5>,
		mpFloat3<3409, -99830, -4690, 5>,
		mpFloat3<3409, 99830, -4690, 5>,
		mpFloat3<-29580, -86409, 40709, 5>,
		mpFloat3<11810, 97960, -16249, 5>,
		mpFloat3<-56879, -25209, 78289, 5>,
		mpFloat3<-56879, 25209, 78289, 5>,
		mpFloat3<11810, -97960, -16249, 5>,
		mpFloat3<-29580, 86409, 40709, 5>,
		mpFloat3<-47859, -86409, 15549, 5>,
		mpFloat3<19099, 97960, -6210, 5>,
		mpFloat3<-92030, -25209, 29899, 5>,
		mpFloat3<-92030, 25209, 29899, 5>,
		mpFloat3<-47859, 86409, 15549, 5>,
		mpFloat3<5510, -99830, -1789, 5>,
		mpFloat3<5510, 99830, -1789, 5>,
		mpFloat3<9849, -99459, 3200, 5>,
		mpFloat3<-92030, 25209, -29899, 5>,
		mpFloat3<19099, -97960, 6210, 5>,
		mpFloat3<-47859, 86409, -15549, 5>,
		mpFloat3<5510, -99830, 1789, 5>,
		mpFloat3<5510, 99830, 1789, 5>,
		mpFloat3<-47859, -86409, -15549, 5>,
		mpFloat3<19099, 97960, 6210, 5>,
		mpFloat3<-92030, -25209, -29899, 5>,
		mpFloat3<9849, 99459, 3200, 5>,
		mpFloat3<6089, -99459, 8380, 5>,
		mpFloat3<6089, 99459, 8380, 5>,
		mpFloat3<0, 99459, 10360, 5>,
		mpFloat3<0, -99459, 10360, 5>,
		mpFloat3<-6089, 99459, 8380, 5>,
		mpFloat3<-6089, -99459, 8380, 5>,
		mpFloat3<-9849, 99459, 3200, 5>,
		mpFloat3<-9849, -99459, 3200, 5>,
		mpFloat3<-9849, 99459, -3200, 5>,
		mpFloat3<-9849, -99459, -3200, 5>,
		mpFloat3<-6089, -99459, -8380, 5>,
		mpFloat3<-6089, 99459, -8380, 5>,
		mpFloat3<0, 99459, -10360, 5>,
		mpFloat3<0, -99459, -10360, 5>,
		mpFloat3<6089, 99459, -8380, 5>,
		mpFloat3<6089, -99459, -8380, 5>,
		mpFloat3<9849, 99459, -3200, 5>,
		mpFloat3<9849, -99459, -3200, 5>,
		mpFloat3<19099, -97960, -6210, 5>
		> ;
}