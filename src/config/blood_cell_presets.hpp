#pragma once

#include "../objects/blood_cells_def_type.hpp"

#include <boost/mp11/list.hpp>
#include <boost/mp11/mpl_tuple.hpp>
#include <glm/vec3.hpp>


namespace preset
{
	using namespace boost::mp11;

	// TODO: add a full-graph preset

	using NoSprings =
		mp_list<>;

	using Dipole =
		mp_list<
		Spring<0, 1, springLength, 1>
		>;

	using Quadrupole =
		mp_list<
		Spring<0, 1, springLength, 1>,
		Spring<1, 2, springLength, 1>,
		Spring<2, 3, springLength, 1>,
		Spring<3, 0, springLength, 1>
		>;

	using Octupole =
		mp_list<
		Spring<0, 1, springLength, 1>,
		Spring<1, 2, springLength, 1>,
		Spring<2, 3, springLength, 1>,
		Spring<3, 0, springLength, 1>,

		Spring<4, 5, springLength, 1>,
		Spring<5, 6, springLength, 1>,
		Spring<6, 7, springLength, 1>,
		Spring<7, 4, springLength, 1>,

		Spring<0, 4, springLength, 1>,
		Spring<1, 5, springLength, 1>,
		Spring<2, 6, springLength, 1>,
		Spring<3, 7, springLength, 1>
		>;

	using Cube =
		mp_list <
		Spring<0, 1, springLength, 1>,
		Spring<1, 2, springLength, 1>,
		Spring<2, 3, springLength, 1>,
		Spring<3, 0, springLength, 1>,
		Spring<0, 2, springLengthDiagonalSmall, 1>,
		Spring<3, 1, springLengthDiagonalSmall, 1>,
		Spring<0, 4, springLength, 1>,
		Spring<1, 5, springLength, 1>,
		Spring<0, 5, springLengthDiagonalSmall, 1>,
		Spring<1, 4, springLengthDiagonalSmall, 1>,
		Spring<1, 6, springLengthDiagonalSmall, 1>,
		Spring<2, 5, springLengthDiagonalSmall, 1>,
		Spring<2, 6, springLength, 1>,
		Spring<3, 7, springLength, 1>,
		Spring<2, 7, springLengthDiagonalSmall, 1>,
		Spring<3, 6, springLengthDiagonalSmall, 1>,
		Spring<3, 4, springLengthDiagonalSmall, 1>,
		Spring<0, 7, springLengthDiagonalSmall, 1>,
		Spring<4, 5, springLength, 1>,
		Spring<5, 6, springLength, 1>,
		Spring<6, 7, springLength, 1>,
		Spring<7, 4, springLength, 1>,
		Spring<4, 6, springLengthDiagonalSmall, 1>,
		Spring<7, 5, springLengthDiagonalSmall, 1>,
		Spring<0, 6, springLengthDiagonal, 1>,
		Spring<1, 7, springLengthDiagonal, 1>,
		Spring<2, 4, springLengthDiagonal, 1>,
		Spring<3, 5, springLengthDiagonal, 1>
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
#pragma region Predefined Blood Cell
	using BloodCellSprings =
		mp_list <
		Spring<4, 14, 63055, 5>,
		Spring<14, 5, 61803, 5>,
		Spring<5, 4, 18527, 5>,
		Spring<9, 17, 38735, 5>,
		Spring<17, 18, 28401, 5>,
		Spring<18, 9, 19098, 5>,
		Spring<3, 13, 56591, 5>,
		Spring<13, 4, 58778, 5>,
		Spring<4, 3, 16237, 5>,
		Spring<7, 17, 48034, 5>,
		Spring<17, 8, 36327, 5>,
		Spring<8, 7, 22156, 5>,
		Spring<2, 12, 48034, 5>,
		Spring<12, 3, 49999, 5>,
		Spring<3, 2, 22156, 5>,
		Spring<6, 16, 56591, 5>,
		Spring<16, 7, 49999, 5>,
		Spring<7, 6, 16237, 5>,
		Spring<2, 10, 38735, 5>,
		Spring<10, 11, 28401, 5>,
		Spring<11, 2, 36327, 5>,
		Spring<6, 14, 63055, 5>,
		Spring<14, 15, 18527, 5>,
		Spring<15, 6, 58778, 5>,
		Spring<18, 26, 38735, 5>,
		Spring<26, 27, 28401, 5>,
		Spring<27, 18, 19098, 5>,
		Spring<12, 22, 56591, 5>,
		Spring<22, 13, 58778, 5>,
		Spring<13, 12, 16237, 5>,
		Spring<17, 25, 48034, 5>,
		Spring<25, 26, 22156, 5>,
		Spring<26, 17, 36327, 5>,
		Spring<11, 21, 48034, 5>,
		Spring<21, 12, 50000, 5>,
		Spring<12, 11, 22156, 5>,
		Spring<15, 25, 56591, 5>,
		Spring<25, 16, 50000, 5>,
		Spring<16, 15, 16237, 5>,
		Spring<11, 19, 38735, 5>,
		Spring<19, 20, 28401, 5>,
		Spring<20, 11, 36327, 5>,
		Spring<14, 24, 63055, 5>,
		Spring<24, 15, 58778, 5>,
		Spring<15, 14, 18527, 5>,
		Spring<13, 23, 63055, 5>,
		Spring<23, 14, 61803, 5>,
		Spring<14, 13, 18527, 5>,
		Spring<26, 34, 48034, 5>,
		Spring<34, 35, 22156, 5>,
		Spring<35, 26, 36327, 5>,
		Spring<20, 30, 48034, 5>,
		Spring<30, 21, 50000, 5>,
		Spring<21, 20, 22156, 5>,
		Spring<24, 34, 56591, 5>,
		Spring<34, 25, 50000, 5>,
		Spring<25, 24, 16237, 5>,
		Spring<20, 28, 38735, 5>,
		Spring<28, 29, 28401, 5>,
		Spring<29, 20, 36327, 5>,
		Spring<23, 33, 63055, 5>,
		Spring<33, 24, 58778, 5>,
		Spring<24, 23, 18527, 5>,
		Spring<23, 31, 63055, 5>,
		Spring<31, 32, 18527, 5>,
		Spring<32, 23, 61803, 5>,
		Spring<26, 36, 38735, 5>,
		Spring<36, 27, 19098, 5>,
		Spring<27, 26, 28401, 5>,
		Spring<22, 30, 56591, 5>,
		Spring<30, 31, 16237, 5>,
		Spring<31, 22, 58778, 5>,
		Spring<33, 43, 56591, 5>,
		Spring<43, 34, 50000, 5>,
		Spring<34, 33, 16237, 5>,
		Spring<28, 38, 38735, 5>,
		Spring<38, 29, 36327, 5>,
		Spring<29, 28, 28401, 5>,
		Spring<32, 42, 63055, 5>,
		Spring<42, 33, 58778, 5>,
		Spring<33, 32, 18527, 5>,
		Spring<31, 41, 63055, 5>,
		Spring<41, 32, 61803, 5>,
		Spring<32, 31, 18527, 5>,
		Spring<36, 44, 38735, 5>,
		Spring<44, 45, 28401, 5>,
		Spring<45, 36, 19098, 5>,
		Spring<31, 39, 56591, 5>,
		Spring<39, 40, 16237, 5>,
		Spring<40, 31, 58778, 5>,
		Spring<35, 43, 48034, 5>,
		Spring<43, 44, 22156, 5>,
		Spring<44, 35, 36327, 5>,
		Spring<29, 39, 48034, 5>,
		Spring<39, 30, 50000, 5>,
		Spring<30, 29, 22156, 5>,
		Spring<41, 51, 63055, 5>,
		Spring<51, 42, 58778, 5>,
		Spring<42, 41, 18527, 5>,
		Spring<41, 49, 63055, 5>,
		Spring<49, 50, 18527, 5>,
		Spring<50, 41, 61803, 5>,
		Spring<45, 53, 38735, 5>,
		Spring<53, 54, 28401, 5>,
		Spring<54, 45, 19098, 5>,
		Spring<39, 49, 56591, 5>,
		Spring<49, 40, 58778, 5>,
		Spring<40, 39, 16237, 5>,
		Spring<44, 52, 48034, 5>,
		Spring<52, 53, 22156, 5>,
		Spring<53, 44, 36327, 5>,
		Spring<38, 48, 48034, 5>,
		Spring<48, 39, 49999, 5>,
		Spring<39, 38, 22156, 5>,
		Spring<42, 52, 56591, 5>,
		Spring<52, 43, 49999, 5>,
		Spring<43, 42, 16237, 5>,
		Spring<37, 47, 38735, 5>,
		Spring<47, 38, 36327, 5>,
		Spring<38, 37, 28401, 5>,
		Spring<50, 58, 63055, 5>,
		Spring<58, 59, 18527, 5>,
		Spring<59, 50, 61803, 5>,
		Spring<53, 63, 38735, 5>,
		Spring<63, 54, 19098, 5>,
		Spring<54, 53, 28401, 5>,
		Spring<49, 57, 56591, 5>,
		Spring<57, 58, 16237, 5>,
		Spring<58, 49, 58778, 5>,
		Spring<53, 61, 48034, 5>,
		Spring<61, 62, 22156, 5>,
		Spring<62, 53, 36327, 5>,
		Spring<47, 57, 48034, 5>,
		Spring<57, 48, 49999, 5>,
		Spring<48, 47, 22156, 5>,
		Spring<52, 60, 56591, 5>,
		Spring<60, 61, 16237, 5>,
		Spring<61, 52, 49999, 5>,
		Spring<47, 55, 38735, 5>,
		Spring<55, 56, 28401, 5>,
		Spring<56, 47, 36327, 5>,
		Spring<50, 60, 63055, 5>,
		Spring<60, 51, 58778, 5>,
		Spring<51, 50, 18527, 5>,
		Spring<62, 73, 38735, 5>,
		Spring<73, 63, 19098, 5>,
		Spring<63, 62, 28401, 5>,
		Spring<58, 67, 56591, 5>,
		Spring<67, 68, 16237, 5>,
		Spring<68, 58, 58778, 5>,
		Spring<62, 71, 48034, 5>,
		Spring<71, 72, 22156, 5>,
		Spring<72, 62, 36327, 5>,
		Spring<56, 67, 48034, 5>,
		Spring<67, 57, 50000, 5>,
		Spring<57, 56, 22156, 5>,
		Spring<61, 70, 56591, 5>,
		Spring<70, 71, 16237, 5>,
		Spring<71, 61, 50000, 5>,
		Spring<55, 66, 38735, 5>,
		Spring<66, 56, 36327, 5>,
		Spring<56, 55, 28401, 5>,
		Spring<59, 70, 63055, 5>,
		Spring<70, 60, 58778, 5>,
		Spring<60, 59, 18527, 5>,
		Spring<59, 68, 63055, 5>,
		Spring<68, 69, 18527, 5>,
		Spring<69, 59, 61803, 5>,
		Spring<72, 81, 48034, 5>,
		Spring<81, 82, 22156, 5>,
		Spring<82, 72, 36327, 5>,
		Spring<66, 77, 48034, 5>,
		Spring<77, 67, 49999, 5>,
		Spring<67, 66, 22156, 5>,
		Spring<71, 80, 56591, 5>,
		Spring<80, 81, 16237, 5>,
		Spring<81, 71, 49999, 5>,
		Spring<65, 76, 38735, 5>,
		Spring<76, 66, 36327, 5>,
		Spring<66, 65, 28401, 5>,
		Spring<69, 80, 63055, 5>,
		Spring<80, 70, 58778, 5>,
		Spring<70, 69, 18527, 5>,
		Spring<68, 79, 63055, 5>,
		Spring<79, 69, 61803, 5>,
		Spring<69, 68, 18527, 5>,
		Spring<72, 83, 38735, 5>,
		Spring<83, 73, 19098, 5>,
		Spring<73, 72, 28401, 5>,
		Spring<67, 78, 56591, 5>,
		Spring<78, 68, 58778, 5>,
		Spring<68, 67, 16237, 5>,
		Spring<80, 90, 56591, 5>,
		Spring<90, 81, 50000, 5>,
		Spring<81, 80, 16237, 5>,
		Spring<75, 85, 38735, 5>,
		Spring<85, 76, 36327, 5>,
		Spring<76, 75, 28401, 5>,
		Spring<79, 89, 63055, 5>,
		Spring<89, 80, 58778, 5>,
		Spring<80, 79, 18527, 5>,
		Spring<79, 87, 63055, 5>,
		Spring<87, 88, 18527, 5>,
		Spring<88, 79, 61803, 5>,
		Spring<77, 87, 56591, 5>,
		Spring<87, 78, 58778, 5>,
		Spring<78, 77, 16237, 5>,
		Spring<82, 90, 48034, 5>,
		Spring<90, 91, 22156, 5>,
		Spring<91, 82, 36327, 5>,
		Spring<76, 86, 48034, 5>,
		Spring<86, 77, 50000, 5>,
		Spring<77, 76, 22156, 5>,
		Spring<64, 0, 31052, 5>,
		Spring<0, 9, 19098, 5>,
		Spring<9, 64, 31052, 5>,
		Spring<88, 4, 63055, 5>,
		Spring<4, 5, 18527, 5>,
		Spring<5, 88, 61803, 5>,
		Spring<0, 8, 38735, 5>,
		Spring<8, 9, 28401, 5>,
		Spring<9, 0, 19098, 5>,
		Spring<86, 4, 56591, 5>,
		Spring<4, 87, 58778, 5>,
		Spring<87, 86, 16237, 5>,
		Spring<91, 7, 48034, 5>,
		Spring<7, 8, 22156, 5>,
		Spring<8, 91, 36327, 5>,
		Spring<85, 3, 48034, 5>,
		Spring<3, 86, 49999, 5>,
		Spring<86, 85, 22156, 5>,
		Spring<89, 7, 56591, 5>,
		Spring<7, 90, 49999, 5>,
		Spring<90, 89, 16237, 5>,
		Spring<85, 1, 38735, 5>,
		Spring<1, 2, 28401, 5>,
		Spring<2, 85, 36327, 5>,
		Spring<89, 5, 63055, 5>,
		Spring<5, 6, 18527, 5>,
		Spring<6, 89, 58778, 5>,
		Spring<84, 74, 31052, 5>,
		Spring<74, 1, 31052, 5>,
		Spring<1, 84, 19098, 5>,
		Spring<64, 9, 31052, 5>,
		Spring<9, 18, 19098, 5>,
		Spring<18, 64, 31052, 5>,
		Spring<1, 74, 31052, 5>,
		Spring<74, 10, 31052, 5>,
		Spring<10, 1, 19098, 5>,
		Spring<10, 74, 31052, 5>,
		Spring<74, 19, 31053, 5>,
		Spring<19, 10, 19098, 5>,
		Spring<64, 18, 31052, 5>,
		Spring<18, 27, 19098, 5>,
		Spring<27, 64, 31053, 5>,
		Spring<19, 74, 31053, 5>,
		Spring<74, 28, 31053, 5>,
		Spring<28, 19, 19098, 5>,
		Spring<64, 27, 31053, 5>,
		Spring<27, 36, 19098, 5>,
		Spring<36, 64, 31052, 5>,
		Spring<28, 74, 31053, 5>,
		Spring<74, 37, 31053, 5>,
		Spring<37, 28, 19098, 5>,
		Spring<64, 36, 31052, 5>,
		Spring<36, 45, 19098, 5>,
		Spring<45, 64, 31052, 5>,
		Spring<37, 74, 31053, 5>,
		Spring<74, 46, 31053, 5>,
		Spring<46, 37, 19098, 5>,
		Spring<64, 45, 31052, 5>,
		Spring<45, 54, 19098, 5>,
		Spring<54, 64, 31052, 5>,
		Spring<64, 54, 31052, 5>,
		Spring<54, 63, 19098, 5>,
		Spring<63, 64, 31053, 5>,
		Spring<46, 74, 31053, 5>,
		Spring<74, 55, 31053, 5>,
		Spring<55, 46, 19098, 5>,
		Spring<55, 74, 31053, 5>,
		Spring<74, 65, 31052, 5>,
		Spring<65, 55, 19098, 5>,
		Spring<64, 63, 31053, 5>,
		Spring<63, 73, 19098, 5>,
		Spring<73, 64, 31052, 5>,
		Spring<65, 74, 31052, 5>,
		Spring<74, 75, 31052, 5>,
		Spring<75, 65, 19098, 5>,
		Spring<64, 73, 31052, 5>,
		Spring<73, 83, 19098, 5>,
		Spring<83, 64, 31052, 5>,
		Spring<75, 74, 31052, 5>,
		Spring<74, 84, 31052, 5>,
		Spring<84, 75, 19098, 5>,
		Spring<64, 83, 31052, 5>,
		Spring<83, 0, 19098, 5>,
		Spring<0, 64, 31052, 5>,
		Spring<82, 0, 38735, 5>,
		Spring<0, 83, 19098, 5>,
		Spring<83, 82, 28401, 5>,
		Spring<4, 13, 58778, 5>,
		Spring<13, 14, 18527, 5>,
		Spring<14, 4, 63055, 5>,
		Spring<9, 8, 28401, 5>,
		Spring<8, 17, 36327, 5>,
		Spring<17, 9, 38735, 5>,
		Spring<3, 12, 49999, 5>,
		Spring<12, 13, 16237, 5>,
		Spring<13, 3, 56591, 5>,
		Spring<7, 16, 49999, 5>,
		Spring<16, 17, 22156, 5>,
		Spring<17, 7, 48034, 5>,
		Spring<2, 11, 36327, 5>,
		Spring<11, 12, 22156, 5>,
		Spring<12, 2, 48034, 5>,
		Spring<6, 15, 58778, 5>,
		Spring<15, 16, 16237, 5>,
		Spring<16, 6, 56591, 5>,
		Spring<2, 1, 28401, 5>,
		Spring<1, 10, 19098, 5>,
		Spring<10, 2, 38735, 5>,
		Spring<6, 5, 18527, 5>,
		Spring<5, 14, 61803, 5>,
		Spring<14, 6, 63055, 5>,
		Spring<18, 17, 28401, 5>,
		Spring<17, 26, 36327, 5>,
		Spring<26, 18, 38735, 5>,
		Spring<12, 21, 50000, 5>,
		Spring<21, 22, 16237, 5>,
		Spring<22, 12, 56591, 5>,
		Spring<17, 16, 22156, 5>,
		Spring<16, 25, 50000, 5>,
		Spring<25, 17, 48034, 5>,
		Spring<11, 20, 36327, 5>,
		Spring<20, 21, 22156, 5>,
		Spring<21, 11, 48034, 5>,
		Spring<15, 24, 58778, 5>,
		Spring<24, 25, 16237, 5>,
		Spring<25, 15, 56591, 5>,
		Spring<11, 10, 28401, 5>,
		Spring<10, 19, 19098, 5>,
		Spring<19, 11, 38735, 5>,
		Spring<14, 23, 61803, 5>,
		Spring<23, 24, 18527, 5>,
		Spring<24, 14, 63055, 5>,
		Spring<13, 22, 58778, 5>,
		Spring<22, 23, 18527, 5>,
		Spring<23, 13, 63055, 5>,
		Spring<26, 25, 22156, 5>,
		Spring<25, 34, 50000, 5>,
		Spring<34, 26, 48034, 5>,
		Spring<20, 29, 36327, 5>,
		Spring<29, 30, 22156, 5>,
		Spring<30, 20, 48034, 5>,
		Spring<24, 33, 58778, 5>,
		Spring<33, 34, 16237, 5>,
		Spring<34, 24, 56591, 5>,
		Spring<20, 19, 28401, 5>,
		Spring<19, 28, 19098, 5>,
		Spring<28, 20, 38735, 5>,
		Spring<23, 32, 61803, 5>,
		Spring<32, 33, 18527, 5>,
		Spring<33, 23, 63055, 5>,
		Spring<23, 22, 18527, 5>,
		Spring<22, 31, 58778, 5>,
		Spring<31, 23, 63055, 5>,
		Spring<26, 35, 36327, 5>,
		Spring<35, 36, 28401, 5>,
		Spring<36, 26, 38735, 5>,
		Spring<22, 21, 16237, 5>,
		Spring<21, 30, 50000, 5>,
		Spring<30, 22, 56591, 5>,
		Spring<33, 42, 58778, 5>,
		Spring<42, 43, 16237, 5>,
		Spring<43, 33, 56591, 5>,
		Spring<28, 37, 19098, 5>,
		Spring<37, 38, 28401, 5>,
		Spring<38, 28, 38735, 5>,
		Spring<32, 41, 61803, 5>,
		Spring<41, 42, 18527, 5>,
		Spring<42, 32, 63055, 5>,
		Spring<31, 40, 58778, 5>,
		Spring<40, 41, 18527, 5>,
		Spring<41, 31, 63055, 5>,
		Spring<36, 35, 28401, 5>,
		Spring<35, 44, 36327, 5>,
		Spring<44, 36, 38735, 5>,
		Spring<31, 30, 16237, 5>,
		Spring<30, 39, 50000, 5>,
		Spring<39, 31, 56591, 5>,
		Spring<35, 34, 22156, 5>,
		Spring<34, 43, 50000, 5>,
		Spring<43, 35, 48034, 5>,
		Spring<29, 38, 36327, 5>,
		Spring<38, 39, 22156, 5>,
		Spring<39, 29, 48034, 5>,
		Spring<41, 50, 61803, 5>,
		Spring<50, 51, 18527, 5>,
		Spring<51, 41, 63055, 5>,
		Spring<41, 40, 18527, 5>,
		Spring<40, 49, 58778, 5>,
		Spring<49, 41, 63055, 5>,
		Spring<45, 44, 28401, 5>,
		Spring<44, 53, 36327, 5>,
		Spring<53, 45, 38735, 5>,
		Spring<39, 48, 49999, 5>,
		Spring<48, 49, 16237, 5>,
		Spring<49, 39, 56591, 5>,
		Spring<44, 43, 22156, 5>,
		Spring<43, 52, 49999, 5>,
		Spring<52, 44, 48034, 5>,
		Spring<38, 47, 36327, 5>,
		Spring<47, 48, 22156, 5>,
		Spring<48, 38, 48034, 5>,
		Spring<42, 51, 58778, 5>,
		Spring<51, 52, 16237, 5>,
		Spring<52, 42, 56591, 5>,
		Spring<37, 46, 19098, 5>,
		Spring<46, 47, 28401, 5>,
		Spring<47, 37, 38735, 5>,
		Spring<50, 49, 18527, 5>,
		Spring<49, 58, 58778, 5>,
		Spring<58, 50, 63055, 5>,
		Spring<53, 62, 36327, 5>,
		Spring<62, 63, 28401, 5>,
		Spring<63, 53, 38735, 5>,
		Spring<49, 48, 16237, 5>,
		Spring<48, 57, 49999, 5>,
		Spring<57, 49, 56591, 5>,
		Spring<53, 52, 22156, 5>,
		Spring<52, 61, 49999, 5>,
		Spring<61, 53, 48034, 5>,
		Spring<47, 56, 36327, 5>,
		Spring<56, 57, 22156, 5>,
		Spring<57, 47, 48034, 5>,
		Spring<52, 51, 16237, 5>,
		Spring<51, 60, 58778, 5>,
		Spring<60, 52, 56591, 5>,
		Spring<47, 46, 28401, 5>,
		Spring<46, 55, 19098, 5>,
		Spring<55, 47, 38735, 5>,
		Spring<50, 59, 61803, 5>,
		Spring<59, 60, 18527, 5>,
		Spring<60, 50, 63055, 5>,
		Spring<62, 72, 36327, 5>,
		Spring<72, 73, 28401, 5>,
		Spring<73, 62, 38735, 5>,
		Spring<58, 57, 16237, 5>,
		Spring<57, 67, 50000, 5>,
		Spring<67, 58, 56591, 5>,
		Spring<62, 61, 22156, 5>,
		Spring<61, 71, 50000, 5>,
		Spring<71, 62, 48034, 5>,
		Spring<56, 66, 36327, 5>,
		Spring<66, 67, 22156, 5>,
		Spring<67, 56, 48034, 5>,
		Spring<61, 60, 16237, 5>,
		Spring<60, 70, 58778, 5>,
		Spring<70, 61, 56591, 5>,
		Spring<55, 65, 19098, 5>,
		Spring<65, 66, 28401, 5>,
		Spring<66, 55, 38735, 5>,
		Spring<59, 69, 61803, 5>,
		Spring<69, 70, 18527, 5>,
		Spring<70, 59, 63055, 5>,
		Spring<59, 58, 18527, 5>,
		Spring<58, 68, 58778, 5>,
		Spring<68, 59, 63055, 5>,
		Spring<72, 71, 22156, 5>,
		Spring<71, 81, 49999, 5>,
		Spring<81, 72, 48034, 5>,
		Spring<66, 76, 36327, 5>,
		Spring<76, 77, 22156, 5>,
		Spring<77, 66, 48034, 5>,
		Spring<71, 70, 16237, 5>,
		Spring<70, 80, 58778, 5>,
		Spring<80, 71, 56591, 5>,
		Spring<65, 75, 19098, 5>,
		Spring<75, 76, 28401, 5>,
		Spring<76, 65, 38735, 5>,
		Spring<69, 79, 61803, 5>,
		Spring<79, 80, 18527, 5>,
		Spring<80, 69, 63055, 5>,
		Spring<68, 78, 58778, 5>,
		Spring<78, 79, 18527, 5>,
		Spring<79, 68, 63055, 5>,
		Spring<72, 82, 36327, 5>,
		Spring<82, 83, 28401, 5>,
		Spring<83, 72, 38735, 5>,
		Spring<67, 77, 49999, 5>,
		Spring<77, 78, 16237, 5>,
		Spring<78, 67, 56591, 5>,
		Spring<80, 89, 58778, 5>,
		Spring<89, 90, 16237, 5>,
		Spring<90, 80, 56591, 5>,
		Spring<75, 84, 19098, 5>,
		Spring<84, 85, 28401, 5>,
		Spring<85, 75, 38735, 5>,
		Spring<79, 88, 61803, 5>,
		Spring<88, 89, 18527, 5>,
		Spring<89, 79, 63055, 5>,
		Spring<79, 78, 18527, 5>,
		Spring<78, 87, 58778, 5>,
		Spring<87, 79, 63055, 5>,
		Spring<77, 86, 50000, 5>,
		Spring<86, 87, 16237, 5>,
		Spring<87, 77, 56591, 5>,
		Spring<82, 81, 22156, 5>,
		Spring<81, 90, 50000, 5>,
		Spring<90, 82, 48034, 5>,
		Spring<76, 85, 36327, 5>,
		Spring<85, 86, 22156, 5>,
		Spring<86, 76, 48034, 5>,
		Spring<88, 87, 18527, 5>,
		Spring<87, 4, 58778, 5>,
		Spring<4, 88, 63055, 5>,
		Spring<0, 91, 28401, 5>,
		Spring<91, 8, 36327, 5>,
		Spring<8, 0, 38735, 5>,
		Spring<86, 3, 49999, 5>,
		Spring<3, 4, 16237, 5>,
		Spring<4, 86, 56591, 5>,
		Spring<91, 90, 22156, 5>,
		Spring<90, 7, 49999, 5>,
		Spring<7, 91, 48034, 5>,
		Spring<85, 2, 36327, 5>,
		Spring<2, 3, 22156, 5>,
		Spring<3, 85, 48034, 5>,
		Spring<89, 6, 58778, 5>,
		Spring<6, 7, 16237, 5>,
		Spring<7, 89, 56591, 5>,
		Spring<85, 84, 28401, 5>,
		Spring<84, 1, 19098, 5>,
		Spring<1, 85, 38735, 5>,
		Spring<89, 88, 18527, 5>,
		Spring<88, 5, 61803, 5>,
		Spring<5, 89, 63055, 5>,
		Spring<82, 91, 36327, 5>,
		Spring<91, 0, 28401, 5>
		> ;

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
#pragma endregion
}