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
		mp_int<0>, mp_int<1>, mp_int<2>,
		mp_int<3>, mp_int<4>, mp_int<5>,
		mp_int<6>, mp_int<7>, mp_int<8>,
		mp_int<9>, mp_int<10>, mp_int<11>,
		mp_int<12>, mp_int<13>, mp_int<14>,
		mp_int<15>, mp_int<16>, mp_int<17>,
		mp_int<18>, mp_int<19>, mp_int<20>,
		mp_int<21>, mp_int<22>, mp_int<23>,
		mp_int<24>, mp_int<25>, mp_int<26>,
		mp_int<27>, mp_int<28>, mp_int<29>,
		mp_int<30>, mp_int<31>, mp_int<32>,
		mp_int<33>, mp_int<34>, mp_int<35>,
		mp_int<36>, mp_int<37>, mp_int<38>,
		mp_int<39>, mp_int<40>, mp_int<41>,
		mp_int<42>, mp_int<43>, mp_int<44>,
		mp_int<45>, mp_int<46>, mp_int<47>,
		mp_int<48>, mp_int<49>, mp_int<50>,
		mp_int<51>, mp_int<52>, mp_int<53>,
		mp_int<54>, mp_int<55>, mp_int<56>,
		mp_int<57>, mp_int<58>, mp_int<59>,
		mp_int<60>, mp_int<61>, mp_int<62>,
		mp_int<63>, mp_int<64>, mp_int<65>,
		mp_int<66>, mp_int<67>, mp_int<68>,
		mp_int<69>, mp_int<70>, mp_int<71>,
		mp_int<72>, mp_int<73>, mp_int<74>,
		mp_int<75>, mp_int<76>, mp_int<77>,
		mp_int<78>, mp_int<79>, mp_int<80>,
		mp_int<81>, mp_int<82>, mp_int<83>,
		mp_int<84>, mp_int<85>, mp_int<86>,
		mp_int<87>, mp_int<88>, mp_int<89>,
		mp_int<90>, mp_int<91>, mp_int<92>,
		mp_int<93>, mp_int<94>, mp_int<95>,
		mp_int<96>, mp_int<97>, mp_int<98>,
		mp_int<99>, mp_int<100>, mp_int<101>,
		mp_int<102>, mp_int<103>, mp_int<104>,
		mp_int<105>, mp_int<106>, mp_int<107>,
		mp_int<108>, mp_int<109>, mp_int<110>,
		mp_int<111>, mp_int<112>, mp_int<113>,
		mp_int<114>, mp_int<115>, mp_int<116>,
		mp_int<117>, mp_int<118>, mp_int<119>,
		mp_int<120>, mp_int<121>, mp_int<122>,
		mp_int<123>, mp_int<124>, mp_int<125>,
		mp_int<126>, mp_int<127>, mp_int<128>,
		mp_int<129>, mp_int<130>, mp_int<131>,
		mp_int<132>, mp_int<133>, mp_int<134>,
		mp_int<135>, mp_int<136>, mp_int<137>,
		mp_int<138>, mp_int<139>, mp_int<140>,
		mp_int<141>, mp_int<142>, mp_int<143>,
		mp_int<144>, mp_int<145>, mp_int<146>,
		mp_int<147>, mp_int<148>, mp_int<149>,
		mp_int<150>, mp_int<151>, mp_int<152>,
		mp_int<153>, mp_int<154>, mp_int<155>,
		mp_int<156>, mp_int<157>, mp_int<158>,
		mp_int<159>, mp_int<160>, mp_int<161>,
		mp_int<162>, mp_int<163>, mp_int<164>,
		mp_int<165>, mp_int<166>, mp_int<167>,
		mp_int<168>, mp_int<169>, mp_int<170>,
		mp_int<171>, mp_int<172>, mp_int<173>,
		mp_int<174>, mp_int<175>, mp_int<176>,
		mp_int<177>, mp_int<178>, mp_int<179>,
		mp_int<180>, mp_int<181>, mp_int<182>,
		mp_int<183>, mp_int<184>, mp_int<185>,
		mp_int<186>, mp_int<187>, mp_int<188>,
		mp_int<189>, mp_int<190>, mp_int<191>,
		mp_int<192>, mp_int<193>, mp_int<194>,
		mp_int<195>, mp_int<196>, mp_int<197>,
		mp_int<198>, mp_int<199>, mp_int<200>,
		mp_int<201>, mp_int<202>, mp_int<203>,
		mp_int<204>, mp_int<205>, mp_int<206>,
		mp_int<207>, mp_int<208>, mp_int<209>,
		mp_int<210>, mp_int<211>, mp_int<212>,
		mp_int<213>, mp_int<214>, mp_int<215>,
		mp_int<216>, mp_int<217>, mp_int<218>,
		mp_int<219>, mp_int<220>, mp_int<221>,
		mp_int<222>, mp_int<223>, mp_int<224>,
		mp_int<225>, mp_int<226>, mp_int<227>,
		mp_int<228>, mp_int<229>, mp_int<230>,
		mp_int<231>, mp_int<232>, mp_int<233>,
		mp_int<234>, mp_int<235>, mp_int<236>,
		mp_int<237>, mp_int<238>, mp_int<239>,
		mp_int<240>, mp_int<241>, mp_int<242>,
		mp_int<243>, mp_int<244>, mp_int<245>,
		mp_int<246>, mp_int<247>, mp_int<248>,
		mp_int<249>, mp_int<250>, mp_int<251>,
		mp_int<252>, mp_int<253>, mp_int<254>,
		mp_int<255>, mp_int<256>, mp_int<257>,
		mp_int<258>, mp_int<259>, mp_int<260>,
		mp_int<261>, mp_int<262>, mp_int<263>,
		mp_int<264>, mp_int<265>, mp_int<266>,
		mp_int<267>, mp_int<268>, mp_int<269>,
		mp_int<270>, mp_int<271>, mp_int<272>,
		mp_int<273>, mp_int<274>, mp_int<275>,
		mp_int<276>, mp_int<277>, mp_int<278>,
		mp_int<279>, mp_int<280>, mp_int<281>,
		mp_int<282>, mp_int<283>, mp_int<284>,
		mp_int<285>, mp_int<286>, mp_int<287>,
		mp_int<288>, mp_int<289>, mp_int<290>,
		mp_int<291>, mp_int<292>, mp_int<293>,
		mp_int<294>, mp_int<295>, mp_int<296>,
		mp_int<297>, mp_int<298>, mp_int<299>,
		mp_int<0>, mp_int<300>, mp_int<1>,
		mp_int<3>, mp_int<301>, mp_int<4>,
		mp_int<6>, mp_int<302>, mp_int<7>,
		mp_int<9>, mp_int<303>, mp_int<10>,
		mp_int<12>, mp_int<304>, mp_int<13>,
		mp_int<15>, mp_int<305>, mp_int<16>,
		mp_int<18>, mp_int<306>, mp_int<19>,
		mp_int<21>, mp_int<307>, mp_int<22>,
		mp_int<24>, mp_int<308>, mp_int<25>,
		mp_int<27>, mp_int<309>, mp_int<28>,
		mp_int<30>, mp_int<310>, mp_int<31>,
		mp_int<33>, mp_int<311>, mp_int<34>,
		mp_int<36>, mp_int<312>, mp_int<37>,
		mp_int<39>, mp_int<313>, mp_int<40>,
		mp_int<42>, mp_int<314>, mp_int<43>,
		mp_int<45>, mp_int<315>, mp_int<46>,
		mp_int<48>, mp_int<316>, mp_int<49>,
		mp_int<51>, mp_int<317>, mp_int<52>,
		mp_int<54>, mp_int<318>, mp_int<55>,
		mp_int<57>, mp_int<319>, mp_int<58>,
		mp_int<60>, mp_int<320>, mp_int<61>,
		mp_int<63>, mp_int<321>, mp_int<64>,
		mp_int<66>, mp_int<322>, mp_int<67>,
		mp_int<69>, mp_int<323>, mp_int<70>,
		mp_int<72>, mp_int<324>, mp_int<73>,
		mp_int<75>, mp_int<325>, mp_int<76>,
		mp_int<78>, mp_int<326>, mp_int<79>,
		mp_int<81>, mp_int<327>, mp_int<82>,
		mp_int<84>, mp_int<328>, mp_int<85>,
		mp_int<87>, mp_int<329>, mp_int<88>,
		mp_int<90>, mp_int<330>, mp_int<91>,
		mp_int<93>, mp_int<331>, mp_int<94>,
		mp_int<96>, mp_int<332>, mp_int<97>,
		mp_int<99>, mp_int<333>, mp_int<100>,
		mp_int<102>, mp_int<334>, mp_int<103>,
		mp_int<105>, mp_int<335>, mp_int<106>,
		mp_int<108>, mp_int<336>, mp_int<109>,
		mp_int<111>, mp_int<337>, mp_int<112>,
		mp_int<114>, mp_int<338>, mp_int<115>,
		mp_int<117>, mp_int<339>, mp_int<118>,
		mp_int<120>, mp_int<340>, mp_int<121>,
		mp_int<123>, mp_int<341>, mp_int<124>,
		mp_int<126>, mp_int<342>, mp_int<127>,
		mp_int<129>, mp_int<343>, mp_int<130>,
		mp_int<132>, mp_int<344>, mp_int<133>,
		mp_int<135>, mp_int<345>, mp_int<136>,
		mp_int<138>, mp_int<346>, mp_int<139>,
		mp_int<141>, mp_int<347>, mp_int<142>,
		mp_int<144>, mp_int<348>, mp_int<145>,
		mp_int<147>, mp_int<349>, mp_int<148>,
		mp_int<150>, mp_int<350>, mp_int<151>,
		mp_int<153>, mp_int<351>, mp_int<154>,
		mp_int<156>, mp_int<352>, mp_int<157>,
		mp_int<159>, mp_int<353>, mp_int<160>,
		mp_int<162>, mp_int<354>, mp_int<163>,
		mp_int<165>, mp_int<355>, mp_int<166>,
		mp_int<168>, mp_int<356>, mp_int<169>,
		mp_int<171>, mp_int<357>, mp_int<172>,
		mp_int<174>, mp_int<358>, mp_int<175>,
		mp_int<177>, mp_int<359>, mp_int<178>,
		mp_int<180>, mp_int<360>, mp_int<181>,
		mp_int<183>, mp_int<361>, mp_int<184>,
		mp_int<186>, mp_int<362>, mp_int<187>,
		mp_int<189>, mp_int<363>, mp_int<190>,
		mp_int<192>, mp_int<364>, mp_int<193>,
		mp_int<195>, mp_int<365>, mp_int<196>,
		mp_int<198>, mp_int<366>, mp_int<199>,
		mp_int<201>, mp_int<367>, mp_int<202>,
		mp_int<204>, mp_int<368>, mp_int<205>,
		mp_int<207>, mp_int<369>, mp_int<208>,
		mp_int<210>, mp_int<370>, mp_int<211>,
		mp_int<216>, mp_int<371>, mp_int<217>,
		mp_int<219>, mp_int<372>, mp_int<220>,
		mp_int<222>, mp_int<373>, mp_int<223>,
		mp_int<225>, mp_int<374>, mp_int<226>,
		mp_int<228>, mp_int<375>, mp_int<229>,
		mp_int<231>, mp_int<376>, mp_int<232>,
		mp_int<234>, mp_int<377>, mp_int<235>,
		mp_int<237>, mp_int<378>, mp_int<238>,
		mp_int<297>, mp_int<379>, mp_int<298>
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