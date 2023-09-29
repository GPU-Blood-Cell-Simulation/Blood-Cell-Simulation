#pragma once

#include "../config/blood_cells_definition.hpp"

#include <array>
#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>
#include <boost/mp11/utility.hpp>


// Helper meta-functor for adding particles
template<class State, class Def>
using Add = mp_int<State::value + Def::count * Def::particlesInCell>;

// Helper meta-functor for calculating the graph size
template<class State, class Def>
using AddSquared = mp_int<State::value + Def::particlesInCell * Def::particlesInCell>;

// Particle count
inline constexpr int particleCount = mp_fold<BloodCellList, mp_int<0>, Add>::value;

// Total particle graph size
inline constexpr int totalGraphSize = mp_fold<BloodCellList, mp_int<0>, AddSquared>::value;

// Distinct blood cell types
inline constexpr int bloodCellTypeCount = mp_size<BloodCellList>::value;

// Fill the particles start array
inline constexpr auto particlesStartsGenerator = []()
{
	std::array<int, bloodCellTypeCount> arr{};

	// Iterate over user-provided definition (particle type)
	using IndexList = mp_iota_c<mp_size<BloodCellList>::value>;
	int state = 0;
	mp_for_each<IndexList>([&](auto i)
	{
		using BloodCellDefinition = mp_at_c<BloodCellList, i>;
		arr[i] = state;
		state += BloodCellDefinition::count * BloodCellDefinition::particlesInCell;
	});
	return arr;
};


// Determine where in the device array a particular stream should start its job (calculate accumulated particlesInCell sums)
inline constexpr auto particlesStarts = particlesStartsGenerator();

// Fill the accumulated graph sizes array
inline constexpr auto graphSizesGenerator = []()
{
	std::array<int, bloodCellTypeCount> arr{};

	// Iterate over user-provided definition (particle type)
	using IndexList = mp_iota_c<mp_size<BloodCellList>::value>;
	int state = 0;
	mp_for_each<IndexList>([&](auto i)
	{
		using BloodCellDefinition = mp_at_c<BloodCellList, i>;
		arr[i] = state;
		state += BloodCellDefinition::particlesInCell * BloodCellDefinition::particlesInCell;
	});
	return arr;
};

// Graph sizes for each type
inline constexpr auto accumulatedGraphSizes = graphSizesGenerator();

// Fill the particle neighborhood graph
inline constexpr auto springGraphGenerator = [&]()
{
	std::array<float, totalGraphSize> arr{};

	// Iterate over user-provided definition (particle type)
	using IndexList = mp_iota<mp_size<BloodCellList>>;
	mp_for_each<IndexList>([&](auto i)
	{
		using BloodCellDefinition = mp_at_c<BloodCellList, i>;
		constexpr int particlesInThisCell = BloodCellDefinition::particlesInCell;

		using SpringList = typename BloodCellDefinition::List;
		constexpr int springCount = mp_size<SpringList>::value;
		constexpr int graphStart = accumulatedGraphSizes[i];

		// For every definition iterate over its particle count
		using IndexListPerBloodCell = mp_iota_c<springCount>;
		mp_for_each<IndexListPerBloodCell>([&](auto j)
		{
			using SpringDefinition = mp_at_c<SpringList, j>;
			arr[graphStart + SpringDefinition::start * particlesInThisCell + SpringDefinition::end] = SpringDefinition::length;
			arr[graphStart + SpringDefinition::end * particlesInThisCell + SpringDefinition::start] = SpringDefinition::length;
		});
	});
	return arr;
};

inline constexpr auto springGraph = springGraphGenerator();