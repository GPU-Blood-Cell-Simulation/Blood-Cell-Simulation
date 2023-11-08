#pragma once

template<int Count, int ParticlesInCell, int IndicesInCell, typename L, typename V, typename I, typename N>
struct BloodCellDef
{
	using List = L;
	using Vertices = V;
	using Indices = I;
	using Normals = N;
	inline static constexpr int count = Count;
	inline static constexpr int particlesInCell = ParticlesInCell;
	inline static constexpr int indicesInCell = IndicesInCell;
};

template<int Start, int End, int Length>
struct Spring
{
	inline static constexpr int start = Start;
	inline static constexpr int end = End;
	inline static constexpr float length = Length;
};