#pragma once

#include <cmath>


class CudaThreads
{
public:
	// TODO :
	// anything above 768 threads (25 warps) trigger an error
	// 'too many resources requested for launch'
	// maybe possible to solve
	static const int maxThreads = 768;

	const int threadsPerBlock;
	const int blocks;

	CudaThreads(int instances) :
		threadsPerBlock(instances > maxThreads ? maxThreads : instances),
		blocks(std::ceil(static_cast<float>(instances) / threadsPerBlock))
	{}
};