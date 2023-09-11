#pragma once

#include <cmath>
#include <stdexcept>


class CudaThreads
{
public:
	// TODO :
	// anything above 768 threads (25 warps) trigger an error
	// 'too many resources requested for launch'
	// maybe possible to solve
	static const int maxThreadsInBlock = 768;
	static const int threadsInWarp = 32;

	const int threadsPerBlock;
	const int blocks;

	CudaThreads(int instances) :
		threadsPerBlock(instances > maxThreadsInBlock ? maxThreadsInBlock : instances),
		blocks(std::ceil(static_cast<float>(instances) / threadsPerBlock))
	{}

	CudaThreads(unsigned int threadsPerBlock, unsigned int blocks) :
		threadsPerBlock(threadsPerBlock), blocks(blocks)
	{
		if (threadsPerBlock > maxThreadsInBlock)
			throw std::invalid_argument("Too many threads per block");
	}
};