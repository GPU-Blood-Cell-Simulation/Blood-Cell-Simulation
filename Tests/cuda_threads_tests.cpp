#include "pch.h"

#include "../src/utilities/cuda_threads.hpp"

TEST(OneInstance, CudaThreadsTests) {
	CudaThreads threads(1);

	EXPECT_EQ(1, threads.blocks);
	EXPECT_EQ(1, threads.threadsPerBlock);
}


TEST(LessThanMaxThreads, CudaThreadsTests) {
	const int instances = CudaThreads::maxThreadsInBlock / 2;

	CudaThreads threads(instances);

	EXPECT_EQ(1, threads.blocks);
	EXPECT_EQ(instances, threads.threadsPerBlock);
}


TEST(MaxThreads, CudaThreadsTests) {
	CudaThreads threads(CudaThreads::maxThreadsInBlock);

	EXPECT_EQ(1, threads.blocks);
	EXPECT_EQ(CudaThreads::maxThreadsInBlock, threads.threadsPerBlock);
}


TEST(TwoTimesMaxThreads, CudaThreadsTests) {
	CudaThreads threads(CudaThreads::maxThreadsInBlock * 2);

	EXPECT_EQ(2, threads.blocks);
	EXPECT_EQ(CudaThreads::maxThreadsInBlock, threads.threadsPerBlock);
}


TEST(TwoAndAHalfTimesMaxThreads, CudaThreadsTests) {
	CudaThreads threads(CudaThreads::maxThreadsInBlock * 2.5);

	EXPECT_EQ(3, threads.blocks);
	EXPECT_EQ(CudaThreads::maxThreadsInBlock, threads.threadsPerBlock);
}
