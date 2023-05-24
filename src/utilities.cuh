#pragma once
#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__host__ __device__ inline float3 operator*(float a, float3 v)
{
	return make_float3(a * v.x, a * v.y, a * v.z);
}

__host__ __device__ inline float3 operator*(float3 a, float3 b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ inline float3 operator+(float3 a, float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator/(float3 v, float a)
{
	return make_float3(v.x / a, v.y / a, v.z / a);
}

__host__ __device__ inline float dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float3 cross(float3 u, float3 v)
{
	return make_float3(u.y * v.z - u.z * v.y,
		u.z * v.x - u.x * v.z,
		u.x * v.y - u.y * v.x);
}

__host__ __device__ inline float length(float3 v)
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__ inline float3 normalize(float3 v) // versor
{
	return v / sqrtf(dot(v, v));
}