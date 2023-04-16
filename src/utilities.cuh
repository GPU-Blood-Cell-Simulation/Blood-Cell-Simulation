#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__host__ __device__ float3 operator*(float a, float3 v);

__host__ __device__ float3 operator*(float3 a, float3 b);

__host__ __device__ float3 operator+(float3 a, float3 b);

__host__ __device__ float3 operator-(float3 a, float3 b);

__host__ __device__ float3 operator/(float3 v, float a);

__host__ __device__ float dot(float3 a, float3 b);

__host__ __device__ float3 cross(float3 u, float3 v);

__host__ __device__ float length(float3 v);

__host__ __device__ float3 normalize(float3 v); // versor