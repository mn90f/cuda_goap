#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <curand_kernel.h>

namespace GOAP_GPGPU{

	class Vector3{
		public:
			__device__ __host__ Vector3(){}

			__device__ __host__ Vector3(float inX, float inY, float inZ){
				x	= inX;
				y	= inY;
				z	= inZ;
			}
			float x,y,z;
	};

}