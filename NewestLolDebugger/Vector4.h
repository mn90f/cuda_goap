#pragma once
#include "Vector3.h"

namespace GOAP_GPGPU{
	class Vector4{
		public:
			Vector4(float inx, float iny, float inz, float inW);
			Vector4(Vector3 invec3,float inW);
			Vector4();	
			float w,x,y,z;
	};
}