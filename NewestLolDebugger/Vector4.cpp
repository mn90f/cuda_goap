#include "Vector4.h"

namespace GOAP_GPGPU{

	Vector4::Vector4(){}

	Vector4::Vector4(float inX, float inY, float inZ, float inW){
		x	= inX;
		y	= inY;
		z	= inZ;
		w	= inW;
	}

	Vector4::Vector4(Vector3 invec3,float inW){
		x	= invec3.x;
		y	= invec3.y;
		z	= invec3.z;
		w	= inW;
	}
}