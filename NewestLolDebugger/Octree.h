#pragma once
#include <iostream>
#include <vector>
#include "Vector3.h"
#include "Vector4.h"

namespace GOAP_GPGPU{
	//Original code from https://github.com/brandonpelfrey/SimpleOctree/blob/master/Octree.h Brandon Pelfrey http://www.brandonpelfrey.com/blog/
	//
	//struct Vector3{
	//	float x,y,z;
	//};

	//So we can make this thing work with most types of data we only have to make sure this returns a Vec3 when we call get data member.
	class OctreeData{
	public:
		OctreeData(){};
		OctreeData(Vector4 dataIn);
		inline Vector4 getData();
	private:
		Vector4 data;
	};

	class Octree{
	public:
		Octree();
		Octree(Vector3 originIn,Vector3 halfDimsIn);
		~Octree();

		int getOctantContainingPoint(const Vector4& point);
		bool isLeafNode();
		void insertNode(OctreeData* newData);
		void convertToArraySTD(std::vector<Vector4> *results);
		void convertToArray(Vector4* results,int *counter);

		Vector3 nodeOrigin;
		Vector3 halfDimentions;
	
		Octree*			children[8];//Coz
		OctreeData*		data;
	};	
};