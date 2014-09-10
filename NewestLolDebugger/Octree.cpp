#include "Octree.h"

//Original code from https://github.com/brandonpelfrey/SimpleOctree/blob/master/Octree.h Brandon Pelfrey http://www.brandonpelfrey.com/blog/
//
//struct Vector3{
//	float x,y,z;
//};

namespace GOAP_GPGPU{
	
	//So we can make this thing work with most types of data we only have to make sure this returns a Vec3 when we call get data member.
	OctreeData::OctreeData(Vector4 dataIn){
		data = dataIn;
	}

	inline Vector4 OctreeData::getData(){
		return data;
	}

	//Octree info
	Octree::Octree(){
		nodeOrigin.x	= 0;
		nodeOrigin.y	= 0;
		nodeOrigin.z	= 0;
		data			= NULL;
		
		for(int i=0;i<8;i++){
			children[i]	=	NULL;
		}
	}

	Octree::Octree(Vector3 originIn,Vector3 halfDimsIn){
		nodeOrigin		=	originIn;
		halfDimentions	=	halfDimsIn;
		data			=	NULL;

		for(int i=0;i<8;i++){
			children[i]	=	NULL;
		}
	}

	Octree::~Octree(){
		delete data;
		data = NULL;
		for(int i=0;i<8;i++){
			delete children[i];
		}
	}

	int Octree::getOctantContainingPoint(const Vector4& point){
		int oct = 0;
		if(point.x >= nodeOrigin.x) oct |= 4;
		if(point.y >= nodeOrigin.y) oct |= 2;
		if(point.z >= nodeOrigin.z) oct |= 1;
		return oct;
	}

	bool Octree::isLeafNode(){
		// We are a leaf if we have no children. Since we either have none, or 
		// all eight, it is sufficient to just check the first.
		return children[0] == NULL;
	}

	/*
		The three cases when inserting to Octree
			a) The node is an interior node tot eh tree, -- Create 8 children, and make a recursive call to add the new node
			b) The node is a leaf (and NO data assigned )-- if this is a leaf nothing much to do, assign the data, move on
			c) Node is a leaf (and HAS data)
	*/
	void Octree::insertNode(OctreeData* newData){
		if(isLeafNode()){
			//If this lead-node does not have any data in it,
			if(data == NULL){
				data = newData;
				return;
			}else{

				//if there's already data in here, then we split it to have 8 children
				//And now insert the old data that was at this node and the new data 
				OctreeData* oldPoint = data;//Save the old data
				data = NULL; // get rid of the data now

				//Init the children 
				for(int i=0;i<8;i++){
					Vector3 newOrigin= nodeOrigin;

					newOrigin.x	+= halfDimentions.x  * (i&4 ? .5f : -.5f);
					newOrigin.y	+= halfDimentions.y  * (i&2 ? .5f : -.5f);
					newOrigin.z	+= halfDimentions.z  * (i&1 ? .5f : -.5f);

					Vector3 newHalfDims = halfDimentions;
					newHalfDims.x *= .5;
					newHalfDims.y *= .5;
					newHalfDims.z *= .5;

					children[i] = new Octree(newOrigin,newHalfDims);
				}

				// Now insert the new and the old points
				children[getOctantContainingPoint(oldPoint->getData())]->insertNode(oldPoint);
				children[getOctantContainingPoint(newData->getData())]->insertNode(newData);
			}
		}else{
			//This means we are at an interior node, i.e. we have children figure-out to which octant the new data member belongs to and insert him.
			children[getOctantContainingPoint(newData->getData())]->insertNode(newData);
		}
	}

	Vector3 nodeOrigin;
	Vector3 halfDimentions;
	
	Octree*			children[8];//Coz
	OctreeData*		data;

	//Collect all the leaf-nodes in to the given std::vector so that it forms an array which is ordered by the locations
	void Octree::convertToArraySTD(std::vector<Vector4> *results){
		//if this is a leaf node
		if(this->data != NULL){
				results->push_back(this->data->getData());
		}else{ // if this is null then we have a friend who has children
			for(int i=0;i<8;i++){
				if(this->children[i] != NULL){
					this->children[i]->convertToArraySTD(results);
				}
			}
		}
	}

	//Collect all the leaf-nodes in to the given std::vector so that it forms an array which is ordered by the locations
	void Octree::convertToArray(Vector4* results,int *counter){ //Counter tells us where to insert the data in our results array on each recursive call
		//if this is a leaf node
		if(this->data != NULL){
			results[*counter] = this->data->getData();
			(*counter)++;
		}else{ // if this is null then we have a friend who has children
			for(int i=0;i<8;i++){
				if(this->children[i] != NULL){
					this->children[i]->convertToArray(results,counter);
				}
			}
		}
	}
}