#pragma once

#include "GOAP_Actions.h"
#include "GOAP_State.h"

namespace GOAP_GPGPU{
	struct AStarNode{
		int h;
		int f;
		int g;
		//AStarNode* parent; are not stored, as this is stored in the shared mem as 1:1 array
		GOAP_ACTION_NAME action;
		GOAP_STATE_COMPACT stateAtNode;

		int padding1;
		int padding2;
	};

	//Remember kids, ORDER is important, when it comes to aligning stuff :D 
	//__align__(16) aligns this so that we will have a good way to pull stuff :D 
	struct  __align__(16) AStarNodeCompact{
		char actionName;
		char isInList;
		char h; // hope to god this wont be more than 127
		char g;
		AStarNodeCompact* parent; 
		GOAP_STATE_COMPACT stateAtNode;
	};

};//endl of GOAP_GPGPU namespace