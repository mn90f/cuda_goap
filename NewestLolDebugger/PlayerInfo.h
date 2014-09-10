//Helps represent a player in the simulation
#pragma once
#include "Vector3.h"
#include "NamedValues.h"

namespace GOAP_GPGPU{
	
	class PlayerInfo{
	public:
		Vector3 position;
		int health;
		int mp;
		PLAYER_FRACTION fraction;
		int state;
		int statusEffects;
		int playerId;
		bool checked;		//A helper var we need when we are trying to rearrange this guy.

		PlayerInfo(){
			checked = false;
		}
	private:
	};

};
