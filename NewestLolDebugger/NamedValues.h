//Contains definitions of enums used for describing the simulation
#pragma once
#include "Vector4.h"

namespace GOAP_GPGPU{

	////Since we might need to support more than 32 actions for our NPCs we have to make classes that contain enums. 
	//enum NPC_BEHAVIOUR_PATTERNS{};

	enum PLAYER_FRACTION{
		Wolves		= 1<<0,
		Goblins		= 1<<1,
		FishPeople	= 1<<2,
		BirdPeople	= 1<<3,
		reptilians	= 1<<4,
	}; // we can define fraction hostility by adding up these enums 

	enum STATUS_EFFECTS{
		DEFENCE_UP	= 1<<0,
		ATTACK_UP	= 1<<1
	};

	enum ITEM_TYPE{
		HEALTH_POTION_ITEM		= 1<<0,
		MP_UP_POTION_ITEM		= 1<<1,
		DEFENCE_UP_POTION_ITEM	= 1<<2,
		ATTACK_UP_POTION_ITEM	= 1<<3
	};

	enum BELONGS_TO_LIST{
		in_no_list			= 1<<0,
		in_open_list		= 1<<1,
		in_closed_list		= 1<<2,
		in_open_closed_list	= 1<<3,
		is_new_node			= 1<<4,
		is_dead_end			= 1<<5,
	};

	GOAP_GPGPU::Vector4 homingPoints[] = {GOAP_GPGPU::Vector4(0,0,0,0),GOAP_GPGPU::Vector4(0,0,10,0),GOAP_GPGPU::Vector4(0,0,20,0),GOAP_GPGPU::Vector4(0,0,30,0),GOAP_GPGPU::Vector4(0,0,40,0),GOAP_GPGPU::Vector4(0,0,50,0),GOAP_GPGPU::Vector4(0,0,60,0),GOAP_GPGPU::Vector4(0,0,70,0),GOAP_GPGPU::Vector4(0,0,0,0)};

};//End namespace