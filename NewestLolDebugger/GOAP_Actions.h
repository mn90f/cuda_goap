#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#include "GOAP_State.h"
#include "GOAP_State.h"
#include "AgentInfo.h"

namespace GOAP_GPGPU{
	/*
		fixed-size array of symbols, implemented as key-value pairs. Keys are represented by enumerated world properties.
		Values are a union of possible data types. 
		//Orkin 2004  Symbolic Representation of Game World State: Toward Real-Time Planning in Games 

		but we are going with one big struct where a value of 
	*/

	//Actions our NPCs can do
	enum GOAP_ACTION_NAME{
		Action_Basic_Stab,					Action_Throw_Stone,
		Action_Bite_Attack,					Action_Knife_Stab,
		Action_Spear_Stab,					Action_Sword_Slash,
		Action_Dive_Attack,					Action_Venom_Bite_Attack,
		Action_Increase_Defense,			Action_Increase_Attack,
		Action_Decrease_Defense,			Action_Decrease_Attack,
		Action_Heal_HP,						Action_Heal_MP,
		Action_Steal_HP,					Action_Heal_Ally_HP,
		Action_Heal_Ally_MP,				Action_Ally_Attack_Up,
		Action_Ally_Defense_Up,				Action_Burn_Enemy,
		Action_Freeze_Enemy,				Action_Quick_Attack,
		Action_Blade_Attack,				Action_Fang_Attack,
		Action_Poison_Attack,				Action_Fart,
		Action_Attack_With_Long_Sword,		Action_Equip_Staff,
		Action_Equip_Knife,					Action_Equip_Spear,
		Action_Equip_Long_Sword,			Action_Pickup_HP_Potion,
		Action_Pick_Up_MP,					Action_Consume_HP_Potion,
		Action_Consume_MP_Potion,			Action_Find_Stone,
		Action_Get_To_Weapon_Attack_Range,	Action_Get_To_Spell_Attack_Range,
		Action_Get_To_Ranged_Attack_Range,	Action_Get_To_Melee_Attack_Range,
		Action_Get_To_Ally_Spell_Range,		Action_Retreat_To_Safety,

		NUMBER_OF_ACTIONS,		//SHOULD BE TWO BEFORE LAST
		NO_ACTION,
		START_ACTION			//SHOULD BE LAST :)
	};

	struct GoapAction{
		GOAP_STATE preCondition;
		GOAP_STATE postCondition;
		GOAP_ACTION_NAME actionName;
		int actionAllowedTo;
		int g;
	};

	struct GoapAction_compact{
		GOAP_STATE_COMPACT preCondition;
		GOAP_STATE_COMPACT postCondition;
		GOAP_ACTION_NAME actionName;
		int actionAllowedTo;
		int g;
	};

	__device__ __host__ void GoapActionToCompact(GoapAction* goapAction,GoapAction_compact *goapActionCompact){
		goapActionCompact->actionName		= goapAction->actionName;
		goapActionCompact->g				= goapAction->g;
		goapActionCompact->actionAllowedTo	= goapAction->actionAllowedTo;

		GoapStateToCompact(&goapAction->preCondition,&goapActionCompact->preCondition);
		GoapStateToCompact(&goapAction->postCondition,&goapActionCompact->postCondition);
	};

	__device__ __host__ void GoapActionCompactToAction(GoapAction_compact *goapActionCompact,GoapAction* goapAction){
		goapAction->actionName		= goapActionCompact->actionName;
		goapAction->g				= goapActionCompact->g;
		goapAction->actionAllowedTo	= goapActionCompact->actionAllowedTo;

		CompactToGoapState(&goapActionCompact->preCondition,&goapAction->preCondition);
		CompactToGoapState(&goapActionCompact->postCondition,&goapAction->postCondition);
	};
	
	GoapAction			allGoapActions[NUMBER_OF_ACTIONS];
	GoapAction_compact	allGoapCompactActions[NUMBER_OF_ACTIONS];

	__host__ void defineGoapCompactActions(){
		for(int i = 0;i<NUMBER_OF_ACTIONS;i++){
			GoapAction			goapAction;
			GoapAction_compact	goapActionCompact;

			goapAction			= allGoapActions[i];
			GoapActionToCompact(&goapAction,&goapActionCompact);


			allGoapCompactActions[i] = goapActionCompact;
		}
	}

	__host__ void defineGoapActions(){
		GOAP_STATE tmpState;
		GOAP_STATE defaultState;

		allGoapActions[Action_Basic_Stab].actionName						= Action_Basic_Stab;
		tmpState.KNIFE_EQUIPPED												= 1;
		tmpState.HAS_LIVE_ENEMY												= 1;
		tmpState.IS_ENEMY_FRACTION											= 1;
		tmpState.ENEMY_IN_WEAPON_ATTACK_RANGE								= 1;
		allGoapActions[Action_Basic_Stab].preCondition						= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.HAS_LIVE_ENEMY												= 0;
		allGoapActions[Action_Basic_Stab].postCondition						= tmpState;
		tmpState															= defaultState;//Clear tmpState
		allGoapActions[Action_Basic_Stab].g									= 20;


		allGoapActions[Action_Throw_Stone].actionName						= Action_Throw_Stone;
		tmpState.STONE_EQUIPPED												= 1;
		tmpState.HAS_LIVE_ENEMY												= 1;
		tmpState.IS_ENEMY_FRACTION											= 1;  
		tmpState.ENEMY_IN_RANGED_ATTACK_RANGE								= 1;
		allGoapActions[Action_Throw_Stone].preCondition						= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.STONE_EQUIPPED												= 0;
		tmpState.HAS_LIVE_ENEMY												= 0;
		allGoapActions[Action_Throw_Stone].postCondition					= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Throw_Stone].g								=	10;

		allGoapActions[Action_Bite_Attack].actionName						= Action_Bite_Attack;
		tmpState.HAS_LIVE_ENEMY												= 0;
		tmpState.IS_ENEMY_FRACTION											= 1;
		tmpState.ENEMY_IN_MELEE_RANGE										= 1;
		allGoapActions[Action_Bite_Attack].preCondition						= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.HAS_LIVE_ENEMY												= 0;
		allGoapActions[Action_Bite_Attack].postCondition					= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Bite_Attack].g								= 10;

		allGoapActions[Action_Knife_Stab].actionName	= Action_Knife_Stab;
		tmpState.KNIFE_EQUIPPED							= 1;
		tmpState.HAS_LIVE_ENEMY							= 1;
		tmpState.IS_ENEMY_FRACTION						= 1;
		tmpState.ENEMY_IN_WEAPON_ATTACK_RANGE			= 1;
		allGoapActions[Action_Knife_Stab].preCondition	= tmpState;
		tmpState										= defaultState; //Clear tmpState
		tmpState.HAS_LIVE_ENEMY							= 0;
		allGoapActions[Action_Knife_Stab].postCondition	= tmpState;
		tmpState										= defaultState; //Clear tmpState
		allGoapActions[Action_Knife_Stab].g				= 10;

		allGoapActions[Action_Spear_Stab].actionName						= Action_Spear_Stab;
		tmpState.SPEAR_EQUIPPED												= 1;
		tmpState.HAS_LIVE_ENEMY												= 1;
		tmpState.IS_ENEMY_FRACTION											= 1;
		tmpState.ENEMY_IN_WEAPON_ATTACK_RANGE								= 1;
		allGoapActions[Action_Spear_Stab].preCondition						= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.HAS_LIVE_ENEMY												= 0;
		allGoapActions[Action_Spear_Stab].postCondition						= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Spear_Stab].g									= 0;

		allGoapActions[Action_Sword_Slash].actionName						= Action_Sword_Slash;
		tmpState.SWORD_EQUIPPED												= 1;
		tmpState.HAS_LIVE_ENEMY												= 1;
		tmpState.IS_ENEMY_FRACTION											= 1;
		tmpState.ENEMY_IN_WEAPON_ATTACK_RANGE								= 1;
		allGoapActions[Action_Sword_Slash].preCondition						= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.HAS_LIVE_ENEMY												= 0;
		allGoapActions[Action_Sword_Slash].postCondition					= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Sword_Slash].g								= 10;

		allGoapActions[Action_Dive_Attack].actionName						= Action_Dive_Attack;
		tmpState.HAS_LIVE_ENEMY												= 1;
		tmpState.IS_ENEMY_FRACTION											= 1;
		tmpState.ENEMY_IN_SPELL_ATTACK_RANGE								= 1;
		allGoapActions[Action_Dive_Attack].preCondition						= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.HAS_LIVE_ENEMY												= 0;
		allGoapActions[Action_Dive_Attack].postCondition					= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Dive_Attack].g								=0;

		allGoapActions[Action_Venom_Bite_Attack].actionName					= Action_Venom_Bite_Attack;
		tmpState.HAS_LIVE_ENEMY												= 1;
		tmpState.IS_ENEMY_FRACTION											= 1;
		tmpState.ENEMY_IN_MELEE_RANGE										= 1;
		allGoapActions[Action_Venom_Bite_Attack].preCondition				= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.HAS_LIVE_ENEMY												= 0;
		allGoapActions[Action_Venom_Bite_Attack].postCondition				= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Venom_Bite_Attack].g							=0;

		allGoapActions[Action_Increase_Defense].actionName					= Action_Increase_Defense;
		tmpState.SELF_DEFENSE_UP											= 0;
		allGoapActions[Action_Increase_Defense].preCondition				= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.SELF_DEFENSE_UP											= 1;
		allGoapActions[Action_Increase_Defense].postCondition				= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Increase_Defense].g							=0;

		allGoapActions[Action_Increase_Attack].actionName					= Action_Increase_Attack;
		tmpState.SELF_ATTACK_UP												= 0;
		allGoapActions[Action_Increase_Attack].preCondition					= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.SELF_ATTACK_UP												= 0;
		allGoapActions[Action_Increase_Attack].postCondition				= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Increase_Attack].g							=0;

		allGoapActions[Action_Decrease_Defense].actionName					= Action_Decrease_Defense;
		tmpState.ENEMY_DEFENSE_UP											= 1;
		tmpState.HAS_LIVE_ENEMY												= 1;
		tmpState.IS_ENEMY_FRACTION											= 1;
		tmpState.ENEMY_IN_SPELL_ATTACK_RANGE								= 1;
		allGoapActions[Action_Decrease_Defense].preCondition				= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.ENEMY_DEFENSE_UP											= 0;
		allGoapActions[Action_Decrease_Defense].postCondition				= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Decrease_Defense].g							= 0;

		allGoapActions[Action_Decrease_Attack].actionName					= Action_Decrease_Attack;
		tmpState.ENEMY_ATTACK_UP											= 1;
		tmpState.HAS_LIVE_ENEMY												= 1;
		tmpState.IS_ENEMY_FRACTION											= 1;
		tmpState.ENEMY_IN_SPELL_ATTACK_RANGE								= 1;
		allGoapActions[Action_Decrease_Attack].preCondition					= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.ENEMY_ATTACK_UP											= 0;
		allGoapActions[Action_Decrease_Attack].postCondition				= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Decrease_Attack].g							= 0;

		allGoapActions[Action_Heal_HP].actionName							= Action_Heal_HP;
		tmpState.SELF_RISK_OF_DEATH											= 1;
		allGoapActions[Action_Heal_HP].preCondition							= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.SELF_RISK_OF_DEATH											= 0;
		allGoapActions[Action_Heal_HP].postCondition						= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Heal_HP].g									= 0;

		allGoapActions[Action_Heal_MP].actionName							= Action_Heal_MP;
		tmpState.SELF_MP_LOW												= 1;
		allGoapActions[Action_Heal_MP].preCondition							= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.SELF_MP_LOW												= 0;
		allGoapActions[Action_Heal_MP].postCondition						= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Heal_MP].g									=0;

		allGoapActions[Action_Steal_HP].actionName							= Action_Steal_HP;
		tmpState.STAFF_EQUIPPED												= 1;
		tmpState.HAS_LIVE_ENEMY												= 1;
		tmpState.IS_ENEMY_FRACTION											= 1;
		tmpState.SELF_HP_LOW												= 1;
		tmpState.ENEMY_IN_SPELL_ATTACK_RANGE								= 1;
		allGoapActions[Action_Steal_HP].preCondition						= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.SELF_HP_LOW												= 0;
		tmpState.HAS_LIVE_ENEMY												= 1;
		allGoapActions[Action_Steal_HP].postCondition						= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Steal_HP].g									=0;

		allGoapActions[Action_Heal_Ally_HP].actionName						= Action_Heal_Ally_HP;
		tmpState.ALLY_HP_LOW												= 1;
		tmpState.ALLY_IN_SPELL_RANGE										= 1;
		allGoapActions[Action_Heal_Ally_HP].preCondition					= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.ALLY_HP_LOW												= 0;
		allGoapActions[Action_Heal_Ally_HP].postCondition					= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Heal_Ally_HP].g								=0;

		allGoapActions[Action_Heal_Ally_MP].actionName						= Action_Heal_Ally_MP;
		tmpState.ALLY_MP_LOW												= 1;
		tmpState.ALLY_IN_SPELL_RANGE										= 1;
		allGoapActions[Action_Heal_Ally_MP].preCondition					= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.ALLY_MP_LOW												= 0;
		allGoapActions[Action_Heal_Ally_MP].postCondition					= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Heal_Ally_MP].g								=0;

		allGoapActions[Action_Ally_Attack_Up].actionName					= Action_Ally_Attack_Up;
		tmpState.ALLY_ATTACK_UP												= 0;
		tmpState.ALLY_IN_SPELL_RANGE										= 1;
		allGoapActions[Action_Ally_Attack_Up].preCondition					= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.ALLY_ATTACK_UP												= 1;
		allGoapActions[Action_Ally_Attack_Up].postCondition					= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Ally_Attack_Up].g								=0;

		allGoapActions[Action_Ally_Defense_Up].actionName					= Action_Ally_Defense_Up;
		tmpState.ALLY_DEFENSE_UP											= 0;
		tmpState.ALLY_IN_SPELL_RANGE										= 1;
		allGoapActions[Action_Ally_Defense_Up].preCondition					= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.ALLY_DEFENSE_UP											= 1;
		allGoapActions[Action_Ally_Defense_Up].postCondition				= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Ally_Defense_Up].g							=0;

		allGoapActions[Action_Burn_Enemy].actionName						= Action_Burn_Enemy;
		tmpState.STAFF_EQUIPPED												= 1;
		tmpState.HAS_LIVE_ENEMY												= 1;
		tmpState.IS_ENEMY_FRACTION											= 1;
		tmpState.ENEMY_IN_SPELL_ATTACK_RANGE								= 1;
		allGoapActions[Action_Burn_Enemy].preCondition						= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.HAS_LIVE_ENEMY												= 0;
		allGoapActions[Action_Burn_Enemy].postCondition						= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Burn_Enemy].g									=0;

		allGoapActions[Action_Freeze_Enemy].actionName						= Action_Freeze_Enemy;
		tmpState.STAFF_EQUIPPED												= 1;
		tmpState.HAS_LIVE_ENEMY												= 1;
		tmpState.IS_ENEMY_FRACTION											= 1;
		tmpState.ENEMY_IN_SPELL_ATTACK_RANGE								= 1;
		allGoapActions[Action_Freeze_Enemy].preCondition					= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.HAS_LIVE_ENEMY												= 0;
		allGoapActions[Action_Freeze_Enemy].postCondition					= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Freeze_Enemy].g								= 0;

		allGoapActions[Action_Quick_Attack].actionName						= Action_Quick_Attack;
		tmpState.HAS_LIVE_ENEMY												= 1;
		tmpState.IS_ENEMY_FRACTION											= 1;
		tmpState.ENEMY_IN_MELEE_RANGE										= 1;
		allGoapActions[Action_Quick_Attack].preCondition					= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.HAS_LIVE_ENEMY												= 0;
		allGoapActions[Action_Quick_Attack].postCondition					= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Quick_Attack].g								= 10;

		allGoapActions[Action_Blade_Attack].actionName						= Action_Blade_Attack;
		tmpState.BLADE_EQUIPPED												= 1;
		tmpState.HAS_LIVE_ENEMY												= 1;
		tmpState.IS_ENEMY_FRACTION											= 1;
		tmpState.ENEMY_IN_WEAPON_ATTACK_RANGE								= 1;
		allGoapActions[Action_Blade_Attack].preCondition					= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.HAS_LIVE_ENEMY												= 0;
		allGoapActions[Action_Blade_Attack].postCondition					= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Blade_Attack].g								= 10;

		allGoapActions[Action_Fang_Attack].actionName						= Action_Fang_Attack;
		tmpState.HAS_LIVE_ENEMY												= 1;
		tmpState.IS_ENEMY_FRACTION											= 1;
		tmpState.ENEMY_IN_MELEE_RANGE										= 1;
		allGoapActions[Action_Fang_Attack].preCondition						= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.HAS_LIVE_ENEMY												= 0;
		allGoapActions[Action_Fang_Attack].postCondition					= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Fang_Attack].g								= 10;

		allGoapActions[Action_Poison_Attack].actionName						= Action_Poison_Attack;
		tmpState.HAS_LIVE_ENEMY												= 1;
		tmpState.IS_ENEMY_FRACTION											= 1;
		tmpState.ENEMY_IN_SPELL_ATTACK_RANGE								= 1;
		allGoapActions[Action_Poison_Attack].preCondition					= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.HAS_LIVE_ENEMY												= 0;
		allGoapActions[Action_Poison_Attack].postCondition					= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Poison_Attack].g								= 10;

		allGoapActions[Action_Fart].actionName								= Action_Fart;
		tmpState.STAFF_EQUIPPED												= 1;
		tmpState.HAS_LIVE_ENEMY												= 1;
		tmpState.IS_ENEMY_FRACTION											= 1;
		tmpState.ENEMY_IN_SPELL_ATTACK_RANGE								= 1;
		allGoapActions[Action_Fart].preCondition							= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.HAS_LIVE_ENEMY												= 0;
		allGoapActions[Action_Fart].postCondition							= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Fart].g										= 10;

		allGoapActions[Action_Attack_With_Long_Sword].actionName			= Action_Attack_With_Long_Sword;
		tmpState.LONG_SWORD_EQUIPPED										= 1;
		tmpState.HAS_LIVE_ENEMY												= 1;
		tmpState.IS_ENEMY_FRACTION											= 1;
		tmpState.ENEMY_IN_SPELL_ATTACK_RANGE								= 1;
		allGoapActions[Action_Attack_With_Long_Sword].preCondition			= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.HAS_LIVE_ENEMY												= 0;
		allGoapActions[Action_Attack_With_Long_Sword].postCondition			= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Attack_With_Long_Sword].g						= 10;

		allGoapActions[Action_Equip_Staff].actionName						= Action_Equip_Staff;
		tmpState.STAFF_EQUIPPED												= 0;
		allGoapActions[Action_Equip_Staff].preCondition						= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.STAFF_EQUIPPED												= 1;
		allGoapActions[Action_Equip_Staff].postCondition					= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Equip_Staff].g								= 10;

		allGoapActions[Action_Equip_Knife].actionName						= Action_Equip_Knife;
		tmpState.KNIFE_EQUIPPED												= 0;
		allGoapActions[Action_Equip_Knife].preCondition						= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.KNIFE_EQUIPPED												= 1;
		allGoapActions[Action_Equip_Knife].postCondition					= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Equip_Knife].g								= 10;

		allGoapActions[Action_Equip_Spear].actionName						= Action_Equip_Spear;
		tmpState.SPEAR_EQUIPPED												= 0;
		allGoapActions[Action_Equip_Spear].preCondition						= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.SPEAR_EQUIPPED												= 1;
		allGoapActions[Action_Equip_Spear].postCondition					= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Equip_Spear].g								= 10;

		allGoapActions[Action_Equip_Long_Sword].actionName					= Action_Equip_Long_Sword;
		tmpState.LONG_SWORD_EQUIPPED										= 0;
		allGoapActions[Action_Equip_Long_Sword].preCondition				= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.LONG_SWORD_EQUIPPED										= 1;
		allGoapActions[Action_Equip_Long_Sword].postCondition				= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Equip_Long_Sword].g							= 0;

		allGoapActions[Action_Pickup_HP_Potion].actionName					= Action_Pickup_HP_Potion;
		tmpState.HAS_HP_POTION												= 0;
		allGoapActions[Action_Pickup_HP_Potion].preCondition				= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.HAS_HP_POTION												= 1;
		allGoapActions[Action_Pickup_HP_Potion].postCondition				= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Pickup_HP_Potion].g							=0;

		allGoapActions[Action_Pick_Up_MP].actionName						= Action_Pick_Up_MP;
		tmpState.HAS_MP_POTION												= 0;
		allGoapActions[Action_Pick_Up_MP].preCondition						= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.HAS_MP_POTION												= 1;
		allGoapActions[Action_Pick_Up_MP].postCondition						= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Pick_Up_MP].g									=0;

		allGoapActions[Action_Consume_HP_Potion].actionName					= Action_Consume_HP_Potion;
		tmpState.HAS_HP_POTION												= 1;
		tmpState.SELF_HP_CRITICAL											= 1;
		allGoapActions[Action_Consume_HP_Potion].preCondition				= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.HAS_HP_POTION												= 0;
		tmpState.SELF_HP_CRITICAL											= 0;
		allGoapActions[Action_Consume_HP_Potion].postCondition				= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Consume_HP_Potion].g							=0;

		allGoapActions[Action_Consume_MP_Potion].actionName					= Action_Consume_MP_Potion;
		tmpState.HAS_MP_POTION												= 1;
		tmpState.SELF_MP_LOW												= 1;
		allGoapActions[Action_Consume_MP_Potion].preCondition				= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.HAS_MP_POTION												= 0;
		tmpState.SELF_MP_LOW												= 0;
		allGoapActions[Action_Consume_MP_Potion].postCondition				= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Consume_MP_Potion].g							= 0;

		allGoapActions[Action_Find_Stone].actionName						= Action_Find_Stone;
		tmpState.STONE_EQUIPPED												= 0;
		allGoapActions[Action_Find_Stone].preCondition						= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.STONE_EQUIPPED												= 1;
		allGoapActions[Action_Find_Stone].postCondition						= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Find_Stone].g									= 5;

		allGoapActions[Action_Get_To_Weapon_Attack_Range].actionName		= Action_Get_To_Weapon_Attack_Range;
		tmpState.ENEMY_IN_WEAPON_ATTACK_RANGE								= 0;
		allGoapActions[Action_Get_To_Weapon_Attack_Range].preCondition		= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.ENEMY_IN_WEAPON_ATTACK_RANGE								= 1;
		allGoapActions[Action_Get_To_Weapon_Attack_Range].postCondition		= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Get_To_Weapon_Attack_Range].g					=0;

		allGoapActions[Action_Get_To_Spell_Attack_Range].actionName			= Action_Get_To_Spell_Attack_Range;
		tmpState.ENEMY_IN_SPELL_ATTACK_RANGE								= 0;
		allGoapActions[Action_Get_To_Spell_Attack_Range].preCondition		= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.ENEMY_IN_SPELL_ATTACK_RANGE								= 1;
		allGoapActions[Action_Get_To_Spell_Attack_Range].postCondition		= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Get_To_Spell_Attack_Range].g					=0;

		allGoapActions[Action_Get_To_Ranged_Attack_Range].actionName		= Action_Get_To_Ranged_Attack_Range;
		tmpState.ENEMY_IN_RANGED_ATTACK_RANGE								= 0;
		allGoapActions[Action_Get_To_Ranged_Attack_Range].preCondition		= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.ENEMY_IN_RANGED_ATTACK_RANGE								= 1;
		allGoapActions[Action_Get_To_Ranged_Attack_Range].postCondition		= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Get_To_Ranged_Attack_Range].g					=0;

		allGoapActions[Action_Get_To_Melee_Attack_Range].actionName			= Action_Get_To_Melee_Attack_Range;
		tmpState.ENEMY_IN_MELEE_RANGE										= 0;
		allGoapActions[Action_Get_To_Melee_Attack_Range].preCondition		= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.ENEMY_IN_MELEE_RANGE										= 1;
		allGoapActions[Action_Get_To_Melee_Attack_Range].postCondition		= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Get_To_Melee_Attack_Range].g					=0;

		allGoapActions[Action_Get_To_Ally_Spell_Range].actionName			= Action_Get_To_Ally_Spell_Range;
		tmpState.ALLY_IN_SPELL_RANGE										= 0;
		allGoapActions[Action_Get_To_Ally_Spell_Range].preCondition			= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.ALLY_IN_SPELL_RANGE										= 1;
		allGoapActions[Action_Get_To_Ally_Spell_Range].postCondition		= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Get_To_Ally_Spell_Range].g					=0;

		allGoapActions[Action_Retreat_To_Safety].actionName					= Action_Retreat_To_Safety;
		tmpState.SELF_HP_CRITICAL											= 1;
		tmpState.HAS_LIVE_ENEMY												= 1;
		tmpState.IS_ENEMY_FRACTION											= 1;
		allGoapActions[Action_Retreat_To_Safety].preCondition				= tmpState;
		tmpState															= defaultState; //Clear tmpState
		tmpState.SELF_HP_CRITICAL											= 0;
		allGoapActions[Action_Retreat_To_Safety].postCondition				= tmpState;
		tmpState															= defaultState; //Clear tmpState
		allGoapActions[Action_Retreat_To_Safety].g							=0;

		//Action allow status
		allGoapActions[	Action_Basic_Stab	].actionAllowedTo	= 	2147483647;
		allGoapActions[	Action_Throw_Stone	].actionAllowedTo	= 	2147483647;
		allGoapActions[	Action_Bite_Attack	].actionAllowedTo	= 	2013233147;
		allGoapActions[	Action_Knife_Stab	].actionAllowedTo	= 	2147483647;
		allGoapActions[	Action_Spear_Stab	].actionAllowedTo	= 	16384;
		allGoapActions[	Action_Sword_Slash	].actionAllowedTo	= 	2130706432;
		allGoapActions[	Action_Dive_Attack	].actionAllowedTo	= 	268435456;
		allGoapActions[	Action_Venom_Bite_Attack	].actionAllowedTo	= 	536870912;
		allGoapActions[	Action_Increase_Defense	].actionAllowedTo	= 	4032;
		allGoapActions[	Action_Increase_Attack	].actionAllowedTo	= 	4032;
		allGoapActions[	Action_Decrease_Defense	].actionAllowedTo	= 	4032;
		allGoapActions[	Action_Decrease_Attack	].actionAllowedTo	= 	4032;
		allGoapActions[	Action_Heal_HP	].actionAllowedTo	= 	2147483647; // used to be 63
		allGoapActions[	Action_Heal_MP	].actionAllowedTo	= 	63;
		allGoapActions[	Action_Steal_HP	].actionAllowedTo	= 	16515072;
		allGoapActions[	Action_Heal_Ally_HP	].actionAllowedTo	= 	4095;
		allGoapActions[	Action_Heal_Ally_MP	].actionAllowedTo	= 	4095;
		allGoapActions[	Action_Ally_Attack_Up	].actionAllowedTo	= 	4032;
		allGoapActions[	Action_Ally_Defense_Up	].actionAllowedTo	= 	4032;
		allGoapActions[	Action_Burn_Enemy	].actionAllowedTo	= 	7340032;
		allGoapActions[	Action_Freeze_Enemy	].actionAllowedTo	= 	7340032;
		allGoapActions[	Action_Quick_Attack	].actionAllowedTo	= 	258048;
		allGoapActions[	Action_Blade_Attack	].actionAllowedTo	= 	20480;
		allGoapActions[	Action_Fang_Attack	].actionAllowedTo	= 	8192;
		allGoapActions[	Action_Poison_Attack	].actionAllowedTo	= 	131072;
		allGoapActions[	Action_Fart	].actionAllowedTo	= 	4194304;
		allGoapActions[	Action_Attack_With_Long_Sword	].actionAllowedTo	= 	134217728;
		allGoapActions[	Action_Equip_Staff	].actionAllowedTo	= 	16515072;
		allGoapActions[	Action_Equip_Knife	].actionAllowedTo	= 	2147483647;
		allGoapActions[	Action_Equip_Spear	].actionAllowedTo	= 	20480;
		allGoapActions[	Action_Equip_Long_Sword	].actionAllowedTo	= 	134217728;
		allGoapActions[	Action_Pickup_HP_Potion	].actionAllowedTo	= 	2147483647;
		allGoapActions[	Action_Pick_Up_MP	].actionAllowedTo	= 	2147483647;
		allGoapActions[	Action_Consume_HP_Potion	].actionAllowedTo	= 	2147483647;
		allGoapActions[	Action_Consume_MP_Potion	].actionAllowedTo	= 	2147483647;
		allGoapActions[	Action_Find_Stone	].actionAllowedTo	= 	2147483647;
		allGoapActions[	Action_Get_To_Weapon_Attack_Range	].actionAllowedTo	= 	2147483647;
		allGoapActions[	Action_Get_To_Spell_Attack_Range	].actionAllowedTo	= 	2147483647;
		allGoapActions[	Action_Get_To_Ranged_Attack_Range	].actionAllowedTo	= 	2147483647;
		allGoapActions[	Action_Get_To_Melee_Attack_Range	].actionAllowedTo	= 	2147483647;
		allGoapActions[	Action_Get_To_Ally_Spell_Range	].actionAllowedTo	= 	2147483647;
		allGoapActions[	Action_Retreat_To_Safety	].actionAllowedTo	= 	2147483647;
 

		defineGoapCompactActions();
	}

};//GOAP_GPGPU:: Namespace end