#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>


//The functions in here should be in-sync with the order and the counts in the GOAP_STATE

namespace GOAP_GPGPU{
	
	const int NUMBER_OF_STATES = 28;
	struct GOAP_STATE{
		/*
			int values representing states, 1 for true, 0 false INT_MIN to ignore

			Every time you make changes to this struct
				a) Update NUMBER_OF_STATES
				b) If NUMBER_OF_STATES has more than 31 states update GOAP_STATE_COMPACT
		*/
		int KNIFE_EQUIPPED;					int STONE_EQUIPPED;		
		int SWORD_EQUIPPED;					int STAFF_EQUIPPED;
		int BLADE_EQUIPPED;					int SPEAR_EQUIPPED;
		int LONG_SWORD_EQUIPPED;			

		int ALLY_DEFENSE_UP;				int ALLY_ATTACK_UP;

		int SELF_DEFENSE_UP;				int SELF_ATTACK_UP;

		int ENEMY_DEFENSE_UP;				int ENEMY_ATTACK_UP;

		int HAS_LIVE_ENEMY;					int IS_ENEMY_FRACTION;

		int SELF_HP_LOW;					int SELF_MP_LOW;
		int SELF_HP_CRITICAL;				int SELF_RISK_OF_DEATH;//19

		int ALLY_MP_LOW;					int ALLY_HP_LOW;

		int HAS_MP_POTION;					int HAS_HP_POTION;
 
		int ALLY_IN_SPELL_RANGE;			int ENEMY_IN_WEAPON_ATTACK_RANGE;
		int ENEMY_IN_SPELL_ATTACK_RANGE;	int ENEMY_IN_RANGED_ATTACK_RANGE;
		int ENEMY_IN_MELEE_RANGE;

		  GOAP_STATE():
							KNIFE_EQUIPPED(INT_MIN),				STONE_EQUIPPED(INT_MIN),
							SWORD_EQUIPPED(INT_MIN),				STAFF_EQUIPPED(INT_MIN),
							BLADE_EQUIPPED(INT_MIN),				SPEAR_EQUIPPED(INT_MIN),
							LONG_SWORD_EQUIPPED(INT_MIN),			ALLY_DEFENSE_UP(INT_MIN),
							ALLY_ATTACK_UP(INT_MIN),				SELF_DEFENSE_UP(INT_MIN),
							SELF_ATTACK_UP(INT_MIN),				ENEMY_DEFENSE_UP(INT_MIN),
							ENEMY_ATTACK_UP(INT_MIN),				HAS_LIVE_ENEMY(INT_MIN),
							IS_ENEMY_FRACTION(INT_MIN),				SELF_HP_LOW(INT_MIN),
							SELF_MP_LOW(INT_MIN),					SELF_HP_CRITICAL(INT_MIN),
							SELF_RISK_OF_DEATH(INT_MIN),			ALLY_MP_LOW(INT_MIN),
							ALLY_HP_LOW(INT_MIN),					HAS_MP_POTION(INT_MIN),
							HAS_HP_POTION(INT_MIN),					ALLY_IN_SPELL_RANGE(INT_MIN),
							ENEMY_IN_WEAPON_ATTACK_RANGE(INT_MIN),	ENEMY_IN_SPELL_ATTACK_RANGE(INT_MIN),
							ENEMY_IN_RANGED_ATTACK_RANGE(INT_MIN),	ENEMY_IN_MELEE_RANGE(INT_MIN){}
	};
	struct GOAP_STATE_COMPACT{
		unsigned int conditions1; //Stores if a given condition is true or false
		unsigned int conditionIgnoreState1; //Stores if a given condition matters or not

		__device__ __host__ GOAP_STATE_COMPACT():conditions1(0),conditionIgnoreState1(0){}
	};

	__device__ __host__ void GoapStateToCompact(GOAP_GPGPU::GOAP_STATE* goapState,GOAP_GPGPU::GOAP_STATE_COMPACT* goapCompact){
		//a value of -2147483648 (10000000000000000000000000000000 in bin 32bit int) means ignore, 1 for true, 0 for false
		unsigned int tmpCondition	= 0;
		unsigned int tmpIgnoreStat	= ~0;
		//tmpIgnoreStat = !tmpIgnoreStat;

		//The algorithm here is simple, take the value on the GOAP_STATE member, shift it so that its at the correct position. 
		//OR it because of the value -2147483648 the 1 gets thrown out and we got nothing to worry about.
		tmpCondition= (unsigned int)goapState->KNIFE_EQUIPPED					<<1;
		tmpCondition= (unsigned int)tmpCondition								>>1;
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->KNIFE_EQUIPPED					>>31; 
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;

		tmpCondition= (unsigned int)goapState->STONE_EQUIPPED					<<1;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->STONE_EQUIPPED					>>30;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;

		tmpCondition= (unsigned int)goapState->SWORD_EQUIPPED					<<2;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->SWORD_EQUIPPED					>>29;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;					

		tmpCondition= (unsigned int)goapState->STAFF_EQUIPPED					<<3;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->STAFF_EQUIPPED					>>28;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;

		tmpCondition= (unsigned int)goapState->BLADE_EQUIPPED					<<4;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->BLADE_EQUIPPED					>>27;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;

		tmpCondition= (unsigned int)goapState->SPEAR_EQUIPPED					<<5;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->SPEAR_EQUIPPED					>>26;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;

		tmpCondition= (unsigned int)goapState->LONG_SWORD_EQUIPPED				<<6;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->LONG_SWORD_EQUIPPED				>>25;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;

		tmpCondition= (unsigned int)goapState->ALLY_DEFENSE_UP					<<7;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->ALLY_DEFENSE_UP					>>24;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;

		tmpCondition= (unsigned int)goapState->ALLY_ATTACK_UP					<<8;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->ALLY_ATTACK_UP					>>23;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;

		tmpCondition= (unsigned int)goapState->SELF_DEFENSE_UP					<<9;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->SELF_DEFENSE_UP					>>22;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;

		tmpCondition= (unsigned int)goapState->SELF_ATTACK_UP					<<10;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->SELF_ATTACK_UP					>>21;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;

		tmpCondition= (unsigned int)goapState->ENEMY_DEFENSE_UP					<<11;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->ENEMY_DEFENSE_UP				>>20;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;

		tmpCondition= (unsigned int)goapState->ENEMY_ATTACK_UP					<<12;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->ENEMY_ATTACK_UP					>>19;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;

		tmpCondition= (unsigned int)goapState->HAS_LIVE_ENEMY					<<13;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->HAS_LIVE_ENEMY					>>18;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;

		tmpCondition= (unsigned int)goapState->IS_ENEMY_FRACTION				<<14;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->IS_ENEMY_FRACTION				>>17;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;

		tmpCondition= (unsigned int)goapState->SELF_HP_LOW						<<15;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->SELF_HP_LOW						>>16;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;

		tmpCondition= (unsigned int)goapState->SELF_MP_LOW						<<16;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->SELF_MP_LOW						>>15;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;

		tmpCondition= (unsigned int)goapState->SELF_HP_CRITICAL					<<17;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->SELF_HP_CRITICAL				>>14;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;

		tmpCondition= (unsigned int)goapState->SELF_RISK_OF_DEATH				<<18;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->SELF_RISK_OF_DEATH				>>13;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;

		tmpCondition= (unsigned int)goapState->ALLY_MP_LOW						<<19;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->ALLY_MP_LOW						>>12;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;					

		tmpCondition= (unsigned int)goapState->ALLY_HP_LOW						<<20;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->ALLY_HP_LOW						>>11;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;

		tmpCondition= (unsigned int)goapState->HAS_MP_POTION					<<21;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->HAS_MP_POTION					>>10;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;					

		tmpCondition= (unsigned int)goapState->HAS_HP_POTION					<<22;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->HAS_HP_POTION					>>9;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;

		tmpCondition= (unsigned int)goapState->ALLY_IN_SPELL_RANGE				<<23;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->ALLY_IN_SPELL_RANGE				>>8;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;		

		tmpCondition= (unsigned int)goapState->ENEMY_IN_WEAPON_ATTACK_RANGE		<<24;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->ENEMY_IN_WEAPON_ATTACK_RANGE	>>7;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;

		tmpCondition= (unsigned int)goapState->ENEMY_IN_SPELL_ATTACK_RANGE		<<25;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->ENEMY_IN_SPELL_ATTACK_RANGE		>>6;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;

		tmpCondition= (unsigned int)goapState->ENEMY_IN_RANGED_ATTACK_RANGE		<<26;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->ENEMY_IN_RANGED_ATTACK_RANGE	>>5;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;

		tmpCondition= (unsigned int)goapState->ENEMY_IN_MELEE_RANGE				<<27;	
		goapCompact->conditions1 = goapCompact->conditions1|tmpCondition;
		tmpIgnoreStat= (unsigned int)goapState->ENEMY_IN_MELEE_RANGE			>>4;	
		goapCompact->conditionIgnoreState1 = goapCompact->conditionIgnoreState1|tmpIgnoreStat;		
	}

	__device__ __host__ void  CompactToGoapState(GOAP_GPGPU::GOAP_STATE_COMPACT* goapCompact,GOAP_GPGPU::GOAP_STATE* goapState){
		int tempIgnore	= 0;
		int tmpVal		= 0;

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<31;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<31;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		//if not an ignore
		if(tempIgnore>-1){
				goapState->KNIFE_EQUIPPED = tmpVal;
		}//lol having else to this if and doing goapState->ENEMY_IS_CLOSE = goapState->ENEMY_IS_CLOSE might be faster ??? :p check please when u got timez :D 

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<30;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<30;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->STONE_EQUIPPED = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<29;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<29;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->SWORD_EQUIPPED = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<28;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<28;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->STAFF_EQUIPPED = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<27;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<27;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->BLADE_EQUIPPED = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<26;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<26;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->SPEAR_EQUIPPED = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<25;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<25;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->LONG_SWORD_EQUIPPED = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<24;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<24;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->ALLY_DEFENSE_UP = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<23;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<23;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->ALLY_ATTACK_UP = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<22;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<22;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->SELF_DEFENSE_UP = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<21;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<21;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->SELF_ATTACK_UP = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<20;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<20;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->ENEMY_DEFENSE_UP = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<19;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<19;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->ENEMY_ATTACK_UP = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<18;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<18;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->HAS_LIVE_ENEMY = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<17;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<17;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->IS_ENEMY_FRACTION = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<16;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<16;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->SELF_HP_LOW = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<15;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<15;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->SELF_MP_LOW = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<14;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<14;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->SELF_HP_CRITICAL = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<13;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<13;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->SELF_RISK_OF_DEATH = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<12;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<12;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->ALLY_MP_LOW = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<11;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<11;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->ALLY_HP_LOW = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<10;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<10;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->HAS_MP_POTION = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<9;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<9;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->HAS_HP_POTION = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<8;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<8;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->ALLY_IN_SPELL_RANGE = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<7;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<7;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->ENEMY_IN_WEAPON_ATTACK_RANGE = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<6;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<6;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->ENEMY_IN_SPELL_ATTACK_RANGE = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<5;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<5;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->ENEMY_IN_RANGED_ATTACK_RANGE = tmpVal;
		}

		tempIgnore = (unsigned int)goapCompact->conditionIgnoreState1	<<4;
		tmpVal	   = (unsigned int)goapCompact->conditions1				<<4;
		tmpVal	   = (unsigned int)tmpVal								>>31;
		if(tempIgnore>-1){
			goapState->ENEMY_IN_MELEE_RANGE = tmpVal;
		}
	}

	/*
		Takes two GOAP_GPGPU::GOAP_STATE_COMPACT and an int*
		to the int* this function will calculate how many differences are there between the two GOAP_STATE_COMPACT(s) RELETIVE to firstStateCompact's 
		ignore state.

		First we take the .conditions1 and xOR it, SO!!! if we have any differences we can see those differences ie, XOR will produce 1 when we have a difference

		then we take fistStateCompact's ignore bits, and for each bit we are interested in, we shift if left and right to make it 0 or 1 
		and then we subtract (-) 

		Sorry about the stupid complexity, but, we need to avoid if's as much as possible
	*/
	//With respect to the first state's ignore state, how many differences are there in the second state
	__device__ __host__ void countDifferences(GOAP_GPGPU::GOAP_STATE_COMPACT* fistStateCompact,GOAP_GPGPU::GOAP_STATE_COMPACT* secondStateCompact, int* differences){
		
		(*differences) = 0;
		unsigned int tempFirstsIgnoreBit	= 0;
//		unsigned int tempFirstsVals			= fistStateCompact->conditions1;
//		unsigned int tempSecondsVals		= secondStateCompact->conditions1;

		for(int i=0;i<GOAP_GPGPU::NUMBER_OF_STATES;i++){
			tempFirstsIgnoreBit = fistStateCompact->conditionIgnoreState1;

			//if, this is something the first state care's about, ie, bit is a 0
			if( (((tempFirstsIgnoreBit) << (31-i))>>31) == 0 ){
				if( (((fistStateCompact->conditions1) << (31-i))>>31) != (((secondStateCompact->conditions1) << (31-i))>>31) ){
					(*differences)++;
				}
			}
		}
	}

	//(*differences) = 0;
		//unsigned int xOredValues = fistStateCompact->conditions1 ^ secondStateCompact->conditions1;

		//unsigned int tempFirstsIgnoreBit = 0;
		//unsigned int tempXoredBit		= 0;
		//int adder				= 0; //someday, some day I should come-up with a better name for this hohoho

		//for(int i=0;i<GOAP_GPGPU::NUMBER_OF_STATES;i++){
		//	tempFirstsIgnoreBit	= (unsigned int)fistStateCompact->conditionIgnoreState1	<< (31-i);	//Clear other bits 
		//	tempFirstsIgnoreBit	= (unsigned int)tempFirstsIgnoreBit>>(31);							//Bring it back so we know the bit we are dealing with :)

		//	tempXoredBit	= (unsigned int)xOredValues		<< (31-i);
		//	tempXoredBit	= (unsigned int)tempXoredBit	>> (31);

		//	//So here, if we are to NOT ignore, and theres a difference, we will be doing a 0 - 1, which is -1, in binary 11111111(32-1ns)
		//	adder	= tempFirstsIgnoreBit - tempXoredBit;

		//	//now shift the bits 1 to the right, so that we should have a 1 if this was not -1 we will have 0, add it to the differences
		//	adder	= (unsigned int)adder>>31;
		//	(*differences) =(*differences) + adder;
		//}


};//Goap state end