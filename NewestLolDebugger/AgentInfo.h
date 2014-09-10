//Helps represent an agent in the simulation
#pragma once
#include "Vector3.h"
#include "NamedValues.h"
#include "Equipment.h"
#include "GOAP_State.h"

namespace GOAP_GPGPU{
	enum NPC_TYPE{
		BloodWolf_Cleric,
		Goblin_Cleric,
		Dullahan_cleric,
		Harpy_cleric,
		Hydra_cleric,
		Hippalectryon_cleric,

		Mermen_Priest,
		BloodWolf_Priest,
		Goblin_Priest,
		Harpy_Priest,
		Hydra_Priest,
		Hippalectryon_Priest,

		Mermen_Attacker,
		BloodWolf_Attacker,
		Goblin_Attacker,
		Dullahan_Attacker,
		Harpy_Attacker,
		Hydra_Attacker,
//		Hippalectryon_Attacker,

		Mermen_Mage,
		BloodWolf_Mage,
		Goblin_Mage,
		Harpy_Mage,
		Hydra_Mage,
		Hippalectryon_Mage,

		Mermen_warrior,
		BloodWolf_warrior,
		Goblin_warrior,
		Dullahan_warrior,
		Harpy_warrior,
		Hydra_warrior,
		Hippalectryon_warrior,

		NUMBER_OF_NPC_TYPES		//THIS SHOULD BE LAST
	};

	struct NPC_ACTION_ALLOW_STAT{
		unsigned int allowBits1;
		unsigned int allowBits2;
	};

	NPC_ACTION_ALLOW_STAT actions_allowedPerNpcType[NUMBER_OF_NPC_TYPES];

	//Fuck common best practices my asshole!!I am tired now I'm gonna hard code this motherfucker!!! fyuck off if it goes to shit!! FUCK!!!
	void defineActionsAllowedPerNPC(NPC_ACTION_ALLOW_STAT* npcActionAllowaStartList){

		//FUCK FUCK FUCK FUCK FCUKCICHIUHIUWHIUDH REALLY FUCK !!! ok , we are going to use windows calculator with this
		//We will decide on the , 1and 0s turned in to ints :)
		npcActionAllowaStartList[BloodWolf_Cleric].allowBits1		= 4027416585;
		npcActionAllowaStartList[BloodWolf_Cleric].allowBits2		= 4290772992;	
		npcActionAllowaStartList[Goblin_Cleric].allowBits1			= 4027416585;
		npcActionAllowaStartList[Goblin_Cleric].allowBits2			= 4290772992;
		npcActionAllowaStartList[Dullahan_cleric].allowBits1		= 3490545673;
		npcActionAllowaStartList[Dullahan_cleric].allowBits2		= 4290772992;
		npcActionAllowaStartList[Harpy_cleric].allowBits1			= 4027416585;
		npcActionAllowaStartList[Harpy_cleric].allowBits2			= 4290772992;
		npcActionAllowaStartList[Hydra_cleric].allowBits1			= 4027416585;
		npcActionAllowaStartList[Hydra_cleric].allowBits2			= 4290772992;
		npcActionAllowaStartList[Hippalectryon_cleric].allowBits1	= 4027416585;
		npcActionAllowaStartList[Hippalectryon_cleric].allowBits2	= 4290772992;
		//npcActionAllowaStartList[];								  
		npcActionAllowaStartList[Mermen_Priest].allowBits1			= 4027416585;
		npcActionAllowaStartList[Mermen_Priest].allowBits2			= 4290772992;
		npcActionAllowaStartList[BloodWolf_Priest].allowBits1		= 4027416585;
		npcActionAllowaStartList[BloodWolf_Priest].allowBits2		= 4290772992;
		npcActionAllowaStartList[Goblin_Priest].allowBits1			= 4027416585;
		npcActionAllowaStartList[Goblin_Priest].allowBits2			= 4290772992;
		npcActionAllowaStartList[Harpy_Priest].allowBits1			= 4027416585;
		npcActionAllowaStartList[Harpy_Priest].allowBits2			= 4290772992;
		npcActionAllowaStartList[Hydra_Priest].allowBits1			= 4027416585;
		npcActionAllowaStartList[Hydra_Priest].allowBits2			= 4290772992;
		npcActionAllowaStartList[Hippalectryon_Priest].allowBits1	= 4027416585;
		npcActionAllowaStartList[Hippalectryon_Priest].allowBits2	= 4290772992;
		//npcActionAllowaStartList[];								  
		npcActionAllowaStartList[Mermen_Attacker].allowBits1		= 4026533389;
		npcActionAllowaStartList[Mermen_Attacker].allowBits2		= 4290772992;
		npcActionAllowaStartList[BloodWolf_Attacker].allowBits1		= 4026533129;
		npcActionAllowaStartList[BloodWolf_Attacker].allowBits2		= 4290772992;
		npcActionAllowaStartList[Goblin_Attacker].allowBits1		= 4160751117;
		npcActionAllowaStartList[Goblin_Attacker].allowBits2		= 4290772992;
		npcActionAllowaStartList[Dullahan_Attacker]	.allowBits1		= 3489661961;
		npcActionAllowaStartList[Dullahan_Attacker]	.allowBits2		= 4290772992;
		npcActionAllowaStartList[Harpy_Attacker].allowBits1			= 4026532873;
		npcActionAllowaStartList[Harpy_Attacker].allowBits2			= 4290772992;
		npcActionAllowaStartList[Hydra_Attacker].allowBits1			= 4026533001;
		npcActionAllowaStartList[Hydra_Attacker].allowBits2			= 4290772992;
		//npcActionAllowaStartList[];								  ;
		npcActionAllowaStartList[Mermen_Mage].allowBits1			= 4026662937;
		npcActionAllowaStartList[Mermen_Mage].allowBits2			= 4290772992;
		npcActionAllowaStartList[BloodWolf_Mage].allowBits1			= 4026662937;
		npcActionAllowaStartList[BloodWolf_Mage].allowBits2			= 4290772992;
		npcActionAllowaStartList[Goblin_Mage].allowBits1			= 4026669081;
		npcActionAllowaStartList[Goblin_Mage].allowBits2			= 4290772992;
		npcActionAllowaStartList[Harpy_Mage].allowBits1				= 4026669081;
		npcActionAllowaStartList[Harpy_Mage].allowBits2				= 4290772992;
		npcActionAllowaStartList[Hydra_Mage].allowBits1				= 4026669145;	
		npcActionAllowaStartList[Hydra_Mage].allowBits2				= 4290772992;	
		npcActionAllowaStartList[Hippalectryon_Mage].allowBits1		= 4026662937;
		npcActionAllowaStartList[Hippalectryon_Mage].allowBits2		= 4290772992;
		//npcActionAllowaStartList[];								  
		npcActionAllowaStartList[Mermen_warrior].allowBits1			= 4093640713;
		npcActionAllowaStartList[Mermen_warrior].allowBits2			= 4290772992;
		npcActionAllowaStartList[BloodWolf_warrior].allowBits1		= 4093640713;
		npcActionAllowaStartList[BloodWolf_warrior].allowBits2		= 4290772992;
		npcActionAllowaStartList[Goblin_warrior].allowBits1			= 4093640713;
		npcActionAllowaStartList[Goblin_warrior].allowBits2			= 4290772992;
		npcActionAllowaStartList[Dullahan_warrior].allowBits1		= 3556769835;
		npcActionAllowaStartList[Dullahan_warrior].allowBits2		= 4290772992;
		npcActionAllowaStartList[Harpy_warrior].allowBits1			= 4127195145;
		npcActionAllowaStartList[Harpy_warrior].allowBits2			= 4290772992;
		npcActionAllowaStartList[Hydra_warrior].allowBits1			= 4110417929;
		npcActionAllowaStartList[Hydra_warrior].allowBits2			= 4290772992;
		npcActionAllowaStartList[Hippalectryon_warrior].allowBits1	= 4093640713;
		npcActionAllowaStartList[Hippalectryon_warrior].allowBits2	= 4290772992;
	}

	class AgentInfo{
	public:
		Vector3 position;
		//Vector3 homingPoint; // lets have homing points for each block
		int health;
		int mp;
		int statusEffects; // things like, if def is up, if attack is up etc etc
		NPC_TYPE agentType;
		EQUIPMENT agentEquipmentInHand;
		//If someone wants help, they come here and ask for help by adding their ID here, no need to do concurrency control here as
		//We give help to the guy who was able to write here before our started his state calculation
		int giveHelpTo;
		int pickedUpItems;	//Helps figure-out if we have hp,mp,def and ORED ITEM_TYPE s
		GOAP_STATE agentGoal;
		//Holds the ID of the enemy who attacked us. If there's no ID here, get a random enemy and attack. 
		//int wasAttackedby; //This value should be a random value if there's no value for our agent
	private:
	};

	class AgentInfoCompact{
	public:
		Vector3 position;
		//NPC_TYPE agentType;
		//int level;
		//int health;
		//int fractionHostility; // An aggregate of PLAYER_FRACTION ORed (|) 
		//int behaviourPattern; //Aggregate of NPC_BEHAVIOURS, Can be a class if we need support for more than 31 flags
		//int state;
		//int allData[];
	private:
	};

};