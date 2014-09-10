#pragma region includes

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <curand_kernel.h>

#include <windows.h>                // for Windows APIs
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>

#include "miscellaneous.h"

#include "Vector4.h"
#include "Vector3.h"
#include "PlayerInfo.h"
#include "AgentInfo.h"
#include "GOAP_State.h"
#include "GOAP_Actions.h"
#include "AStarNode.h"


#pragma endregion 

#pragma region Global vars and defines

#define DEBUG_MODE						1
#define WITH_BLOCKING					1  //States if blocking is used in shared memory

//Player, agent and object info
//const int PLAYER_TOTAL					= 49212;
//const int AGENT_TOTAL					= 49984;
//const dim3 NUMBER_OF_THREADS			= dim3(32,1,1);
//const dim3 NUMBER_OF_BLOCKS				= dim3(1562,1,1);
//const int NUMBER_OF_BLOCKS_IN_X			= 1562;
const dim3 NUMBER_OF_THREADS			= dim3(32,1,1);


const int PLAYER_TOTAL					= 65536;
const int AGENT_TOTAL					= 65536;
const dim3 NUMBER_OF_BLOCKS				= dim3(2048,1,1);
const int NUMBER_OF_BLOCKS_IN_X			= 2048;

//const int PLAYER_INT_ATTRIBS			= 5;
//const int AGENT_INT_ATTRIBS			= 9;

const int PARTY_SIZE					= 32;
const int NUMBER_OF_THREADS_X			= 32;

const int PLAYER_HEALTH_BEGIN			= 0;
const int PLAYER_MP_BEGIN				= PLAYER_HEALTH_BEGIN			+ PLAYER_TOTAL;
const int PLAYER_FRACTION_BEGIN			= PLAYER_MP_BEGIN				+ PLAYER_TOTAL;
const int PLAYER_STATE_BEGIN			= PLAYER_FRACTION_BEGIN			+ PLAYER_TOTAL;
const int PLAYER_STATUS_EFFECTS_BEGIN	= PLAYER_STATE_BEGIN			+ PLAYER_TOTAL;

const int AGENT_HEALTH_BEGIN			= PLAYER_STATUS_EFFECTS_BEGIN	+ PLAYER_TOTAL;
const int AGENT_MP_BEGIN				= AGENT_HEALTH_BEGIN			+ AGENT_TOTAL;
const int AGENT_STATUS_EFFECTS_BEGIN	= AGENT_MP_BEGIN				+ AGENT_TOTAL;
const int AGENT_TYPE_BEGIN				= AGENT_STATUS_EFFECTS_BEGIN	+ AGENT_TOTAL;
const int AGENT_EQUIPMENT_IN_HAND_BEGIN = AGENT_TYPE_BEGIN				+ AGENT_TOTAL;
const int AGENT_PICKED_UP_ITEMS			= AGENT_EQUIPMENT_IN_HAND_BEGIN	+ AGENT_TOTAL;
const int AGENT_GOAL_VALUE_BEGIN		= AGENT_PICKED_UP_ITEMS			+ AGENT_TOTAL;
const int AGENT_GOAL_IGNORE_BEGIN		= AGENT_GOAL_VALUE_BEGIN		+ AGENT_TOTAL;
const int AGENT_STATE_VALUES_BEGIN		= AGENT_GOAL_IGNORE_BEGIN		+ AGENT_TOTAL; 
const int AGENT_STATE_IGNORE_BEGIN		= AGENT_STATE_VALUES_BEGIN		+ AGENT_TOTAL;

//GOAP information 
const int GOAP_ACTION_NAMES_BEGIN					= AGENT_STATE_IGNORE_BEGIN					+ AGENT_TOTAL;
const int GOAP_ACTION_PRECONDITIONS_VALUES_BEGIN	= GOAP_ACTION_NAMES_BEGIN					+ GOAP_GPGPU::NUMBER_OF_ACTIONS;
const int GOAP_ACTION_PRECONDITIONS_IGNORE_BEGIN	= GOAP_ACTION_PRECONDITIONS_VALUES_BEGIN	+ GOAP_GPGPU::NUMBER_OF_ACTIONS;
const int GOAP_ACTION_POSTCONDITIONS_VALUES_BEGIN	= GOAP_ACTION_PRECONDITIONS_IGNORE_BEGIN	+ GOAP_GPGPU::NUMBER_OF_ACTIONS;
const int GOAP_ACTION_POSTCONDITIONS_IGNORE_BEGIN	= GOAP_ACTION_POSTCONDITIONS_VALUES_BEGIN	+ GOAP_GPGPU::NUMBER_OF_ACTIONS;
const int GOAP_ACTION_G_COST_BEGIN					= GOAP_ACTION_POSTCONDITIONS_IGNORE_BEGIN	+ GOAP_GPGPU::NUMBER_OF_ACTIONS;
const int GOAP_ACTION_ALLOWED_TO					= GOAP_ACTION_G_COST_BEGIN					+ GOAP_GPGPU::NUMBER_OF_ACTIONS;

const int NUMBER_OF_ASTAR_COMPACT_NODES_IN_GLOB_MEM = AGENT_TOTAL * 1024; //at 20 bytes For 256 agents thats 5mb :S 

const int TOTAL_INT_ATTRIBS							= GOAP_ACTION_ALLOWED_TO + GOAP_GPGPU::NUMBER_OF_ACTIONS;


#define SPHERE_RADIUS_SQ				10.0f //2 for each sphere

//Timer global vars
LARGE_INTEGER frequency;        //Ticks per second
LARGE_INTEGER t1, t2;           //Ticks

#pragma endregion

#pragma region function prototypes

__host__ void hardcodedPopulate(GOAP_GPGPU::AgentInfo* agentArr,GOAP_GPGPU::PlayerInfo* playerArr);
__host__ void arrangePlayersByBlock(GOAP_GPGPU::PlayerInfo* players, GOAP_GPGPU::Vector4* playerAndHomingLocations);
__host__ void createIntArrayFromAllData(GOAP_GPGPU::PlayerInfo* playerArr, GOAP_GPGPU::AgentInfo*, int* results_array);

__host__ void startTimer();
__host__ double getElapsedTime();
__host__ void writeResultsToFile(int *dataYo,int intSize, std::string fileName);
__host__ void getAgentPositions(GOAP_GPGPU::AgentInfo* agents, GOAP_GPGPU::Vector3* agent_positions);
__host__ void generateData(int numberOfPlayers,int numberOfAgents,GOAP_GPGPU::PlayerInfo* playersArr,GOAP_GPGPU::AgentInfo* agentsArr);

__global__ void calculateCurrentState(int threadsPerBlock,int memBankCount,GOAP_GPGPU::Vector4* d_playerAndHomingPositions,GOAP_GPGPU::Vector3* d_agentPositions,int* d_gm_allIntData);
__global__ void runPlanner(int* d_gm_allIntData,int* d_results,GOAP_GPGPU::AStarNodeCompact* d_allAstarNodeCompacts);

__device__ __host__ GOAP_GPGPU::AStarNodeCompact* findLowestFCostNodeInList(GOAP_GPGPU::AStarNodeCompact* list, int sizeOfList,int startAt,int *foundAt);
__device__ void calculateAgentGoapState(int* agentIntMemLocation,int* enemyIntMemLocation,GOAP_GPGPU::Vector3* agentPosition,
										GOAP_GPGPU::Vector3* allyPos,GOAP_GPGPU::Vector4* enemyPos,GOAP_GPGPU::Vector4* homingPoint,
										int agentID,int allyId, int numberOfAgentDataInInt, int* resultState,int* resultStateIgnore);
__device__ void applyStateAtoB(GOAP_GPGPU::GOAP_STATE_COMPACT* stateA,GOAP_GPGPU::GOAP_STATE_COMPACT* stateB,GOAP_GPGPU::GOAP_STATE_COMPACT* answer);
__device__ void generateActionsForCurruntState(GOAP_GPGPU::AStarNodeCompact* nodesStore,GOAP_GPGPU::GOAP_STATE_COMPACT* currState,GOAP_GPGPU::GOAP_STATE_COMPACT* goalState,int* intData,int globalThreadId,int* numberOfNodesInNodeStore,int agentId);
__device__ bool isExactMatch(GOAP_GPGPU::GOAP_STATE_COMPACT* firstState,GOAP_GPGPU::GOAP_STATE_COMPACT* secondState);

#pragma endregion

int main()
{
	//DEBUGSTUFF		
	//std::cout<<sizeof(GOAP_GPGPU::AStarNodeCompact);

#pragma region setting up stuff
	startTimer();

	//We pass 1 here to set the 2nd coda device to active
	if(!hasCUDADevice(1)){ 
		int temp;
		std::cout<<std::endl<<"Enter something to quit...."<<std::endl;
		std::cin>>temp;
		return -1;
	}
	//Create the arrays that hold info about our players, agents and objects on the HOST side
	GOAP_GPGPU::PlayerInfo*	players								= new GOAP_GPGPU::PlayerInfo[PLAYER_TOTAL];
	GOAP_GPGPU::Vector4*	playersPosRearrangedAndHomingPoints	= new GOAP_GPGPU::Vector4 [PLAYER_TOTAL+NUMBER_OF_BLOCKS_IN_X +1]; //+ 1 to store info about the players not close to us

	GOAP_GPGPU::AgentInfo*	agents			 = new GOAP_GPGPU::AgentInfo[AGENT_TOTAL];
	GOAP_GPGPU::Vector3*	agent_positions	 = new GOAP_GPGPU::Vector3[AGENT_TOTAL];

	//Populate players and agents data
	//hardcodedPopulate(agents,players);
	generateData(PLAYER_TOTAL,AGENT_TOTAL,players,agents);
	GOAP_GPGPU::defineGoapActions();//Helper function which populates actions with their preconditions and post conditions

	//Save all the int type data as one huge chunk of data
	int* allIntData = new int[TOTAL_INT_ATTRIBS];
	createIntArrayFromAllData(players,agents,allIntData);

	std::cout<<"\n Startup time : "<<getElapsedTime()<<"  ms\n";

#pragma endregion


#pragma region rearranging and doing the setting up of arrays

	arrangePlayersByBlock(players,playersPosRearrangedAndHomingPoints);
	getAgentPositions(agents, agent_positions);

#pragma endregion

#pragma region allocate memory on cuda

	startTimer();

	cudaError_t cudaStatus;

	//Player Positions and the homingPoints as vec4's and their(homingPoints) w's as the number of players for each block
	GOAP_GPGPU::Vector4* d_playerAndHomingPositions = NULL;
	cudaStatus = cudaMalloc((void**)&d_playerAndHomingPositions, (PLAYER_TOTAL + NUMBER_OF_BLOCKS_IN_X) * sizeof(GOAP_GPGPU::Vector4));
	if (cudaStatus != cudaSuccess) {
		std::cout<<"\n cudaMalloc failed! at all d_playerAndHomingPositions \n"<<std::endl;
		return -1;
	}

	//Agents Positions 
	GOAP_GPGPU::Vector3* d_agentPositions = NULL;
	cudaStatus = cudaMalloc((void**)&d_agentPositions, (AGENT_TOTAL) * sizeof(GOAP_GPGPU::Vector3));
	if (cudaStatus != cudaSuccess) {
		std::cout<<"\n cudaMalloc failed! at all d_agentPositions \n"<<std::endl;
		return -1;
	}
	
	//All the data as one int array to minimize the number of cudaMallocs and cudaMemcpy
	int* d_allIntData = NULL;
	cudaStatus = cudaMalloc((void**)&d_allIntData, TOTAL_INT_ATTRIBS * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		std::cout<<"\n cudaMalloc failed! at all int d_allIntData \n"<<std::endl;
		return -1;
	}

	//All the data as one int array to minimize the number of cudaMallocs and cudaMemcpy
	const int RESULTS_PER_AGENT = 48*2;
	int* d_results = NULL;
	int* h_results = new int[(AGENT_TOTAL * RESULTS_PER_AGENT)]; 
	for(int i=0;i<(AGENT_TOTAL * RESULTS_PER_AGENT);i++){
		h_results[i] = -888;
	}

	cudaStatus = cudaMalloc((void**)&d_results, ((AGENT_TOTAL * RESULTS_PER_AGENT) * sizeof(int)));
	if (cudaStatus != cudaSuccess) {
		std::cout<<"\n cudaMalloc failed! at all int d_results \n"<<std::endl;
		return -1;
	}

	GOAP_GPGPU::AStarNodeCompact* d_allAstarNodeCompact = NULL;
	cudaStatus = cudaMalloc((void**)&d_allAstarNodeCompact, NUMBER_OF_ASTAR_COMPACT_NODES_IN_GLOB_MEM * sizeof(GOAP_GPGPU::AStarNodeCompact));
	if (cudaStatus != cudaSuccess) {
		std::cout<<"\n cudaMalloc failed! at all int d_allAstarNodeCompact \n"<<std::endl;
		return -1;
	}


	std::cout<<"\n Time taken to allocate memory on the GPU : "<<getElapsedTime()<<"  ms\n";
#pragma endregion	

#pragma region memcopy host to device
	startTimer();

	cudaMemcpy(d_playerAndHomingPositions	,playersPosRearrangedAndHomingPoints,(PLAYER_TOTAL + NUMBER_OF_BLOCKS_IN_X) * sizeof(GOAP_GPGPU::Vector4),	cudaMemcpyHostToDevice);
	cudaMemcpy(d_agentPositions				,agent_positions					,AGENT_TOTAL							* sizeof(GOAP_GPGPU::Vector3),	cudaMemcpyHostToDevice);
	cudaMemcpy(d_allIntData					,allIntData							,TOTAL_INT_ATTRIBS						* sizeof(int)				,	cudaMemcpyHostToDevice);

	std::cout<<"\n Time taken to copy memory on to the GPU from HOST : "<<getElapsedTime()<<" ms\n";
#pragma endregion

#pragma region call to CUDA

#if DEBUG_MODE == 1
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	cudaEventRecord(start, 0);
#endif
	//(int threadsPerBlock,	int memBankCount,	GOAP_GPGPU::Vector4* d_playerAndHomingPositions,GOAP_GPGPU::Vector3* d_agentPositions,	int* d_gm_allIntData)
	calculateCurrentState<<<NUMBER_OF_BLOCKS,NUMBER_OF_THREADS>>>(NUMBER_OF_THREADS.x,32,d_playerAndHomingPositions,d_agentPositions,d_allIntData);

	//cudaFuncSetCacheConfig(runPlanner, cudaFuncCachePreferShared);//To get 48K of mem
	runPlanner<<<NUMBER_OF_BLOCKS,NUMBER_OF_THREADS>>>(d_allIntData,d_results,d_allAstarNodeCompact);

	cudaMemcpy(h_results					,d_results							,((AGENT_TOTAL * (48*2)) * sizeof(int)) 	,	cudaMemcpyDeviceToHost);

#if DEBUG_MODE == 1
	// Stop event
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
	// Clean up:
	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	std::cout<<"\n It took CUDA "<<elapsedTime<<" ms to run the program.\n";
#endif


#pragma endregion

#pragma region ending program

	writeResultsToFile(h_results,(AGENT_TOTAL * (96)),"lol");

	//delete octree;				
	cudaFree(d_playerAndHomingPositions); 
	cudaFree(d_agentPositions);
	cudaFree(d_allIntData);

	return 0;

#pragma endregion
}

/* 
Makes a big dump of ints using all the individual int data from players, agents and objectsf

Note!! We do not care about the way the main array is ordered, since ordering in this way will help us to make sure we can keep similar agents close to eachother and therefore we can
decrease thread divergence in thread warps
*/
void createIntArrayFromAllData(GOAP_GPGPU::PlayerInfo* playerArr,GOAP_GPGPU::AgentInfo* agentArr,int* results_array){

	/*
	PLAYER_HEALTH_BEGIN		
	PLAYER_MP_BEGIN			
	PLAYER_FRACTION_BEGIN		
	PLAYER_STATE_BEGIN		
	PLAYER_STATUS_EFFECTS_BEGIN
	*/

	for(int i=0;i<PLAYER_TOTAL;i++){
		results_array[PLAYER_HEALTH_BEGIN			+ (i)] = playerArr[i].health;
		results_array[PLAYER_MP_BEGIN				+ (i)] = playerArr[i].mp;
		results_array[PLAYER_FRACTION_BEGIN			+ (i)] = (int)playerArr[i].fraction;
		results_array[PLAYER_STATE_BEGIN			+ (i)] = playerArr[i].state;
		results_array[PLAYER_STATUS_EFFECTS_BEGIN	+ (i)] = playerArr[i].statusEffects;
	}
	/*
		AGENT_HEALTH_BEGIN			
		AGENT_MP_BEGIN				
		AGENT_STATUS_EFFECTS_BEGIN	
		AGENT_TYPE_BEGIN				
		AGENT_EQUIPMENT_IN_HAND_BEGIN 
		AGENT_PICKED_UP_ITEMS			
		AGENT_GOAL_VALUE_BEGIN		
		AGENT_GOAL_IGNORE_BEGIN		
		AGENT_STATE_VALUES_BEGIN		
		AGENT_STATE_IGNORE_BEGIN		
	*/

	for(int i=0;i<AGENT_TOTAL;i++){
		results_array[AGENT_HEALTH_BEGIN			+ (i)]	= agentArr[i].health;
		results_array[AGENT_MP_BEGIN				+ (i)]	= agentArr[i].mp; 
		results_array[AGENT_STATUS_EFFECTS_BEGIN	+ (i)]	= agentArr[i].statusEffects; 
		results_array[AGENT_TYPE_BEGIN				+ (i)]	= agentArr[i].agentType;
		results_array[AGENT_EQUIPMENT_IN_HAND_BEGIN	+ (i)]	= agentArr[i].agentEquipmentInHand;
		results_array[AGENT_PICKED_UP_ITEMS			+ (i)]	= agentArr[i].pickedUpItems;
			GOAP_GPGPU::GOAP_STATE_COMPACT	tmpCompact;	
			GOAP_GPGPU::GoapStateToCompact(&(agentArr[i].agentGoal),&tmpCompact);
		results_array[AGENT_GOAL_VALUE_BEGIN		+ (i)]	= tmpCompact.conditions1;
		results_array[AGENT_GOAL_IGNORE_BEGIN		+ (i)]	= tmpCompact.conditionIgnoreState1;
		
		/*
			AGENT_STATE_VALUES_BEGIN
			AGENT_STATE_IGNORE_BEGIN
		*/
	}

	//GOAP action details
	/*
		GOAP_ACTION_NAMES_BEGIN					
		GOAP_ACTION_PRECONDITIONS_VALUES_BEGIN	
		GOAP_ACTION_PRECONDITIONS_IGNORE_BEGIN	
		GOAP_ACTION_POSTCONDITIONS_VALUES_BEGIN	
		GOAP_ACTION_POSTCONDITIONS_IGNORE_BEGIN	
		GOAP_ACTION_G_COST_BEGIN

		allGoapCompactActions has the compact actions in it 
	*/
	for(int i=0;i<GOAP_GPGPU::NUMBER_OF_ACTIONS;i++){
		results_array[GOAP_ACTION_NAMES_BEGIN					+ (i)]	= GOAP_GPGPU::allGoapCompactActions[i].actionName; //LOLOLOLOL
		results_array[GOAP_ACTION_PRECONDITIONS_VALUES_BEGIN	+ (i)]	= GOAP_GPGPU::allGoapCompactActions[i].preCondition.conditions1; 
		results_array[GOAP_ACTION_PRECONDITIONS_IGNORE_BEGIN	+ (i)]	= GOAP_GPGPU::allGoapCompactActions[i].preCondition.conditionIgnoreState1; 
		results_array[GOAP_ACTION_POSTCONDITIONS_VALUES_BEGIN	+ (i)]	= GOAP_GPGPU::allGoapCompactActions[i].postCondition.conditions1; 
		results_array[GOAP_ACTION_POSTCONDITIONS_IGNORE_BEGIN	+ (i)]	= GOAP_GPGPU::allGoapCompactActions[i].postCondition.conditionIgnoreState1;
		results_array[GOAP_ACTION_G_COST_BEGIN					+ (i)]	= GOAP_GPGPU::allGoapCompactActions[i].g;
		results_array[GOAP_ACTION_ALLOWED_TO					+ (i)]	= GOAP_GPGPU::allGoapCompactActions[i].actionAllowedTo; // who can perform the action
		//
	}


}

__global__ void calculateCurrentState(int threadsPerBlock,	int memBankCount,	GOAP_GPGPU::Vector4* d_playerAndHomingPositions,
										GOAP_GPGPU::Vector3* d_agentPositions,	int* d_gm_allIntData){	

	int globalThreadId =  ((blockDim.x*blockIdx.x)+threadIdx.x);
	int numberOfPlayersInArea = d_playerAndHomingPositions[blockIdx.x + PLAYER_TOTAL].w;
	int playerIndexOffset;
	int playerIndex;

	if(numberOfPlayersInArea>0){
		playerIndex = (numberOfPlayersInArea + threadIdx.x)%NUMBER_OF_THREADS_X;
		//if there are players (enemies) for us to deal with then 
		for(int i=0;i<blockIdx.x;i++){
			//Then go through the list of players and try to figureout which player to access;
			playerIndexOffset = playerIndexOffset + d_playerAndHomingPositions[PLAYER_TOTAL+i].w;
		}
	}
	playerIndexOffset = playerIndexOffset + playerIndex;

	
#if WITH_BLOCKING == 1 //If we are using blocking

	__shared__ int shared_agent_int_data[PARTY_SIZE*8]; //We store data for 64 agents only the first 8 mem blocks for the agents are used.
	__shared__ GOAP_GPGPU::Vector3 shared_agent_positions[NUMBER_OF_THREADS_X];
	__syncthreads();
	
	//populate shared mem!!!!!!!
	//need for loop here 
	//All int data
	shared_agent_int_data[threadIdx.x + (PARTY_SIZE* 0)]	= d_gm_allIntData[globalThreadId + AGENT_HEALTH_BEGIN];
	shared_agent_int_data[threadIdx.x + (PARTY_SIZE* 1)]	= d_gm_allIntData[globalThreadId + AGENT_MP_BEGIN];
	shared_agent_int_data[threadIdx.x + (PARTY_SIZE* 2)]	= d_gm_allIntData[globalThreadId + AGENT_STATUS_EFFECTS_BEGIN];
	shared_agent_int_data[threadIdx.x + (PARTY_SIZE* 3)]	= d_gm_allIntData[globalThreadId + AGENT_TYPE_BEGIN];
	shared_agent_int_data[threadIdx.x + (PARTY_SIZE* 4)]	= d_gm_allIntData[globalThreadId + AGENT_EQUIPMENT_IN_HAND_BEGIN];
	shared_agent_int_data[threadIdx.x + (PARTY_SIZE* 5)]	= d_gm_allIntData[globalThreadId + AGENT_PICKED_UP_ITEMS];
	shared_agent_int_data[threadIdx.x + (PARTY_SIZE* 6)]	= d_gm_allIntData[globalThreadId + AGENT_GOAL_VALUE_BEGIN];
	shared_agent_int_data[threadIdx.x + (PARTY_SIZE* 7)]	= d_gm_allIntData[globalThreadId + AGENT_GOAL_IGNORE_BEGIN];
	//All the pos data for our agents
	shared_agent_positions[threadIdx.x]						= d_agentPositions[globalThreadId];


	//We have no idea how players are going to come in, and how many players belong to each party, so let it live in global mem
	//Also this function directly saves the result in global memory location we have provided to it as the last arg :)
	//So we dont have to do a seperate step to store stuff back to the global mem :)
	calculateAgentGoapState(
					shared_agent_int_data,	d_gm_allIntData,	&(shared_agent_positions[threadIdx.x]),
					&(d_agentPositions[(threadIdx.x+1)%NUMBER_OF_THREADS_X]),	&(d_playerAndHomingPositions[playerIndexOffset]),	&(d_playerAndHomingPositions[blockIdx.x + PLAYER_TOTAL]),
					threadIdx.x,playerIndexOffset,NUMBER_OF_THREADS_X,&(d_gm_allIntData[globalThreadId + AGENT_STATE_VALUES_BEGIN]),&(d_gm_allIntData[globalThreadId + AGENT_STATE_IGNORE_BEGIN]) );

#else  //if we dont use blocking
	//__shared__ int shared_agent_int_data[NUMBER_OF_THREADS_X*8]; //We store data for 64 agents only the first 8 mem blocks for the agents are used.
	//__shared__ GOAP_GPGPU::Vector3 shared_agent_positions[NUMBER_OF_THREADS_X];
	__syncthreads();
	
	//populate shared mem!!!!!!!
	//need for loop here 
	//All int data
	/*shared_agent_int_data[threadIdx.x + (PARTY_SIZE* 0)]	= d_gm_allIntData[globalThreadId + AGENT_HEALTH_BEGIN];
	shared_agent_int_data[threadIdx.x + (PARTY_SIZE* 1)]	= d_gm_allIntData[globalThreadId + AGENT_MP_BEGIN];
	shared_agent_int_data[threadIdx.x + (PARTY_SIZE* 2)]	= d_gm_allIntData[globalThreadId + AGENT_STATUS_EFFECTS_BEGIN];
	shared_agent_int_data[threadIdx.x + (PARTY_SIZE* 3)]	= d_gm_allIntData[globalThreadId + AGENT_TYPE_BEGIN];
	shared_agent_int_data[threadIdx.x + (PARTY_SIZE* 4)]	= d_gm_allIntData[globalThreadId + AGENT_EQUIPMENT_IN_HAND_BEGIN];
	shared_agent_int_data[threadIdx.x + (PARTY_SIZE* 5)]	= d_gm_allIntData[globalThreadId + AGENT_PICKED_UP_ITEMS];
	shared_agent_int_data[threadIdx.x + (PARTY_SIZE* 6)]	= d_gm_allIntData[globalThreadId + AGENT_GOAL_VALUE_BEGIN];
	shared_agent_int_data[threadIdx.x + (PARTY_SIZE* 7)]	= d_gm_allIntData[globalThreadId + AGENT_GOAL_IGNORE_BEGIN];
	//All the pos data for our agents
	shared_agent_positions[threadIdx.x]						= d_agentPositions[globalThreadId];*/


	//We have no idea how players are going to come in, and how many players belong to each party, so let it live in global mem
	//Also this function directly saves the result in global memory location we have provided to it as the last arg :)
	//So we dont have to do a seperate step to store stuff back to the global mem :)
	calculateAgentGoapState(
					d_gm_allIntData,	d_gm_allIntData,	&(d_agentPositions[globalThreadId]),
					&(d_agentPositions[(threadIdx.x+1)%NUMBER_OF_THREADS_X]),	&(d_playerAndHomingPositions[playerIndexOffset]),	&(d_playerAndHomingPositions[blockIdx.x + PLAYER_TOTAL]),
					threadIdx.x,playerIndexOffset,AGENT_TOTAL,&(d_gm_allIntData[globalThreadId + AGENT_STATE_VALUES_BEGIN]),&(d_gm_allIntData[globalThreadId + AGENT_STATE_IGNORE_BEGIN]) );

#endif
}


__global__ void runPlanner(int* d_gm_allIntData,int* d_results,GOAP_GPGPU::AStarNodeCompact* d_allAstarNodeCompacts){
	const int NUMBER_OF_NODES = 48*2;

	int globalThreadId =  ((blockDim.x*blockIdx.x)+threadIdx.x);
	__shared__ GOAP_GPGPU::AStarNodeCompact nodeList[3072];//(49152_shared_mem / 16_bytes_per_node)/64threads_per_block = 48 vars per thread

	__syncthreads();

	GOAP_GPGPU::AStarNodeCompact nodeCreator; //Used to create nodes and copy to shared mem

	nodeCreator.actionName							= GOAP_GPGPU::NO_ACTION;
	nodeCreator.g									= 126;
	nodeCreator.h									= 126;
	nodeCreator.isInList							= GOAP_GPGPU::in_no_list;
	nodeCreator.parent								= NULL;
	nodeCreator.stateAtNode.conditionIgnoreState1	= 0;
	nodeCreator.stateAtNode.conditions1				= 0;
	//init the shared mem
	for(int i=0;i<NUMBER_OF_NODES;i++){
		nodeList[i+(threadIdx.x*NUMBER_OF_NODES)] = nodeCreator;
	}
	for(int i=0;i<48*2;i++){
		d_results[(globalThreadId*(48*2)) + i] = -9999;
	}

	__syncthreads();

	GOAP_GPGPU::AStarNodeCompact* currNode;
	GOAP_GPGPU::GOAP_STATE_COMPACT goalState;
	GOAP_GPGPU::GOAP_STATE_COMPACT currState;

	goalState.conditions1			= d_gm_allIntData[AGENT_GOAL_VALUE_BEGIN  + globalThreadId];
	goalState.conditionIgnoreState1	= d_gm_allIntData[AGENT_GOAL_IGNORE_BEGIN + globalThreadId];
									  
	currState.conditions1			= d_gm_allIntData[AGENT_STATE_VALUES_BEGIN + globalThreadId];
	currState.conditionIgnoreState1	= d_gm_allIntData[AGENT_STATE_IGNORE_BEGIN + globalThreadId];

	int differences			= 0;

	nodeCreator.actionName	= GOAP_GPGPU::START_ACTION;

	countDifferences(&goalState,&currState,&differences);

	nodeCreator.h			= differences;
	nodeCreator.g			= 0;
	nodeCreator.stateAtNode	= currState;
	nodeCreator.isInList	= GOAP_GPGPU::in_open_list;

	nodeList[threadIdx.x*NUMBER_OF_NODES] = nodeCreator;

	//__syncthreads();

	int nodeFoundAt;
	int numbOfNodesInStore	= 1;//we already have a node, the start node

	int myCounter = 0;
	while(true){
		myCounter++;
		differences			= 0;
		nodeFoundAt			= 0;

		currNode = findLowestFCostNodeInList(nodeList,numbOfNodesInStore,threadIdx.x*NUMBER_OF_NODES,&nodeFoundAt);
		currState = currNode->stateAtNode;
		nodeList[nodeFoundAt].isInList = GOAP_GPGPU::in_closed_list;

		//Get the differences between the currunt state and the goal

		//How long from curr to goal
		GOAP_GPGPU::countDifferences(&goalState,&(currState),&differences);

		//Write results in to G array so we can see them in the CPU side
		if(differences == 0){
			int resultsCounter = 0;
			GOAP_GPGPU::AStarNodeCompact* tmpNode = currNode;
			do{
				d_results[resultsCounter + (globalThreadId*NUMBER_OF_NODES)] = (int)tmpNode->actionName;
				tmpNode = tmpNode->parent;
				resultsCounter++;
			}while(tmpNode!=NULL && tmpNode->parent != NULL);

			break; // how to send back the data to the home planet ie cpu side
		}


		generateActionsForCurruntState(nodeList,&currState,&goalState,d_gm_allIntData,globalThreadId,&numbOfNodesInStore,globalThreadId);

		//Go through the node list and find all the new nodes we have added and then add these guys to the openList
		int nodesInOpenList = 0;

		for(int allNodesCounter=0;allNodesCounter<numbOfNodesInStore;allNodesCounter++){
			//If we have a new node in the list, then calculate their g costs
			if( (nodeList[(threadIdx.x*NUMBER_OF_NODES) + allNodesCounter].isInList) == GOAP_GPGPU::is_new_node ){
				//nodeList[(threadIdx.x*NUMBER_OF_NODES) + allNodesCounter].g = (nodeList[(threadIdx.x*NUMBER_OF_NODES) + allNodesCounter].g) + (currNode->g);
				differences = 0;
				countDifferences(&currState,&(nodeList[(threadIdx.x*NUMBER_OF_NODES) + allNodesCounter].stateAtNode),&differences);
				int cost	= currNode->g + differences;

				for(int oListCounter=0;oListCounter<numbOfNodesInStore;oListCounter++){
					if( (nodeList[(threadIdx.x*NUMBER_OF_NODES) + oListCounter].isInList) == GOAP_GPGPU::in_open_list ){
						nodesInOpenList++;
						if(isExactMatch(&(nodeList[(threadIdx.x*NUMBER_OF_NODES) + allNodesCounter].stateAtNode),&(nodeList[(threadIdx.x*NUMBER_OF_NODES) + oListCounter].stateAtNode))){//if there are no differences, ie they are the same state nodes
							//If there are no differences check the g costs, if the one in the open list has a higher g than the one in the all nodes list, then remove it
							if( cost < (nodeList[(threadIdx.x*NUMBER_OF_NODES) + oListCounter].g)){
								(nodeList[(threadIdx.x*NUMBER_OF_NODES) + oListCounter]).isInList = GOAP_GPGPU::in_no_list;
							}
						}
					}
				}

				for(int cListCounter=0;cListCounter<numbOfNodesInStore;cListCounter++){
					if( (nodeList[(threadIdx.x*NUMBER_OF_NODES) + cListCounter].isInList) == GOAP_GPGPU::in_closed_list){
						if(isExactMatch(&(nodeList[(threadIdx.x*NUMBER_OF_NODES) + allNodesCounter].stateAtNode),&(nodeList[(threadIdx.x*NUMBER_OF_NODES) + cListCounter].stateAtNode))){//if there are no differences, ie they are the same state nodes
							//If there are no differences check the g costs, if the one in the CLOSED list has a higher g than the one in the all nodes list, then remove it
							if(  cost < (nodeList[(threadIdx.x*NUMBER_OF_NODES) + cListCounter].g)){
								(nodeList[(threadIdx.x*NUMBER_OF_NODES) + cListCounter]).isInList = GOAP_GPGPU::in_no_list;
							}//
						}
					}
				}

				
				(nodeList[(threadIdx.x*NUMBER_OF_NODES) + allNodesCounter]).isInList	= GOAP_GPGPU::in_open_list;
				(nodeList[(threadIdx.x*NUMBER_OF_NODES) + allNodesCounter]).parent		= currNode;
				(nodeList[(threadIdx.x*NUMBER_OF_NODES) + allNodesCounter]).g			= cost;
			}
		}
		//__syncthreads();
		if(nodesInOpenList==0){
			//printf("Quit without anything Thread %i Block %i \n",threadIdx.x,blockIdx.x);
			break;
		}
	}
	
}

__device__ bool isExactMatch(GOAP_GPGPU::GOAP_STATE_COMPACT* firstState,GOAP_GPGPU::GOAP_STATE_COMPACT* secondState){
	//AND it, if not all 1,s then puff!! 
	unsigned int conditions		= (unsigned int)(firstState->conditions1)^(unsigned int)(secondState->conditions1);
	unsigned int ignores		= (unsigned int)(firstState->conditionIgnoreState1)^(unsigned int)(secondState->conditionIgnoreState1);

	//if this is the same thing, then we need to return true, if this is true the value here should be 0
	return !(conditions+ignores);
}



//We are passing the mem location, since, we can apply blocking and non blocking by simply changing the physical location of mem pointed (ie shared vs global)
//Agent id is normally the global thread id

/*
	int* intMemLocation									= Points to where the int type mem are stored. This could be shared or global memory.
	GOAP_GPGPU::Vector4* playerPositions,				= Points to where the rearranged played positions are at, again, could be shared, or global.
	GOAP_GPGPU::Vector3* allyPos						= pointer to the position of the ally in 3d space
	homingPoint											= 
	int agentID,										= Id of the agent with reference to the intMemlocation
	int allyId											= Id of the ally in int data
*/

__device__ void calculateAgentGoapState(
	int* agentIntMemLocation,int* enemyIntMemLocation,GOAP_GPGPU::Vector3* agentPosition,GOAP_GPGPU::Vector3* allyPos,GOAP_GPGPU::Vector4* enemyPos,
	GOAP_GPGPU::Vector4* homingPoint,int agentID,int allyId,int numberOfAgentDataInInt,int* resultState,int* resultStateIgnore){

	//The #defines helps "define" :P in which order the vars are stored in the players
	#define AGNT_HEALTH_LOC			0
	#define AGNT_MP_LOC				1
	#define AGNT_STATUS_EFFECTS_LOC	2
	#define AGNT_NPC_TYPE_LOC		3
	#define AGNT_EQUIPMENT_LOC		4
	#define AGNT_PICKED_UP_ITEMS	5
	#define AGENT_GOAL_VALUE_LOC	6
	#define	AGENT_GOAL_IGNORE_LOC	7

	#define PLR_HEALTH_LOC			0
	#define PLR_MP_LOC				1
	#define PLR_FRACTION			2
	#define PLR_STATE_LOC			3
	#define PLR_STATUS_EFFECTS_LOC	4

	#define LOW_OUT_OF_100			51
	#define CRITICAL_OUT_OF_100     31
	#define RISK_OF_DEATH_OUT_OF100	5

	#define CAST_ON_ALLY_RADIUS		5
	#define CAST_ON_ENEMY_RADIUS	5
	#define WEAPON_ON_ENEMY_RADIUS	2
	#define RANGED_ON_ENEMY_RADIUS	5
	#define MELEE_ON_ENEMY_RADIUS	1

	unsigned int state			= 0; //start with all zeros, we are not going to bother about the ignores here, as we are calculating everything for the agent here
	unsigned int tmpBitHolder	= 0; //HoHo change name later lol cant think now :P

	int tmpIndex = 0;

	//This one or will calculate the equipment *_EQUIPPED stat for many in one or, this works well because we have equipment stats first
	//And in an order thats insync with EQUIPMENT(enum) KNIFE_EQUIPPED;STONE_EQUIPPED;SWORD_EQUIPPED;STAFF_EQUIPPED;BLADE_EQUIPPED;SPEAR_EQUIPPED;LONG_SWORD_EQUIPPED;
	state = state | agentIntMemLocation[(AGNT_EQUIPMENT_LOC * numberOfAgentDataInInt) + agentID];

	//Ally defense up and attack up state for ALLY
	tmpBitHolder = agentIntMemLocation[(AGNT_STATUS_EFFECTS_LOC * numberOfAgentDataInInt) + allyId];
	tmpBitHolder = ((tmpBitHolder<<30)>>30)<<7;// clear left and right except for what we want, def and attack
	state = state | tmpBitHolder;

	//Now the AGENT's OWN own def and attack stats
	tmpBitHolder = agentIntMemLocation[(AGNT_STATUS_EFFECTS_LOC * numberOfAgentDataInInt) + agentID];
	tmpBitHolder = ((tmpBitHolder<<30)>>30)<<9;
	state = state | tmpBitHolder;

	//state = state | ( ((bool)numberOfLiveEnemies)<<13 );//So anything thats above 0 will tell us there are enemies nearby

	//HP and MP - Low, And Crit. Crit is less than 30, low is Less than 50
	
	//SELF_HP_LOW;					
	tmpBitHolder = agentIntMemLocation[(AGNT_HEALTH_LOC * numberOfAgentDataInInt) + agentID];
	tmpBitHolder = (int)tmpBitHolder - (int)LOW_OUT_OF_100;// Treat the values as signed int
	//if - then low hp, if + ok
	state = state | ( (tmpBitHolder>>31)<<15 );
	
	//SELF_MP_LOW; // same as before
	tmpBitHolder = agentIntMemLocation[(AGNT_MP_LOC * numberOfAgentDataInInt) + agentID];
	tmpBitHolder = (int)tmpBitHolder - (int)LOW_OUT_OF_100;// Treat the values as signed int otherwise people will eb sad
	state = state | ( (tmpBitHolder>>31)<<16 );
	
	//SELF_HP_CRITICAL;				
	tmpBitHolder = agentIntMemLocation[(AGNT_HEALTH_LOC * numberOfAgentDataInInt) + agentID];
	tmpBitHolder = (int)tmpBitHolder - (int)CRITICAL_OUT_OF_100;// Treat the values as signed int otherwise people will eb sad
	state = state | ( (tmpBitHolder>>31)<<17 );

	//SELF_RISK_OF_DEATH;
	tmpBitHolder = agentIntMemLocation[(AGNT_HEALTH_LOC * numberOfAgentDataInInt) + agentID];
	tmpBitHolder = (int)tmpBitHolder - (int)RISK_OF_DEATH_OUT_OF100;// Treat the values as signed int otherwise people will eb sad
	state = state | ( (tmpBitHolder>>31)<<18 );

	//ALLY_MP_LOW;					
	tmpBitHolder = agentIntMemLocation[(AGNT_MP_LOC * numberOfAgentDataInInt) + agentID];
	tmpBitHolder = (int)tmpBitHolder - (int)LOW_OUT_OF_100;// Treat the values as signed int otherwise people will eb sad
	state = state | ( (tmpBitHolder>>31)<<19 );

	//ALLY_HP_LOW;
	tmpBitHolder = agentIntMemLocation[(AGNT_HEALTH_LOC * numberOfAgentDataInInt) + agentID];
	tmpBitHolder = (int)tmpBitHolder - (int)LOW_OUT_OF_100;// Treat the values as signed int otherwise people will eb sad
	state = state | ( (tmpBitHolder>>31)<<20 );

	//AGNT_PICKED_UP_ITEMS
	//HAS_MP_POTION; is 	1<<0				
	//HAS_HP_POTION;		1<<1
	tmpBitHolder = agentIntMemLocation[(AGNT_PICKED_UP_ITEMS * numberOfAgentDataInInt) + agentID];
	tmpBitHolder = ((tmpBitHolder<<30)>>30)<<21;// clear left and right except for what we want, def and attack
	state = state | tmpBitHolder;


	//ALLY_IN_SPELL_RANGE;
	float x =  allyPos->x - agentPosition->x;
	float y =  allyPos->y - agentPosition->y;
	float z =  allyPos->z - agentPosition->z;
	float lengthSq	= (x*x)+(y*y)+(z*z);
	float sumRadius_sq	= CAST_ON_ALLY_RADIUS * 2;
	sumRadius_sq=sumRadius_sq*sumRadius_sq;
	//sooo if lengthSq < sumRadius_sq is true then set the state to true
	int stupidInt;
	stupidInt		= sumRadius_sq - lengthSq; // this is okay since we are only worried about the size of stuff man  // you are going to get a warning, but fuckit
	tmpBitHolder	= stupidInt;
	state = state | ((tmpBitHolder>>31)<<23);

	if(enemyPos){ //if we have an enemy to deal with 
		//Now ENEMY def and attack stats
		tmpIndex		= (PLAYER_STATUS_EFFECTS_BEGIN + (enemyPos->w));//w is the original player id

		tmpBitHolder	= enemyIntMemLocation[tmpIndex];
		tmpBitHolder	= ((tmpBitHolder<<30)>>30)<<11;
		state			= state | tmpBitHolder;

		//To see if someone is alive, we take their HP, and add 127 to it, since the max HP we can have 
		//is 100, it takes at least 7 bits of data to represent a 100 in base 2.
		//127 is the largest number we can store in 7 bits, so, if we have a +ve HP value 127 + 
		//(+HP) means the 8th bit will be 1 :D and -ve HP means 8th bit will be 0 
		//HAS_LIVE_ENEMY 
		
		//Find the location where the player's(enemy's) health is stored
		tmpIndex		= (PLAYER_HEALTH_BEGIN + (enemyPos->w)); 

		tmpBitHolder	= enemyIntMemLocation[tmpIndex];
		//The 8th bit now hold's our answer, so shift<< shift>>
		tmpBitHolder	= tmpBitHolder + 127;
		state			= state|(((tmpBitHolder<<24)>>31)<<13);

		//IS_ENEMY_FRACTION
		//fractions are not implemented now we will do this later, for now everyone hates everyone :D
		tmpBitHolder = 1;
		tmpBitHolder = tmpBitHolder<<14;
		state	= state|tmpBitHolder;

		//ENEMY_IN_WEAPON_ATTACK_RANGE;
		x =  enemyPos->x - agentPosition->x;
		y =  enemyPos->y - agentPosition->y;
		z =  enemyPos->z - agentPosition->z;
		lengthSq	= (x*x)+(y*y)+(z*z);
		sumRadius_sq	= WEAPON_ON_ENEMY_RADIUS * 2;
		sumRadius_sq=sumRadius_sq*sumRadius_sq;
		stupidInt	= 0;
		stupidInt		= sumRadius_sq - lengthSq; // this is okay since we are only worried about the size of stuff man  // you are going to get a warning, but fuckit
		tmpBitHolder	= stupidInt;
		state = state | ((tmpBitHolder>>31)<<24);

		//ENEMY_IN_SPELL_ATTACK_RANGE;	
		sumRadius_sq	= CAST_ON_ENEMY_RADIUS * 2;
		sumRadius_sq=sumRadius_sq*sumRadius_sq;
		stupidInt	= 0;
		stupidInt		= sumRadius_sq - lengthSq; // this is okay since we are only worried about the size of stuff man  // you are going to get a warning, but fuckit
		tmpBitHolder	= stupidInt;
		state = state | ((tmpBitHolder>>31)<<25);

		//ENEMY_IN_RANGED_ATTACK_RANGE;
		sumRadius_sq	= RANGED_ON_ENEMY_RADIUS * 2;
		sumRadius_sq=sumRadius_sq*sumRadius_sq;
		stupidInt	= 0;
		stupidInt		= sumRadius_sq - lengthSq; // this is okay since we are only worried about the size of stuff man  // you are going to get a warning, but fuckit
		tmpBitHolder	= stupidInt;
		state = state | ((tmpBitHolder>>31)<<26);

		//ENEMY_IN_MELEE_RANGE;			
		sumRadius_sq	= MELEE_ON_ENEMY_RADIUS * 2;
		sumRadius_sq=sumRadius_sq*sumRadius_sq;
		stupidInt	= 0;
		stupidInt		= sumRadius_sq - lengthSq; // this is okay since we are only worried about the size of stuff man  // you are going to get a warning, but fuckit
		tmpBitHolder	= stupidInt;
		state = state | ((tmpBitHolder>>31)<<27);
	}//No need to do any thing to else, coz shifting 0s means nothing :D 
	*resultState		= state;
	*resultStateIgnore	= 0;
}

/*
Take two states A and B, C
apply state A to B, return the result on C
*/
__device__ void applyStateAtoB(GOAP_GPGPU::GOAP_STATE_COMPACT* stateA,GOAP_GPGPU::GOAP_STATE_COMPACT* stateB,GOAP_GPGPU::GOAP_STATE_COMPACT* answer){
	*answer = *stateB;

	unsigned int stateAtmpBit = stateA->conditionIgnoreState1;

	for(int i=0;i<GOAP_GPGPU::NUMBER_OF_STATES;i++){

		stateAtmpBit = stateA->conditionIgnoreState1;
		stateAtmpBit = ((stateAtmpBit<<(31-i))>>31);

		//IF is non ignore value for A (meaning, its important to state A)
		if(stateAtmpBit == 0){
			//We need to toggle the bit if the bits are not the same noh :)
			if( (((stateA->conditions1)<<(31-i))>>31) != (((stateB->conditions1)<<(31-i))>>31) ){
				(answer->conditions1) = (answer->conditions1) ^ (1 << i);
			}
			//if this guy does not care right now, make him care :P
			if( (((stateB->conditionIgnoreState1)<<(31-i))>>31) == 1 ){
				//number =number 1 << x; -- setting a bit
				//number =number & ( ~(1 << x));
				//(answer->conditionIgnoreState1) = (answer->conditionIgnoreState1) ^ (1 << i);
				(answer->conditionIgnoreState1) = (answer->conditionIgnoreState1) & ( ~(1 << i));
			}
		}
	}

	//if(goapStateIn.hasBullets != -1){
	//		this->hasBullets = goapStateIn.hasBullets;
	//	}


//	unsigned int valueDifferences		= 0; //Stores the values that are different in each of our bits
//	unsigned int curruntIgnoreStateOfA	= 0;
//	int tmpBit							= 0;
//	unsigned int tmpBit2				= 0;
////	unsigned int valid_valueDifferences	= 0; 
//
//	valueDifferences = (stateA->conditions1) ^ (stateB->conditions1); //This step gives us the differences of the values in A and B
//	curruntIgnoreStateOfA = stateA->conditionIgnoreState1;
//
//	//Now the trick is applying the differences to B, ie, changing whatever value is at B's values, to the opposite if there's a difference.
//	//But there's a catch!!! We have to APPLY THIS DIFFERENCE, If and ONLY IF!! ignore state at A for this change is 0 (ie, dont ignore for 0)
//	unsigned int tmpBsbit = 0;
//
//	for(int i=0;i<GOAP_GPGPU::NUMBER_OF_STATES;i++){
//		//We are looking for a condition where, you have 0 for ignore state, 1 for change
//		//ie. 0 -1, will leave a negative number, meaning the first bit will be 1 coz -1
//		tmpBit = ((curruntIgnoreStateOfA<<(31-i))>>31) - ((valueDifferences<<(31-i))>>31); //Now if we have a -1 at temp bit, we know we are dealing with a change 
//				
//		tmpBsbit = (stateB->conditions1 << (31-i));
//
//
//		if(	tmpBit == -1){
//			//This means a change we want is here then flip the bit in question in B
//			tmpBsbit = ~tmpBsbit;
//		}
//
//		tmpBsbit = tmpBsbit >> 31;
//		tmpBsbit = tmpBsbit << i;
//
//		tmpBit2 = tmpBit2 | tmpBsbit;
//		
//		tmpBsbit	= 0;
//		tmpBit		= 0;
//	}
//	//AND we need to set the ignore bit of tmpBit2 to not ignore T_T ie, true
//	answer->conditionIgnoreState1 = stateB->conditionIgnoreState1|stateA->conditionIgnoreState1;
//	answer->conditions1 = tmpBit2;



}

__host__ void writeResultsToFile(int *dataYo,int intSize, std::string fileName){
	std::ofstream myfile;
	fileName = "C:/Users/B3057489/Desktop/hohoho.txt";
	myfile.open (fileName);


	for(int i=0;i<intSize;i++){
		myfile<<(dataYo[i])<<",";
		if((intSize%(48*2))==0){ //OMG HARDCODING LAZY!! BITCH!!!
			myfile<<std::endl;
		} //d
	}
}


/*
	PURPOSE: Takes the ID of an agent, looks up at the actions available to him, and then builds new nodes which can be run by this agent at this time. 
	GOAP_GPGPU::AStarNodeCompact* nodesStore	==> give the mem location to store our 
	int* intData								==> To access GOAP information 
	int agentId									==> with reference to the mem locations in *intData
*/
__device__ void generateActionsForCurruntState(GOAP_GPGPU::AStarNodeCompact* nodesStore,GOAP_GPGPU::GOAP_STATE_COMPACT* currState,GOAP_GPGPU::GOAP_STATE_COMPACT* goalState,int* intData,int globalThreadId,int* numberOfNodesInNodeStore,int agentId){
	//First 32 actions
	unsigned int tmpBitVar = 0;
	const int NUMBER_OF_NODES = 48 * 2;
	//Go through each action
	tmpBitVar = 0;
	tmpBitVar = 1<<(intData[AGENT_TYPE_BEGIN + agentId]);

	for(int i=0;i<32;i++){
		//Check if this agent is allowed to run this action
		if( ((tmpBitVar) & (intData[GOAP_ACTION_ALLOWED_TO + i]))  ){

			GOAP_GPGPU::GOAP_STATE_COMPACT currActionsPostondition;
			currActionsPostondition.conditions1				= intData[GOAP_ACTION_POSTCONDITIONS_VALUES_BEGIN+i];
			currActionsPostondition.conditionIgnoreState1	= intData[GOAP_ACTION_POSTCONDITIONS_IGNORE_BEGIN+i];

			GOAP_GPGPU::GOAP_STATE_COMPACT currActionsPrecondition;
			currActionsPrecondition.conditions1				= intData[GOAP_ACTION_PRECONDITIONS_VALUES_BEGIN+i];
			currActionsPrecondition.conditionIgnoreState1	= intData[GOAP_ACTION_PRECONDITIONS_IGNORE_BEGIN+i];

			int differences = 0;
			//Count differences between the current state and the state of the action we are checking , can we run the action?
			countDifferences(&currActionsPrecondition,currState, &differences);

			if(!differences){ //if there are no differences then apply the condition
				GOAP_GPGPU::AStarNodeCompact tempNode;

				//Apply to the current state, the state of the post condition of the action we are dealing with
				GOAP_GPGPU::GOAP_STATE_COMPACT currentStateAfterPostconditionApplied; //Good god this var name :D :D 
				applyStateAtoB(&currActionsPostondition,currState,&currentStateAfterPostconditionApplied);

				tempNode.stateAtNode = currentStateAfterPostconditionApplied;
				tempNode.actionName = intData[GOAP_ACTION_NAMES_BEGIN + i];
				tempNode.g = intData[GOAP_ACTION_G_COST_BEGIN + i]; //Run the G calculator here'

				differences = 0;
				countDifferences(goalState,&(currentStateAfterPostconditionApplied), &differences);
				tempNode.h = differences;
				tempNode.parent = 0;
				tempNode.isInList = GOAP_GPGPU::is_new_node; //0 means NOT in any LIST, 1= OPEN LIST 2 = CLOSED LIST 4 = OPEN AND CLOSED LIST
				

				bool alreadyExsist = false;
				//Check if this new node is in the shared mem already, if its there why add.
				for(int k=0;k<(*numberOfNodesInNodeStore);k++){
					if((isExactMatch(&(nodesStore[(threadIdx.x*NUMBER_OF_NODES)+(k)].stateAtNode),&(tempNode.stateAtNode))) && ((nodesStore[(threadIdx.x*NUMBER_OF_NODES)+(k)].h) == tempNode.h ) && ((nodesStore[(threadIdx.x*NUMBER_OF_NODES)+(k)].actionName) == tempNode.actionName ) ){
						alreadyExsist = true;
						break;
					}
				}
				if(!alreadyExsist){
					nodesStore[(threadIdx.x*NUMBER_OF_NODES)+(*numberOfNodesInNodeStore)] = tempNode;
					if(*numberOfNodesInNodeStore < (NUMBER_OF_NODES-1)){
						(*numberOfNodesInNodeStore)++;
					}else{
						//printf("Exceded Thread: %i, Block %i \n",threadIdx.x,blockIdx.x);
					}
				}
			}
		}
	}
	

	//The next actions (actions from 32 to 42)
	for(int i=32;i<GOAP_GPGPU::NUMBER_OF_ACTIONS;i++){
		//Check if this agent is allowed to run this action
		if( (tmpBitVar) & (intData[GOAP_ACTION_ALLOWED_TO + i]) ){

			GOAP_GPGPU::GOAP_STATE_COMPACT currActionsPostondition;
			currActionsPostondition.conditions1				= intData[GOAP_ACTION_POSTCONDITIONS_VALUES_BEGIN+i];
			currActionsPostondition.conditionIgnoreState1	= intData[GOAP_ACTION_POSTCONDITIONS_IGNORE_BEGIN+i];

			GOAP_GPGPU::GOAP_STATE_COMPACT currActionsPrecondition;
			currActionsPrecondition.conditions1				= intData[GOAP_ACTION_PRECONDITIONS_VALUES_BEGIN+i];
			currActionsPrecondition.conditionIgnoreState1	= intData[GOAP_ACTION_PRECONDITIONS_IGNORE_BEGIN+i];

			int differences = 0;
			//Count differences between the current state and the state of the action we are checking 
			countDifferences(&currActionsPrecondition,currState, &differences);
			if(!differences){ //if there are no differences then apply the condition
				GOAP_GPGPU::AStarNodeCompact tempNode;

				//Apply to the current state, the state of the post condition of the action we are dealing with
				GOAP_GPGPU::GOAP_STATE_COMPACT currentStateAfterPostconditionApplied; //Good god this var name :D :D 
				applyStateAtoB(&currActionsPostondition,currState,&currentStateAfterPostconditionApplied);

				tempNode.stateAtNode = currentStateAfterPostconditionApplied;
				tempNode.actionName = intData[GOAP_ACTION_NAMES_BEGIN + i];
				tempNode.g = intData[GOAP_ACTION_G_COST_BEGIN + i]; //Run the G calculator here'

				differences = 0;
				countDifferences(goalState,&(currentStateAfterPostconditionApplied), &differences);
				tempNode.h = differences;
				tempNode.parent = 0;
				tempNode.isInList = GOAP_GPGPU::is_new_node; //0 means NOT in any LIST, 1= OPEN LIST 2 = CLOSED LIST 4 = OPEN AND CLOSED LIST
			
				bool alreadyExsist = false;
				//Check if this new node is in the shared mem already, if its there why add.
				for(int k=0;k<(*numberOfNodesInNodeStore);k++){
					if((isExactMatch(&(nodesStore[(threadIdx.x*NUMBER_OF_NODES)+(k)].stateAtNode),&(tempNode.stateAtNode))) && ((nodesStore[(threadIdx.x*NUMBER_OF_NODES)+(k)].h) == tempNode.h ) && ((nodesStore[(threadIdx.x*NUMBER_OF_NODES)+(k)].actionName) == tempNode.actionName ) ){
						alreadyExsist = true;
						break;
					}
				}
				if(!alreadyExsist){
					nodesStore[(threadIdx.x*NUMBER_OF_NODES)+(*numberOfNodesInNodeStore)] = tempNode;
					if(*numberOfNodesInNodeStore < (NUMBER_OF_NODES-1)){
						(*numberOfNodesInNodeStore)++;
					}else{
						//printf("Exceded Thread: %i, Block %i \n",threadIdx.x,blockIdx.x);
					}
				}
			}
		}
	}
}



__device__ __host__ GOAP_GPGPU::AStarNodeCompact* findLowestFCostNodeInList(GOAP_GPGPU::AStarNodeCompact* list, int sizeOfList,int startAt, int *foundAt){
	//go through all the nodes which are not NULL, and find an answer.
	int lowest_f_found_so_far	= 999;
	*foundAt					= startAt;
	//Find the first non NULL occurrence so we have somewhere to start with
	for(int i=startAt;i<(startAt+sizeOfList);i++){
		//if its in the open list
		if(list[i].isInList == GOAP_GPGPU::in_open_list){
			//has a low cost
			if((int)((int)(list[i].g)+(int)(list[i].h)) < lowest_f_found_so_far){
				*foundAt				= i;
				lowest_f_found_so_far	= ((int)(list[i].g)+(int)(list[i].h));
			}
		}
	}
	(list[*foundAt]).isInList = GOAP_GPGPU::in_closed_list;
	return &(list[*foundAt]);
}

/*
	PURPOSE: This function will go through a list of players, and arrange them according to the blocks(agent party's home radius). 
	starting with block 0 till NUMBER_OF_BLOCKS_IN_X and in the end will append the homing coordinates to the end, and each w of the homing coordinate vec4 will
	tell us how many players are in each block

	so we have a Vec4 array that looks like,
		[player x-y-z and in w player ID][playerxyzw][playerxyzw][playerxyzw][homingPoints][homingPoints][0,0,0 homingPoint which stores how many agents are not anywhere near]
*/
__host__ void arrangePlayersByBlock(GOAP_GPGPU::PlayerInfo* players, GOAP_GPGPU::Vector4* playerAndHomingLocations){
	//append the homingPoints to the end of the playerAndHomingLocations array
	for(int i=0;i<NUMBER_OF_BLOCKS_IN_X;i++){
		playerAndHomingLocations[PLAYER_TOTAL + i] = GOAP_GPGPU::homingPoints[i];
	}
	
	//For each predefined homing point, check how many players(enemies) are in that area and add it to the respectable homing point's w,
	//so we know, for each given homing point how many enemies we have
	int counter = 0;// the counter is here to help us order our players(enemies) in to groups.
	float sumRadius_sq	= SPHERE_RADIUS_SQ*2;
	sumRadius_sq=sumRadius_sq*sumRadius_sq;

	for(int i = 0;i<NUMBER_OF_BLOCKS_IN_X;i++){

		int playersPerBlockCounter = 0;

		for(int j=0;j<PLAYER_TOTAL;j++){
			if(!(players[j].checked)){//if not yet checked check for collision :)

				//We are accessing the homing location here PLAYER_TOTAL + i 
				float x = playerAndHomingLocations[PLAYER_TOTAL + i].x - players[j].position.x;
				float y = playerAndHomingLocations[PLAYER_TOTAL + i].y - players[j].position.y;
				float z = playerAndHomingLocations[PLAYER_TOTAL + i].z - players[j].position.z;

				float lengthSq	= (x*x)+(y*y)+(z*z);
				//if intersection, that means the player is in the block attack area

				if(lengthSq < sumRadius_sq){
					playerAndHomingLocations[counter]	= GOAP_GPGPU::Vector4(players[j].position,(float)j);
					counter++;

					players[j].checked = true;
					playersPerBlockCounter++;
				}
			}
		}
		playerAndHomingLocations[PLAYER_TOTAL + i].w = (float)playersPerBlockCounter;
	}

	int wanderingPlayers = 0 ;

	for(int i=0;i<PLAYER_TOTAL;i++){
		//If player not yet added the add the player to the ist and remember how many players are walking around doing nothing
		if(!(players[i].checked)){
			playerAndHomingLocations[counter]	= GOAP_GPGPU::Vector4(players[i].position, (float)i);
			players[i].checked					= true;
			counter++;
			wanderingPlayers++;
		}
	}
	playerAndHomingLocations[PLAYER_TOTAL + NUMBER_OF_BLOCKS_IN_X] = GOAP_GPGPU::Vector4((float)wanderingPlayers,(float)wanderingPlayers,(float)wanderingPlayers,(float)wanderingPlayers);
}

__host__ void getAgentPositions(GOAP_GPGPU::AgentInfo* agents, GOAP_GPGPU::Vector3* agent_positions){
	for(int j=0;j<AGENT_TOTAL;j++){
		agent_positions[j] = agents[j].position;
	}
}

void startTimer(){
	// get ticks per second
	QueryPerformanceFrequency(&frequency);
	// start timer
	QueryPerformanceCounter(&t1);
}

double getElapsedTime(){
	double elapsedTime =0.0f;

	// stop timer
	QueryPerformanceCounter(&t2);

	// compute and print the elapsed time in millisec
	elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;

	return elapsedTime;
}

__host__ void generateData(int numberOfPlayers,int numberOfAgents,GOAP_GPGPU::PlayerInfo* playersArr,GOAP_GPGPU::AgentInfo* agentsArr){
	std::vector<std::string> myStrings;

	/*myStrings.push_back("BloodWolf_Cleric");
	myStrings.push_back("Goblin_Cleric");
	myStrings.push_back("Dullahan_cleric");
	myStrings.push_back("Harpy_cleric");
	myStrings.push_back("Hydra_cleric");
	myStrings.push_back("Hippalectryon_cleric");
	myStrings.push_back("Mermen_Priest");
	myStrings.push_back("BloodWolf_Priest");
	myStrings.push_back("Goblin_Priest");
	myStrings.push_back("Harpy_Priest");
	myStrings.push_back("Hydra_Priest");
	myStrings.push_back("Hippalectryon_Priest");
	myStrings.push_back("Mermen_Attacker");
	myStrings.push_back("BloodWolf_Attacker");
	myStrings.push_back("Goblin_Attacker");
	myStrings.push_back("Dullahan_Attacker");
	myStrings.push_back("Harpy_Attacker");
	myStrings.push_back("Hydra_Attacker");
	myStrings.push_back("Hippalectryon_Attacker");
	myStrings.push_back("Mermen_Mage");
	myStrings.push_back("BloodWolf_Mage");
	myStrings.push_back("Goblin_Mage");
	myStrings.push_back("Harpy_Mage");
	myStrings.push_back("Hydra_Mage");
	myStrings.push_back("Hippalectryon_Mage");
	myStrings.push_back("Mermen_warrior");
	myStrings.push_back("BloodWolf_warrior");*/
	myStrings.push_back("Goblin_warrior");
	myStrings.push_back("Dullahan_warrior");
	myStrings.push_back("Harpy_warrior");
	myStrings.push_back("Hydra_warrior");
	myStrings.push_back("Hippalectryon_warrior");

	std::vector<std::string> myStrings2;
	myStrings2.push_back("knife");
	myStrings2.push_back("stone");
	myStrings2.push_back("sword");
	myStrings2.push_back("staff");
	myStrings2.push_back("blade");
	myStrings2.push_back("spear");
	myStrings2.push_back("longSword");

	int counterPlayer	= 0;
	int counterAgent	= 0;

	for(int i=0;i<100;i++){
		for(int j=0;j<100;j++){
			
			if(rand() % 2){
				if(counterAgent >= numberOfPlayers)
					break;
				playersArr[counterPlayer].position		= GOAP_GPGPU::Vector3(i,0,j);
				playersArr[counterPlayer].health		= (rand() % 101);
				playersArr[counterPlayer].mp			= (rand() % 101);

					int randomz = rand()%6;
					if(randomz ==1){
						playersArr[counterPlayer].fraction			= GOAP_GPGPU::Wolves;
					}else if(randomz ==2){
						playersArr[counterPlayer].fraction			= GOAP_GPGPU::Goblins;
					}else if(randomz ==3){
						playersArr[counterPlayer].fraction			= GOAP_GPGPU::FishPeople;
					}else if(randomz ==4){
						playersArr[counterPlayer].fraction			= GOAP_GPGPU::BirdPeople;
					}else{
						playersArr[counterPlayer].fraction			= GOAP_GPGPU::reptilians;
					}
				
					randomz = rand()%3;
					if(randomz==1){
						playersArr[counterPlayer].statusEffects		= 1;
					}else if(randomz==2){
						playersArr[counterPlayer].statusEffects		= 2;
					}else{
						playersArr[counterPlayer].statusEffects		= 3;
					}
					counterPlayer++;
				
			}else{
				if(counterAgent >= numberOfAgents)
					break;
				agentsArr[counterAgent].position						= GOAP_GPGPU::Vector3(i,0,j);
				agentsArr[counterAgent].health							= (rand() % 101);
				agentsArr[counterAgent].mp								= (rand() % 101);
				agentsArr[counterAgent].statusEffects					= (rand() % 2+1);
				agentsArr[counterAgent].agentType						= GOAP_GPGPU::NPC_TYPE(rand() % myStrings.size());
				agentsArr[counterAgent].agentEquipmentInHand			= EQUIPMENT(rand() % 7);
				agentsArr[counterAgent].pickedUpItems					= (rand() % 2+1);
				agentsArr[counterAgent].agentGoal.HAS_LIVE_ENEMY		= 0;
				agentsArr[counterAgent].agentGoal.SELF_RISK_OF_DEATH	= 0;
				counterAgent++;
			};
		}
	}
}

__host__ void hardcodedPopulate(GOAP_GPGPU::AgentInfo* agentArr,GOAP_GPGPU::PlayerInfo* playerArr){
	using namespace GOAP_GPGPU;
///* 64 threads
	//Dullahan_warrior
	
	agentArr[0].position					= Vector3(100,880,22);
	agentArr[0].health						= 86;
	agentArr[0].mp							= 55;
	agentArr[0].statusEffects	 				= 1;
	agentArr[0].agentType						= Dullahan_warrior;
	agentArr[0].agentEquipmentInHand		 = staff;
	agentArr[0].pickedUpItems					= 2;
	agentArr[0].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[0].agentGoal.SELF_RISK_OF_DEATH	= 0;
	agentArr[1].position						= Vector3(0,0,1);
	agentArr[1].health						= 86;
	agentArr[1].mp							= 55;
	agentArr[1].statusEffects	 				= 1;
	agentArr[1].agentType						= Dullahan_warrior;
	agentArr[1].agentEquipmentInHand   = staff;
	agentArr[1].pickedUpItems					= 2;
	agentArr[1].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[1].agentGoal.SELF_RISK_OF_DEATH	= 0;
	agentArr[2].position						= Vector3(0,0,2);
	agentArr[2].health						= 86;
	agentArr[2].mp							= 55;
	agentArr[2].statusEffects	 				= 1;
	agentArr[2].agentType						= Dullahan_warrior;
	agentArr[2].agentEquipmentInHand   = staff;
	agentArr[2].pickedUpItems					= 2;
	agentArr[2].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[2].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[0].position			= Vector3(0,0,3);

	playerArr[0].health					= 86;
	playerArr[0].mp					= 55;
	playerArr[0].fraction			= FishPeople;
	playerArr[0].statusEffects				= 2;
	playerArr[1].position			= Vector3(0,0,4);

	playerArr[1].health					= 86;
	playerArr[1].mp					= 55;
	playerArr[1].fraction			= FishPeople;
	playerArr[1].statusEffects				= 2;
	playerArr[2].position			= Vector3(0,0,5);

	playerArr[2].health					= 86;
	playerArr[2].mp					= 55;
	playerArr[2].fraction			= FishPeople;
	playerArr[2].statusEffects				= 2;
	agentArr[3].position						= Vector3(0,0,6);
	agentArr[3].health						= 86;
	agentArr[3].mp							= 55;
	agentArr[3].statusEffects	 				= 1;
	agentArr[3].agentType						= Dullahan_warrior;
	agentArr[3].agentEquipmentInHand   = staff;
	agentArr[3].pickedUpItems					= 2;
	agentArr[3].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[3].agentGoal.SELF_RISK_OF_DEATH	= 0;
	agentArr[4].position						= Vector3(0,0,7);
	agentArr[4].health						= 86;
	agentArr[4].mp							= 55;
	agentArr[4].statusEffects	 				= 1;
	agentArr[4].agentType						= Dullahan_warrior;
	agentArr[4].agentEquipmentInHand   = staff;
	agentArr[4].pickedUpItems					= 2;
	agentArr[4].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[4].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[3].position			= Vector3(0,0,8);

	playerArr[3].health					= 86;
	playerArr[3].mp					= 55;
	playerArr[3].fraction			= FishPeople;
	playerArr[3].statusEffects				= 2;
	agentArr[5].position						= Vector3(0,0,9);
	agentArr[5].health						= 86;
	agentArr[5].mp							= 55;
	agentArr[5].statusEffects	 				= 1;
	agentArr[5].agentType						= Dullahan_warrior;
	agentArr[5].agentEquipmentInHand   = staff;
	agentArr[5].pickedUpItems					= 2;
	agentArr[5].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[5].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[4].position			= Vector3(0,0,10);

	playerArr[4].health					= 86;
	playerArr[4].mp					= 55;
	playerArr[4].fraction			= FishPeople;
	playerArr[4].statusEffects				= 2;
	playerArr[5].position			= Vector3(0,0,11);

	playerArr[5].health					= 86;
	playerArr[5].mp					= 55;
	playerArr[5].fraction			= FishPeople;
	playerArr[5].statusEffects				= 2;
	playerArr[6].position			= Vector3(0,0,12);

	playerArr[6].health					= 86;
	playerArr[6].mp					= 55;
	playerArr[6].fraction			= FishPeople;
	playerArr[6].statusEffects				= 2;
	playerArr[7].position			= Vector3(0,0,13);

	playerArr[7].health					= 86;
	playerArr[7].mp					= 55;
	playerArr[7].fraction			= FishPeople;
	playerArr[7].statusEffects				= 2;
	agentArr[6].position						= Vector3(0,0,14);
	agentArr[6].health						= 86;
	agentArr[6].mp							= 55;
	agentArr[6].statusEffects	 				= 1;
	agentArr[6].agentType						= Dullahan_warrior;
	agentArr[6].agentEquipmentInHand   = staff;
	agentArr[6].pickedUpItems					= 2;
	agentArr[6].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[6].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[8].position			= Vector3(0,0,15);

	playerArr[8].health					= 86;
	playerArr[8].mp					= 55;
	playerArr[8].fraction			= FishPeople;
	playerArr[8].statusEffects				= 2;
	playerArr[9].position			= Vector3(0,0,16);

	playerArr[9].health					= 86;
	playerArr[9].mp					= 55;
	playerArr[9].fraction			= FishPeople;
	playerArr[9].statusEffects				= 2;
	agentArr[7].position						= Vector3(0,0,17);
	agentArr[7].health						= 86;
	agentArr[7].mp							= 55;
	agentArr[7].statusEffects	 				= 1;
	agentArr[7].agentType						= Dullahan_warrior;
	agentArr[7].agentEquipmentInHand   = staff;
	agentArr[7].pickedUpItems					= 2;
	agentArr[7].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[7].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[10].position			= Vector3(0,0,18);

	playerArr[10].health					= 86;
	playerArr[10].mp					= 55;
	playerArr[10].fraction			= FishPeople;
	playerArr[10].statusEffects				= 2;
	playerArr[11].position			= Vector3(0,0,19);

	playerArr[11].health					= 86;
	playerArr[11].mp					= 55;
	playerArr[11].fraction			= FishPeople;
	playerArr[11].statusEffects				= 2;
	playerArr[12].position			= Vector3(0,0,20);

	playerArr[12].health					= 86;
	playerArr[12].mp					= 55;
	playerArr[12].fraction			= FishPeople;
	playerArr[12].statusEffects				= 2;
	playerArr[13].position			= Vector3(0,0,21);

	playerArr[13].health					= 86;
	playerArr[13].mp					= 55;
	playerArr[13].fraction			= FishPeople;
	playerArr[13].statusEffects				= 2;
	agentArr[8].position						= Vector3(0,0,22);
	agentArr[8].health						= 86;
	agentArr[8].mp							= 55;
	agentArr[8].statusEffects	 				= 1;
	agentArr[8].agentType						= Dullahan_warrior;
	agentArr[8].agentEquipmentInHand   = staff;
	agentArr[8].pickedUpItems					= 2;
	agentArr[8].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[8].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[14].position			= Vector3(0,0,23);

	playerArr[14].health					= 86;
	playerArr[14].mp					= 55;
	playerArr[14].fraction			= FishPeople;
	playerArr[14].statusEffects				= 2;
	agentArr[9].position						= Vector3(0,0,24);
	agentArr[9].health						= 86;
	agentArr[9].mp							= 55;
	agentArr[9].statusEffects	 				= 1;
	agentArr[9].agentType						= Dullahan_warrior;
	agentArr[9].agentEquipmentInHand   = staff;
	agentArr[9].pickedUpItems					= 2;
	agentArr[9].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[9].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[15].position			= Vector3(0,0,25);

	playerArr[15].health					= 86;
	playerArr[15].mp					= 55;
	playerArr[15].fraction			= FishPeople;
	playerArr[15].statusEffects				= 2;
	playerArr[16].position			= Vector3(0,0,26);

	playerArr[16].health					= 86;
	playerArr[16].mp					= 55;
	playerArr[16].fraction			= FishPeople;
	playerArr[16].statusEffects				= 2;
	playerArr[17].position			= Vector3(0,0,27);

	playerArr[17].health					= 86;
	playerArr[17].mp					= 55;
	playerArr[17].fraction			= FishPeople;
	playerArr[17].statusEffects				= 2;
	agentArr[10].position						= Vector3(0,0,28);
	agentArr[10].health						= 86;
	agentArr[10].mp							= 55;
	agentArr[10].statusEffects	 				= 1;
	agentArr[10].agentType						= Dullahan_warrior;
	agentArr[10].agentEquipmentInHand   = staff;
	agentArr[10].pickedUpItems					= 2;
	agentArr[10].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[10].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[18].position			= Vector3(0,0,29);

	playerArr[18].health					= 86;
	playerArr[18].mp					= 55;
	playerArr[18].fraction			= FishPeople;
	playerArr[18].statusEffects				= 2;
	playerArr[19].position			= Vector3(0,0,30);

	playerArr[19].health					= 86;
	playerArr[19].mp					= 55;
	playerArr[19].fraction			= FishPeople;
	playerArr[19].statusEffects				= 2;
	playerArr[20].position			= Vector3(0,0,31);

	playerArr[20].health					= 86;
	playerArr[20].mp					= 55;
	playerArr[20].fraction			= FishPeople;
	playerArr[20].statusEffects				= 2;
	playerArr[21].position			= Vector3(0,0,32);

	playerArr[21].health					= 86;
	playerArr[21].mp					= 55;
	playerArr[21].fraction			= FishPeople;
	playerArr[21].statusEffects				= 2;
	agentArr[11].position						= Vector3(0,0,33);
	agentArr[11].health						= 86;
	agentArr[11].mp							= 55;
	agentArr[11].statusEffects	 				= 1;
	agentArr[11].agentType						= Dullahan_warrior;
	agentArr[11].agentEquipmentInHand   = staff;
	agentArr[11].pickedUpItems					= 2;
	agentArr[11].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[11].agentGoal.SELF_RISK_OF_DEATH	= 0;
	agentArr[12].position						= Vector3(0,0,34);
	agentArr[12].health						= 86;
	agentArr[12].mp							= 55;
	agentArr[12].statusEffects	 				= 1;
	agentArr[12].agentType						= Dullahan_warrior;
	agentArr[12].agentEquipmentInHand   = staff;
	agentArr[12].pickedUpItems					= 2;
	agentArr[12].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[12].agentGoal.SELF_RISK_OF_DEATH	= 0;
	agentArr[13].position						= Vector3(0,0,35);
	agentArr[13].health						= 86;
	agentArr[13].mp							= 55;
	agentArr[13].statusEffects	 				= 1;
	agentArr[13].agentType						= Dullahan_warrior;
	agentArr[13].agentEquipmentInHand   = staff;
	agentArr[13].pickedUpItems					= 2;
	agentArr[13].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[13].agentGoal.SELF_RISK_OF_DEATH	= 0;
	agentArr[14].position						= Vector3(0,0,36);
	agentArr[14].health						= 86;
	agentArr[14].mp							= 55;
	agentArr[14].statusEffects	 				= 1;
	agentArr[14].agentType						= Dullahan_warrior;
	agentArr[14].agentEquipmentInHand   = staff;
	agentArr[14].pickedUpItems					= 2;
	agentArr[14].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[14].agentGoal.SELF_RISK_OF_DEATH	= 0;
	agentArr[15].position						= Vector3(0,0,37);
	agentArr[15].health						= 86;
	agentArr[15].mp							= 55;
	agentArr[15].statusEffects	 				= 1;
	agentArr[15].agentType						= Dullahan_warrior;
	agentArr[15].agentEquipmentInHand   = staff;
	agentArr[15].pickedUpItems					= 2;
	agentArr[15].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[15].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[22].position			= Vector3(0,0,38);

	playerArr[22].health					= 86;
	playerArr[22].mp					= 55;
	playerArr[22].fraction			= FishPeople;
	playerArr[22].statusEffects				= 2;
	agentArr[16].position						= Vector3(0,0,39);
	agentArr[16].health						= 86;
	agentArr[16].mp							= 55;
	agentArr[16].statusEffects	 				= 1;
	agentArr[16].agentType						= Dullahan_warrior;
	agentArr[16].agentEquipmentInHand   = staff;
	agentArr[16].pickedUpItems					= 2;
	agentArr[16].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[16].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[23].position			= Vector3(0,0,40);

	playerArr[23].health					= 86;
	playerArr[23].mp					= 55;
	playerArr[23].fraction			= FishPeople;
	playerArr[23].statusEffects				= 2;
	playerArr[24].position			= Vector3(0,0,41);

	playerArr[24].health					= 86;
	playerArr[24].mp					= 55;
	playerArr[24].fraction			= FishPeople;
	playerArr[24].statusEffects				= 2;
	agentArr[17].position						= Vector3(0,0,42);
	agentArr[17].health						= 86;
	agentArr[17].mp							= 55;
	agentArr[17].statusEffects	 				= 1;
	agentArr[17].agentType						= Dullahan_warrior;
	agentArr[17].agentEquipmentInHand   = staff;
	agentArr[17].pickedUpItems					= 2;
	agentArr[17].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[17].agentGoal.SELF_RISK_OF_DEATH	= 0;
	agentArr[18].position						= Vector3(0,0,43);
	agentArr[18].health						= 86;
	agentArr[18].mp							= 55;
	agentArr[18].statusEffects	 				= 1;
	agentArr[18].agentType						= Dullahan_warrior;
	agentArr[18].agentEquipmentInHand   = staff;
	agentArr[18].pickedUpItems					= 2;
	agentArr[18].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[18].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[25].position			= Vector3(0,0,44);

	playerArr[25].health					= 86;
	playerArr[25].mp					= 55;
	playerArr[25].fraction			= FishPeople;
	playerArr[25].statusEffects				= 2;
	playerArr[26].position			= Vector3(0,0,45);

	playerArr[26].health					= 86;
	playerArr[26].mp					= 55;
	playerArr[26].fraction			= FishPeople;
	playerArr[26].statusEffects				= 2;
	agentArr[19].position						= Vector3(0,0,46);
	agentArr[19].health						= 86;
	agentArr[19].mp							= 55;
	agentArr[19].statusEffects	 				= 1;
	agentArr[19].agentType						= Dullahan_warrior;
	agentArr[19].agentEquipmentInHand   = staff;
	agentArr[19].pickedUpItems					= 2;
	agentArr[19].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[19].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[27].position			= Vector3(0,0,47);

	playerArr[27].health					= 86;
	playerArr[27].mp					= 55;
	playerArr[27].fraction			= FishPeople;
	playerArr[27].statusEffects				= 2;
	playerArr[28].position			= Vector3(0,0,48);

	playerArr[28].health					= 86;
	playerArr[28].mp					= 55;
	playerArr[28].fraction			= FishPeople;
	playerArr[28].statusEffects				= 2;
	playerArr[29].position			= Vector3(0,0,49);

	playerArr[29].health					= 86;
	playerArr[29].mp					= 55;
	playerArr[29].fraction			= FishPeople;
	playerArr[29].statusEffects				= 2;
	playerArr[30].position			= Vector3(0,0,50);

	playerArr[30].health					= 86;
	playerArr[30].mp					= 55;
	playerArr[30].fraction			= FishPeople;
	playerArr[30].statusEffects				= 2;
	agentArr[20].position						= Vector3(0,0,51);
	agentArr[20].health						= 86;
	agentArr[20].mp							= 55;
	agentArr[20].statusEffects	 				= 1;
	agentArr[20].agentType						= Dullahan_warrior;
	agentArr[20].agentEquipmentInHand   = staff;
	agentArr[20].pickedUpItems					= 2;
	agentArr[20].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[20].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[31].position			= Vector3(0,0,52);

	playerArr[31].health					= 86;
	playerArr[31].mp					= 55;
	playerArr[31].fraction			= FishPeople;
	playerArr[31].statusEffects				= 2;
	agentArr[21].position						= Vector3(0,0,53);
	agentArr[21].health						= 86;
	agentArr[21].mp							= 55;
	agentArr[21].statusEffects	 				= 1;
	agentArr[21].agentType						= Dullahan_warrior;
	agentArr[21].agentEquipmentInHand   = staff;
	agentArr[21].pickedUpItems					= 2;
	agentArr[21].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[21].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[32].position			= Vector3(0,0,54);

	playerArr[32].health					= 86;
	playerArr[32].mp					= 55;
	playerArr[32].fraction			= FishPeople;
	playerArr[32].statusEffects				= 2;
	playerArr[33].position			= Vector3(0,0,55);

	playerArr[33].health					= 86;
	playerArr[33].mp					= 55;
	playerArr[33].fraction			= FishPeople;
	playerArr[33].statusEffects				= 2;
	agentArr[22].position						= Vector3(0,0,56);
	agentArr[22].health						= 86;
	agentArr[22].mp							= 55;
	agentArr[22].statusEffects	 				= 1;
	agentArr[22].agentType						= Dullahan_warrior;
	agentArr[22].agentEquipmentInHand   = staff;
	agentArr[22].pickedUpItems					= 2;
	agentArr[22].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[22].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[34].position			= Vector3(0,0,57);

	playerArr[34].health					= 86;
	playerArr[34].mp					= 55;
	playerArr[34].fraction			= FishPeople;
	playerArr[34].statusEffects				= 2;
	playerArr[35].position			= Vector3(0,0,58);

	playerArr[35].health					= 86;
	playerArr[35].mp					= 55;
	playerArr[35].fraction			= FishPeople;
	playerArr[35].statusEffects				= 2;
	playerArr[36].position			= Vector3(0,0,59);

	playerArr[36].health					= 86;
	playerArr[36].mp					= 55;
	playerArr[36].fraction			= FishPeople;
	playerArr[36].statusEffects				= 2;
	agentArr[23].position						= Vector3(0,0,60);
	agentArr[23].health						= 86;
	agentArr[23].mp							= 55;
	agentArr[23].statusEffects	 				= 1;
	agentArr[23].agentType						= Dullahan_warrior;
	agentArr[23].agentEquipmentInHand   = staff;
	agentArr[23].pickedUpItems					= 2;
	agentArr[23].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[23].agentGoal.SELF_RISK_OF_DEATH	= 0;
	agentArr[24].position						= Vector3(0,0,61);
	agentArr[24].health						= 86;
	agentArr[24].mp							= 55;
	agentArr[24].statusEffects	 				= 1;
	agentArr[24].agentType						= Dullahan_warrior;
	agentArr[24].agentEquipmentInHand   = staff;
	agentArr[24].pickedUpItems					= 2;
	agentArr[24].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[24].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[37].position			= Vector3(0,0,62);

	playerArr[37].health					= 86;
	playerArr[37].mp					= 55;
	playerArr[37].fraction			= FishPeople;
	playerArr[37].statusEffects				= 2;
	playerArr[38].position			= Vector3(0,0,63);

	playerArr[38].health					= 86;
	playerArr[38].mp					= 55;
	playerArr[38].fraction			= FishPeople;
	playerArr[38].statusEffects				= 2;
	agentArr[25].position						= Vector3(0,0,64);
	agentArr[25].health						= 86;
	agentArr[25].mp							= 55;
	agentArr[25].statusEffects	 				= 1;
	agentArr[25].agentType						= Dullahan_warrior;
	agentArr[25].agentEquipmentInHand   = staff;
	agentArr[25].pickedUpItems					= 2;
	agentArr[25].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[25].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[39].position			= Vector3(0,0,65);

	playerArr[39].health					= 86;
	playerArr[39].mp					= 55;
	playerArr[39].fraction			= FishPeople;
	playerArr[39].statusEffects				= 2;
	playerArr[40].position			= Vector3(0,0,66);

	playerArr[40].health					= 86;
	playerArr[40].mp					= 55;
	playerArr[40].fraction			= FishPeople;
	playerArr[40].statusEffects				= 2;
	playerArr[41].position			= Vector3(0,0,67);

	playerArr[41].health					= 86;
	playerArr[41].mp					= 55;
	playerArr[41].fraction			= FishPeople;
	playerArr[41].statusEffects				= 2;
	agentArr[26].position						= Vector3(0,0,68);
	agentArr[26].health						= 86;
	agentArr[26].mp							= 55;
	agentArr[26].statusEffects	 				= 1;
	agentArr[26].agentType						= Dullahan_warrior;
	agentArr[26].agentEquipmentInHand   = staff;
	agentArr[26].pickedUpItems					= 2;
	agentArr[26].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[26].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[42].position			= Vector3(0,0,69);

	playerArr[42].health					= 86;
	playerArr[42].mp					= 55;
	playerArr[42].fraction			= FishPeople;
	playerArr[42].statusEffects				= 2;
	agentArr[27].position						= Vector3(0,0,70);
	agentArr[27].health						= 86;
	agentArr[27].mp							= 55;
	agentArr[27].statusEffects	 				= 1;
	agentArr[27].agentType						= Dullahan_warrior;
	agentArr[27].agentEquipmentInHand   = staff;
	agentArr[27].pickedUpItems					= 2;
	agentArr[27].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[27].agentGoal.SELF_RISK_OF_DEATH	= 0;
	agentArr[28].position						= Vector3(0,0,71);
	agentArr[28].health						= 86;
	agentArr[28].mp							= 55;
	agentArr[28].statusEffects	 				= 1;
	agentArr[28].agentType						= Dullahan_warrior;
	agentArr[28].agentEquipmentInHand   = staff;
	agentArr[28].pickedUpItems					= 2;
	agentArr[28].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[28].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[43].position			= Vector3(0,0,72);

	playerArr[43].health					= 86;
	playerArr[43].mp					= 55;
	playerArr[43].fraction			= FishPeople;
	playerArr[43].statusEffects				= 2;
	agentArr[29].position						= Vector3(0,0,73);
	agentArr[29].health						= 86;
	agentArr[29].mp							= 55;
	agentArr[29].statusEffects	 				= 1;
	agentArr[29].agentType						= Dullahan_warrior;
	agentArr[29].agentEquipmentInHand   = staff;
	agentArr[29].pickedUpItems					= 2;
	agentArr[29].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[29].agentGoal.SELF_RISK_OF_DEATH	= 0;
	agentArr[30].position						= Vector3(0,0,74);
	agentArr[30].health						= 86;
	agentArr[30].mp							= 55;
	agentArr[30].statusEffects	 				= 1;
	agentArr[30].agentType						= Dullahan_warrior;
	agentArr[30].agentEquipmentInHand   = staff;
	agentArr[30].pickedUpItems					= 2;
	agentArr[30].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[30].agentGoal.SELF_RISK_OF_DEATH	= 0;
	agentArr[31].position						= Vector3(0,0,75);
	agentArr[31].health						= 86;
	agentArr[31].mp							= 55;
	agentArr[31].statusEffects	 				= 1;
	agentArr[31].agentType						= Dullahan_warrior;
	agentArr[31].agentEquipmentInHand   = staff;
	agentArr[31].pickedUpItems					= 2;
	agentArr[31].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[31].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[44].position			= Vector3(0,0,76);

	playerArr[44].health					= 86;
	playerArr[44].mp					= 55;
	playerArr[44].fraction			= FishPeople;
	playerArr[44].statusEffects				= 2;
	playerArr[45].position			= Vector3(0,0,77);

	playerArr[45].health					= 86;
	playerArr[45].mp					= 55;
	playerArr[45].fraction			= FishPeople;
	playerArr[45].statusEffects				= 2;
	playerArr[46].position			= Vector3(0,0,78);

	playerArr[46].health					= 86;
	playerArr[46].mp					= 55;
	playerArr[46].fraction			= FishPeople;
	playerArr[46].statusEffects				= 2;
	playerArr[47].position			= Vector3(0,0,79);

	playerArr[47].health					= 86;
	playerArr[47].mp					= 55;
	playerArr[47].fraction			= FishPeople;
	playerArr[47].statusEffects				= 2;
	playerArr[48].position			= Vector3(0,0,80);

	playerArr[48].health					= 86;
	playerArr[48].mp					= 55;
	playerArr[48].fraction			= FishPeople;
	playerArr[48].statusEffects				= 2;
	agentArr[32].position						= Vector3(0,0,81);
	agentArr[32].health						= 86;
	agentArr[32].mp							= 55;
	agentArr[32].statusEffects	 				= 1;
	agentArr[32].agentType						= Dullahan_warrior;
	agentArr[32].agentEquipmentInHand   = staff;
	agentArr[32].pickedUpItems					= 2;
	agentArr[32].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[32].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[49].position			= Vector3(0,0,82);

	playerArr[49].health					= 86;
	playerArr[49].mp					= 55;
	playerArr[49].fraction			= FishPeople;
	playerArr[49].statusEffects				= 2;
	playerArr[50].position			= Vector3(0,0,83);

	playerArr[50].health					= 86;
	playerArr[50].mp					= 55;
	playerArr[50].fraction			= FishPeople;
	playerArr[50].statusEffects				= 2;
	agentArr[33].position						= Vector3(0,0,84);
	agentArr[33].health						= 86;
	agentArr[33].mp							= 55;
	agentArr[33].statusEffects	 				= 1;
	agentArr[33].agentType						= Dullahan_warrior;
	agentArr[33].agentEquipmentInHand   = staff;
	agentArr[33].pickedUpItems					= 2;
	agentArr[33].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[33].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[51].position			= Vector3(0,0,85);

	playerArr[51].health					= 86;
	playerArr[51].mp					= 55;
	playerArr[51].fraction			= FishPeople;
	playerArr[51].statusEffects				= 2;
	playerArr[52].position			= Vector3(0,0,86);

	playerArr[52].health					= 86;
	playerArr[52].mp					= 55;
	playerArr[52].fraction			= FishPeople;
	playerArr[52].statusEffects				= 2;
	playerArr[53].position			= Vector3(0,0,87);

	playerArr[53].health					= 86;
	playerArr[53].mp					= 55;
	playerArr[53].fraction			= FishPeople;
	playerArr[53].statusEffects				= 2;
	agentArr[34].position						= Vector3(0,0,88);
	agentArr[34].health						= 86;
	agentArr[34].mp							= 55;
	agentArr[34].statusEffects	 				= 1;
	agentArr[34].agentType						= Dullahan_warrior;
	agentArr[34].agentEquipmentInHand   = staff;
	agentArr[34].pickedUpItems					= 2;
	agentArr[34].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[34].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[54].position			= Vector3(0,0,89);

	playerArr[54].health					= 86;
	playerArr[54].mp					= 55;
	playerArr[54].fraction			= FishPeople;
	playerArr[54].statusEffects				= 2;
	agentArr[35].position						= Vector3(0,0,90);
	agentArr[35].health						= 86;
	agentArr[35].mp							= 55;
	agentArr[35].statusEffects	 				= 1;
	agentArr[35].agentType						= Dullahan_warrior;
	agentArr[35].agentEquipmentInHand   = staff;
	agentArr[35].pickedUpItems					= 2;
	agentArr[35].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[35].agentGoal.SELF_RISK_OF_DEATH	= 0;
	agentArr[36].position						= Vector3(0,0,91);
	agentArr[36].health						= 86;
	agentArr[36].mp							= 55;
	agentArr[36].statusEffects	 				= 1;
	agentArr[36].agentType						= Dullahan_warrior;
	agentArr[36].agentEquipmentInHand   = staff;
	agentArr[36].pickedUpItems					= 2;
	agentArr[36].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[36].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[55].position			= Vector3(0,0,92);

	playerArr[55].health					= 86;
	playerArr[55].mp					= 55;
	playerArr[55].fraction			= FishPeople;
	playerArr[55].statusEffects				= 2;
	playerArr[56].position			= Vector3(0,0,93);

	playerArr[56].health					= 86;
	playerArr[56].mp					= 55;
	playerArr[56].fraction			= FishPeople;
	playerArr[56].statusEffects				= 2;
	agentArr[37].position						= Vector3(0,0,94);
	agentArr[37].health						= 86;
	agentArr[37].mp							= 55;
	agentArr[37].statusEffects	 				= 1;
	agentArr[37].agentType						= Dullahan_warrior;
	agentArr[37].agentEquipmentInHand   = staff;
	agentArr[37].pickedUpItems					= 2;
	agentArr[37].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[37].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[57].position			= Vector3(0,0,95);

	playerArr[57].health					= 86;
	playerArr[57].mp					= 55;
	playerArr[57].fraction			= FishPeople;
	playerArr[57].statusEffects				= 2;
	agentArr[38].position						= Vector3(0,0,96);
	agentArr[38].health						= 86;
	agentArr[38].mp							= 55;
	agentArr[38].statusEffects	 				= 1;
	agentArr[38].agentType						= Dullahan_warrior;
	agentArr[38].agentEquipmentInHand   = staff;
	agentArr[38].pickedUpItems					= 2;
	agentArr[38].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[38].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[58].position			= Vector3(0,0,97);

	playerArr[58].health					= 86;
	playerArr[58].mp					= 55;
	playerArr[58].fraction			= FishPeople;
	playerArr[58].statusEffects				= 2;
	agentArr[39].position						= Vector3(0,0,98);
	agentArr[39].health						= 86;
	agentArr[39].mp							= 55;
	agentArr[39].statusEffects	 				= 1;
	agentArr[39].agentType						= Dullahan_warrior;
	agentArr[39].agentEquipmentInHand   = staff;
	agentArr[39].pickedUpItems					= 2;
	agentArr[39].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[39].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[59].position			= Vector3(0,0,99);

	playerArr[59].health					= 86;
	playerArr[59].mp					= 55;
	playerArr[59].fraction			= FishPeople;
	playerArr[59].statusEffects				= 2;
	agentArr[40].position						= Vector3(1,0,0);
	agentArr[40].health						= 86;
	agentArr[40].mp							= 55;
	agentArr[40].statusEffects	 				= 1;
	agentArr[40].agentType						= Dullahan_warrior;
	agentArr[40].agentEquipmentInHand   = staff;
	agentArr[40].pickedUpItems					= 2;
	agentArr[40].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[40].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[60].position			= Vector3(1,0,1);

	playerArr[60].health					= 86;
	playerArr[60].mp					= 55;
	playerArr[60].fraction			= FishPeople;
	playerArr[60].statusEffects				= 2;
	agentArr[41].position						= Vector3(1,0,2);
	agentArr[41].health						= 86;
	agentArr[41].mp							= 55;
	agentArr[41].statusEffects	 				= 1;
	agentArr[41].agentType						= Dullahan_warrior;
	agentArr[41].agentEquipmentInHand   = staff;
	agentArr[41].pickedUpItems					= 2;
	agentArr[41].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[41].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[61].position			= Vector3(1,0,3);

	playerArr[61].health					= 86;
	playerArr[61].mp					= 55;
	playerArr[61].fraction			= FishPeople;
	playerArr[61].statusEffects				= 2;
	playerArr[62].position			= Vector3(1,0,4);

	playerArr[62].health					= 86;
	playerArr[62].mp					= 55;
	playerArr[62].fraction			= FishPeople;
	playerArr[62].statusEffects				= 2;
	agentArr[42].position						= Vector3(1,0,5);
	agentArr[42].health						= 86;
	agentArr[42].mp							= 55;
	agentArr[42].statusEffects	 				= 1;
	agentArr[42].agentType						= Dullahan_warrior;
	agentArr[42].agentEquipmentInHand   = staff;
	agentArr[42].pickedUpItems					= 2;
	agentArr[42].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[42].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[63].position			= Vector3(1,0,6);

	playerArr[63].health					= 86;
	playerArr[63].mp					= 55;
	playerArr[63].fraction			= FishPeople;
	playerArr[63].statusEffects				= 2;
	agentArr[43].position						= Vector3(1,0,7);
	agentArr[43].health						= 86;
	agentArr[43].mp							= 55;
	agentArr[43].statusEffects	 				= 1;
	agentArr[43].agentType						= Dullahan_warrior;
	agentArr[43].agentEquipmentInHand   = staff;
	agentArr[43].pickedUpItems					= 2;
	agentArr[43].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[43].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[64].position			= Vector3(1,0,8);

	playerArr[64].health					= 86;
	playerArr[64].mp					= 55;
	playerArr[64].fraction			= FishPeople;
	playerArr[64].statusEffects				= 2;
	playerArr[65].position			= Vector3(1,0,9);

	playerArr[65].health					= 86;
	playerArr[65].mp					= 55;
	playerArr[65].fraction			= FishPeople;
	playerArr[65].statusEffects				= 2;
	playerArr[66].position			= Vector3(1,0,10);

	playerArr[66].health					= 86;
	playerArr[66].mp					= 55;
	playerArr[66].fraction			= FishPeople;
	playerArr[66].statusEffects				= 2;
	agentArr[44].position						= Vector3(1,0,11);
	agentArr[44].health						= 86;
	agentArr[44].mp							= 55;
	agentArr[44].statusEffects	 				= 1;
	agentArr[44].agentType						= Dullahan_warrior;
	agentArr[44].agentEquipmentInHand   = staff;
	agentArr[44].pickedUpItems					= 2;
	agentArr[44].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[44].agentGoal.SELF_RISK_OF_DEATH	= 0;
	agentArr[45].position						= Vector3(1,0,12);
	agentArr[45].health						= 86;
	agentArr[45].mp							= 55;
	agentArr[45].statusEffects	 				= 1;
	agentArr[45].agentType						= Dullahan_warrior;
	agentArr[45].agentEquipmentInHand   = staff;
	agentArr[45].pickedUpItems					= 2;
	agentArr[45].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[45].agentGoal.SELF_RISK_OF_DEATH	= 0;
	agentArr[46].position						= Vector3(1,0,13);
	agentArr[46].health						= 86;
	agentArr[46].mp							= 55;
	agentArr[46].statusEffects	 				= 1;
	agentArr[46].agentType						= Dullahan_warrior;
	agentArr[46].agentEquipmentInHand   = staff;
	agentArr[46].pickedUpItems					= 2;
	agentArr[46].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[46].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[67].position			= Vector3(1,0,14);

	playerArr[67].health					= 86;
	playerArr[67].mp					= 55;
	playerArr[67].fraction			= FishPeople;
	playerArr[67].statusEffects				= 2;
	playerArr[68].position			= Vector3(1,0,15);

	playerArr[68].health					= 86;
	playerArr[68].mp					= 55;
	playerArr[68].fraction			= FishPeople;
	playerArr[68].statusEffects				= 2;
	agentArr[47].position						= Vector3(1,0,16);
	agentArr[47].health						= 86;
	agentArr[47].mp							= 55;
	agentArr[47].statusEffects	 				= 1;
	agentArr[47].agentType						= Dullahan_warrior;
	agentArr[47].agentEquipmentInHand   = staff;
	agentArr[47].pickedUpItems					= 2;
	agentArr[47].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[47].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[69].position			= Vector3(1,0,17);

	playerArr[69].health					= 86;
	playerArr[69].mp					= 55;
	playerArr[69].fraction			= FishPeople;
	playerArr[69].statusEffects				= 2;
	agentArr[48].position						= Vector3(1,0,18);
	agentArr[48].health						= 86;
	agentArr[48].mp							= 55;
	agentArr[48].statusEffects	 				= 1;
	agentArr[48].agentType						= Dullahan_warrior;
	agentArr[48].agentEquipmentInHand   = staff;
	agentArr[48].pickedUpItems					= 2;
	agentArr[48].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[48].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[70].position			= Vector3(1,0,19);

	playerArr[70].health					= 86;
	playerArr[70].mp					= 55;
	playerArr[70].fraction			= FishPeople;
	playerArr[70].statusEffects				= 2;
	playerArr[71].position			= Vector3(1,0,20);

	playerArr[71].health					= 86;
	playerArr[71].mp					= 55;
	playerArr[71].fraction			= FishPeople;
	playerArr[71].statusEffects				= 2;
	playerArr[72].position			= Vector3(1,0,21);

	playerArr[72].health					= 86;
	playerArr[72].mp					= 55;
	playerArr[72].fraction			= FishPeople;
	playerArr[72].statusEffects				= 2;
	playerArr[73].position			= Vector3(1,0,22);

	playerArr[73].health					= 86;
	playerArr[73].mp					= 55;
	playerArr[73].fraction			= FishPeople;
	playerArr[73].statusEffects				= 2;
	playerArr[74].position			= Vector3(1,0,23);

	playerArr[74].health					= 86;
	playerArr[74].mp					= 55;
	playerArr[74].fraction			= FishPeople;
	playerArr[74].statusEffects				= 2;
	agentArr[49].position						= Vector3(1,0,24);
	agentArr[49].health						= 86;
	agentArr[49].mp							= 55;
	agentArr[49].statusEffects	 				= 1;
	agentArr[49].agentType						= Dullahan_warrior;
	agentArr[49].agentEquipmentInHand   = staff;
	agentArr[49].pickedUpItems					= 2;
	agentArr[49].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[49].agentGoal.SELF_RISK_OF_DEATH	= 0;
	agentArr[50].position						= Vector3(1,0,25);
	agentArr[50].health						= 86;
	agentArr[50].mp							= 55;
	agentArr[50].statusEffects	 				= 1;
	agentArr[50].agentType						= Dullahan_warrior;
	agentArr[50].agentEquipmentInHand   = staff;
	agentArr[50].pickedUpItems					= 2;
	agentArr[50].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[50].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[75].position			= Vector3(1,0,26);

	playerArr[75].health					= 86;
	playerArr[75].mp					= 55;
	playerArr[75].fraction			= FishPeople;
	playerArr[75].statusEffects				= 2;
	agentArr[51].position						= Vector3(1,0,27);
	agentArr[51].health						= 86;
	agentArr[51].mp							= 55;
	agentArr[51].statusEffects	 				= 1;
	agentArr[51].agentType						= Dullahan_warrior;
	agentArr[51].agentEquipmentInHand   = staff;
	agentArr[51].pickedUpItems					= 2;
	agentArr[51].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[51].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[76].position			= Vector3(1,0,28);

	playerArr[76].health					= 86;
	playerArr[76].mp					= 55;
	playerArr[76].fraction			= FishPeople;
	playerArr[76].statusEffects				= 2;
	playerArr[77].position			= Vector3(1,0,29);

	playerArr[77].health					= 86;
	playerArr[77].mp					= 55;
	playerArr[77].fraction			= FishPeople;
	playerArr[77].statusEffects				= 2;
	agentArr[52].position						= Vector3(1,0,30);
	agentArr[52].health						= 86;
	agentArr[52].mp							= 55;
	agentArr[52].statusEffects	 				= 1;
	agentArr[52].agentType						= Dullahan_warrior;
	agentArr[52].agentEquipmentInHand   = staff;
	agentArr[52].pickedUpItems					= 2;
	agentArr[52].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[52].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[78].position			= Vector3(1,0,31);

	playerArr[78].health					= 86;
	playerArr[78].mp					= 55;
	playerArr[78].fraction			= FishPeople;
	playerArr[78].statusEffects				= 2;
	playerArr[79].position			= Vector3(1,0,32);

	playerArr[79].health					= 86;
	playerArr[79].mp					= 55;
	playerArr[79].fraction			= FishPeople;
	playerArr[79].statusEffects				= 2;
	agentArr[53].position						= Vector3(1,0,33);
	agentArr[53].health						= 86;
	agentArr[53].mp							= 55;
	agentArr[53].statusEffects	 				= 1;
	agentArr[53].agentType						= Dullahan_warrior;
	agentArr[53].agentEquipmentInHand   = staff;
	agentArr[53].pickedUpItems					= 2;
	agentArr[53].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[53].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[80].position			= Vector3(1,0,34);

	playerArr[80].health					= 86;
	playerArr[80].mp					= 55;
	playerArr[80].fraction			= FishPeople;
	playerArr[80].statusEffects				= 2;
	agentArr[54].position						= Vector3(1,0,35);
	agentArr[54].health						= 86;
	agentArr[54].mp							= 55;
	agentArr[54].statusEffects	 				= 1;
	agentArr[54].agentType						= Dullahan_warrior;
	agentArr[54].agentEquipmentInHand   = staff;
	agentArr[54].pickedUpItems					= 2;
	agentArr[54].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[54].agentGoal.SELF_RISK_OF_DEATH	= 0;
	agentArr[55].position						= Vector3(1,0,36);
	agentArr[55].health						= 86;
	agentArr[55].mp							= 55;
	agentArr[55].statusEffects	 				= 1;
	agentArr[55].agentType						= Dullahan_warrior;
	agentArr[55].agentEquipmentInHand   = staff;
	agentArr[55].pickedUpItems					= 2;
	agentArr[55].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[55].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[81].position			= Vector3(1,0,37);

	playerArr[81].health					= 86;
	playerArr[81].mp					= 55;
	playerArr[81].fraction			= FishPeople;
	playerArr[81].statusEffects				= 2;
	playerArr[82].position			= Vector3(1,0,38);

	playerArr[82].health					= 86;
	playerArr[82].mp					= 55;
	playerArr[82].fraction			= FishPeople;
	playerArr[82].statusEffects				= 2;
	agentArr[56].position						= Vector3(1,0,39);
	agentArr[56].health						= 86;
	agentArr[56].mp							= 55;
	agentArr[56].statusEffects	 				= 1;
	agentArr[56].agentType						= Dullahan_warrior;
	agentArr[56].agentEquipmentInHand   = staff;
	agentArr[56].pickedUpItems					= 2;
	agentArr[56].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[56].agentGoal.SELF_RISK_OF_DEATH	= 0;
	agentArr[57].position						= Vector3(1,0,40);
	agentArr[57].health						= 86;
	agentArr[57].mp							= 55;
	agentArr[57].statusEffects	 				= 1;
	agentArr[57].agentType						= Dullahan_warrior;
	agentArr[57].agentEquipmentInHand   = staff;
	agentArr[57].pickedUpItems					= 2;
	agentArr[57].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[57].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[83].position			= Vector3(1,0,41);

	playerArr[83].health					= 86;
	playerArr[83].mp					= 55;
	playerArr[83].fraction			= FishPeople;
	playerArr[83].statusEffects				= 2;
	playerArr[84].position			= Vector3(1,0,42);

	playerArr[84].health					= 86;
	playerArr[84].mp					= 55;
	playerArr[84].fraction			= FishPeople;
	playerArr[84].statusEffects				= 2;
	agentArr[58].position						= Vector3(1,0,43);
	agentArr[58].health						= 86;
	agentArr[58].mp							= 55;
	agentArr[58].statusEffects	 				= 1;
	agentArr[58].agentType						= Dullahan_warrior;
	agentArr[58].agentEquipmentInHand   = staff;
	agentArr[58].pickedUpItems					= 2;
	agentArr[58].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[58].agentGoal.SELF_RISK_OF_DEATH	= 0;
	agentArr[59].position						= Vector3(1,0,44);
	agentArr[59].health						= 86;
	agentArr[59].mp							= 55;
	agentArr[59].statusEffects	 				= 1;
	agentArr[59].agentType						= Dullahan_warrior;
	agentArr[59].agentEquipmentInHand   = staff;
	agentArr[59].pickedUpItems					= 2;
	agentArr[59].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[59].agentGoal.SELF_RISK_OF_DEATH	= 0;
	playerArr[85].position			= Vector3(1,0,45);

	playerArr[85].health					= 86;
	playerArr[85].mp					= 55;
	playerArr[85].fraction			= FishPeople;
	playerArr[85].statusEffects				= 2;
	playerArr[86].position			= Vector3(1,0,46);

	playerArr[86].health					= 86;
	playerArr[86].mp					= 55;
	playerArr[86].fraction			= FishPeople;
	playerArr[86].statusEffects				= 2;
	agentArr[60].position						= Vector3(1,0,47);
	agentArr[60].health						= 86;
	agentArr[60].mp							= 55;
	agentArr[60].statusEffects	 				= 1;
	agentArr[60].agentType						= Dullahan_warrior;
	agentArr[60].agentEquipmentInHand   = staff;
	agentArr[60].pickedUpItems					= 2;
	agentArr[60].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[60].agentGoal.SELF_RISK_OF_DEATH	= 0;
	agentArr[61].position						= Vector3(1,0,48);
	agentArr[61].health						= 86;
	agentArr[61].mp							= 55;
	agentArr[61].statusEffects	 				= 1;
	agentArr[61].agentType						= Dullahan_warrior;
	agentArr[61].agentEquipmentInHand   = staff;
	agentArr[61].pickedUpItems					= 2;
	agentArr[61].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[61].agentGoal.SELF_RISK_OF_DEATH	= 0;
	agentArr[62].position						= Vector3(1,0,49);
	agentArr[62].health						= 86;
	agentArr[62].mp							= 55;
	agentArr[62].statusEffects	 				= 1;
	agentArr[62].agentType						= Dullahan_warrior;
	agentArr[62].agentEquipmentInHand   = staff;
	agentArr[62].pickedUpItems					= 2;
	agentArr[62].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[62].agentGoal.SELF_RISK_OF_DEATH	= 0;
	agentArr[63].position						= Vector3(1,0,50);
	agentArr[63].health						= 86;
	agentArr[63].mp							= 55;
	agentArr[63].statusEffects	 				= 1;
	agentArr[63].agentType						= Dullahan_warrior;
	agentArr[63].agentEquipmentInHand   = staff;
	agentArr[63].pickedUpItems					= 2;
	agentArr[63].agentGoal.HAS_LIVE_ENEMY		= 0;
	agentArr[63].agentGoal.SELF_RISK_OF_DEATH	= 0;
}