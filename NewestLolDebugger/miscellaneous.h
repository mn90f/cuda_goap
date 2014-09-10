#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

//Checks if we have CUDA capable device(s) on our computer. Else we cant run anything.
inline bool hasCUDADevice(int deviceUsed = 0)
{
	cudaSetDevice(deviceUsed);
	int numberOfDevices = 0;
	cudaGetDeviceCount(&numberOfDevices);

	if(numberOfDevices == 0){
		std::cout<<"\n\n"<<"CUDA enabled devices not found! Application is quitting now.";
		return false;
	}

	//If we got here lets print some info about the device which we are going to use.
	cudaDeviceProp devProperties;
	cudaGetDeviceProperties(&devProperties, deviceUsed);

	std::cout<<"==========================================================================="<<std::endl<<std::endl;

	std::cout<<std::endl<<"Global memory available on device in MB :"<<((devProperties.totalGlobalMem)/1024)/1024;             
	std::cout<<std::endl<<"Shared memory available per block in bytes : "<<devProperties.sharedMemPerBlock;          
	std::cout<<std::endl<<"32-bit registers available per block : "<<devProperties.regsPerBlock;               
	std::cout<<std::endl<<"Warp size in threads : "<<devProperties.warpSize;                   
	std::cout<<std::endl<<"Maximum pitch in bytes allowed by memory copies : "<<devProperties.memPitch;                   
	std::cout<<std::endl<<"Maximum number of threads per block : "<<devProperties.maxThreadsPerBlock;          
	std::cout<<std::endl<<"Clock frequency in kilohertz : "<<devProperties.clockRate;                  
	std::cout<<std::endl<<"Constant memory available on device in bytes : "<<devProperties.totalConstMem;              
	std::cout<<std::endl<<"Compute capability : "<<devProperties.major <<"."<<devProperties.minor;                      
	std::cout<<std::endl<<"Number of multiprocessors on device : "<<devProperties.multiProcessorCount;        
	std::cout<<std::endl<<"Specified whether there is a run time limit on kernels : "<<devProperties.kernelExecTimeoutEnabled;   
	std::cout<<std::endl<<"Device is integrated as opposed to discrete : "<<devProperties.integrated;                 
	std::cout<<std::endl<<"Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer : "<<devProperties.canMapHostMemory;           
	std::cout<<std::endl<<"Device can possibly execute multiple kernels concurrently : "<<devProperties.concurrentKernels;          
	std::cout<<std::endl<<"Peak memory clock frequency in kilohertz : "<<devProperties.memoryClockRate;            
	std::cout<<std::endl<<"Global memory bus width in bits : "<<devProperties.memoryBusWidth;             
	std::cout<<std::endl<<"Size of L2 cache in bytes : "<<devProperties.l2CacheSize;                
	std::cout<<std::endl<<"Maximum resident threads per multiprocessor : "<<devProperties.maxThreadsPerMultiProcessor;
	std::cout<<std::endl<<"Device supports stream priorities : "<<devProperties.streamPrioritiesSupported;  

	std::cout<<std::endl<<std::endl<<"===========================================================================";

	if(devProperties.computeMode == cudaComputeModeProhibited){
		std::cout<<std::endl<<"Cuda cannot run, compute mode is prohibited. The application is quitting now..";
		return false;
	}

	int* d_agentState = NULL;
	cudaMalloc((void**)&d_agentState, 1 * sizeof(int));	

	return true;
}