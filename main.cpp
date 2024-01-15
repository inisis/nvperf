
#include <iostream>

#include <cuda.h>
#include "cuda_runtime.h"


void printHelpMessage(const char* programName) {
    std::cout << "Usage: " << programName << " [device_id]" << std::endl;
    std::cout << "  device_id (optional): Specify the CUDA device ID. Default is 0." << std::endl;
}

extern "C" bool runPerf(int device_id);


int main(int argc, char **argv)
{
    if (argc > 2) {
        std::cerr << "Invalid number of arguments." << std::endl;
        printHelpMessage(argv[0]);
        return 1;  // Return an error code
    }


    if (argc == 2 && (argv[1][0] == '-' || argv[1][0] == '/')) {
        if (argv[1][1] == 'h' || argv[1][1] == 'H') {
            printHelpMessage(argv[0]);
            return 0;
        }
    }

    int defaultDeviceID = 0;
    int requestedDeviceID = (argc == 2) ? std::stoi(argv[1]) : defaultDeviceID;

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed with error code " << error_id << std::endl;
        return 1;
    }

    if (requestedDeviceID < 0 || requestedDeviceID >= deviceCount) {
        std::cerr << "Invalid device ID. Please provide a device ID between 0 and " << deviceCount - 1 << std::endl;
        return 1;
    }

    runPerf(requestedDeviceID);

    return 0;
}