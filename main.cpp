
#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"


extern "C" bool runPerf(int device_id);


int main(int argc, char ** argv)
{
    runPerf(0);
}