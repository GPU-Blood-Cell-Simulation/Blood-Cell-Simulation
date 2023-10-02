#include "../defines.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>


class binary_radix_tree
{
    unsigned int* radixData;
    unsigned int* internalNodes;
    unsigned int differentKeysCount;
    unsigned int treeDepth;

    binary_radix_tree(unsigned int* sortedRadixData, unsigned int differentKeysCount, unsigned int treeDepth)
    {
        this->radixData = sortedRadixData;
        this->differentKeysCount = differentKeysCount;
        this->treeDepth = treeDepth;
        //this->nodesCount = pow(8, treeDepth + 1)/7;
        HANDLE_ERROR(cudaMalloc((void**)&internalNodes, (differentKeysCount - 1) * sizeof(unsigned int)));
    }

    __device__ int delta(unsigned int i, unsigned int j)
    {
        if(j < 0 || j > differentKeysCount - 1)
            return -1;

        return __clz(radixData[i] ^ radixData[j]);
    }
}


__global__ void constructBinaryRadixTree(binary_radix_tree tree)
{
    unsigned int id;
    if(id > tree.differentKeysCount - 1)
        return;
    
    int d = sign(tree.delta(id, id+1) - tree.delta(id, id-1));

    int delta_min = tree.delta(id, id-d);
    unsigned int l_max = 2;
    while(tree.delta(i, i+l_max*d) > delta_min)
        l_max <<= 1; // *2

    unsigned int l = 0;
    for(unsigned int t = l_max >> 1; t >= 1; t >>= 1)
        if(tree.delta(i, i+(l+t)*d) > delta_min)
            l += t;

    int j = id + l*d;
    int delta_node = tree.delta(i, j);

    unsigned int s = 0;
    for(unsigned int t = ceil(float(l)/2.0f); t >= 1; t = ceil(float(t)/2.0f) )
        if(tree.delta(i, i+(s+t)*d) > delta_node)
            s += t;

    int lambda = id + s*d + min(d, 0);
    unsigned int left, right;

}