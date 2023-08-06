#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// CUDA kernel to count words in parallel
__global__ void countWordsGPU(const char* words, int* count, int numWords) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int partialCount = 0;

    for (int i = tid; i < numWords; i += stride) {
        if (words[i] == ' ' || words[i] == '\n') {
            partialCount++;
        }
    }

    atomicAdd(count, partialCount);
}

int main() {
    // Read the document and split it into words (CPU task)
    std::string document = "Your document contents here Your document contents here ";
    std::string delimiter = " "; // Assuming words are separated by space characters

    // Convert the document to a char array to transfer to GPU
    char* wordsGPU;
    cudaMalloc(&wordsGPU, document.size() + 1);
    cudaMemcpy(wordsGPU, document.c_str(), document.size() + 1, cudaMemcpyHostToDevice);

    // Count the number of words on the GPU
    int* wordCountGPU;
    cudaMalloc(&wordCountGPU, sizeof(int));
    cudaMemset(wordCountGPU, 0, sizeof(int));

    // GPU configuration
    int threadsPerBlock = 4;
    int numBlocks = 2;

    // Launch the GPU kernel
    countWordsGPU <<<numBlocks, threadsPerBlock >>> (wordsGPU, wordCountGPU, document.size());

    // Transfer the result back to CPU
    int wordCount;
    cudaMemcpy(&wordCount, wordCountGPU, sizeof(int), cudaMemcpyDeviceToHost);

    // Free memory on GPU
    cudaFree(wordsGPU);
    cudaFree(wordCountGPU);

    std::cout << "Total number of words in the document: " << wordCount << std::endl;

    return 0;
}
