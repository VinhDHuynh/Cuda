#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

const int GRID_SIZE = 256;
const int NUM_GENERATIONS = 100;
const int BLOCK_SIZE = 16;

// Kernel to update the state of each cell in the grid
__global__ void updateGrid(int* currentGrid, int* nextGrid) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * GRID_SIZE + x;

    int numNeighbors = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0)
                continue;

            int neighborX = (x + dx + GRID_SIZE) % GRID_SIZE;
            int neighborY = (y + dy + GRID_SIZE) % GRID_SIZE;
            int neighborIndex = neighborY * GRID_SIZE + neighborX;

            numNeighbors += currentGrid[neighborIndex];
        }
    }

    if (currentGrid[index] == 1) {
        if (numNeighbors < 2 || numNeighbors > 3)
            nextGrid[index] = 0;
        else
            nextGrid[index] = 1;
    }
    else {
        if (numNeighbors == 3)
            nextGrid[index] = 1;
        else
            nextGrid[index] = 0;
    }
}

int main() {
    // Allocate memory for the grids on the host
    int* currentGridHost = new int[GRID_SIZE * GRID_SIZE];
    int* nextGridHost = new int[GRID_SIZE * GRID_SIZE];

    // Initialize the grids randomly
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (int i = 0; i < GRID_SIZE * GRID_SIZE; ++i) {
        currentGridHost[i] = rand() % 2;
    }

    // Allocate memory for the grids on the GPU
    int* currentGridDevice;
    int* nextGridDevice;
    cudaMalloc(&currentGridDevice, GRID_SIZE * GRID_SIZE * sizeof(int));
    cudaMalloc(&nextGridDevice, GRID_SIZE * GRID_SIZE * sizeof(int));

    // Copy the initial state from host to device
    cudaMemcpy(currentGridDevice, currentGridHost, GRID_SIZE * GRID_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid sizes for the CUDA kernel
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(GRID_SIZE / BLOCK_SIZE, GRID_SIZE / BLOCK_SIZE);

    // Run the simulation
    for (int generation = 0; generation < NUM_GENERATIONS; ++generation) {
        updateGrid << <gridSize, blockSize >> > (currentGridDevice, nextGridDevice);
        std::swap(currentGridDevice, nextGridDevice);
    }

    // Copy the final state from device back to host
    cudaMemcpy(currentGridHost, currentGridDevice, GRID_SIZE * GRID_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the final state
    for (int y = 0; y < GRID_SIZE; ++y) {
        for (int x = 0; x < GRID_SIZE; ++x) {
            std::cout << (currentGridHost[y * GRID_SIZE + x] ? "#" : " ");
        }
        std::cout << std::endl;
    }

    // Free memory on the GPU and host
    cudaFree(currentGridDevice);
    cudaFree(nextGridDevice);
    delete[] currentGridHost;
    delete[] nextGridHost;

    return 0;
}
