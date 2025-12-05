/*
CISC372 HW 4
Jason Domingo
Andrew Orlov
*/
#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#define THREADS_PER_BLOCK 256
#define BLOCK_DIM_XY 16

// Kernel to compute accelerations
__global__ void computeAccels(vector3 *hPos, double *mass, vector3 *accels) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row index
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // col index

    // shared memory declarations
    __shared__ vector3 sharedPosI[BLOCK_DIM_XY];
    __shared__ vector3 sharedPosJ[BLOCK_DIM_XY];
    __shared__ double sharedMassJ[BLOCK_DIM_XY];

    // Only load data into shared memory once per row
    if (i < NUMENTITIES && threadIdx.x == 0) {
        for (int k = 0; k < 3; k++) {
            sharedPosI[threadIdx.y][k] = hPos[i][k];
        }
    }
    // Only load data into shared memory once per col
    if (j < NUMENTITIES && threadIdx.y == 0) {
        for (int k = 0; k < 3; k++) {
            sharedPosJ[threadIdx.x][k] = hPos[j][k];
        }
        sharedMassJ[threadIdx.x] = mass[j];
    }
    __syncthreads();

    if (i < NUMENTITIES && j < NUMENTITIES) {
        int accelIndex = i * NUMENTITIES + j;
        if (i == j) {
            accels[accelIndex][0] = 0.0;
            accels[accelIndex][1] = 0.0;
            accels[accelIndex][2] = 0.0;
        } else {
            vector3 distance;
            for (int k=0;k<3;k++) distance[k]=sharedPosI[threadIdx.y][k]-sharedPosJ[threadIdx.x][k];
            double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
            double magnitude=sqrt(magnitude_sq);
            double accelmag=-1*GRAV_CONSTANT*sharedMassJ[threadIdx.x]/magnitude_sq;

            for (int k = 0; k < 3; k++) {
                accels[accelIndex][k] = accelmag * distance[k] / magnitude;
            }
        }
    }
}

// Reduction Kernel to add accelerations to velocities; update velocities and positions
__global__ void getEffect(vector3 *hPos, vector3 *hVel, vector3 *accels) {
    int i = blockIdx.x;
    int stride = blockDim.x;
    vector3 accel_sum = {0.0, 0.0, 0.0};

    // Each thread computes a partial sum of accelerations
    for (int j = threadIdx.x; j < NUMENTITIES; j+= stride) {
        int matrixIndex = i * NUMENTITIES + j;
        for (int k = 0; k < 3; k++) {
            accel_sum[k] += accels[matrixIndex][k];
        }
    }

    // Stores partial sums in shared memory
    __shared__ vector3 sharedAccelSum[THREADS_PER_BLOCK];
    for (int k = 0; k < 3; k++) {
        sharedAccelSum[threadIdx.x][k] = accel_sum[k];
    }
    __syncthreads();

    // Reduce partial sums to get total acceleration
    for (int offset = stride / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            for (int k = 0; k < 3; k++) {
                sharedAccelSum[threadIdx.x][k] += sharedAccelSum[threadIdx.x + offset][k];
            }
        }
        __syncthreads();
    }

    // First thread updates velocity and position
    if (threadIdx.x == 0) {
        for (int k = 0; k < 3; k++) {
            hVel[i][k] += sharedAccelSum[0][k] * INTERVAL;
			hPos[i][k]+=hVel[i][k] * INTERVAL;
        }
    }
}

// Compute function to launch kernels
void compute() {
    dim3 dimBlock(BLOCK_DIM_XY, BLOCK_DIM_XY);
    dim3 dimGrid((NUMENTITIES + BLOCK_DIM_XY - 1) / BLOCK_DIM_XY, (NUMENTITIES + BLOCK_DIM_XY - 1) / BLOCK_DIM_XY);

    computeAccels<<<dimGrid, dimBlock>>>(d_hPos, d_mass, d_accels);

    int numBlocks = NUMENTITIES;  // one block per entity
    getEffect<<<numBlocks, THREADS_PER_BLOCK>>>(d_hPos, d_hVel, d_accels);

    cudaDeviceSynchronize();
}