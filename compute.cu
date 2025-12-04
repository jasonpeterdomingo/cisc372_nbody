#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#define THREADS_PER_BLOCK 256
#define BLOCK_DIM_XY 16

__global__ void computeAccels(vector3 *hPos, double *mass, vector3 *accels) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NUMENTITIES && j < NUMENTITIES) {
        int accelIndex = i * NUMENTITIES + j;
        if (i == j) {
            accels[accelIndex][0] = 0.0;
            accels[accelIndex][1] = 0.0;
            accels[accelIndex][2] = 0.0;
        } else {
            vector3 distance;
            for (int k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
            double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
            double magnitude=sqrt(magnitude_sq);
            double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;

            for (int k = 0; k < 3; k++) {
                accels[accelIndex][k] = accelmag * distance[k] / magnitude;
            }
        }
    }
}

__global__ void getEffect(vector3 *hPos, vector3 *hVel, vector3 *accels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NUMENTITIES) {
        vector3 accel_sum = {0.0, 0.0, 0.0};

        for (int j = 0; j < NUMENTITIES; j++) {
            int matrixIndex = i * NUMENTITIES + j;
            for (int k = 0; k < 3; k++) {
                accel_sum[k] += accels[matrixIndex][k];
            }
        }

        for (int k=0;k<3;k++){
			hVel[i][k]+=accel_sum[k]*INTERVAL;
			hPos[i][k]+=hVel[i][k]*INTERVAL;
		}
    }
}

void compute() {
    dim3 dimBlock(BLOCK_DIM_XY, BLOCK_DIM_XY);
    dim3 dimGrid((NUMENTITIES * BLOCK_DIM_XY - 1) / BLOCK_DIM_XY, (NUMENTITIES * BLOCK_DIM_XY - 1) / BLOCK_DIM_XY);

    computeAccels<<<dimGrid, dimBlock>>>(d_hPos, d_mass, d_accels);

    int numBlocks = (NUMENTITIES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    getEffect<<<numBlocks, THREADS_PER_BLOCK>>>(d_hPos, d_hVel, d_accels);

    cudaDeviceSynchronize();
}