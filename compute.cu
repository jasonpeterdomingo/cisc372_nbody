#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#define THREADS_PER_BLOCK 256

__global__ void computeAccels(vector3 *pos, vector3 *vel, double *mass) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NUMENTITIES) {
        // vector3 accel_sum = {1.0, 1.0, 1.0};
        pos[i][0] += 1.0;
        // vector3 myPos;
    
        // FILL_VECTOR(myPos, pos[i][0], pos[i][1], pos[i][2]);

        // vel[i][0] += accel_sum[0] * INTERVAL;
        // vel[i][1] += accel_sum[1] * INTERVAL;
        // vel[i][2] += accel_sum[2] * INTERVAL;

        // pos[i][0] += vel[i][0] * INTERVAL;
        // pos[i][1] += vel[i][1] * INTERVAL;
        // pos[i][2] += vel[i][2] * INTERVAL;
    }
}

void compute() {
    int numBlocks = (NUMENTITIES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    computeAccels<<<numBlocks, THREADS_PER_BLOCK>>>(d_hPos, d_hVel, d_mass);
}