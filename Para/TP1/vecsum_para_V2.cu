#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 1024

__global__
void ksomme(float *input, float *partial_sums, int size) {
    __shared__ float shared[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared[tid] = (global_idx < size) ? input[global_idx] : 0.0f;
    __syncthreads();

    // Réduction parallèle dans le bloc
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }
}

int main(int argc, char **argv) {

    clock_t start;
    clock_t end;
    

    if (argc < 2) {
        printf("Usage: %s <filename>\n", argv[0]);
        return -1;
    }

    unsigned int log2size, size;
    float *h_vec, *d_vec, *d_partial, *h_partial;

    FILE *f = fopen(argv[1], "r");
    if (!f) {
        perror("Erreur ouverture fichier");
        return -1;
    }

    fscanf(f, "%u\n", &log2size);
    if (log2size > 20) {
        printf("Max log2size is 20\n");
        fclose(f);
        return -1;
    }

    size = 1 << log2size;
    unsigned int bytes = size * sizeof(float);
    h_vec = (float*)malloc(bytes);
    assert(h_vec);

    for (unsigned int i = 0; i < size; i++) {
        unsigned int tmp;
        fscanf(f, "%u\n", &tmp);
        h_vec[i] = (float)tmp;
    }
    fclose(f);

    // CUDA memory allocation
    cudaMalloc((void**)&d_vec, bytes);
    cudaMemcpy(d_vec, h_vec, bytes, cudaMemcpyHostToDevice);

    int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaMalloc((void**)&d_partial, blocks * sizeof(float));
    h_partial = (float*)malloc(blocks * sizeof(float));

   
    start=clock();

    ksomme<<<blocks, THREADS_PER_BLOCK>>>(d_vec, d_partial, size);
    
   
    // Copie des résultats partiels vers CPU
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Réduction finale CPU
    float total = 0.0f;
    for (int i = 0; i < blocks; i++) {
        total += h_partial[i];
    }
    end=clock();
    double temps=((double)end - (double)start)/CLOCKS_PER_SEC;
    printf("temps execution %lf \n",temps);


    printf("Somme totale : %.0f\n", total);

    // Libération
    cudaFree(d_vec);
    cudaFree(d_partial);
    free(h_vec);
    free(h_partial);

    return 0;
}
