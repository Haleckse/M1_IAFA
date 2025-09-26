
#include <iostream>
#include <cuda_runtime.h>


__global__ 
void ksomme1(float * vec, int size) {
    int idx = threadidx.x; 
    for(int offset = 1; offset < size; offset *=2) {
        if (idw + offeset < size && idw%(2*offset)) {
            vec[idx] += vec[idx + offset]; 
        }
        __syncthread(); 
    }
}


// Kernel pour faire une réduction (somme) parallèle
__global__
void ksomme1(float *vec, int size) {
    int idx = threadIdx.x;

    for (int offset = 1; offset < size; offset *= 2) {
        if (idx + offset < size && idx % (2 * offset) == 0) {
            vec[idx] += vec[idx + offset];
        }
        __syncthreads(); // synchronise tous les threads du bloc
    }
}

int main() {
    const int size = 8;
    float h_vec[size] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float *d_vec;

    // Allocation mémoire sur le device
    cudaMalloc((void**)&d_vec, size * sizeof(float));
    cudaMemcpy(d_vec, h_vec, size * sizeof(float), cudaMemcpyHostToDevice);

    // Lancer le kernel avec 1 bloc et size threads
    ksomme1<<<1, size>>>(d_vec, size);
    cudaDeviceSynchronize();

    // Copier le résultat vers l'hôte
    cudaMemcpy(h_vec, d_vec, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Afficher le vecteur résultant
    std::cout << "Résultat du vecteur après ksomme1 (réduction):" << std::endl;
    for (int i = 0; i < size; i++) {
        std::cout << h_vec[i] << " ";
    }
    std::cout << std::endl;

    // Résultat attendu : h_vec[0] contiendra la somme totale
    std::cout << "Somme totale : " << h_vec[0] << std::endl;

    // Libération mémoire
    cudaFree(d_vec);
    return 0;
}
