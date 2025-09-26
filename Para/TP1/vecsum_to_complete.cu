#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BSIZE 1024


__global__
void ksomme(float *d_vec, int size) {
    int idx = threadIdx.x; 
    for (int offset = size/2; offset > 0; offset /=2) {
        if (idx + offset < size) {
            d_vec[idx] += d_vec[idx + offset]; 
        }
        __syncthreads(); 
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <filename>\n", argv[0]);
        exit(-1);
    }

    unsigned int log2size, size;
    float *vec;
    float *d_vec;

    // Lire le fichier
    FILE *f = fopen(argv[1], "r");
    if (!f) {
        perror("Erreur ouverture fichier");
        exit(EXIT_FAILURE);
    }

    fscanf(f, "%u\n", &log2size);
    if (log2size > 10) {
        printf("Size (%u) is too large: size is limited to 2^10\n", log2size);
        fclose(f);
        exit(-1);
    }

    size = 1 << log2size;
    unsigned int bytes = size * sizeof(float);
    vec = (float *)malloc(bytes);
    assert(vec);

    for (unsigned int i = 0; i < size; i++) {
        unsigned int tmp;
        fscanf(f, "%u\n", &tmp);
        vec[i] = (float)tmp;
    }
    fclose(f);

    // Afficher les valeurs lues
    printf("Données en entrée :\n");
    for (int i = 0; i < size; i++) {
        printf("%.0f ", vec[i]);
    }
    printf("\n");

    // Allocation et copie sur le device
    cudaMalloc((void**)&d_vec, bytes);
    cudaMemcpy(d_vec, vec, bytes, cudaMemcpyHostToDevice);

    // Lancer le kernel avec 1 bloc de "size" threads
    ksomme<<<1, size>>>(d_vec, size);
    
    // Copier le résultat du device vers host
    cudaMemcpy(vec, d_vec, bytes, cudaMemcpyDeviceToHost);

    // Afficher le vecteur après réduction
    printf("Résultat après réduction (ksomme):\n");
    for (int i = 0; i < size; i++) {
        printf("%.0f ", vec[i]);
    }
    printf("\n");

    printf("Somme totale : %.0f\n", vec[0]);

    // Nettoyage
    cudaFree(d_vec);
    free(vec);

    return 0;
}

