#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

int main(int argc, char **argv) {

    clock_t start;
    clock_t end;
    

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
    // if (log2size > 10) {
    //     printf("Size (%u) is too large: size is limited to 2^10\n", log2size);
    //     fclose(f);
    //     exit(-1);
    // }

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

    float sum = 0; 
    start=clock();
    for (int i = 0; i < size; i++) {
        sum += vec[i]; 
    }
    end=clock();
    printf("somme totale : %.0f\n", sum); 
    double temps=((double)end - (double)start)/CLOCKS_PER_SEC;
    printf("temps execution %lf \n",temps);
}