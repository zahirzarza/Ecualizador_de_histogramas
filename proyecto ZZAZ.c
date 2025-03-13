/*
Proyecto: Ecualización de imágenes

Zarza Zurita Axel Zahir */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define NUM_BINS 256

// Prototipos de funciones
int equalizeImage(unsigned char *imgPath);
void generateGrayscaleHistogram(unsigned char *image, int width, int height, int *histogram);
void generateGrayscaleCDF(const int *histogram, int *cdf);
int findNonZeroMin(const int *cdf);
void generateGrayscaleEqCDF(const int *cdf, int *eqCdf, int cdfmin, int size);
unsigned char* generateGrayscaleEqImage(unsigned char *srcImage, int width, int height, int *eqCdf, int size);
void createGrayscaleImage(unsigned char *eqImage, int width, int height, const char *original_filename, bool parallel);
void generateCSV(int *histogram_original, int *histogram_equalized, const char *original_filename, bool parallel);

unsigned char *equalizeImageParallel(unsigned char *imgPath, int width, int height, int size, int *histogram, int *cdf, int *eqCdf, int *eqHistogram, int cdfmin);

int main(int argc, char *argv[]) {
    if (argc != 2) { /*El parámetro argument count debe ser exactamente igual a 2 (el nombre del 
                    ejecutable y la ruta o nombre de la imagen)*/
        printf("Error: Se espera un argumento con el nombre de la imagen.\n");
        return 1;
    }

    equalizeImage(argv[1]);

    return 0;

}

int equalizeImage(unsigned char *imgPath) {
    double startLoad, endLoad, timeLoad;
    int width, height, channels;
    startLoad = omp_get_wtime();
    unsigned char *image_data = stbi_load(imgPath, &width, &height, &channels, 0);
    endLoad = omp_get_wtime();
    timeLoad = endLoad - startLoad;

    if (image_data == NULL) { /*stbi_load devuelve un puntero NULL si no puede cargar la imagen
                            correctamente desde el archivo especificado*/
        printf("Error: No se pudo cargar la imagen '%s'.\n", imgPath);
        return 1;
    }

    printf("Imagen cargada con éxito.\n");

    int num_procs = omp_get_num_procs();
    omp_set_num_threads(num_procs);

    // Resolución
    int size = width * height * channels;
    printf("\nResolución de la imagen:\n");
    printf("Ancho: %d\n", width);
    printf("Alto: %d\n", height);
    printf("Tamaño de la imagen: %d bytes\n", size);
    printf("Canales: %d\n", channels);

    // Declaración de variables necesarias para las métricas
    double startSeq, endSeq, startPar, endPar, timeSeq, timePar, speedup, efficiency, overhead, startIm, endIm, startCSV, endCSV, timeIm, timeCSV;
    startSeq = endSeq = startPar = endPar = timeSeq = timePar = speedup = efficiency = overhead = startIm = endIm = startCSV = endCSV = timeIm = timeCSV = 0;

    if(channels == 1) {
        int histogram[NUM_BINS], eqHistogram[NUM_BINS], cdf[NUM_BINS], eqCdf[NUM_BINS];
        unsigned char *eqImage;
        int cdfmin;
        // Ecualización secuencial
        startSeq = omp_get_wtime();
        generateGrayscaleHistogram(image_data, width, height, histogram); // Obtener histograma de la imagen        
        generateGrayscaleCDF(histogram, cdf); // Generar el cdf (Cumulative Distributive Function)
        cdfmin = findNonZeroMin(cdf); // Encontrar cfdmin
        generateGrayscaleEqCDF(cdf, eqCdf, cdfmin, size); // Generar el nuevo cdf, que será el arreglo eqCdf
        eqImage = generateGrayscaleEqImage(image_data, width, height, eqCdf, size); // Generar el arreglo de pixeles para la nueva imagen, eqImage
        generateGrayscaleHistogram(eqImage, width, height, eqHistogram); // Generar el nuevo histograma
        endSeq = omp_get_wtime();
        startIm = omp_get_wtime();
        createGrayscaleImage(eqImage, width, height, imgPath, false); // Generar la nueva imagen
        endIm = omp_get_wtime();
        timeIm = endIm - startIm;
        startCSV = omp_get_wtime();
        generateCSV(histogram, eqHistogram, imgPath, false); // Genere el archivo csv
        endCSV = omp_get_wtime();
        timeCSV = endCSV - startCSV;

        // Ecualización paralela

        startPar = omp_get_wtime();
        eqImage = equalizeImageParallel(image_data, width, height, size, histogram, cdf, eqCdf, eqHistogram, cdfmin);
        endPar = omp_get_wtime();
        createGrayscaleImage(eqImage, width, height, imgPath, true);
        generateCSV(histogram, eqHistogram, imgPath, true);

    }

    // Cálculo de métricas 
    // Obtención de los tiempos y número de procesadores
    printf("\nSobre la ecualización:\n");
    printf("Número de procesadores: %d\n", num_procs);
    printf("Numero de hilos usados: %d\n", omp_get_max_threads());

    // Cálculo de speedup, eficiencia y overhead
    timeSeq = (endSeq - startSeq);
    timePar = (endPar - startPar);
    speedup = timeSeq / timePar;
    efficiency = speedup / num_procs;
    overhead = num_procs * (1 - efficiency);
    printf("\nMétricas:\n");
    printf("Tiempo de ejecución en serie: %f [s]\n", timeSeq);
    printf("Tiempo de ejecución en paralelo: %f [s]\n", timePar);
    printf("Speedup: %f\n", speedup);
    printf("Eficiencia: %f\n", efficiency);
    printf("Tiempo de Overhead: %f [s]\n", overhead);
    // Otros Tiempos:
    printf("\nOtros tiempos:\n");
    printf("Tiempo de carga de imagen: %f\n", timeLoad);
    printf("Tiempo de generación de imagen: %f\n", timeIm);
    printf("Tiempo de generación CSV: %f\n", timeCSV);
    
    stbi_image_free(image_data); // Liberar la memoria de la imagen cargada
}

void generateGrayscaleHistogram(unsigned char *image, int width, int height, int *histogram) {
    // Inicializar el histograma a 0
    for (int i = 0; i < NUM_BINS; ++i) {
        histogram[i] = 0;
    }

    // Calcular el histograma directamente para la imagen de escala de grises
    for (int i = 0; i < width * height; ++i) {
        histogram[image[i]]++;
    }
}

void generateGrayscaleCDF(const int *histogram, int *cdf) {
    cdf[0] = histogram[0]; // El primer valor del CDF es el primer valor del histograma

    // Calcular el CDF
    for (int i = 1; i < NUM_BINS; ++i) {
        cdf[i] = histogram[i] + cdf[i - 1];
    }
}

int findNonZeroMin(const int *cdf) {
    int minNonZero = cdf[0]; // Inicializa el mínimo con el primer valor del CDF

    // Busca el primer valor diferente de cero en el CDF
    for (int i = 1; i < NUM_BINS; ++i) {
        if (cdf[i] != 0) {
            minNonZero = cdf[i];
            break; // Se encontró el primer valor no cero, se puede detener el bucle
        }
    }

    return minNonZero;
}

void generateGrayscaleEqCDF(const int *cdf, int *eqCdf, int cdfmin, int size) {
    for (int i = 0; i < NUM_BINS; ++i) {
        double eqcdfv = round(((double)(cdf[i] - cdfmin) / (size - cdfmin)) * ((double) NUM_BINS - 2)) + 1;
        eqCdf[i] = (int) eqcdfv;
    }
}

unsigned char* generateGrayscaleEqImage(unsigned char *srcImage, int width, int height, int *eqCdf, int size) {
    
    // Verificar si eqCdf es un puntero válido y srcImage no es NULL
    if (eqCdf == NULL || srcImage == NULL) {
        return NULL;
    }

    // Crear el arreglo para la nueva imagen eqImage
    unsigned char *eqImage = malloc(size);

    // Verificar si la asignación de memoria fue exitosa
    if (eqImage == NULL) {
        return NULL;
    }

    // Generar la nueva imagen eqImage utilizando el CDF ecualizado eqCdf
    for (int i = 0; i < size; ++i) {
        eqImage[i] = (unsigned char)eqCdf[srcImage[i]];
    }

    return eqImage;
}

void createGrayscaleImage(unsigned char *eqImage, int width, int height, const char *original_filename, bool parallel) {
    char *output_filename = malloc(strlen(original_filename) + strlen("_eq_secuencial.jpg") + 1);
    strcpy(output_filename, original_filename);

    // Encontrar la extensión del archivo (asumiendo que el nombre de archivo original tiene la extensión)
    char *dot = strrchr(output_filename, '.');
    if (NULL != dot) {
        // Si se encuentra una extensión, se agrega el sufijo antes de la extensión
        *dot = '\0';
        if(parallel) {
            strcat(output_filename, "_eq_paralelo.jpg");
        } else {
            strcat(output_filename, "_eq_secuencial.jpg");
        }  
    }

    // Escribir la imagen utilizando la librería STB
    stbi_write_jpg(output_filename, width, height, 1, eqImage, 100);

    // Liberar la memoria del arreglo eqImage
    stbi_image_free(eqImage);
    free(output_filename);
}

void generateCSV(int *histogram_original, int *histogram_equalized, const char *original_filename, bool parallel) {

    char *output_filename = malloc(strlen(original_filename) + strlen("_histo_secuencial.csv") + 1);
    strcpy(output_filename, original_filename);
    // Encontrar la extensión del archivo (asumiendo que el nombre de archivo original tiene la extensión)
    char *dot = strrchr(output_filename, '.');
    if (NULL != dot) {
        // Si se encuentra una extensión, se agrega el sufijo antes de la extensión
        *dot = '\0';
        if(parallel) {
            strcat(output_filename, "_histo_paralelo.csv");
        } else {
            strcat(output_filename, "_histo_secuencial.csv");
        }
    }
    FILE *csv_file = fopen(output_filename, "w"); // Abre el archivo en modo escritura

    if (csv_file == NULL) {
        printf("Error al abrir el archivo.");
        return;
    }

    fprintf(csv_file, "valor, histo, eqHisto\n");

    for (int i = 0; i < 256; ++i) {
        fprintf(csv_file, "%d, %d, %d\n", i, histogram_original[i], histogram_equalized[i]);
    }

    fclose(csv_file); // Cierra el archivo
}

unsigned char* equalizeImageParallel(unsigned char *image_data, int width, int height, int size,
                                     int *histogram, int *cdf, int *eqCdf, int *eqHistogram, int cdfmin) {

    // Obtener el histograma en paralelo usando reduction
    #pragma omp parallel
    {
        #pragma omp for
        for(int i=0; i < NUM_BINS; i++)
            histogram[i] = 0;

        #pragma omp barrier

        #pragma omp for reduction(+:histogram[:NUM_BINS])
        for(int i=0; i < size; i++)
            histogram[image_data[i]]++;
    }

    // Calcular el CDF en secuencia con un solo hilo
    #pragma omp single
    {
        cdf[0] = histogram[0];
        for (int i = 1; i < NUM_BINS; ++i) {
            cdf[i] = histogram[i] + cdf[i - 1];
        }

        cdfmin = cdf[0];
        for (int i = 1; i < NUM_BINS; ++i) {
            if (cdf[i] != 0) {
                cdfmin = cdf[i];
                break;
            }
        }
    }

    // Generar el CDF ecualizado en paralelo
    #pragma omp for nowait
    for (int i = 0; i < NUM_BINS; ++i) {
        double eqcdfv = round(((double)(cdf[i] - cdfmin) / (size - cdfmin)) * ((double) NUM_BINS - 2)) + 1;
        eqCdf[i] = (int) eqcdfv;
    }

    // Generar la nueva imagen en paralelo
    #pragma omp for nowait
    for (int i = 0; i < size; ++i) {
        image_data[i] = (unsigned char)eqCdf[image_data[i]];
    }

    // Generar el nuevo histograma en paralelo usando reduction
    #pragma omp parallel
    {
        #pragma omp for
        for(int i=0; i < NUM_BINS; i++)
            eqHistogram[i] = 0;

        #pragma omp barrier

        #pragma omp for reduction(+:eqHistogram[:NUM_BINS])
        for(int i = 0; i < size; i++)
            eqHistogram[image_data[i]]++;
    }

    return image_data;
}
