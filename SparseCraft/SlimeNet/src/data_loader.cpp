#include "data_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

#define IMAGE_SIZE 16
#define MAX_CLASSES 100
#define MAX_SAMPLES_PER_CLASS 100000

const char *FIXED_LABELS[] = {
    "COO", "CSR", "ELL", "HYB", "DRW", "DCL", "DNS"};

Dataset *load_sparse_matrix_dataset(const char *data_dir)
{
    Dataset *dataset = (Dataset *)calloc(1, sizeof(Dataset));
    if (!dataset)
        return NULL;

    dataset->num_classes = NUM_CLASSES;
    dataset->label_names = (char **)malloc(NUM_CLASSES * sizeof(char *));
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        dataset->label_names[i] = (char *)malloc(256);
        strcpy(dataset->label_names[i], FIXED_LABELS[i]);
    }

    int total_samples = 0;

    const char *filenames[] = {
        "COO", "CSR", "ELL", "HYB",
        "DRW", "DCL", "DNS"};

    for (int label = 0; label < 7; ++label)
    {
        char filepath[512];
        sprintf(filepath, "%s/%s.txt", data_dir, filenames[label]);

        FILE *fp = fopen(filepath, "r");
        if (!fp)
            continue;

        int line_count = 0;
        char line[512];
        while (fgets(line, sizeof(line), fp))
            line_count++;
        
        total_samples += line_count;
        fclose(fp);
    }

    size_t image_size = (size_t)IMAGE_SIZE * IMAGE_SIZE;
    size_t max_samples = (size_t)MAX_CLASSES * total_samples;
    size_t total_elements = max_samples * image_size;
    size_t total_bytes = total_elements * sizeof(double);

    // double *temp_images = (double *)malloc(total_bytes);
    // int *temp_labels = (int *)malloc(max_samples * sizeof(int));

    dataset->num_samples = total_samples;
    dataset->images = (double *)malloc(total_samples * IMAGE_SIZE * IMAGE_SIZE * sizeof(double));
    dataset->labels = (int *)malloc(total_samples * sizeof(int));

    if (!dataset->images || !dataset->labels)
    {
        fprintf(stderr, "Memory allocation failed; temp_images[%llu] (%.3lf MB) = %p, temp_labels[%llu] (%.3lf MB) = %p\n",
                total_elements, total_bytes / (1024.0 * 1024.0), dataset->images,
                max_samples, max_samples * sizeof(int) / (1024.0 * 1024.0), dataset->labels);
        free(dataset);
        return NULL;
    }

    int counter = 0;
    for (int label = 0; label < 7; ++label)
    {
        char filepath[512];
        sprintf(filepath, "%s/%s.txt", data_dir, filenames[label]);

        FILE *fp = fopen(filepath, "r");
        if (!fp)
            continue;
        char line[512];

        while (fgets(line, sizeof(line), fp))
        {
            size_t len = strlen(line);
            if (len < IMAGE_SIZE * IMAGE_SIZE)
            {
                fprintf(stderr, "Warning: Line length insufficient (%zu < %d)\n", len, IMAGE_SIZE * IMAGE_SIZE);
                continue;
            }

            for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++)
            {
                dataset->images[counter * IMAGE_SIZE * IMAGE_SIZE + i] = (line[i] == '1') ? 1.0 : 0.0;
            }
            dataset->labels[counter] = label;
            counter++;
        }
        fclose(fp);
    }

    return dataset;
}

void free_dataset(Dataset *dataset)
{
    if (dataset)
    {
        free(dataset->images);
        free(dataset->labels);
        free(dataset->train_indices);
        free(dataset->val_indices);
        for (int i = 0; i < MAX_CLASSES; i++)
        {
            free(dataset->label_names[i]);
        }
        free(dataset->label_names);
        free(dataset);
    }
}

void print_sample(Dataset *dataset, int index)
{
    if (!dataset || index < 0 || index >= dataset->num_samples)
    {
        fprintf(stderr, "Invalid sample index\n");
        return;
    }

    fprintf(stderr, "Sample %d (Label: %s):\n",
            index, dataset->label_names[dataset->labels[index]]);

    double *img = &dataset->images[index * IMAGE_SIZE * IMAGE_SIZE];
    for (int i = 0; i < IMAGE_SIZE; i++)
    {
        for (int j = 0; j < IMAGE_SIZE; j++)
        {
            fprintf(stderr, "%c", img[i * IMAGE_SIZE + j] > 0.5 ? '1' : '0');
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
}