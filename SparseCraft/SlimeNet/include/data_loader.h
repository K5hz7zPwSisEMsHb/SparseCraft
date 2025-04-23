#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#define IMAGE_SIZE 16
#define NUM_CLASSES 7

typedef struct {
    double* images;
    double* features;
    int* labels;
    char** label_names;
    int num_samples;
    int num_classes;
    int image_size;
    int num_features;
    int* train_indices;
    int* val_indices;
} Dataset;

void extract_matrix_features(const double* image, double* features);
double calculate_std(const double* values, int n);
double calculate_dispersion(const double* matrix, int size);
double calculate_clustering(const double* matrix, int size);

extern const char* FIXED_LABELS[];

Dataset* load_sparse_matrix_dataset(const char* data_dir);
void free_dataset(Dataset* dataset);
void print_sample(Dataset* dataset, int index);

#endif // DATA_LOADER_H