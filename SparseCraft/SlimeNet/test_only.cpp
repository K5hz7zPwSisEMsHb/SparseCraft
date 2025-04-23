#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cnn.h"
#include "data_loader.h"
#include "msg.h"
#include <unistd.h>
#include <limits.h>


char* get_absolute_path(const char* path) {
    char* abs_path = (char*)malloc(PATH_MAX);
    if (path[0] == '/') {
        strcpy(abs_path, path);
    } else {
        char cwd[PATH_MAX];
        if (getcwd(cwd, sizeof(cwd)) != NULL) {
            snprintf(abs_path, PATH_MAX, "%s/%s", cwd, path);
        } else {
            strcpy(abs_path, path);
        }
    }
    return abs_path;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        echo(error, "Usage: %s <model_path> <test_data_dir>", argv[0]);
        return 1;
    }

    char* model_path = get_absolute_path(argv[1]);
    char* test_dir = get_absolute_path(argv[2]);

    echo(info, "Loading test dataset: %s", test_dir);
    Dataset* test_dataset = load_sparse_matrix_dataset(test_dir);
    if (!test_dataset) {
        echo(error, "Failed to load test dataset: %s", test_dir);
        free(model_path);
        free(test_dir);
        return 1;
    }
    if (test_dataset->num_samples == 0) {
        echo(error, "Test dataset is empty");
        free(test_dataset);
        return 1;
    }
    echo(success, "Successfully loaded test dataset, samples: %d", test_dataset->num_samples);

    Layer* linput = Layer_create_input(1, 16, 16);
    Layer* lconv1 = Layer_create_conv(linput, 32, 16, 16, 3, 1, 1, 0.1);
    Layer* lpool1 = Layer_create_pool(lconv1, 8, 8, 2);
    Layer* lconv2 = Layer_create_conv(lpool1, 64, 8, 8, 3, 1, 1, 0.1);
    Layer* lpool2 = Layer_create_pool(lconv2, 4, 4, 2);
    Layer* lconv3 = Layer_create_conv(lpool2, 128, 4, 4, 3, 1, 1, 0.1);
    Layer* lpool3 = Layer_create_pool(lconv3, 2, 2, 2);
    Layer* lfull1 = Layer_create_full(lpool3, 256, 0.1);
    Layer* lfull2 = Layer_create_full(lfull1, 128, 0.1);
    Layer* loutput = Layer_create_full(lfull2, test_dataset->num_classes, 0.1);
    loutput->lnext = NULL;

    echo(info, "Loading model file: %s", model_path);
    FILE* model_file = fopen(model_path, "rb");
    if (!model_file) {
        echo(error, "Failed to open model file: %s", model_path);
        free(test_dataset);
        Layer_destroy(linput);
        return 1;
    }

    Layer* layers[] = {lconv1, lconv2, lconv3, lfull1, lfull2, loutput};
    for (int i = 0; i < 6; i++) {
        Layer_load_weights(layers[i], model_file);
    }
    fclose(model_file);
    echo(success, "Model loaded successfully");

    int* true_positives = (int*)calloc(test_dataset->num_classes, sizeof(int));
    int* false_positives = (int*)calloc(test_dataset->num_classes, sizeof(int));
    int* false_negatives = (int*)calloc(test_dataset->num_classes, sizeof(int));
    int* total_samples = (int*)calloc(test_dataset->num_classes, sizeof(int));

    echo(info, "Starting test...");
    int test_correct = 0;

    for (int i = 0; i < test_dataset->num_samples; i++) {
        double* img_data = &test_dataset->images[i * 16 * 16];
        double* y = (double*)calloc(test_dataset->num_classes, sizeof(double));
        
        double mean = 0, std = 0;
        for (int j = 0; j < 16 * 16; j++) {
            img_data[j] = (img_data[j] > 0.5) ? 1.0 : 0.0;
            mean += img_data[j];
        }
        mean /= (16 * 16);
        
        for (int j = 0; j < 16 * 16; j++) {
            double diff = img_data[j] - mean;
            std += diff * diff;
        }
        std = sqrt(std / (16 * 16));
        
        if (std > 1e-6) {
            for (int j = 0; j < 16 * 16; j++) {
                img_data[j] = (img_data[j] - mean) / std;
            }
        }
        
        Layer_setInputs(linput, img_data);
        Layer_getOutputs(loutput, y);
        
        int predicted = 0;
        for (int j = 1; j < test_dataset->num_classes; j++) {
            if (y[j] > y[predicted]) {
                predicted = j;
            }
        }
        
        if (predicted == test_dataset->labels[i]) {
            test_correct++;
            true_positives[predicted]++;
        } else {
            false_positives[predicted]++;
            false_negatives[test_dataset->labels[i]]++;
        }
        total_samples[test_dataset->labels[i]]++;

        
        free(y);
    }

    double test_accuracy = 100.0 * test_correct / test_dataset->num_samples;
    echo(success, "\nTest completed, overall accuracy: %.2f%%", test_accuracy);

    free(true_positives);
    free(false_positives);
    free(false_negatives);
    free(total_samples);
    free(test_dataset);
    Layer_destroy(linput);
    Layer_destroy(lconv1);
    Layer_destroy(lpool1);
    Layer_destroy(lconv2);
    Layer_destroy(lpool2);
    Layer_destroy(lconv3);
    Layer_destroy(lpool3);
    Layer_destroy(lfull1);
    Layer_destroy(lfull2);
    Layer_destroy(loutput);

    free(model_path);
    free(test_dir);
    return 0;
}