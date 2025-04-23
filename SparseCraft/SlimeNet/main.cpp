#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <math.h>
#include "msg.h"
#include "data_loader.h"
#include "cnn.h"

/* main */
int main(int argc, char* argv[])
{
    if (argc < 3) {
        echo(error, "Usage: %s <training_data_dir> <output_model_path> [test_data_dir]", argv[0]);
        return 1;
    }

    const char* train_dir = argv[1];
    const char* best_model_path = argv[2];
    const char* test_dir = (argc > 3) ? argv[3] : NULL;

    Dataset* train_dataset = NULL;
    Dataset* test_dataset = NULL;

    echo(info, "Loading training dataset: %s", train_dir);
    train_dataset = load_sparse_matrix_dataset(train_dir);
    if (!train_dataset) {
        echo(error, "Failed to load training dataset: %s", train_dir);
        return 1;
    }
    echo(success, "Successfully loaded training dataset, samples: %d", train_dataset->num_samples);

    if (test_dir) {
        echo(info, "Loading test dataset: %s", test_dir);
        test_dataset = load_sparse_matrix_dataset(test_dir);
        if (!test_dataset) {
            echo(error, "Failed to load test dataset: %s", test_dir);
            free(train_dataset);
            return 1;
        }
        echo(success, "Successfully loaded test dataset, samples: %d", test_dataset->num_samples);
    }

    srand(0);

    echo(info, "Training dataset statistics:");
    for (int i = 0; i < train_dataset->num_classes; i++) {
        int count = 0;
        for (int j = 0; j < train_dataset->num_samples; j++) {
            if (train_dataset->labels[j] == i) count++;
        }
        echo(info, "Class %s: %d samples", train_dataset->label_names[i], count);
    }

    Layer* linput = Layer_create_input(1, 16, 16);
    Layer* lconv1 = Layer_create_conv(linput, 32, 16, 16, 3, 1, 1, 0.1);  
    Layer* lpool1 = Layer_create_pool(lconv1, 8, 8, 2);  
    Layer* lconv2 = Layer_create_conv(lpool1, 64, 8, 8, 3, 1, 1, 0.1);  
    Layer* lpool2 = Layer_create_pool(lconv2, 4, 4, 2);  
    Layer* lconv3 = Layer_create_conv(lpool2, 128, 4, 4, 3, 1, 1, 0.1);
    Layer* lpool3 = Layer_create_pool(lconv3, 2, 2, 2);
    Layer* lfull1 = Layer_create_full(lpool3, 256, 0.1);
    Layer* lfull2 = Layer_create_full(lfull1, 128, 0.1);
    Layer* loutput = Layer_create_full(lfull2, 7, 0.1);
    loutput->lnext = NULL;

    int num_total = train_dataset->num_samples;
    int* all_indices = (int*)malloc(num_total * sizeof(int));
    for (int i = 0; i < num_total; i++) {
        all_indices[i] = i;
    }
    
    double rate = 0.01;
    int nepoch = 50;
    int batch_size = 32;
    double rate_decay = 0.98;
    double best_accuracy = 0.0;

    for (int epoch = 0; epoch < nepoch; epoch++) {
        double current_rate = rate * pow(rate_decay, epoch);
        double train_loss = 0;
        int train_samples = 0;
        
        for (int i = num_total - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = all_indices[i];
            all_indices[i] = all_indices[j];
            all_indices[j] = temp;
        }
        
        int processed = 0;
        while (processed < num_total) {
            for (int b = 0; b < batch_size && processed < num_total; b++) {
                int idx = all_indices[processed++];
                
                double* img_data = &train_dataset->images[idx * 16 * 16];
                double* y = (double*)calloc(train_dataset->num_classes, sizeof(double));
                
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
                
                for (int j = 0; j < train_dataset->num_classes; j++) {
                    y[j] = (j == train_dataset->labels[idx]) ? 1.0 : 0.0;
                }
                
                Layer_learnOutputs(loutput, y);
                train_loss += Layer_getErrorTotal(loutput);
                train_samples++;
                
                free(y);
            }
            
            Layer* layer = loutput;
            while (layer != NULL) {
                Layer_update(layer, current_rate/batch_size);
                layer = layer->lprev;
            }
            
            if (processed % 1000 == 0) {
                echo(info, "Epoch %d: Processed %d/%d samples, current loss: %.4f", 
                    epoch, processed, num_total, train_loss/train_samples);
            }
        }
        
        echo(info, "Epoch %d completed, loss: %.4f", epoch, train_loss/train_samples);

        int correct = 0;
        for (int i = 0; i < num_total; i++) {
            double* img_data = &train_dataset->images[i * 16 * 16];
            double* y = (double*)calloc(train_dataset->num_classes, sizeof(double));
            
            Layer_setInputs(linput, img_data);
            Layer_getOutputs(loutput, y);
            
            int predicted = 0;
            for (int j = 1; j < train_dataset->num_classes; j++) {
                if (y[j] > y[predicted]) predicted = j;
            }
            
            if (predicted == train_dataset->labels[i]) correct++;
            free(y);
        }
        
        double accuracy = 100.0 * correct / num_total;
        echo(info, "Epoch %d training accuracy: %.2f%%", epoch, accuracy);

        if (accuracy > best_accuracy) {
            best_accuracy = accuracy;
            FILE* model_file = fopen(best_model_path, "wb");
            if (model_file != NULL) {
                Layer* layers[] = {lconv1, lconv2, lconv3, lfull1, lfull2, loutput};
                for (int i = 0; i < 6; i++) {
                    Layer_save_weights(layers[i], model_file);
                }
                fclose(model_file);
                echo(success, "Saved best model, accuracy: %.2f%%", accuracy);
            }
        }
    }

    if (test_dataset) {
        echo(info, "Evaluating model performance on test set...");
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
                if (y[j] > y[predicted]) predicted = j;
            }
            
            if (predicted == test_dataset->labels[i]) test_correct++;
            free(y);
            
            if ((i + 1) % 100 == 0) {
                echo(info, "Tested %d/%d samples", i + 1, test_dataset->num_samples);
            }
        }
        
        double test_accuracy = 100.0 * test_correct / test_dataset->num_samples;
        echo(success, "Test set accuracy: %.2f%%", test_accuracy);
    }

    if (train_dataset) free(train_dataset);
    if (test_dataset) free(test_dataset);
    free(all_indices);
    
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

    return 0;
}