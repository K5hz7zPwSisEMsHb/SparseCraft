#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>  
#include <stdint.h>  

typedef struct Node {
    int is_leaf;
    int feature_index;
    double threshold;
    int predicted_class;
    struct Node* left;
    struct Node* right;
} Node;

void save_tree_node(Node* node, FILE* fp, int depth);  
void save_decision_tree(Node* root, const char* filename);
Node* load_tree_node(FILE* fp, int idx);
Node* load_decision_tree(const char* filename);

#define MAX_FEATURES 10
#define MAX_SAMPLES 1000
#define MAX_DEPTH 25
#define MIN_SAMPLES_SPLIT 10
#define MIN_GAIN 0.0005

typedef struct {
    double** features;
    int* labels;
    int n_samples;
    int n_features;
    int n_classes;
} Dataset;

double calculate_gini(int* labels, int n_samples, int n_classes) {
    if (n_samples == 0) return 0.0;
    
    int* counts = (int*)calloc(n_classes, sizeof(int));
    double gini = 1.0;
    
    for (int i = 0; i < n_samples; i++) {
        counts[labels[i]]++;
    }
    
    for (int i = 0; i < n_classes; i++) {
        double p = (double)counts[i] / n_samples;
        gini -= p * p;
    }
    
    free(counts);
    return gini;
}

void find_best_split(Dataset* dataset, int* sample_indices, int n_samples,
                    int* best_feature, double* best_threshold, double* best_gini) {
    *best_gini = 1.0;
    *best_threshold = 0.0;
    *best_feature = 0;
    
    double parent_gini = calculate_gini(dataset->labels, n_samples, dataset->n_classes);
    printf("  Finding best split...\n");
    
    for (int feature = 0; feature < dataset->n_features; feature++) {
        printf("    Evaluating feature %d/%d\r", feature + 1, dataset->n_features);
        fflush(stdout);
        
        int valid_values = 0;
        for (int i = 0; i < n_samples; i++) {
            double val = dataset->features[sample_indices[i]][feature];
            if (!isnan(val) && !isinf(val)) {
                valid_values++;
            }
        }
        
        if (valid_values < n_samples * 0.5) {
            printf("Warning: Feature %d has too few valid values (%d/%d), skipping\n", 
                   feature, valid_values, n_samples);
            continue;
        }
        
        double* feature_values = (double*)malloc(n_samples * sizeof(double));
        int* sorted_indices = (int*)malloc(n_samples * sizeof(int));
        
        for (int i = 0; i < n_samples; i++) {
            feature_values[i] = dataset->features[sample_indices[i]][feature];
            sorted_indices[i] = i;
        }
        
        for (int i = 0; i < n_samples - 1; i++) {
            for (int j = 0; j < n_samples - i - 1; j++) {
                if (feature_values[j] > feature_values[j + 1]) {
                    double temp_val = feature_values[j];
                    feature_values[j] = feature_values[j + 1];
                    feature_values[j + 1] = temp_val;
                    
                    int temp_idx = sorted_indices[j];
                    sorted_indices[j] = sorted_indices[j + 1];
                    sorted_indices[j + 1] = temp_idx;
                }
            }
        }
        
        for (int i = 0; i < n_samples - 1; i++) {
            if (i > 0 && feature_values[i] == feature_values[i - 1]) {
                continue;
            }
            
            double threshold;
            if (feature_values[i] == feature_values[i + 1]) {
                threshold = feature_values[i];
            } else {
                threshold = (feature_values[i] + feature_values[i + 1]) / 2.0;
            }
            
            if (isnan(threshold) || isinf(threshold)) {
                printf("Warning: Invalid threshold calculation for feature %d at index %d\n", feature, i);
                continue;
            }
            
            int left_count = i + 1;
            int right_count = n_samples - left_count;
            
            if (left_count < MIN_SAMPLES_SPLIT/2 || right_count < MIN_SAMPLES_SPLIT/2) {
                continue;
            }
            
            int* left_labels = (int*)malloc(left_count * sizeof(int));
            int* right_labels = (int*)malloc(right_count * sizeof(int));
            
            for (int j = 0; j < left_count; j++) {
                left_labels[j] = dataset->labels[sample_indices[sorted_indices[j]]];
            }
            for (int j = 0; j < right_count; j++) {
                right_labels[j] = dataset->labels[sample_indices[sorted_indices[left_count + j]]];
            }
            
            double left_gini = calculate_gini(left_labels, left_count, dataset->n_classes);
            double right_gini = calculate_gini(right_labels, right_count, dataset->n_classes);
            double weighted_gini = (left_count * left_gini + right_count * right_gini) / n_samples;
            
            double gain = parent_gini - weighted_gini;
            
            if (weighted_gini < *best_gini && gain > MIN_GAIN) {
                if (left_count >= MIN_SAMPLES_SPLIT/2 && right_count >= MIN_SAMPLES_SPLIT/2) {
                    double left_purity = calculate_gini(left_labels, left_count, dataset->n_classes);
                    double right_purity = calculate_gini(right_labels, right_count, dataset->n_classes);
                    
                    if (left_purity < parent_gini * 0.9 || right_purity < parent_gini * 0.9) {
                        *best_gini = weighted_gini;
                        *best_feature = feature;
                        *best_threshold = threshold;
                    }
                }
            }
            
            free(left_labels);
            free(right_labels);
        }
        
        free(feature_values);
        free(sorted_indices);
    }
    printf("\n");
}

Node* create_node(Dataset* dataset, int* sample_indices, int n_samples, int depth) {
    static int node_count = 0;
    static Node* root = NULL;
    
    Node* node = (Node*)malloc(sizeof(Node));
    if (!node) {
        printf("[ERROR] Memory allocation failed\n");
        return NULL;
    }
    
    node->is_leaf = 0;
    node->feature_index = 0;
    node->threshold = 0.0;
    node->predicted_class = 0;
    node->left = NULL;
    node->right = NULL;

    node_count++;
    
    if (depth == 0) {
        root = node;
    }
    
    if (n_samples == 0) {
        node->is_leaf = 1;
        node->predicted_class = 0;
        node->left = NULL;
        node->right = NULL;
        return node;
    }
    
    printf("\n=== Node Creation [%d] ===\n", node_count);
    printf("Depth: %d\nSample count: %d\n", depth, n_samples);
    
    int* class_counts = (int*)calloc(dataset->n_classes, sizeof(int));
    for (int i = 0; i < n_samples; i++) {
        class_counts[dataset->labels[sample_indices[i]]]++;
    }
    
    int max_count = 0;
    int dominant_class = 0;
    for (int i = 0; i < dataset->n_classes; i++) {
        if (class_counts[i] > max_count) {
            max_count = class_counts[i];
            dominant_class = i;
        }
    }
    
    double purity = (double)max_count / n_samples;
    
    printf("Class distribution:\n");
    const char* formats[] = {"COO", "CSR", "DCL", "DRW", "DNS", "ELL", "HYB"};
    for (int i = 0; i < dataset->n_classes; i++) {
        if (class_counts[i] > 0) {
            printf("  %s: %d (%.1f%%)\n", 
                   formats[i], class_counts[i], 
                   (double)class_counts[i] / n_samples * 100);
        }
    }
    
    if (depth >= MAX_DEPTH || n_samples < MIN_SAMPLES_SPLIT || purity > 0.98) {
        node->is_leaf = 1;
        node->predicted_class = dominant_class;
        printf("\nStop splitting reason: %s\n", 
               depth >= MAX_DEPTH ? "Max depth reached" : 
               (purity > 0.98 ? "Sufficient purity reached" : "Insufficient samples"));
        printf("Create leaf node, predicted class: %s (%.1f%% samples)\n", 
               formats[node->predicted_class], purity * 100);
        free(class_counts);
        return node;
    }
    
    int best_feature;
    double best_threshold, best_gini;
    find_best_split(dataset, sample_indices, n_samples, &best_feature, &best_threshold, &best_gini);
    
    printf("\nBest split:\n");
    printf("Feature index: %d\n", best_feature);
    printf("Split threshold: %.4f\n", best_threshold);
    printf("Gini impurity: %.4f\n", best_gini);
    
    if (best_gini >= 0.99) {
        node->is_leaf = 1;
        node->predicted_class = dominant_class;
        free(class_counts);
        return node;
    }
    
    node->is_leaf = 0;
    node->feature_index = best_feature;
    node->threshold = best_threshold;
    node->left = NULL;
    node->right = NULL;
    
    if (root) {
        printf("\nSaving current tree (node count: %d)...\n", node_count);
        save_decision_tree(root, "decision_tree_model.bin");
    }
    
    int left_count = 0, right_count = 0;
    int* left_indices = (int*)malloc(n_samples * sizeof(int));
    int* right_indices = (int*)malloc(n_samples * sizeof(int));
    
    for (int i = 0; i < n_samples; i++) {
        if (dataset->features[sample_indices[i]][best_feature] <= best_threshold) {
            left_indices[left_count++] = sample_indices[i];
        } else {
            right_indices[right_count++] = sample_indices[i];
        }
    }
    
    printf("\nLeft child sample count: %d\n", left_count);
    printf("Right child sample count: %d\n", right_count);
    
    printf("\nCreating left child...\n");
    node->left = create_node(dataset, left_indices, left_count, depth + 1);
    printf("\nCreating right child...\n");
    node->right = create_node(dataset, right_indices, right_count, depth + 1);
    
    free(left_indices);
    free(right_indices);
    free(class_counts);
    
    return node;
}

Node* train_decision_tree(Dataset* dataset) {
    printf("\n========== Starting Decision Tree Training ==========\n");
    printf("Dataset information:\n");
    printf("  Total samples: %d\n", dataset->n_samples);
    printf("  Feature count: %d\n", dataset->n_features);
    printf("  Class count: %d\n", dataset->n_classes);
    printf("Training parameters:\n");
    printf("  Max depth: %d\n", MAX_DEPTH);
    printf("  Min split samples: %d\n\n", MIN_SAMPLES_SPLIT);
    
    int* sample_indices = (int*)malloc(dataset->n_samples * sizeof(int));
    for (int i = 0; i < dataset->n_samples; i++) {
        sample_indices[i] = i;
    }
    
    printf("Starting root node construction...\n");
    Node* root = create_node(dataset, sample_indices, dataset->n_samples, 0);
    free(sample_indices);
    
    printf("\n========== Decision Tree Training Complete ==========\n");
    return root;
}

int predict_sample(Node* node, double* features) {
    while (!node->is_leaf) {
        if (features[node->feature_index] <= node->threshold) {
            node = node->left;
        } else {
            node = node->right;
        }
    }
    return node->predicted_class;
}

int seg_tree_predict_sample(Node* segTree, double* features) {
    int cursor = 1;
    while (!segTree[cursor].is_leaf) {
        if (features[segTree[cursor].feature_index] <= segTree[cursor].threshold) {
            cursor = cursor << 1;
        } else {
            cursor = cursor << 1 | 1;
        }
    }
    return segTree[cursor].predicted_class;
}

void free_tree(Node* node) {
    if (node == NULL) return;
    free_tree(node->left);
    free_tree(node->right);
    free(node);
}

#define MATRIX_SIZE 16
#define NUM_FEATURES 16
#define NUM_FORMATS 7

typedef enum {
    FORMAT_COO = 0,
    FORMAT_CSR,
    FORMAT_DCL,
    FORMAT_DRW,
    FORMAT_DNS,
    FORMAT_ELL,
    FORMAT_HYB
} MatrixFormat;

double calculate_std(double* values, int n) {
    if (n <= 1) return 0.0;
    double mean = 0.0, std = 0.0;
    
    for (int i = 0; i < n; i++) {
        mean += values[i];
    }
    mean /= n;
    
    for (int i = 0; i < n; i++) {
        double diff = values[i] - mean;
        std += diff * diff;
    }
    std = sqrt(std / (n - 1));
    return std;
}

// 特征提取函数
// 在特征提取函数中修改第8和第9个特征的计算
void extract_features(const char* binary_str, double* features) {
    int matrix[MATRIX_SIZE][MATRIX_SIZE] = {0};
    
    // 将二进制字符串转换为矩阵
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            matrix[i][j] = binary_str[i * MATRIX_SIZE + j] - '0';
        }
    }
    
    // 1. 行数
    features[0] = MATRIX_SIZE;
    
    // 2. 非零元素的数量
    int nnz = 0;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            if (matrix[i][j] != 0) nnz++;
        }
    }
    features[1] = nnz;
    
    // 3. 非空行数
    int non_empty_rows = 0;
    int* row_nnz = (int*)calloc(MATRIX_SIZE, sizeof(int));
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            if (matrix[i][j] != 0) {
                row_nnz[i]++;
                if (row_nnz[i] == 1) non_empty_rows++;
            }
        }
    }
    features[2] = non_empty_rows;
    
    // 4. DIA格式中对角线的数量
    int dia_count = 0;
    for (int d = -MATRIX_SIZE + 1; d < MATRIX_SIZE; d++) {
        int has_nonzero = 0;
        for (int i = 0; i < MATRIX_SIZE; i++) {
            int j = i + d;
            if (j >= 0 && j < MATRIX_SIZE && matrix[i][j] != 0) {
                has_nonzero = 1;
                break;
            }
        }
        dia_count += has_nonzero;
    }
    features[3] = dia_count;
    
    // 5. 平均每行非零元的数量
    features[4] = (double)nnz / MATRIX_SIZE;
    
    // 6. 非零元的密度
    features[5] = (double)nnz / (MATRIX_SIZE * MATRIX_SIZE);
    
    // 7. 每行非零块的数量的标准差
    double* blocks_per_row = (double*)calloc(MATRIX_SIZE, sizeof(double));
    int total_blocks = 0;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        int blocks = 0;
        int in_block = 0;
        for (int j = 0; j < MATRIX_SIZE; j++) {
            if (matrix[i][j] != 0) {
                if (!in_block) {
                    blocks++;
                    in_block = 1;
                }
            } else {
                in_block = 0;
            }
        }
        blocks_per_row[i] = blocks;
        total_blocks += blocks;
    }
    features[6] = calculate_std(blocks_per_row, MATRIX_SIZE);
    
    // 8. 每行非零块的大小标准差
    double* block_sizes = (double*)malloc(total_blocks * sizeof(double));
    int block_idx = 0;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        int size = 0;
        for (int j = 0; j < MATRIX_SIZE; j++) {
            if (matrix[i][j] != 0) {
                size++;
            } else if (size > 0) {
                block_sizes[block_idx++] = size;
                size = 0;
            }
        }
        if (size > 0) {
            block_sizes[block_idx++] = size;
        }
    }
    features[7] = calculate_std(block_sizes, total_blocks);
    
    // 9. 每行非零元数量的变异系数
    double row_nnz_mean = (double)nnz / MATRIX_SIZE;
    double row_nnz_std = 0.0;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        double diff = row_nnz[i] - row_nnz_mean;
        row_nnz_std += diff * diff;
    }
    row_nnz_std = sqrt(row_nnz_std / (MATRIX_SIZE - 1));
    // 添加防止除零检查
    features[8] = (row_nnz_mean > 0) ? (row_nnz_std / row_nnz_mean) : 0.0;  // 变异系数
    
    // 10. relative_range
    int max_nnz = 0, min_nnz = MATRIX_SIZE;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        if (row_nnz[i] > max_nnz) max_nnz = row_nnz[i];
        if (row_nnz[i] < min_nnz && row_nnz[i] > 0) min_nnz = row_nnz[i];
    }
    // 添加防止除零检查
    features[9] = (max_nnz > 0) ? ((double)(max_nnz - min_nnz) / max_nnz) : 0.0;
    
    // 11. 每行非零元的最大值
    features[10] = 1.0;  // 对于二值矩阵，最大值始终为1
    
    // 12. 相邻行的非零元的差的平均差
    double avg_diff = 0.0;
    int diff_count = 0;
    for (int i = 0; i < MATRIX_SIZE - 1; i++) {
        int diff = abs(row_nnz[i] - row_nnz[i + 1]);
        avg_diff += diff;
        diff_count++;
    }
    features[11] = diff_count > 0 ? avg_diff / diff_count : 0;
    
    // 13. 非零块数
    features[12] = total_blocks;
    
    // 14. bandwidth index (BWI) 的标准差
    double* bwi = (double*)malloc(MATRIX_SIZE * sizeof(double));
    for (int i = 0; i < MATRIX_SIZE; i++) {
        int min_col = MATRIX_SIZE, max_col = -1;
        for (int j = 0; j < MATRIX_SIZE; j++) {
            if (matrix[i][j] != 0) {
                if (j < min_col) min_col = j;
                if (j > max_col) max_col = j;
            }
        }
        bwi[i] = (max_col >= min_col) ? (max_col - min_col + 1) : 0;
    }
    features[13] = calculate_std(bwi, MATRIX_SIZE);
    
    // 15. dispersion的平均值
    double total_dispersion = 0.0;
    int dispersion_count = 0;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        if (row_nnz[i] > 0) {
            total_dispersion += (double)row_nnz[i] / bwi[i];
            dispersion_count++;
        }
    }
    features[14] = dispersion_count > 0 ? total_dispersion / dispersion_count : 0;
    
    // 16. clustering
    double clustering = 0.0;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            if (matrix[i][j] != 0) {
                int neighbors = 0;
                if (i > 0 && matrix[i-1][j] != 0) neighbors++;
                if (i < MATRIX_SIZE-1 && matrix[i+1][j] != 0) neighbors++;
                if (j > 0 && matrix[i][j-1] != 0) neighbors++;
                if (j < MATRIX_SIZE-1 && matrix[i][j+1] != 0) neighbors++;
                clustering += neighbors;
            }
        }
    }
    features[15] = nnz > 0 ? clustering / (4 * nnz) : 0;  // 归一化
    
    // 清理内存
    free(row_nnz);
    free(blocks_per_row);
    free(block_sizes);
    free(bwi);
}

// 读取数据集函数
// 添加数据集划分结构
typedef struct {
    Dataset* train;
    Dataset* test;
} SplitDataset;

// 修改数据集加载函数
SplitDataset* load_and_split_dataset(const char* data_dir) {
    // 首先计算总样本数
    char filepath[256];
    int total_samples = 0;
    const char* formats[] = {"COO", "CSR", "DCL", "DRW", "DNS", "ELL", "HYB"};
    
    for (int i = 0; i < NUM_FORMATS; i++) {
        sprintf(filepath, "%s/%s.txt", data_dir, formats[i]);
        FILE* fp = fopen(filepath, "r");
        if (fp) {
            char line[257];
            while (fgets(line, sizeof(line), fp)) {
                if (strlen(line) >= 256) total_samples++;
            }
            fclose(fp);
        }
    }
    
    // 计算训练集和测试集大小
    int test_size = total_samples / 10;  // 10%作为测试集
    int train_size = total_samples - test_size;
    
    // 创建训练集和测试集
    Dataset* train_dataset = (Dataset*)malloc(sizeof(Dataset));
    Dataset* test_dataset = (Dataset*)malloc(sizeof(Dataset));
    
    train_dataset->n_features = NUM_FEATURES;
    train_dataset->n_classes = NUM_FORMATS;
    train_dataset->n_samples = train_size;
    test_dataset->n_features = NUM_FEATURES;
    test_dataset->n_classes = NUM_FORMATS;
    test_dataset->n_samples = test_size;
    
    // 分配内存
    train_dataset->features = (double**)malloc(train_size * sizeof(double*));
    train_dataset->labels = (int*)malloc(train_size * sizeof(int));
    test_dataset->features = (double**)malloc(test_size * sizeof(double*));
    test_dataset->labels = (int*)malloc(test_size * sizeof(int));
    
    for (int i = 0; i < train_size; i++) {
        train_dataset->features[i] = (double*)malloc(NUM_FEATURES * sizeof(double));
    }
    for (int i = 0; i < test_size; i++) {
        test_dataset->features[i] = (double*)malloc(NUM_FEATURES * sizeof(double));
    }
    
    // 读取数据并随机划分
    int train_idx = 0, test_idx = 0;
    srand(42);  // 设置随机种子
    
    for (int format = 0; format < NUM_FORMATS; format++) {
        sprintf(filepath, "%s/%s.txt", data_dir, formats[format]);
        FILE* fp = fopen(filepath, "r");
        if (fp) {
            char line[257];
            while (fgets(line, sizeof(line), fp)) {
                if (strlen(line) >= 256) {
                    line[256] = '\0';
                    // 随机决定是否放入测试集
                    if ((rand() % 10) == 0 && test_idx < test_size) {
                        extract_features(line, test_dataset->features[test_idx]);
                        test_dataset->labels[test_idx] = format;
                        test_idx++;
                    } else if (train_idx < train_size) {
                        extract_features(line, train_dataset->features[train_idx]);
                        train_dataset->labels[train_idx] = format;
                        train_idx++;
                    }
                }
            }
            fclose(fp);
        }
    }
    
    SplitDataset* split = (SplitDataset*)malloc(sizeof(SplitDataset));
    split->train = train_dataset;
    split->test = test_dataset;
    return split;
}


void save_tree_node(Node* node, FILE* fp, int depth) {
    printf("  [DEBUG] Saving node at depth %d (address: %p)\n", depth, (void*)node);
    
    if (!fp) {
        printf("  [ERROR] File pointer is NULL\n");
        return;
    }
    
    int is_null = (node == NULL);
    printf("  [DEBUG] Writing node null flag: %d\n", is_null);
    size_t write_size = fwrite(&is_null, sizeof(int), 1, fp);
    if (write_size != 1) {
        printf("  [ERROR] Failed to write node null flag: wrote %zu bytes\n", write_size);
        return;
    }
    
    if (node) {
        if ((uintptr_t)node < 0x1000) {
            printf("  [ERROR] Invalid node pointer detected: %p\n", (void*)node);
            return;
        }
        
        printf("  [DEBUG] Node info: is_leaf=%d, feature_index=%d, threshold=%.6f, predicted_class=%d\n",
               node->is_leaf, node->feature_index, node->threshold, node->predicted_class);
        
        write_size = fwrite(&node->is_leaf, sizeof(int), 1, fp);
        printf("  [DEBUG] Wrote is_leaf: %zu bytes\n", write_size);
        
        write_size = fwrite(&node->feature_index, sizeof(int), 1, fp);
        printf("  [DEBUG] Wrote feature_index: %zu bytes\n", write_size);
        
        write_size = fwrite(&node->threshold, sizeof(double), 1, fp);
        printf("  [DEBUG] Wrote threshold: %zu bytes\n", write_size);
        
        write_size = fwrite(&node->predicted_class, sizeof(int), 1, fp);
        printf("  [DEBUG] Wrote predicted_class: %zu bytes\n", write_size);
        
        if (node->left && (uintptr_t)node->left < 0x1000) {
            printf("  [ERROR] Invalid left child pointer detected: %p\n", (void*)node->left);
            node->left = NULL;
        }
        if (node->right && (uintptr_t)node->right < 0x1000) {
            printf("  [ERROR] Invalid right child pointer detected: %p\n", (void*)node->right);
            node->right = NULL;
        }
        
        printf("  [DEBUG] Preparing to save left child (address: %p)...\n", (void*)node->left);
        save_tree_node(node->left, fp, depth + 1);
        
        printf("  [DEBUG] Preparing to save right child (address: %p)...\n", (void*)node->right);
        save_tree_node(node->right, fp, depth + 1);
    }
    
    printf("  [DEBUG] Node at depth %d saved\n", depth);
}

void save_decision_tree(Node* root, const char* filename) {
    printf("[DEBUG] Starting to save decision tree\n");
    printf("[DEBUG] Root node address: %p\n", (void*)root);
    printf("[DEBUG] Save path: %s\n", filename);
    
    if (!root || !filename) {
        fprintf(stderr, "[ERROR] Invalid parameters: root=%p, filename=%s\n", 
                (void*)root, filename ? filename : "NULL");
        return;
    }
    
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "[ERROR] Cannot create file: %s (errno=%d: %s)\n", 
                filename, errno, strerror(errno));
        return;
    }
    printf("[DEBUG] File opened successfully\n");
    
    save_tree_node(root, fp, 0);
    
    printf("[DEBUG] Flushing file buffer...\n");
    if (fflush(fp) != 0) {
        fprintf(stderr, "[ERROR] Failed to flush file buffer (errno=%d: %s)\n", 
                errno, strerror(errno));
    }
    
    printf("[DEBUG] Closing file...\n");
    if (fclose(fp) != 0) {
        fprintf(stderr, "[ERROR] Failed to close file (errno=%d: %s)\n", 
                errno, strerror(errno));
    }
    
    FILE* test = fopen(filename, "rb");
    if (test) {
        fclose(test);
        printf("[DEBUG] Model saved successfully, file can be opened\n");
    } else {
        fprintf(stderr, "[ERROR] Model save failed, cannot open file (errno=%d: %s)\n", 
                errno, strerror(errno));
    }
}

Node* load_tree_node(FILE* fp, int idx) {
    if (!fp) return NULL;
    
    int is_null;
    if (fread(&is_null, sizeof(int), 1, fp) != 1) return NULL;
    if (is_null) return NULL;
    
    Node* node = (Node*)malloc(sizeof(Node));
    if (!node) return NULL;
    
    fread(&node->is_leaf, sizeof(int), 1, fp);
    fread(&node->feature_index, sizeof(int), 1, fp);
    fread(&node->threshold, sizeof(double), 1, fp);
    fread(&node->predicted_class, sizeof(int), 1, fp);
    printf("  [DEBUG] Loading node (idx: %d): is_leaf=%d feature_index=%d, threshold=%.6f, predicted_class=%d\n", idx,
           node->is_leaf, node->feature_index, 
           node->threshold, node->predicted_class);
    fflush(stdout);
    
    node->left = load_tree_node(fp, idx << 1);
    node->right = load_tree_node(fp, idx << 1 | 1);
    
    return node;
}

Node* load_decision_tree(const char* filename) {
    if (!filename) return NULL;
    
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        return NULL;
    }
    
    printf("\nLoading decision tree...\n");
    Node* root = load_tree_node(fp, 1);
    fclose(fp);
    
    if (root) {
        printf("Model loaded successfully\n");
    } else {
        fprintf(stderr, "Model loading failed\n");
    }
    
    return root;
}

void load_seg_tree_node(FILE* fp, Node* seg_tree, int idx) {
    if (!fp) return;
    
    int is_null;
    if (fread(&is_null, sizeof(int), 1, fp) != 1) return;
    if (is_null) return;
    
    fread(&seg_tree[idx].is_leaf, sizeof(int), 1, fp);
    fread(&seg_tree[idx].feature_index, sizeof(int), 1, fp);
    fread(&seg_tree[idx].threshold, sizeof(double), 1, fp);
    fread(&seg_tree[idx].predicted_class, sizeof(int), 1, fp);
    printf("  [DEBUG] Loading node (idx: %d): is_leaf=%d feature_index=%d, threshold=%.6f, predicted_class=%d\n", idx,
        seg_tree[idx].is_leaf, seg_tree[idx].feature_index, 
        seg_tree[idx].threshold, seg_tree[idx].predicted_class);
    fflush(stdout);
    
    load_seg_tree_node(fp, seg_tree, idx << 1);
    load_seg_tree_node(fp, seg_tree, idx << 1 | 1);
}

void load_decision_seg_tree(const char* filename, Node* segTree) {
    if (!filename) return;
    
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        return;
    }
    load_seg_tree_node(fp, segTree, 1);
    fclose(fp);
}


int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <data_dir>\n", argv[0]);
        return 1;
    }
    
    Node* tree = NULL;
    Node segTree[6];
    char choice;
    
    printf("Load existing model? (y/n): ");
    scanf(" %c", &choice);
    
    SplitDataset* split = NULL;
    if (choice == 'y' || choice == 'Y') {
        const char *model_path = "decision_tree_model.bin";  
        tree = load_decision_tree(model_path);
        load_decision_seg_tree(model_path, segTree);
        if (!tree) {
            fprintf(stderr, "Model loading failed\n");
            return 1;
        }
        split = load_and_split_dataset(argv[1]);
    } else {
        split = load_and_split_dataset(argv[1]);
        if (!split || !split->train || !split->test) {
            fprintf(stderr, "Failed to load dataset\n");
            return 1;
        }
        
        tree = train_decision_tree(split->train);
        
        printf("Save model? (y/n): ");
        scanf(" %c", &choice);
        if (choice == 'y' || choice == 'Y') {
            const char* save_path = "decision_tree_model.bin";
            printf("\nSaving model to: %s\n", save_path);
            save_decision_tree(tree, save_path);
        }
    }
    
    printf("\n========== Model Evaluation ==========\n");
    int correct = 0;
    int total = split->train->n_samples + split->test->n_samples;
    int confusion_matrix[7][7] = {0};
    
    printf("\nEvaluating training set...\n");
    for (int i = 0; i < split->train->n_samples; i++) {
        int predicted = predict_sample(tree, split->train->features[i]);
        int check_segTree_predicted = seg_tree_predict_sample(segTree, split->train->features[i]);
        if (predicted != check_segTree_predicted) {
            printf("Training set: Prediction mismatch: %d != %d\n", predicted, check_segTree_predicted);
            return 0;
        }
        int actual = split->train->labels[i];
        
        if (predicted == actual) {
            correct++;
        }
        confusion_matrix[actual][predicted]++;
        
        if ((i + 1) % 100 == 0) {
            printf("\rTraining set progress: %d/%d (%.1f%%)", 
                   i + 1, split->train->n_samples, 
                   (i + 1) * 100.0 / split->train->n_samples);
            fflush(stdout);
        }
    }
    
    printf("\n\nEvaluating test set...\n");
    for (int i = 0; i < split->test->n_samples; i++) {
        int predicted = predict_sample(tree, split->test->features[i]);
        int check_segTree_predicted = seg_tree_predict_sample(segTree, split->test->features[i]);
        if (predicted != check_segTree_predicted) {
            printf("Test set[%d]: Prediction mismatch: %d != %d\n", i, predicted, check_segTree_predicted);
            return 0;
        }
        int actual = split->test->labels[i];
        
        if (predicted == actual) {
            correct++;
        }
        confusion_matrix[actual][predicted]++;
        
        if ((i + 1) % 100 == 0) {
            printf("\rTest set progress: %d/%d (%.1f%%)", 
                   i + 1, split->test->n_samples, 
                   (i + 1) * 100.0 / split->test->n_samples);
            fflush(stdout);
        }
    }
    printf("\n\n");
    
    double accuracy = (double)correct / total * 100;
    printf("Total samples: %d\n", total);
    printf("Training set samples: %d\n", split->train->n_samples);
    printf("Test set samples: %d\n", split->test->n_samples);
    printf("Correct predictions: %d\n", correct);
    printf("Overall accuracy: %.2f%%\n\n", accuracy);
    
    const char* formats[] = {"COO", "CSR", "DCL", "DRW", "DNS", "ELL", "HYB"};
    printf("Performance by class:\n");
    for (int i = 0; i < 7; i++) {
        int true_pos = confusion_matrix[i][i];
        int total_actual = 0;
        int total_predicted = 0;
        
        for (int j = 0; j < 7; j++) {
            total_actual += confusion_matrix[i][j];
            total_predicted += confusion_matrix[j][i];
        }
        
        if (total_actual > 0) {
            double precision = total_predicted > 0 ? 
                (double)true_pos / total_predicted * 100 : 0;
            double recall = (double)true_pos / total_actual * 100;
            double f1_score = (precision + recall > 0) ? 
                2 * precision * recall / (precision + recall) : 0;
            
            printf("%s:\n", formats[i]);
            printf("  Samples: %d\n", total_actual);
            printf("  Precision: %.2f%%\n", precision);
            printf("  Recall: %.2f%%\n", recall);
            printf("  F1 Score: %.2f%%\n", f1_score);
        }
    }
    
    free_tree(tree);
    for (int i = 0; i < split->train->n_samples; i++) {
        free(split->train->features[i]);
    }
    for (int i = 0; i < split->test->n_samples; i++) {
        free(split->test->features[i]);
    }
    free(split->train->features);
    free(split->train->labels);
    free(split->train);
    free(split->test->features);
    free(split->test->labels);
    free(split->test);
    free(split);
    
    return 0;
}