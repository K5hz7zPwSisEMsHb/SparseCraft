#include <tmatrix/DataStructure/predict.cuh>

#define rnd() ((double)rand() / RAND_MAX)
#define nrnd() ((rnd()+rnd()+rnd()+rnd()-2.0) * 1.724)
#define tanh_g(y) (1.0 - (y)*(y))
#define relu(x) ((0 < (x))? (x) : 0)
#define relu_g(y) ((0 < (y))? 1 : 0)

typedef enum {
    LAYER_INPUT,
    LAYER_FULL,
    LAYER_CONV,
    LAYER_POOL  // 新增
} LayerType;

void double2float_cpy(float *dst, double *src, int size)
{
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        dst[i] = src[i];
    }
}

__host__ __device__ __forceinline__ float bitmap_get(const uint64_t *bitmap, uint8_t idx)
{
    int row = idx >> 4;
    int col = idx & 15;
    int row8 = row >> 3;
    row &= 7;
    int col8 = col >> 3;
    col &= 7;
    return bitmap[(row8 << 1) | col8] >> ((row << 3) | col) & 1;
}

struct IO_sequense_slime_net
{
    float conv_0_outputs[8192], conv_1_outputs[4096], conv_2_outputs[2048];
    float pool_0_outputs[2048], pool_1_outputs[1024], pool_2_outputs[512];
    float full_0_outputs[256], full_1_outputs[128], full_2_outputs[7];
};

struct IO_sequense_slime_net_buffer
{
    half buffer0[8192], buffer1[2048];
};

struct sequence_slime_net
{
    float conv_0_biases[32], conv_0_weights[288], conv_0_gradients[8192], conv_0_errors[8192], conv_0_u_biases[32], conv_0_u_weights[288];
    float conv_1_biases[64], conv_1_weights[18432], conv_1_gradients[4096], conv_1_errors[4096], conv_1_u_biases[64], conv_1_u_weights[18432];
    float conv_2_biases[128], conv_2_weights[73728], conv_2_gradients[2048], conv_2_errors[2048], conv_2_u_biases[128], conv_2_u_weights[73728];

    float pool_0_gradients[2048], pool_0_errors[2048];
    float pool_1_gradients[1024], pool_1_errors[1024];
    float pool_2_gradients[512], pool_2_errors[512];

    float full_0_biases[256], full_0_weights[131072], full_0_gradients[256], full_0_errors[256], full_0_u_biases[256], full_0_u_weights[131072];
    float full_1_biases[128], full_1_weights[32768], full_1_gradients[128], full_1_errors[128], full_1_u_biases[128], full_1_u_weights[32768];
    float full_2_biases[7], full_2_weights[896], full_2_gradients[7], full_2_errors[7], full_2_u_biases[7], full_2_u_weights[896];

    const int conv0_depth = 32, conv1_depth = 64, conv2_depth = 128;
    const int conv0_width = 16, conv1_width = 8, conv2_width = 4;
    const int conv0_height = 16, conv1_height = 8, conv2_height = 4;
    const int full0_nnodes = 256, full1_nnodes = 128, full2_nnodes = 7;
    const int full0_nweights = 131072, full1_nweights = 32768, full2_nweights = 896;
    const int full0_nbiases = 256, full1_nbiases = 128, full2_nbiases = 7;
    const int pool_0_depth = 32, pool_1_depth = 64, pool_2_depth = 128;
    const int pool_0_width = 8, pool_1_width = 4, pool_2_width = 2;
    const int pool_0_height = 8, pool_1_height = 4, pool_2_height = 2;

    const int conv0_kernsize = 3, conv0_padding = 1, conv0_stride = 1;
    const int conv1_kernsize = 3, conv1_padding = 1, conv1_stride = 1;
    const int conv2_kernsize = 3, conv2_padding = 1, conv2_stride = 1;
    const int conv0_nweights = 288, conv1_nweights = 18432, conv2_nweights = 73728;
    const int conv0_nbiases = 32, conv1_nbiases = 64, conv2_nbiases = 128;
    const int pool_0_size = 2, pool_1_size = 2, pool_2_size = 2;
};

void load_model_layer_head(FILE* fp)
{
    LayerType type;
    int ignore[3];
    fread(&type, sizeof(LayerType), 1, fp);
    fread(ignore, sizeof(int), 3, fp);
}

void load_model_conv(sequence_slime_net*network, int k, FILE* fp)
{
    int ignore[3];
    load_model_layer_head(fp);
    int nweights = k == 0 ? network->conv0_nweights : (k == 1 ? network->conv1_nweights : network->conv2_nweights);
    int nbiases = k == 0 ? network->conv0_nbiases : (k == 1 ? network->conv1_nbiases : network->conv2_nbiases);
    double*weight = (double*)malloc(nweights * sizeof(double));
    double*bias = (double*)malloc(nbiases * sizeof(double));
    fread(weight, sizeof(double), nweights, fp);
    fread(bias, sizeof(double), nbiases, fp);
    float*n_weight = k == 0 ? network->conv_0_weights : (k == 1 ? network->conv_1_weights : network->conv_2_weights);
    float*n_bias = k == 0 ? network->conv_0_biases : (k == 1 ? network->conv_1_biases : network->conv_2_biases);
    double2float_cpy(n_weight, weight, nweights);
    double2float_cpy(n_bias, bias, nbiases);
    fread(ignore, sizeof(int), 3, fp);
    free(weight);
    free(bias);
}

void load_model_pool(sequence_slime_net*network, int k, FILE*fp)
{
    load_model_layer_head(fp);
    fread(&k, sizeof(int), 1, fp);
}

void load_model_full(sequence_slime_net*network, int k, FILE*fp)
{
    load_model_layer_head(fp);
    int nweights = k == 0 ? network->full0_nweights : (k == 1 ? network->full1_nweights : network->full2_nweights);
    int nbiases = k == 0 ? network->full0_nbiases : (k == 1 ? network->full1_nbiases : network->full2_nbiases);
    double*weight = (double*)malloc(nweights * sizeof(double));
    double*bias = (double*)malloc(nbiases * sizeof(double));
    fread(weight, sizeof(double), nweights, fp);
    fread(bias, sizeof(double), nbiases, fp);
    float*n_weight = k == 0 ? network->full_0_weights : (k == 1 ? network->full_1_weights : network->full_2_weights);
    float*n_bias = k == 0 ? network->full_0_biases : (k == 1 ? network->full_1_biases : network->full_2_biases);
    double2float_cpy(n_weight, weight, nweights);
    double2float_cpy(n_bias, bias, nbiases);
    free(weight);
    free(bias);
}

sequence_slime_net* load_slimenet(const char *model_path)
{
    FILE *fp = fopen(model_path, "rb");
    if (!fp)
    {
        fprintf(stderr, "无法打开模型文件: %s\n", model_path);
        return 0;
    }

    sequence_slime_net *network = new sequence_slime_net();
    load_model_conv(network, 0, fp);
    load_model_pool(network, 0, fp);
    load_model_conv(network, 1, fp);
    load_model_pool(network, 1, fp);
    load_model_conv(network, 2, fp);
    load_model_pool(network, 2, fp);
    load_model_full(network, 0, fp);
    load_model_full(network, 1, fp);
    load_model_full(network, 2, fp);
    fclose(fp);
    return network;
}

void seq_slime_layer_feedForw_conv(sequence_slime_net *network, IO_sequense_slime_net *sout, float *input, int k)
{
    int prev_depth = k == 0 ? 1 : (k == 1 ? network->pool_0_depth : network->pool_1_depth);
    int prev_width = k == 0 ? 16 : (k == 1 ? network->pool_0_width : network->pool_1_width);
    int prev_height = k == 0 ? 16 : (k == 1 ? network->pool_0_height : network->pool_1_height);

    int kernsize = k == 0 ? network->conv0_kernsize : (k == 1 ? network->conv1_kernsize : network->conv2_kernsize);
    int padding = k == 0 ? network->conv0_padding : (k == 1 ? network->conv1_padding : network->conv2_padding);
    int stride = k == 0 ? network->conv0_stride : (k == 1 ? network->conv1_stride : network->conv2_stride);

    int depth = k == 0 ? network->conv0_depth : (k == 1 ? network->conv1_depth : network->conv2_depth);
    int width = k == 0 ? network->conv0_width : (k == 1 ? network->conv1_width : network->conv2_width);
    int height = k == 0 ? network->conv0_height : (k == 1 ? network->conv1_height : network->conv2_height);

    float *outputs = k == 0 ? sout->conv_0_outputs : (k == 1 ? sout->conv_1_outputs : sout->conv_2_outputs);
    float *gradients = k == 0 ? network->conv_0_gradients : (k == 1 ? network->conv_1_gradients : network->conv_2_gradients);
    float *biases = k == 0 ? network->conv_0_biases : (k == 1 ? network->conv_1_biases : network->conv_2_biases);
    float *weights = k == 0 ? network->conv_0_weights : (k == 1 ? network->conv_1_weights : network->conv_2_weights);
    float *prev_outputs = k == 0 ? input : (k == 1 ? sout->pool_0_outputs : sout->pool_1_outputs);

    for (int z1 = 0; z1 < depth; z1++)
    {
        /* z1: dst matrix */
        /* qbase: kernel matrix base index */
        int qbase = z1 * prev_depth * kernsize * kernsize;
        for (int y1 = 0; y1 < height; y1++)
        {
            int y0 = stride * y1 - padding;
            for (int x1 = 0; x1 < width; x1++)
            {
                int x0 = stride * x1 - padding;
                /* Compute the kernel at (x1,y1) */
                /* (x0,y0): src pixel */
                double v = biases[z1];
                for (int z0 = 0; z0 < prev_depth; z0++)
                {
                    /* z0: src matrix */
                    /* pbase: src matrix base index */
                    int pbase = z0 * prev_width * prev_height;
                    for (int dy = 0; dy < kernsize; dy++)
                    {
                        int y = y0 + dy;
                        if (0 <= y && y < prev_height)
                        {
                            int p = pbase + y * prev_width;
                            int q = qbase + dy * kernsize;
                            for (int dx = 0; dx < kernsize; dx++)
                            {
                                int x = x0 + dx;
                                if (0 <= x && x < prev_width)
                                {
                                    v += prev_outputs[p + x] * weights[q + dx];
                                }
                            }
                        }
                    }
                }
                /* Apply the activation function. */
                v = relu(v);
                outputs[z1 * width * height + y1 * width + x1] = v;
                gradients[z1 * width * height + y1 * width + x1] = relu_g(v);
            }
        }
    }
}

void seq_slime_layer_feedForw_pool(sequence_slime_net *network, IO_sequense_slime_net *sout, int k)
{
    int pool_size = k == 0 ? network->pool_0_size : (k == 1 ? network->pool_1_size : network->pool_2_size);
    int depth = k == 0 ? network->pool_0_depth : (k == 1 ? network->pool_1_depth : network->pool_2_depth);
    int width = k == 0 ? network->pool_0_width : (k == 1 ? network->pool_1_width : network->pool_2_width);
    int height = k == 0 ? network->pool_0_height : (k == 1 ? network->pool_1_height : network->pool_2_height);
    float *outputs = k == 0 ? sout->pool_0_outputs : (k == 1 ? sout->pool_1_outputs : sout->pool_2_outputs);
    float *gradients = k == 0 ? network->pool_0_gradients : (k == 1 ? network->pool_1_gradients : network->pool_2_gradients);

    int prev_width = k == 0 ? network->conv0_width : (k == 1 ? network->conv1_width : network->conv2_width);
    int prev_height = k == 0 ? network->conv0_height : (k == 1 ? network->conv1_height : network->conv2_height);
    float *prev_outputs = k == 0 ? sout->conv_0_outputs : (k == 1 ? sout->conv_1_outputs : sout->conv_2_outputs);

    for (int z = 0; z < depth; z++)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double maxval = -1e9;
                int py = y * pool_size;
                int px = x * pool_size;

                // 在池化窗口中找最大值
                for (int dy = 0; dy < pool_size; dy++)
                {
                    for (int dx = 0; dx < pool_size; dx++)
                    {
                        int prev_idx = z * (prev_width * prev_height) +
                                       (py + dy) * prev_width + (px + dx);
                        if (prev_outputs[prev_idx] > maxval)
                        {
                            maxval = prev_outputs[prev_idx];
                        }
                    }
                }

                outputs[z * width * height + y * width + x] = maxval;
                gradients[z * width * height + y * width + x] = 1.0;
            }
        }
    }
}

void seq_slime_layer_feedForw_full(sequence_slime_net *network, IO_sequense_slime_net *sout, int k)
{
    int nnodes = k == 0 ? network->full0_nnodes : (k == 1 ? network->full1_nnodes : network->full2_nnodes);
    float *outputs = k == 0 ? sout->full_0_outputs : (k == 1 ? sout->full_1_outputs : sout->full_2_outputs);
    float *biases = k == 0 ? network->full_0_biases : (k == 1 ? network->full_1_biases : network->full_2_biases);
    float *weights = k == 0 ? network->full_0_weights : (k == 1 ? network->full_1_weights : network->full_2_weights);
    float *gradients = k == 0 ? network->full_0_gradients : (k == 1 ? network->full_1_gradients : network->full_2_gradients);

    int prev_nnodes = k == 0 ? network->pool_2_depth * network->pool_2_width * network->pool_2_height : (k == 1 ? network->full0_nnodes : network->full1_nnodes);
    float *prev_outputs = k == 0 ? sout->pool_2_outputs : (k == 1 ? sout->full_0_outputs : sout->full_1_outputs);

    for (int i = 0; i < nnodes; i++)
    {
        /* Compute Y = (W * X + B) without activation function. */
        double x = biases[i];
        for (int j = 0; j < prev_nnodes; j++)
        {
            x += (prev_outputs[j] * weights[i * prev_nnodes + j]);
        }
        outputs[i] = x;
    }
    if (k == 2)
    {
        double m = -1;
        for (int i = 0; i < nnodes; i++)
        {
            double x = outputs[i];
            if (m < x)
            {
                m = x;
            }
        }
        // printf("[CPU] m = %f\n", m);
        double t = 0;
        for (int i = 0; i < nnodes; i++)
        {
            double y = exp(outputs[i] - m);
            outputs[i] = y;
            t += y;
        }
        for (int i = 0; i < nnodes; i++)
        {
            outputs[i] /= t;
            /* This isn't right, but set the same value to all the gradients. */
            gradients[i] = 1;
        }
    }
    else
    {
        for (int i = 0; i < nnodes; i++)
        {
            double x = outputs[i];
            double y = tanh(x);
            outputs[i] = y;
            gradients[i] = tanh_g(y);
        }
    }
}

uint8_t run_inference(sequence_slime_net *network, uint64_t *bitmap)
{
    float input[256];
    IO_sequense_slime_net soutput;
    #pragma omp parallel for
    for (int i = 0; i < 256; ++i) 
    {
        input[i] = bitmap_get(bitmap, i);
    }
    seq_slime_layer_feedForw_conv(network, &soutput, input, 0);
    seq_slime_layer_feedForw_pool(network, &soutput, 0);
    seq_slime_layer_feedForw_conv(network, &soutput, NULL, 1);
    seq_slime_layer_feedForw_pool(network, &soutput, 1);
    seq_slime_layer_feedForw_conv(network, &soutput, NULL, 2);
    seq_slime_layer_feedForw_pool(network, &soutput, 2);
    seq_slime_layer_feedForw_full(network, &soutput, 0);
    seq_slime_layer_feedForw_full(network, &soutput, 1);
    seq_slime_layer_feedForw_full(network, &soutput, 2);
    uint8_t max_class = 0;
    float max_prob = soutput.full_2_outputs[0];
    for (int i = 1; i < 7; i++)
    {
        if (soutput.full_2_outputs[i] > max_prob)
        {
            max_prob = soutput.full_2_outputs[i];
            max_class = i;
        }
    }
    return max_class;
}

__device__ void cuda_seq_slime_layer_feedForw_conv(sequence_slime_net *network, IO_sequense_slime_net_buffer *sout, uint64_t *input, int k, int lane_id)
{
    int prev_depth = k == 0 ? 1 : (k == 1 ? network->pool_0_depth : network->pool_1_depth);
    int prev_width = k == 0 ? 16 : (k == 1 ? network->pool_0_width : network->pool_1_width);
    int prev_height = k == 0 ? 16 : (k == 1 ? network->pool_0_height : network->pool_1_height);

    int kernsize = k == 0 ? network->conv0_kernsize : (k == 1 ? network->conv1_kernsize : network->conv2_kernsize);
    int padding = k == 0 ? network->conv0_padding : (k == 1 ? network->conv1_padding : network->conv2_padding);
    int stride = k == 0 ? network->conv0_stride : (k == 1 ? network->conv1_stride : network->conv2_stride);

    int depth = k == 0 ? network->conv0_depth : (k == 1 ? network->conv1_depth : network->conv2_depth);
    int width = k == 0 ? network->conv0_width : (k == 1 ? network->conv1_width : network->conv2_width);
    int height = k == 0 ? network->conv0_height : (k == 1 ? network->conv1_height : network->conv2_height);

    half *outputs = k == 0 ? sout->buffer0 : (k == 1 ? sout->buffer0 : sout->buffer0);
    float *gradients = k == 0 ? network->conv_0_gradients : (k == 1 ? network->conv_1_gradients : network->conv_2_gradients);
    float *biases = k == 0 ? network->conv_0_biases : (k == 1 ? network->conv_1_biases : network->conv_2_biases);
    float *weights = k == 0 ? network->conv_0_weights : (k == 1 ? network->conv_1_weights : network->conv_2_weights);
    half *prev_outputs = k == 0? nullptr: (k == 1 ? sout->buffer1 : sout->buffer1);

    const int total_iterations = depth * height * width;
    for (int idx = lane_id; idx < total_iterations; idx += 32)
    {
        // 从一维索引还原三维坐标
        int z1 = idx / (height * width);        // 计算z1
        int remainder = idx % (height * width); // 计算余数
        int y1 = remainder / width;             // 计算y1
        int x1 = remainder % width;             // 计算x1

        // 原来的计算逻辑
        int qbase = z1 * prev_depth * kernsize * kernsize;
        int y0 = stride * y1 - padding;
        int x0 = stride * x1 - padding;

        float v = biases[z1];
        for (int z0 = 0; z0 < prev_depth; z0++)
        {
            int pbase = z0 * prev_width * prev_height;
            for (int dy = 0; dy < kernsize; dy++)
            {
                int y = y0 + dy;
                if (0 <= y && y < prev_height)
                {
                    int p = pbase + y * prev_width;
                    int q = qbase + dy * kernsize;
                    for (int dx = 0; dx < kernsize; dx++)
                    {
                        int x = x0 + dx;
                        if (0 <= x && x < prev_width)
                        {
                            v += (k? __half2float(prev_outputs[p + x]): bitmap_get(input, p + x)) * weights[q + dx];
                        }
                    }
                }
            }
        }

        // 计算输出索引
        int output_idx = z1 * width * height + y1 * width + x1;

        // 应用激活函数并存储结果
        v = relu(v);
        outputs[output_idx] = v;
        gradients[output_idx] = relu_g(v);
    }
}

__device__ void cuda_seq_slime_layer_feedForw_pool(sequence_slime_net *network, IO_sequense_slime_net_buffer *sout, int k, int lane_id)
{
    int pool_size = k == 0 ? network->pool_0_size : (k == 1 ? network->pool_1_size : network->pool_2_size);
    int depth = k == 0 ? network->pool_0_depth : (k == 1 ? network->pool_1_depth : network->pool_2_depth);
    int width = k == 0 ? network->pool_0_width : (k == 1 ? network->pool_1_width : network->pool_2_width);
    int height = k == 0 ? network->pool_0_height : (k == 1 ? network->pool_1_height : network->pool_2_height);
    half *outputs = sout->buffer1;
    float *gradients = k == 0 ? network->pool_0_gradients : (k == 1 ? network->pool_1_gradients : network->pool_2_gradients);

    int prev_width = k == 0 ? network->conv0_width : (k == 1 ? network->conv1_width : network->conv2_width);
    int prev_height = k == 0 ? network->conv0_height : (k == 1 ? network->conv1_height : network->conv2_height);
    half *prev_outputs = sout->buffer0;

    const int total_iterations = depth * height * width;

    for (int idx = lane_id; idx < total_iterations; idx += 32) {
        // 从一维索引还原三维坐标
        int z = idx / (height * width);        // 计算z
        int remainder = idx % (height * width); // 计算余数
        int y = remainder / width;             // 计算y
        int x = remainder % width;             // 计算x
        float maxval = -1e3;
        int py = y * pool_size;
        int px = x * pool_size;

        // 在池化窗口中找最大值
        for (int dy = 0; dy < pool_size; dy++)
        {
            for (int dx = 0; dx < pool_size; dx++)
            {
                int prev_idx = z * (prev_width * prev_height) + (py + dy) * prev_width + (px + dx);
                // if (prev_outputs[prev_idx] > maxval)
                if (__hgt(prev_outputs[prev_idx], __float2half(maxval)))
                {
                    // maxval = prev_outputs[prev_idx];
                    maxval = __half2float(prev_outputs[prev_idx]);
                }
            }
        }

        outputs[z * width * height + y * width + x] = maxval;
        gradients[z * width * height + y * width + x] = 1.0;
    }
}

__device__ void cuda_seq_slime_layer_feedForw_full(sequence_slime_net *network, IO_sequense_slime_net_buffer *sout, int k, int lane_id)
{
    int nnodes = k == 0 ? network->full0_nnodes : (k == 1 ? network->full1_nnodes : network->full2_nnodes);
    int nweights = k == 0 ? network->full0_nweights : (k == 1 ? network->full1_nweights : network->full2_nweights);
    int nbiases = k == 0 ? network->full0_nbiases : (k == 1 ? network->full1_nbiases : network->full2_nbiases);
    half *outputs = k == 0 ? sout->buffer0 : (k == 1 ? sout->buffer1 : sout->buffer0);
    float *biases = k == 0 ? network->full_0_biases : (k == 1 ? network->full_1_biases : network->full_2_biases);
    float *weights = k == 0 ? network->full_0_weights : (k == 1 ? network->full_1_weights : network->full_2_weights);
    float *gradients = k == 0 ? network->full_0_gradients : (k == 1 ? network->full_1_gradients : network->full_2_gradients);

    int prev_nnodes = k == 0 ? network->pool_2_depth * network->pool_2_width * network->pool_2_height : (k == 1 ? network->full0_nnodes : network->full1_nnodes);
    half *prev_outputs = k == 0 ? sout->buffer1 : (k == 1 ? sout->buffer0 : sout->buffer1);

    // int kk = 0;
    for (int i = lane_id; i < nnodes; i += 32)
    {
        /* Compute Y = (W * X + B) without activation function. */
        float x = biases[i];
        for (int j = 0; j < prev_nnodes; j++)
        {
            x += (__half2float(prev_outputs[j]) * weights[i * prev_nnodes + j]);
        }
        outputs[i] = x;
    }
    if (k == 2)
    {
        float m = -1;
        for (int i = lane_id; i < nnodes; i += 32)
        {
            float x = outputs[i];
            if (m < x)
            {
                m = x;
            }
        }
        // Warp内最大值归约
        for (int offset = 16; offset > 0; offset /= 2)
        {
            m = fmaxf(m, __shfl_down_sync(0xFFFFFFFF, m, offset));
        }
        m = __shfl_sync(0xFFFFFFFF, m, 0);
        float t = 0;
        for (int i = lane_id; i < nnodes; i += 32)
        {
            float y = exp(__half2float(outputs[i]) - m);
            outputs[i] = y;
            t += y;
        }
        // Warp内归约求和
        for (int offset = 16; offset > 0; offset /= 2)
        {
            t += __shfl_down_sync(0xFFFFFFFF, t, offset);
        }
        t = __shfl_sync(0xFFFFFFFF, t, 0);
        for (int i = lane_id; i < nnodes; i += 32)
        {
            outputs[i] /= t;
            gradients[i] = 1;
        }
    }
    else
    {
        for (int i = lane_id; i < nnodes; i += 32)
        {
            float x = outputs[i];
            float y = tanh(x);
            outputs[i] = y;
            gradients[i] = tanh_g(y);
        }
    }
}

__device__ uint8_t cuda_warp_run_inference(sequence_slime_net *network, IO_sequense_slime_net_buffer *sio, uint64_t *input, int lane_id)
{
    cuda_seq_slime_layer_feedForw_conv(network, sio, input, 0, lane_id);
    cuda_seq_slime_layer_feedForw_pool(network, sio, 0, lane_id);
    cuda_seq_slime_layer_feedForw_conv(network, sio, NULL, 1, lane_id);
    cuda_seq_slime_layer_feedForw_pool(network, sio, 1, lane_id);
    cuda_seq_slime_layer_feedForw_conv(network, sio, NULL, 2, lane_id);
    cuda_seq_slime_layer_feedForw_pool(network, sio, 2, lane_id);
    cuda_seq_slime_layer_feedForw_full(network, sio, 0, lane_id);
    cuda_seq_slime_layer_feedForw_full(network, sio, 1, lane_id);
    cuda_seq_slime_layer_feedForw_full(network, sio, 2, lane_id);
    uint8_t max_class = 0;
    if (lane_id == 0) {
        half max_prob = sio->buffer0[0];
        for (int i = 1; i < 7; i++)
        {
            if (__hgt(sio->buffer0[i], max_prob))
            {
                max_prob = sio->buffer0[i];
                max_class = i;
            }
        }
    }
    return max_class;
}

sequence_slime_net *slimenet_to_device(sequence_slime_net *model)
{
    sequence_slime_net *device_model;
    cudaMalloc(&device_model, sizeof(sequence_slime_net));
    cudaMemcpy(device_model, model, sizeof(sequence_slime_net), cudaMemcpyHostToDevice);
    return device_model;
}

__global__ void pixel_format_prediction_single(sequence_slime_net *model, Tile*tiles, int n)
{
    int gwarp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (gwarp_id >= n)
        return;
    __shared__ uint64_t s_bitmap[8];
    __shared__ IO_sequense_slime_net_buffer sios[2];
    int lane_id = threadIdx.x & 31, warp_id = threadIdx.x >> 5;
    if (lane_id < 4) s_bitmap[warp_id * 4 + lane_id] = tiles[gwarp_id].bitmap[lane_id];
    __syncwarp();
    uint8_t res = cuda_warp_run_inference(model, sios + warp_id, s_bitmap + warp_id * 4, lane_id);
    if (lane_id == 0) tiles[gwarp_id].fmt = res;
}

struct oaktree{
    int is_leaf;
    int feature_index;
    double threshold;
    int predicted_class;
};

void load_seg_tree_node(FILE* fp, oaktree*seg_tree, int idx) {
    if (!fp) return;
    // 读取节点是否为空的标志
    int is_null;
    if (fread(&is_null, sizeof(int), 1, fp) != 1) return;
    if (is_null) return;
    
    // 读取节点数据
    fread(&seg_tree[idx].is_leaf, sizeof(int), 1, fp);
    fread(&seg_tree[idx].feature_index, sizeof(int), 1, fp);
    fread(&seg_tree[idx].threshold, sizeof(double), 1, fp);
    fread(&seg_tree[idx].predicted_class, sizeof(int), 1, fp);
    
    // 递归加载子节点
    load_seg_tree_node(fp, seg_tree, idx << 1);
    load_seg_tree_node(fp, seg_tree, idx << 1 | 1);
}

oaktree* load_oaktree(const char *model_path)
{
    oaktree* model = new oaktree[14];
    FILE *fp = fopen(model_path, "rb");
    if (!fp)
    {
        fprintf(stderr, "无法打开模型文件: %s\n", model_path);
        return 0;
    }
    load_seg_tree_node(fp, model, 1);
    fclose(fp);
    return model;
}

oaktree* oaktree_to_device(oaktree *model)
{
    oaktree *device_model;
    cudaMalloc(&device_model, sizeof(oaktree) * 14);
    cudaMemcpy(device_model, model, sizeof(oaktree) * 14, cudaMemcpyHostToDevice);
    return device_model;
}

__device__ __forceinline__ bool _bitmap_check(const uint64_t *bitmap, TileIndex row, TileIndex col)
{
    int row8 = row >> 3;
    row &= 7;
    int col8 = col >> 3;
    col &= 7;
    return bitmap[(row8 << 1) | col8] >> ((row << 3) | col) & 1;
}

__device__ bool _bitmap_check(const uint64_t *bitmap, TileIndex idx)
{
    int row = idx >> 4;
    int col = idx & 15;
    return _bitmap_check(bitmap, row, col);
}

__host__ __device__ uint16_t _bitmap_get_row(const uint64_t *bitmap, TileIndex row)
{
    int row8 = row >> 3;
    int _row = row & 7;
    uint64_t row_upper = (bitmap[row8 << 1] >> (_row << 3)) & 0xff;
    uint64_t row_lower = (bitmap[(row8 << 1) + 1] >> (_row << 3)) & 0xff;
    return (row_lower << 8) | row_upper;
}

__host__ __device__ uint16_t _bitmap_get_col(const uint64_t *bitmap, TileIndex col)
{
    int col8 = col >> 3;
    col &= 7;
    uint64_t col_upper = (bitmap[col8] >> col) & 0x0101010101010101;
    uint64_t col_lower = (bitmap[col8 + 2] >> col) & 0x0101010101010101;
    uint16_t res = 0;
    for (int i = 0; i < 8; i++)
    {
        uint16_t bits = (col_upper & 1) | ((col_lower & 1) << 8);
        res |= bits << i;
        col_upper >>= 8;
        col_lower >>= 8;
    }
    return res;
}

__device__ float calculate_std(float* values, int n) {
    if (n <= 1) return 0.0;
    float mean = 0.0, std = 0.0;
    
    // 计算平均值
    for (int i = 0; i < n; i++) {
        mean += values[i];
    }
    mean /= n;
    
    // 计算标准差
    for (int i = 0; i < n; i++) {
        float diff = values[i] - mean;
        std += diff * diff;
    }
    std = sqrt(std / (n - 1));
    return std;
}

__device__ __forceinline__ void extract_features(uint64_t*bitmap, float* features) {
    // 1. 行数
    features[0] = TILE_N;
    
    // 2. 非零元素的数量
    int nnz = __popcll(bitmap[0]) + __popcll(bitmap[1]) + __popcll(bitmap[2]) + __popcll(bitmap[3]);
    features[1] = nnz;
    
    // 3. 非空行数
    int non_empty_rows = 0;
    int row_nnz[16] = {0};
    for (int i = 0; i < TILE_N; i++) {
        uint16_t row = _bitmap_get_row(bitmap, i);
        row_nnz[i] = __popc(row);
        non_empty_rows += row_nnz[i]? 1: 0;
    }
    features[2] = non_empty_rows;
    
    // // 4. DIA格式中对角线的数量
    // int dia_count = 0;
    // // for (int d = -TILE_N + 1; d < TILE_N; d++) {
    // //     int has_nonzero = 0;
    // //     for (int i = 0; i < TILE_N; i++) {
    // //         int j = i + d;
    // //         if (j >= 0 && j < TILE_N && matrix[i][j] != 0) {
    // //             has_nonzero = 1;
    // //             break;
    // //         }
    // //     }
    // //     dia_count += has_nonzero;
    // // }
    // features[3] = dia_count;
    
    // // 5. 平均每行非零元的数量
    // features[4] = (float)nnz / TILE_N;
    
    // // 6. 非零元的密度
    // features[5] = (float)nnz / (TILE_N * TILE_N);
    
    // // 7. 每行非零块的数量的标准差
    // float blocks_per_row[TILE_N]; 
    // int total_blocks = 0;
    // for (int i = 0; i < TILE_N; i++) {
    //     int blocks = (row_nnz[i] + 1) >> 1;
    //     blocks_per_row[i] = blocks;
    //     total_blocks += blocks;
    // }
    // features[6] = calculate_std(blocks_per_row, TILE_N);
    
    // // 8. 每行非零块的大小标准差
    // // double block_sizes[128];
    // // int block_idx = 0;
    // // for (int i = 0; i < TILE_N; i++) {
    // //     int size = 0;
    // //     for (int j = 0; j < TILE_N; j++) {
    // //         if (matrix[i][j] != 0) {
    // //             size++;
    // //         } else if (size > 0) {
    // //             block_sizes[block_idx++] = size;
    // //             size = 0;
    // //         }
    // //     }
    // //     if (size > 0) {
    // //         block_sizes[block_idx++] = size;
    // //     }
    // // }
    // // features[7] = calculate_std(block_sizes, total_blocks);
    // features[7] = 0;
    
    // // 9. 每行非零元数量的变异系数
    // float row_nnz_mean = (float)nnz / TILE_N;
    // float row_nnz_std = 0.0;
    // for (int i = 0; i < TILE_N; i++) {
    //     float diff = row_nnz[i] - row_nnz_mean;
    //     row_nnz_std += diff * diff;
    // }
    // row_nnz_std = sqrt(row_nnz_std / (TILE_N - 1));
    // // 添加防止除零检查
    // features[8] = (row_nnz_mean > 0) ? (row_nnz_std / row_nnz_mean) : 0.0;  // 变异系数
    
    // // 10. relative_range
    // int max_nnz = 0, min_nnz = TILE_N;
    // for (int i = 0; i < TILE_N; i++) {
    //     if (row_nnz[i] > max_nnz) max_nnz = row_nnz[i];
    //     if (row_nnz[i] < min_nnz && row_nnz[i] > 0) min_nnz = row_nnz[i];
    // }
    // // 添加防止除零检查
    // features[9] = (max_nnz > 0) ? ((float)(max_nnz - min_nnz) / max_nnz) : 0.0;
    
    // // 11. 每行非零元的最大值
    // features[10] = 1.0;  // 对于二值矩阵，最大值始终为1
    
    // // 12. 相邻行的非零元的差的平均差
    float avg_diff = 0.0;
    int diff_count = 0;
    for (int i = 0; i < TILE_N - 1; i++) {
        int diff = abs(row_nnz[i] - row_nnz[i + 1]);
        avg_diff += diff;
        diff_count++;
    }
    features[11] = diff_count > 0 ? avg_diff / diff_count : 0;
    
    // // 13. 非零块数
    // features[12] = total_blocks;
    
    // // 14. bandwidth index (BWI) 的标准差
    // // double* bwi = (double*)malloc(TILE_N * sizeof(double));
    // float bwi[TILE_N];
    // for (int i = 0; i < TILE_N; i++) {
    //     int min_col = TILE_N, max_col = -1;
    //     for (int j = 0; j < TILE_N; j++) {
    //         if (_bitmap_check(bitmap, i, j)) {
    //             if (j < min_col) min_col = j;
    //             if (j > max_col) max_col = j;
    //         }
    //     }
    //     bwi[i] = (max_col >= min_col) ? (max_col - min_col + 1) : 0;
    // }
    // features[13] = calculate_std(bwi, TILE_N);
    
    // // 15. dispersion的平均值
    // float total_dispersion = 0.0;
    // int dispersion_count = 0;
    // for (int i = 0; i < TILE_N; i++) {
    //     if (row_nnz[i] > 0) {
    //         total_dispersion += (float)row_nnz[i] / bwi[i];
    //         dispersion_count++;
    //     }
    // }
    // features[14] = dispersion_count > 0 ? total_dispersion / dispersion_count : 0;
    
    // // 16. clustering
    // float clustering = 0.0;
    // for (int i = 0; i < TILE_N; i++) {
    //     for (int j = 0; j < TILE_N; j++) {
    //         if (_bitmap_check(bitmap, i, j)) {
    //             int neighbors = 0;
    //             if (i > 0 && _bitmap_check(bitmap, i - 1, j) != 0) neighbors++;
    //             if (i < TILE_N-1 && _bitmap_check(bitmap, i + 1, j) != 0) neighbors++;
    //             if (j > 0 && _bitmap_check(bitmap, i, j - 1) != 0) neighbors++;
    //             if (j < TILE_N-1 && _bitmap_check(bitmap, i, j + 1) != 0) neighbors++;
    //             clustering += neighbors;
    //         }
    //     }
    // }
    // features[15] = nnz > 0 ? clustering / (4 * nnz) : 0;  // 归一化
}

__global__ void oaktree_prediction_single(oaktree *model, Tile*tiles, int n)
{
    int gthread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (gthread_id >= n)
        return;
    int cursor = 1;
    float features[16] = {0};
    extract_features(tiles[gthread_id].bitmap, features);

    while (!model[cursor].is_leaf) {
        if (features[model[cursor].feature_index] <= model[cursor].threshold) {
            cursor = cursor << 1;  // 左子节点
        } else {
            cursor = cursor << 1 | 1;  // 右子节点
        }
    }
    tiles[gthread_id].fmt = model[cursor].predicted_class;
}