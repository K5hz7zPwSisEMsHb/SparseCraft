#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cnn.h"

#define DEBUG_LAYER 0

static inline double rnd()
{
    return ((double)rand() / RAND_MAX);
}

static inline double nrnd()
{
    return (rnd()+rnd()+rnd()+rnd()-2.0) * 1.724;
}

#if 0
static inline double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

static inline double sigmoid_g(double y)
{
    return y * (1.0 - y);
}
#endif

#if 0
static inline double tanh(double x)
{
    return 2.0 / (1.0 + exp(-2*x)) - 1.0;
}
#endif

static inline double tanh_g(double y)
{
    return 1.0 - y*y;
}

static inline double relu(double x)
{
    return (0 < x)? x : 0;
}

static inline double relu_g(double y)
{
    return (0 < y)? 1 : 0;
}

void Layer_feedForw_pool(Layer* self);
void Layer_feedBack_pool(Layer* self);

static Layer* Layer_create(
    Layer* lprev, LayerType ltype,
    int depth, int width, int height,
    int nbiases, int nweights)
{
    Layer* self = (Layer*)calloc(1, sizeof(Layer));
    if (self == NULL) return NULL;

    self->lprev = lprev;
    self->lnext = NULL;
    self->ltype = ltype;
    self->lid = 0;
    if (lprev != NULL) {
        assert (lprev->lnext == NULL);
        lprev->lnext = self;
        self->lid = lprev->lid+1;
    }
    self->depth = depth;
    self->width = width;
    self->height = height;

    self->nnodes = depth * width * height;
    self->outputs = (double*)calloc(self->nnodes, sizeof(double));
    self->gradients = (double*)calloc(self->nnodes, sizeof(double));
    self->errors = (double*)calloc(self->nnodes, sizeof(double));

    self->nbiases = nbiases;
    self->biases = (double*)calloc(self->nbiases, sizeof(double));
    self->u_biases = (double*)calloc(self->nbiases, sizeof(double));

    self->nweights = nweights;
    self->weights = (double*)calloc(self->nweights, sizeof(double));
    self->u_weights = (double*)calloc(self->nweights, sizeof(double));

    return self;
}

void Layer_destroy(Layer* self)
{
    assert (self != NULL);

    free(self->outputs);
    free(self->gradients);
    free(self->errors);

    free(self->biases);
    free(self->u_biases);
    free(self->weights);
    free(self->u_weights);

    free(self);
}

void Layer_dump(const Layer* self, FILE* fp)
{
    assert (self != NULL);
    Layer* lprev = self->lprev;
    fprintf(fp, "Layer%d ", self->lid);
    if (lprev != NULL) {
        fprintf(fp, "(lprev=Layer%d) ", lprev->lid);
    }
    fprintf(fp, "shape=(%d,%d,%d), nodes=%d\n",
            self->depth, self->width, self->height, self->nnodes);
    {
        int i = 0;
        for (int z = 0; z < self->depth; z++) {
            fprintf(fp, "  %d:\n", z);
            for (int y = 0; y < self->height; y++) {
                fprintf(fp, "    [");
                for (int x = 0; x < self->width; x++) {
                    fprintf(fp, " %.4f", self->outputs[i++]);
                }
                fprintf(fp, "]\n");
            }
        }
    }

    switch (self->ltype) {
    case LAYER_FULL:
        assert (lprev != NULL);
        fprintf(fp, "  biases = [");
        for (int i = 0; i < self->nnodes; i++) {
            fprintf(fp, " %.4f", self->biases[i]);
        }
        fprintf(fp, "]\n");
        fprintf(fp, "  weights = [\n");
        {
            int k = 0;
            for (int i = 0; i < self->nnodes; i++) {
                fprintf(fp, "    [");
                for (int j = 0; j < lprev->nnodes; j++) {
                    fprintf(fp, " %.4f", self->weights[k++]);
                }
                fprintf(fp, "]\n");
            }
        }
        fprintf(fp, "  ]\n");
        break;

    case LAYER_CONV:
        assert (lprev != NULL);
        fprintf(fp, "  stride=%d, kernsize=%d\n",
                self->conv.stride, self->conv.kernsize);
        {
            int k = 0;
            for (int z = 0; z < self->depth; z++) {
                fprintf(fp, "  %d: bias=%.4f, weights = [", z, self->biases[z]);
                for (int j = 0; j < lprev->depth * self->conv.kernsize * self->conv.kernsize; j++) {
                    fprintf(fp, " %.4f", self->weights[k++]);
                }
                fprintf(fp, "]\n");
            }
        }
        break;

    default:
        break;
    }
}

static void Layer_feedForw_full(Layer* self)
{
    assert (self->ltype == LAYER_FULL);
    assert (self->lprev != NULL);
    Layer* lprev = self->lprev;

    int k = 0;
    #pragma omp parallel for
    for (int i = 0; i < self->nnodes; i++) {
        double x = self->biases[i];

        for (int j = 0; j < lprev->nnodes; j++) {
            x += (lprev->outputs[j] * self->weights[k++]);
        }
        self->outputs[i] = x;
    }

    if (self->lnext == NULL) {
        double m = -1;
        #pragma omp parallel for reduction(max:m)
        for (int i = 0; i < self->nnodes; i++) {
            double x = self->outputs[i];
            if (m < x) { m = x; }
        }
        double t = 0;
        #pragma omp parallel for reduction(+:t)
        for (int i = 0; i < self->nnodes; i++) {
            double x = self->outputs[i];
            double y = exp(x-m);
            self->outputs[i] = y;
            t += y;
        }
        #pragma omp parallel for
        for (int i = 0; i < self->nnodes; i++) {
            self->outputs[i] /= t;
            self->gradients[i] = 1;
        }
    } else {
        #pragma omp parallel for
        for (int i = 0; i < self->nnodes; i++) {
            double x = self->outputs[i];
            double y = tanh(x);
            self->outputs[i] = y;
            self->gradients[i] = tanh_g(y);
        }
    }

#if DEBUG_LAYER
    fprintf(stderr, "Layer_feedForw_full(Layer%d):\n", self->lid);
    fprintf(stderr, "  outputs = [");
    for (int i = 0; i < self->nnodes; i++) {
        fprintf(stderr, " %.4f", self->outputs[i]);
    }
    fprintf(stderr, "]\n  gradients = [");
    for (int i = 0; i < self->nnodes; i++) {
        fprintf(stderr, " %.4f", self->gradients[i]);
    }
    fprintf(stderr, "]\n");
#endif
}

static void Layer_feedBack_full(Layer* self)
{
    assert (self->ltype == LAYER_FULL);
    assert (self->lprev != NULL);
    Layer* lprev = self->lprev;

    #pragma omp parallel for
    for (int j = 0; j < lprev->nnodes; j++) {
        lprev->errors[j] = 0;
    }

    #pragma omp parallel for
    for (int i = 0; i < self->nnodes; i++) {
        double dnet = self->errors[i] * self->gradients[i];
        for (int j = 0; j < lprev->nnodes; j++) {
            lprev->errors[j] += self->weights[i * lprev->nnodes + j] * dnet;
            self->u_weights[i * lprev->nnodes + j] += dnet * lprev->outputs[j];
        }
        self->u_biases[i] += dnet;
    }

#if DEBUG_LAYER
    fprintf(stderr, "Layer_feedBack_full(Layer%d):\n", self->lid);
    for (int i = 0; i < self->nnodes; i++) {
        double dnet = self->errors[i] * self->gradients[i];
        fprintf(stderr, "  dnet = %.4f, dw = [", dnet);
        for (int j = 0; j < lprev->nnodes; j++) {
            double dw = dnet * lprev->outputs[j];
            fprintf(stderr, " %.4f", dw);
        }
        fprintf(stderr, "]\n");
    }
#endif
}

static void Layer_feedForw_conv(Layer* self)
{
    assert (self->ltype == LAYER_CONV);
    assert (self->lprev != NULL);
    Layer* lprev = self->lprev;

    int kernsize = self->conv.kernsize;

    const int total_iterations = self->depth * self->height * self->width;
    #pragma omp parallel for
    for (int i = 0; i < total_iterations; i++) {
        int z1 = i / (self->height * self->width);
        int y1 = (i / self->width) % self->height;
        int x1 = i % self->width;

        int qbase = z1 * lprev->depth * kernsize * kernsize;
        int y0 = self->conv.stride * y1 - self->conv.padding;
        int x0 = self->conv.stride * x1 - self->conv.padding;

        double v = self->biases[z1];
        for (int z0 = 0; z0 < lprev->depth; z0++) {
            int pbase = z0 * lprev->width * lprev->height;
            for (int dy = 0; dy < kernsize; dy++) {
                int y = y0+dy;
                if (0 <= y && y < lprev->height) {
                    int p = pbase + y*lprev->width;
                    int q = qbase + dy*kernsize;
                    for (int dx = 0; dx < kernsize; dx++) {
                        int x = x0+dx;
                        if (0 <= x && x < lprev->width) {
                            v += lprev->outputs[p+x] * self->weights[q+dx];
                        }
                    }
                }
            }
        }
        v = relu(v);
        self->outputs[i] = v;
        self->gradients[i] = relu_g(v);
    }

#if DEBUG_LAYER
    fprintf(stderr, "Layer_feedForw_conv(Layer%d):\n", self->lid);
    fprintf(stderr, "  outputs = [");
    for (int i = 0; i < self->nnodes; i++) {
        fprintf(stderr, " %.4f", self->outputs[i]);
    }
    fprintf(stderr, "]\n  gradients = [");
    for (int i = 0; i < self->nnodes; i++) {
        fprintf(stderr, " %.4f", self->gradients[i]);
    }
    fprintf(stderr, "]\n");
#endif
}

static void Layer_feedBack_conv(Layer* self)
{
    assert (self->ltype == LAYER_CONV);
    assert (self->lprev != NULL);
    Layer* lprev = self->lprev;

    #pragma omp parallel for
    for (int j = 0; j < lprev->nnodes; j++) {
        lprev->errors[j] = 0;
    }

    int kernsize = self->conv.kernsize;

    const int total_iterations = self->depth * self->height * self->width;
    #pragma omp parallel for
    for (int i = 0; i < total_iterations; i++) {
        int z1 = i / (self->height * self->width);
        int y1 = (i / self->width) % self->height;
        int x1 = i % self->width;

        int qbase = z1 * lprev->depth * kernsize * kernsize;
        int y0 = self->conv.stride * y1 - self->conv.padding;
        int x0 = self->conv.stride * x1 - self->conv.padding;

        double dnet = self->errors[i] * self->gradients[i];
        for (int z0 = 0; z0 < lprev->depth; z0++) {
            int pbase = z0 * lprev->width * lprev->height;
            for (int dy = 0; dy < kernsize; dy++) {
                int y = y0+dy;
                if (0 <= y && y < lprev->height) {
                    int p = pbase + y*lprev->width;
                    int q = qbase + dy*kernsize;
                    for (int dx = 0; dx < kernsize; dx++) {
                        int x = x0+dx;
                        if (0 <= x && x < lprev->width) {
                            lprev->errors[p+x] += self->weights[q+dx] * dnet;
                            self->u_weights[q+dx] += dnet * lprev->outputs[p+x];
                        }
                    }
                }
            }
        }
        self->u_biases[z1] += dnet;
    }

#if DEBUG_LAYER
    fprintf(stderr, "Layer_feedBack_conv(Layer%d):\n", self->lid);
    for (int i = 0; i < self->nnodes; i++) {
        double dnet = self->errors[i] * self->gradients[i];
        fprintf(stderr, "  dnet=%.4f, dw=[", dnet);
        for (int j = 0; j < lprev->nnodes; j++) {
            double dw = dnet * lprev->outputs[j];
            fprintf(stderr, " %.4f", dw);
        }
        fprintf(stderr, "]\n");
    }
#endif
}

void Layer_setInputs(Layer* self, const double* values)
{
    assert (self != NULL);
    assert (self->ltype == LAYER_INPUT);
    assert (self->lprev == NULL);

#if DEBUG_LAYER
    fprintf(stderr, "Layer_setInputs(Layer%d): values = [", self->lid);
    for (int i = 0; i < self->nnodes; i++) {
        fprintf(stderr, " %.4f", values[i]);
    }
    fprintf(stderr, "]\n");
#endif

    #pragma omp parallel for
    for (int i = 0; i < self->nnodes; i++) {
        self->outputs[i] = values[i];
    }

    Layer* layer = self->lnext;
    while (layer != NULL) {
        switch (layer->ltype) {
        case LAYER_FULL:
            Layer_feedForw_full(layer);
            break;
        case LAYER_CONV:
            Layer_feedForw_conv(layer);
            break;
        case LAYER_POOL:
            Layer_feedForw_pool(layer);
            break;
        default:
            break;
        }
        layer = layer->lnext;
    }
}

void Layer_getOutputs(const Layer* self, double* outputs)
{
    assert (self != NULL);
    #pragma omp parallel for
    for (int i = 0; i < self->nnodes; i++) {
        outputs[i] = self->outputs[i];
    }
}

double Layer_getErrorTotal(const Layer* self)
{
    assert (self != NULL);
    double total = 0;
    #pragma omp parallel for reduction(+:total)
    for (int i = 0; i < self->nnodes; i++) {
        double e = self->errors[i];
        total += e*e;
    }
    return (total / self->nnodes);
}

void Layer_learnOutputs(Layer* self, const double* values)
{
    assert (self != NULL);
    assert (self->ltype != LAYER_INPUT);
    assert (self->lprev != NULL);
    
    if (self->lnext == NULL) {
        #pragma omp parallel for
        for (int i = 0; i < self->nnodes; i++) {
            self->errors[i] = self->outputs[i] - values[i];
        }
    } else {
        #pragma omp parallel for
        for (int i = 0; i < self->nnodes; i++) {
            self->errors[i] = (self->outputs[i] - values[i]);
        }
    }

#if DEBUG_LAYER
    fprintf(stderr, "Layer_learnOutputs(Layer%d): errors = [", self->lid);
    for (int i = 0; i < self->nnodes; i++) {
        fprintf(stderr, " %.4f", self->errors[i]);
    }
    fprintf(stderr, "]\n");
#endif

    Layer* layer = self;
    while (layer != NULL) {
        switch (layer->ltype) {
        case LAYER_FULL:
            Layer_feedBack_full(layer);
            break;
        case LAYER_CONV:
            Layer_feedBack_conv(layer);
            break;
        case LAYER_POOL:
            Layer_feedBack_pool(layer);
            break;
        default:
            break;
        }
        layer = layer->lprev;
    }
}

void Layer_update(Layer* self, double rate)
{
    #pragma omp parallel for
    for (int i = 0; i < self->nbiases; i++) {
        self->biases[i] -= rate * self->u_biases[i];
        self->u_biases[i] = 0;
    }
    #pragma omp parallel for
    for (int i = 0; i < self->nweights; i++) {
        self->weights[i] -= rate * self->u_weights[i];
        self->u_weights[i] = 0;
    }
    if (self->lprev != NULL) {
        Layer_update(self->lprev, rate);
    }
}

Layer* Layer_create_input(int depth, int width, int height)
{
    return Layer_create(
        NULL, LAYER_INPUT, depth, width, height, 0, 0);
}

Layer* Layer_create_full(Layer* lprev, int nnodes, double std)
{
    assert (lprev != NULL);
    Layer* self = Layer_create(
        lprev, LAYER_FULL, nnodes, 1, 1,
        nnodes, nnodes * lprev->nnodes);
    assert (self != NULL);

    for (int i = 0; i < self->nweights; i++) {
        self->weights[i] = std * nrnd();
    }

#if DEBUG_LAYER
    Layer_dump(self, stderr);
#endif
    return self;
}

Layer* Layer_create_conv(
    Layer* lprev, int depth, int width, int height,
    int kernsize, int padding, int stride, double std)
{
    assert (lprev != NULL);
    assert ((kernsize % 2) == 1);
    assert ((width-1) * stride + kernsize <= lprev->width + padding*2);
    assert ((height-1) * stride + kernsize <= lprev->height + padding*2);

    Layer* self = Layer_create(
        lprev, LAYER_CONV, depth, width, height,
        depth, depth * lprev->depth * kernsize * kernsize);
    assert (self != NULL);

    self->conv.kernsize = kernsize;
    self->conv.padding = padding;
    self->conv.stride = stride;

    for (int i = 0; i < self->nweights; i++) {
        self->weights[i] = std * nrnd();
    }

#if DEBUG_LAYER
    Layer_dump(self, stderr);
#endif
    return self;
}

void Layer_feedForw_pool(Layer* self)
{
    assert(self->ltype == LAYER_POOL);
    assert(self->lprev != NULL);
    Layer* lprev = self->lprev;
    
    int pool_size = self->pool.size;
    const int total_iterations = self->depth * self->height * self->width;

    #pragma omp parallel for
    for (int idx = 0; idx < total_iterations; idx++) {
        int z = idx / (self->height * self->width);
        int y = (idx / self->width) % self->height;
        int x = idx % self->width;

        double maxval = -1e9;
        int py = y * pool_size;
        int px = x * pool_size;

        for (int dy = 0; dy < pool_size; dy++) {
            for (int dx = 0; dx < pool_size; dx++) {
                int prev_idx = z * (lprev->width * lprev->height) + 
                             (py + dy) * lprev->width + (px + dx);
                if (lprev->outputs[prev_idx] > maxval) {
                    maxval = lprev->outputs[prev_idx];
                }
            }
        }

        self->outputs[idx] = maxval;
        self->gradients[idx] = 1.0;
    }
}

void Layer_feedBack_pool(Layer* self)
{
    assert(self->ltype == LAYER_POOL);
    assert(self->lprev != NULL);
    Layer* lprev = self->lprev;
    
    for (int j = 0; j < lprev->nnodes; j++) {
        lprev->errors[j] = 0;
    }
    
    int pool_size = self->pool.size;
    const int total_iterations = self->depth * self->height * self->width;
    
    #pragma omp parallel for
    for (int idx = 0; idx < total_iterations; idx++) {
        int z = idx / (self->height * self->width);
        int y = (idx / self->width) % self->height;
        int x = idx % self->width;

        double maxval = -1e9;
        int max_idx = -1;
        int py = y * pool_size;
        int px = x * pool_size;

        for (int dy = 0; dy < pool_size; dy++) {
            for (int dx = 0; dx < pool_size; dx++) {
                int prev_idx = z * (lprev->width * lprev->height) + 
                             (py + dy) * lprev->width + (px + dx);
                if (lprev->outputs[prev_idx] > maxval) {
                    maxval = lprev->outputs[prev_idx];
                    max_idx = prev_idx;
                }
            }
        }

        if (max_idx >= 0) {
            lprev->errors[max_idx] += self->errors[idx];
        }
    }
}

Layer* Layer_create_pool(Layer* lprev, int width, int height, int pool_size)
{
    assert(lprev != NULL);
    assert(pool_size > 0);
    assert(lprev->width % pool_size == 0);
    assert(lprev->height % pool_size == 0);
    
    Layer* self = Layer_create(
        lprev, LAYER_POOL, lprev->depth, width, height,
        0, 0);  
    
    self->pool.size = pool_size;
    
    return self;
}

void Layer_save_weights(const Layer* layer, FILE* fp) {
    if (layer == NULL || fp == NULL) return;
    
    fwrite(&layer->ltype, sizeof(LayerType), 1, fp);
    fwrite(&layer->depth, sizeof(int), 1, fp);
    fwrite(&layer->width, sizeof(int), 1, fp);
    fwrite(&layer->height, sizeof(int), 1, fp);
    
    if (layer->nweights > 0) {
        fwrite(layer->weights, sizeof(double), layer->nweights, fp);
    }
    if (layer->nbiases > 0) {
        fwrite(layer->biases, sizeof(double), layer->nbiases, fp);
    }
    
    if (layer->ltype == LAYER_CONV) {
        fwrite(&layer->conv.kernsize, sizeof(int), 1, fp);
        fwrite(&layer->conv.padding, sizeof(int), 1, fp);
        fwrite(&layer->conv.stride, sizeof(int), 1, fp);
    }
    else if (layer->ltype == LAYER_POOL) {
        fwrite(&layer->pool.size, sizeof(int), 1, fp);
    }
    
    if (layer->lnext) {
        Layer_save_weights(layer->lnext, fp);
    }
}

void Layer_load_weights(Layer* layer, FILE* fp) {
    if (layer == NULL || fp == NULL) return;
    
    LayerType saved_type;
    int saved_depth, saved_width, saved_height;
    
    if (fread(&saved_type, sizeof(LayerType), 1, fp) != 1 ||
        fread(&saved_depth, sizeof(int), 1, fp) != 1 ||
        fread(&saved_width, sizeof(int), 1, fp) != 1 ||
        fread(&saved_height, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "Failed to read model basic information\n");
        return;
    }
    
    assert(saved_type == layer->ltype);
    assert(saved_depth == layer->depth);
    assert(saved_width == layer->width);
    assert(saved_height == layer->height);
    
    if (layer->nweights > 0) {
        if (fread(layer->weights, sizeof(double), layer->nweights, fp) != layer->nweights) {
            fprintf(stderr, "Failed to read weights\n");
            return;
        }
    }
    if (layer->nbiases > 0) {
        if (fread(layer->biases, sizeof(double), layer->nbiases, fp) != layer->nbiases) {
            fprintf(stderr, "Failed to read biases\n");
            return;
        }
    }
    
    if (layer->ltype == LAYER_CONV) {
        int saved_kernsize, saved_padding, saved_stride;
        if (fread(&saved_kernsize, sizeof(int), 1, fp) != 1 ||
            fread(&saved_padding, sizeof(int), 1, fp) != 1 ||
            fread(&saved_stride, sizeof(int), 1, fp) != 1) {
            fprintf(stderr, "Failed to read convolution layer parameters\n");
            return;
        }
        assert(saved_kernsize == layer->conv.kernsize);
        assert(saved_padding == layer->conv.padding);
        assert(saved_stride == layer->conv.stride);
    }
    else if (layer->ltype == LAYER_POOL) {
        int saved_pool_size;
        if (fread(&saved_pool_size, sizeof(int), 1, fp) != 1) {
            fprintf(stderr, "Failed to read pooling layer parameters\n");
            return;
        }
        assert(saved_pool_size == layer->pool.size);
    }
    
    if (layer->lnext) {
        Layer_load_weights(layer->lnext, fp);
    }
}