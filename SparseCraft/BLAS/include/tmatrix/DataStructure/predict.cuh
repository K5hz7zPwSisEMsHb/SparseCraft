#include <tmatrix/DataStructure/TileMatrix.h>
#include <cuda_fp16.h>

struct sequence_slime_net;
struct oaktree;

oaktree* load_oaktree(const char *model_path);
oaktree* oaktree_to_device(oaktree *model);
sequence_slime_net* load_slimenet(const char *model_path);
sequence_slime_net* slimenet_to_device(sequence_slime_net *model);
__global__ void oaktree_prediction_single(oaktree *model, Tile*tiles, int n);
__global__ void pixel_format_prediction_single(sequence_slime_net *model, Tile*tiles, int n);