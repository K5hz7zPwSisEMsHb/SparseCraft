#include <tmatrix/Calculation/spgemm.cuh>
#include <tmatrix/Calculation/kernel_matrix/spgemm_kernels.cuh>
// #include <tmatrix/Calculation/kernel_matrix/spgemm_csr.cuh>

/*

// __global__ void SpGEMM_CalculateOnly(
//     Tile*At, Tile*Bt, Tile*Ct, char*Ad, char*Bd, char*Cd, MatValue*tmp_buffer
// )
// {
//     MatIndex gwarp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
//     int lane_id = threadIdx.x & 31, lwarp_id = threadIdx.x >> 5;
//     int vwarp_id = lane_id >> 1, hwarp_id = lane_id >> 4;
//     int vlane_id = lane_id & 1, hlane_id = lane_id & 15;

//     __shared__ TileIndex s_adata_all[9280], s_bdata_all[9280], s_cdata_all[9280];
//     TileIndex* s_adata = s_adata_all + (lwarp_id * 2320);
//     TileIndex* s_bdata = s_bdata_all + (lwarp_id * 2320);
//     TileIndex* s_cdata = s_cdata_all + (lwarp_id * 2320);

//     int bytes = At[0].bitslen * 16 + At[0].valslen * 8;
//     for (int i = lane_id; i < bytes; i += 32) s_adata[i] = Ad[i];
//     bytes = Bt[0].bitslen * 16 + Bt[0].valslen * 8;
//     for (int i = lane_id; i < bytes; i += 32) s_bdata[i] = Bd[i];
//     bytes = Ct[0].bitslen * 16 + Ct[0].valslen * 8;
//     for (int i = lane_id; i < bytes; i += 32) s_cdata[i] = Cd[i];
//     __syncwarp();
    
//     kernel_matrix[At[0].fmt][Bt[0].fmt][Ct[0].fmt](
//         s_adata + At[0].bits_off, (MatValue*) (s_adata + At[0].bits_off + At[0].bitslen * 16), 
//         s_bdata + Bt[0].bits_off, (MatValue*) (s_bdata + Bt[0].bits_off + Bt[0].bitslen * 16),
//         s_cdata + Ct[0].bits_off, (MatValue*) (s_cdata + Ct[0].bits_off + Ct[0].bitslen * 16), Ct->bitmap,
//         lane_id, vwarp_id, vlane_id, hwarp_id, hlane_id, false
//     );

//     __syncwarp();
//     if (gwarp_id == 0 && lane_id == 0) tmp_buffer[0] = s_cdata[lane_id];
// }
*/

__device__ void (*kernel_matrix_B[7][7])(
        TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
        const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag) = 
{
    {coo_x_coo_2_dns, coo_x_csr_2_dns, coo_x_ell_2_dns, coo_x_hyb_2_dns, coo_x_drw_2_dns, coo_x_dcl_2_dns, coo_x_dns_2_dns},
    {csr_x_coo_2_dns, csr_x_csr_2_dns, csr_x_ell_2_dns, csr_x_hyb_2_dns, csr_x_drw_2_dns, csr_x_dcl_2_dns, csr_x_dns_2_dns},
    {ell_x_coo_2_dns, ell_x_csr_2_dns, ell_x_ell_2_dns, ell_x_hyb_2_dns, ell_x_drw_2_dns, ell_x_dcl_2_dns, ell_x_dns_2_dns},
    {hyb_x_coo_2_dns, hyb_x_csr_2_dns, hyb_x_ell_2_dns, hyb_x_hyb_2_dns, hyb_x_drw_2_dns, hyb_x_dcl_2_dns, hyb_x_dns_2_dns},
    {drw_x_coo_2_dns, drw_x_csr_2_dns, drw_x_ell_2_dns, drw_x_hyb_2_dns, drw_x_drw_2_dns, drw_x_dcl_2_dns, drw_x_dns_2_dns},
    {dcl_x_coo_2_dns, dcl_x_csr_2_dns, dcl_x_ell_2_dns, dcl_x_hyb_2_dns, dcl_x_drw_2_dns, dcl_x_dcl_2_dns, dcl_x_dns_2_dns},
    {dns_x_coo_2_dns, dns_x_csr_2_dns, dns_x_ell_2_dns, dns_x_hyb_2_dns, dns_x_drw_2_dns, dns_x_dcl_2_dns, dns_x_dns_2_dns}
};

__global__ void SpGEMM_B_CalculateOnly(
    Tile*At, char*Ad, Tile*Bt, char*Bd, MatValue*tmp_buffer
)
{
    MatIndex gwarp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane_id = threadIdx.x & 31, lwarp_id = threadIdx.x >> 5;
    int vwarp_id = lane_id >> 1, hwarp_id = lane_id >> 4;
    int vlane_id = lane_id & 1, hlane_id = lane_id & 15;

    __shared__ MatValue s_cdata_all[1024];
    __shared__ TileIndex s_bdata_all[9280];
    TileIndex* s_bdata = s_bdata_all + (lwarp_id * 2320);
    MatValue* s_cdata = s_cdata_all + (lwarp_id * 256);

    kernel_matrix_B[At[0].fmt][Bt[0].fmt](
        (TileIndex*) Ad, (MatValue*) (Ad + At[0].bitslen * 16), 
        s_bdata + Bt[0].bits_off, (MatValue*) (s_bdata + Bt[0].bits_off + Bt[0].bitslen * 16),
        NULL, s_cdata, NULL,
        lane_id, vwarp_id, vlane_id, hwarp_id, hlane_id, false
    );

    __syncwarp();
    if (gwarp_id == 0 && lane_id == 0) tmp_buffer[0] = s_cdata[lane_id];
}