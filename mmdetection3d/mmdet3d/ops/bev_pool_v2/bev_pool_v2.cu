#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void bev_pool_v2_kernel(
    int c, int n_intervals,
    const float* __restrict__ depth,
    const float* __restrict__ feat,
    const int* __restrict__ ranks_depth,
    const int* __restrict__ ranks_feat,
    const int* __restrict__ ranks_bev,
    const int* __restrict__ interval_starts,
    const int* __restrict__ interval_lengths,
    float* __restrict__ out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int index = idx / c;
    int cur_c = idx % c;
    if (index >= n_intervals) return;
    int interval_start = interval_starts[index];
    int interval_length = interval_lengths[index];
    float psum = 0;
    for (int i = interval_start; i < interval_start + interval_length; i++) {
        psum += depth[ranks_depth[i]] * feat[ranks_feat[i] * c + cur_c];
    }
    out[ranks_bev[interval_start] * c + cur_c] = psum;
}

void bev_pool_v2_forward(
    const torch::Tensor depth,
    const torch::Tensor feat,
    torch::Tensor out,
    const torch::Tensor ranks_depth,
    const torch::Tensor ranks_feat,
    const torch::Tensor ranks_bev,
    const torch::Tensor interval_lengths,
    const torch::Tensor interval_starts
) {
    int n_intervals = interval_lengths.size(0);
    int c = feat.size(feat.dim() - 1);
    auto out_flat = out.view({-1, c});
    bev_pool_v2_kernel<<<(n_intervals * c + 255) / 256, 256>>>(
        c, n_intervals,
        depth.data_ptr<float>(),
        feat.data_ptr<float>(),
        ranks_depth.data_ptr<int>(),
        ranks_feat.data_ptr<int>(),
        ranks_bev.data_ptr<int>(),
        interval_starts.data_ptr<int>(),
        interval_lengths.data_ptr<int>(),
        out_flat.data_ptr<float>()
    );
}

__global__ void bev_pool_v2_grad_kernel(
    int c, int n_intervals,
    const float* __restrict__ out_grad,
    const float* __restrict__ depth,
    const float* __restrict__ feat,
    const int* __restrict__ ranks_depth,
    const int* __restrict__ ranks_feat,
    const int* __restrict__ ranks_bev,
    const int* __restrict__ interval_starts,
    const int* __restrict__ interval_lengths,
    float* __restrict__ depth_grad,
    float* __restrict__ feat_grad
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int index = idx / c;
    int cur_c = idx % c;
    if (index >= n_intervals) return;
    int interval_start = interval_starts[index];
    int interval_length = interval_lengths[index];
    const float* cur_out_grad = out_grad + ranks_bev[interval_start] * c + cur_c;
    for (int i = interval_start; i < interval_start + interval_length; i++) {
        atomicAdd(&feat_grad[ranks_feat[i] * c + cur_c], depth[ranks_depth[i]] * (*cur_out_grad));
        atomicAdd(&depth_grad[ranks_depth[i]], feat[ranks_feat[i] * c + cur_c] * (*cur_out_grad));
    }
}

void bev_pool_v2_backward(
    const torch::Tensor out_grad,
    torch::Tensor depth_grad,
    torch::Tensor feat_grad,
    const torch::Tensor depth,
    const torch::Tensor feat,
    const torch::Tensor ranks_depth,
    const torch::Tensor ranks_feat,
    const torch::Tensor ranks_bev,
    const torch::Tensor interval_lengths,
    const torch::Tensor interval_starts
) {
    int n_intervals = interval_lengths.size(0);
    int c = feat.size(feat.dim() - 1);
    auto out_grad_flat = out_grad.contiguous().view({-1, c});
    bev_pool_v2_grad_kernel<<<(n_intervals * c + 255) / 256, 256>>>(
        c, n_intervals,
        out_grad_flat.data_ptr<float>(),
        depth.data_ptr<float>(),
        feat.data_ptr<float>(),
        ranks_depth.data_ptr<int>(),
        ranks_feat.data_ptr<int>(),
        ranks_bev.data_ptr<int>(),
        interval_starts.data_ptr<int>(),
        interval_lengths.data_ptr<int>(),
        depth_grad.data_ptr<float>(),
        feat_grad.data_ptr<float>()
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bev_pool_v2_forward", &bev_pool_v2_forward);
    m.def("bev_pool_v2_backward", &bev_pool_v2_backward);
}
