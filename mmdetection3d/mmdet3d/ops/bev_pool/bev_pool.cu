#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void bev_pool_forward_kernel(
    int c, int n_intervals,
    const float* __restrict__ x,
    const int* __restrict__ geom_feats,
    const int* __restrict__ interval_starts,
    const int* __restrict__ interval_lengths,
    float* __restrict__ out,
    int b, int d, int h, int w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int index = idx / c;
    int cur_c = idx % c;
    if (index >= n_intervals) return;
    int interval_start = interval_starts[index];
    int interval_length = interval_lengths[index];
    const int* cur_geom_feats = geom_feats + interval_start * 4;
    float* cur_out = out + (cur_geom_feats[3] * d * h * w + cur_geom_feats[2] * h * w + cur_geom_feats[1] * w + cur_geom_feats[0]) * c + cur_c;
    float psum = 0;
    const float* cur_x = x + interval_start * c + cur_c;
    for (int i = 0; i < interval_length; i++) {
        psum += cur_x[i * c];
    }
    *cur_out = psum;
}

__global__ void bev_pool_backward_kernel(
    int c, int n_points,
    const float* __restrict__ out_grad,
    const int* __restrict__ geom_feats,
    const int* __restrict__ interval_starts,
    const int* __restrict__ interval_lengths,
    float* __restrict__ x_grad,
    int b, int d, int h, int w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int index = idx / c;
    int cur_c = idx % c;
    if (index >= n_points) return;
    const int* cur_geom_feats = geom_feats + index * 4;
    const float* cur_out_grad = out_grad + (cur_geom_feats[3] * d * h * w + cur_geom_feats[2] * h * w + cur_geom_feats[1] * w + cur_geom_feats[0]) * c + cur_c;
    x_grad[index * c + cur_c] = *cur_out_grad;
}

torch::Tensor bev_pool_forward(
    const torch::Tensor x,
    const torch::Tensor geom_feats,
    const torch::Tensor interval_lengths,
    const torch::Tensor interval_starts,
    int b, int d, int h, int w
) {
    int n_intervals = interval_lengths.size(0);
    int c = x.size(1);
    auto out = torch::zeros({b, d, h, w, c}, x.options());
    bev_pool_forward_kernel<<<(n_intervals * c + 255) / 256, 256>>>(
        c, n_intervals,
        x.data_ptr<float>(),
        geom_feats.data_ptr<int>(),
        interval_starts.data_ptr<int>(),
        interval_lengths.data_ptr<int>(),
        out.data_ptr<float>(),
        b, d, h, w
    );
    return out;
}

torch::Tensor bev_pool_backward(
    const torch::Tensor out_grad,
    const torch::Tensor geom_feats,
    const torch::Tensor interval_lengths,
    const torch::Tensor interval_starts,
    int b, int d, int h, int w
) {
    int n_points = geom_feats.size(0);
    int c = out_grad.size(4);
    auto x_grad = torch::zeros({n_points, c}, out_grad.options());
    bev_pool_backward_kernel<<<(n_points * c + 255) / 256, 256>>>(
        c, n_points,
        out_grad.data_ptr<float>(),
        geom_feats.data_ptr<int>(),
        interval_starts.data_ptr<int>(),
        interval_lengths.data_ptr<int>(),
        x_grad.data_ptr<float>(),
        b, d, h, w
    );
    return x_grad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bev_pool_forward", &bev_pool_forward);
    m.def("bev_pool_backward", &bev_pool_backward);
}
