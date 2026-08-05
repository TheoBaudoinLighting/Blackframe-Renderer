#include <Blackframe/XPU/CUDA/SampleStreamDevice.cuh>
#include <Blackframe/XPU/CUDA/SampleStreamKernel.hpp>
#include <cstdint>
#include <cuda_runtime_api.h>

#if !defined(__CUDACC__)
#error "The CUDA SampleStream kernel must be compiled by the CUDA compiler."
#endif

static_assert(__cplusplus == 202002L);

#if defined(__cpp_pack_indexing)
#error "C++26 features are forbidden in CUDA SampleStream code."
#endif

namespace {

namespace cuda = blackframe::xpu::cuda;
namespace sample_stream = blackframe::xpu::cuda::sample_stream;

constexpr auto ThreadsPerBlock = std::uint32_t{256U};

__global__ void sample_stream_dump_kernel(const cuda::SampleStreamDumpRequest* const requests,
                                          const std::uint32_t request_count,
                                          cuda::SampleStreamDimensionManifest* const manifest,
                                          cuda::SampleStreamDumpResult* const results) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index == 0U) {
        *manifest = sample_stream::dimension_manifest();
    }
    if (index >= request_count) {
        return;
    }

    const auto& request = requests[index];
    auto result = cuda::SampleStreamDumpResult{};
    result.status =
        sample_stream::dimensions_for_bounce(request.bounce_index, result.bounce_dimensions);
    result.indexed_bits = sample_stream::indexed_bits(request.sample_stream, request.dimension);
    result.sample = sample_stream::sample_1d(request.sample_stream, request.dimension);
    results[index] = result;
}

} // namespace

extern "C" int blackframe_cuda_launch_sample_stream_dump(
    const std::uint32_t requested_schema_version,
    const blackframe::xpu::cuda::SampleStreamDumpRequest* const requests,
    const std::uint32_t request_count,
    blackframe::xpu::cuda::SampleStreamDimensionManifest* const manifest,
    blackframe::xpu::cuda::SampleStreamDumpResult* const results) noexcept {
    if (requested_schema_version !=
            blackframe::xpu::cuda::sample_stream::DimensionMapSchemaVersion ||
        requests == nullptr || request_count == 0U || manifest == nullptr || results == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const auto block_count = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(request_count) + ThreadsPerBlock - 1U) / ThreadsPerBlock);
    sample_stream_dump_kernel<<<block_count, ThreadsPerBlock>>>(requests, request_count, manifest,
                                                                results);
    return static_cast<int>(cudaGetLastError());
}
