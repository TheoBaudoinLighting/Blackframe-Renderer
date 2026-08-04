#include <Blackframe/XPU/CUDA/SceneSoaHash.hpp>
#include <Blackframe/XPU/Shared/SceneSoaAbi.hpp>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

#if !defined(__CUDACC__)
#error "The scene SoA hash kernel must be compiled by the CUDA compiler."
#endif

static_assert(__cplusplus == 202002L);

#if defined(__cpp_pack_indexing)
#error "C++26 features are forbidden in host/device scene code."
#endif

namespace {

__global__ void hash_scene_soa_kernel(const std::uint8_t* const scene_bytes,
                                      const std::size_t scene_byte_count,
                                      std::uint64_t* const output_hash) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) {
        return;
    }

    auto hash = blackframe::xpu::shared::SceneSoaFnv1aOffsetBasis;
    constexpr auto hash_begin = blackframe::xpu::shared::SceneSoaContentHashOffset;
    constexpr auto hash_end = hash_begin + sizeof(std::uint64_t);
    for (auto index = std::size_t{0}; index < scene_byte_count; ++index) {
        const auto value =
            index >= hash_begin && index < hash_end ? std::uint8_t{0} : scene_bytes[index];
        hash ^= value;
        hash *= blackframe::xpu::shared::SceneSoaFnv1aPrime;
    }
    *output_hash = hash;
}

} // namespace

extern "C" int blackframe_cuda_launch_scene_soa_hash(const std::uint8_t* const scene_bytes,
                                                     const std::size_t scene_byte_count,
                                                     std::uint64_t* const output_hash) noexcept {
    if (scene_bytes == nullptr ||
        scene_byte_count < sizeof(blackframe::xpu::shared::SceneSoaHeader) ||
        output_hash == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    hash_scene_soa_kernel<<<1, 1>>>(scene_bytes, scene_byte_count, output_hash);
    return static_cast<int>(cudaGetLastError());
}
