#include <Blackframe/XPU/CUDA/SmokeKernel.hpp>
#include <Blackframe/XPU/CUDA/SmokeKernelPayload.hpp>
#include <cstdint>
#include <type_traits>

#if !defined(__CUDACC__)
#error "The shared-header contract must be compiled by the CUDA compiler."
#endif

static_assert(__cplusplus == 202002L);

#if defined(__cpp_pack_indexing)
#error "C++26 features are forbidden in shared host/device headers."
#endif

static_assert(std::is_standard_layout_v<blackframe::xpu::cuda::SmokeKernelPayload>);
static_assert(std::is_trivially_copyable_v<blackframe::xpu::cuda::SmokeKernelPayload>);

extern "C" std::uint64_t blackframe_cuda_shared_header_language_level() noexcept {
    return __cplusplus;
}
