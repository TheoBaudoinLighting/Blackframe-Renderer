#include <Blackframe/XPU/Shared/SceneBvhAbi.hpp>
#include <type_traits>

#if !defined(__CUDACC__)
#error "The scene BVH ABI contract must be compiled by the CUDA compiler."
#endif

static_assert(__cplusplus == 202002L);

#if defined(__cpp_pack_indexing)
#error "C++26 features are forbidden in host/device scene BVH code."
#endif

static_assert(std::is_standard_layout_v<blackframe::xpu::shared::SceneBvhHeader>);
static_assert(std::is_trivially_copyable_v<blackframe::xpu::shared::SceneBvhHeader>);
static_assert(std::is_standard_layout_v<blackframe::xpu::shared::SceneBvhNode>);
static_assert(std::is_trivially_copyable_v<blackframe::xpu::shared::SceneBvhNode>);
static_assert(std::is_standard_layout_v<blackframe::xpu::shared::SceneBvhBlas>);
static_assert(std::is_trivially_copyable_v<blackframe::xpu::shared::SceneBvhBlas>);
static_assert(std::is_standard_layout_v<blackframe::xpu::shared::SceneBvhInstanceReference>);
static_assert(std::is_trivially_copyable_v<blackframe::xpu::shared::SceneBvhInstanceReference>);
