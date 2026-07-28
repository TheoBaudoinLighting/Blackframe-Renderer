#pragma once

#include <cstdint>
#include <type_traits>

namespace blackframe::xpu::cuda {

struct SmokeKernelPayload final {
    std::uint32_t input;
    std::uint32_t xor_mask;
};

static_assert(std::is_standard_layout_v<SmokeKernelPayload>);
static_assert(std::is_trivially_copyable_v<SmokeKernelPayload>);

} // namespace blackframe::xpu::cuda
