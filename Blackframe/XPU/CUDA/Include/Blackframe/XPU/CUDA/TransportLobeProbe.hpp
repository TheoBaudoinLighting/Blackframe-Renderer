#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace blackframe::xpu::cuda {

inline constexpr auto TransportLobeProbeCheckCount = std::uint32_t{11U};
inline constexpr auto TransportLobeProbeExpectedMask =
    (std::uint32_t{1U} << TransportLobeProbeCheckCount) - 1U;

struct TransportLobeProbeResult final {
    std::uint32_t passed_mask;
    std::uint32_t device_cxx_standard;
    float representative_values[TransportLobeProbeCheckCount];
    std::uint32_t reserved[3U];
};

static_assert(std::is_standard_layout_v<TransportLobeProbeResult>);
static_assert(std::is_trivially_copyable_v<TransportLobeProbeResult>);
static_assert(sizeof(TransportLobeProbeResult) == 64U);
static_assert(alignof(TransportLobeProbeResult) == 4U);
static_assert(offsetof(TransportLobeProbeResult, passed_mask) == 0U);
static_assert(offsetof(TransportLobeProbeResult, device_cxx_standard) == 4U);
static_assert(offsetof(TransportLobeProbeResult, representative_values) == 8U);
static_assert(offsetof(TransportLobeProbeResult, reserved) == 52U);

} // namespace blackframe::xpu::cuda

extern "C" int
blackframe_cuda_run_transport_lobe_probe(blackframe::xpu::cuda::TransportLobeProbeResult* output,
                                         int* device_count) noexcept;
