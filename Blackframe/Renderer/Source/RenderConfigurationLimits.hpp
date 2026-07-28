#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace blackframe::renderer::limits {

inline constexpr auto maximum_encoded_configuration_size = std::size_t{64U * 1024U};
inline constexpr auto maximum_key_size = std::size_t{64};
inline constexpr auto maximum_xpu_device_id_size = std::size_t{1024};
inline constexpr auto maximum_dimension = std::uint32_t{1U << 20U};
inline constexpr auto maximum_pixel_count =
    static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max());
inline constexpr auto maximum_samples_per_pixel = std::uint32_t{1U << 24U};
inline constexpr auto maximum_path_depth = std::uint32_t{4096U};
inline constexpr auto maximum_tile_edge_length = std::uint32_t{4096U};
inline constexpr auto maximum_sample_work_items = std::uint64_t{1ULL << 48U};
inline constexpr auto maximum_transport_work_items = std::uint64_t{1ULL << 56U};

} // namespace blackframe::renderer::limits
