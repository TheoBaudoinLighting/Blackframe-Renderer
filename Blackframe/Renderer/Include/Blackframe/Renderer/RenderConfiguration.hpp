#pragma once

#include <Blackframe/Core/Status.hpp>
#include <cstdint>
#include <string>

namespace blackframe::renderer {

struct RenderExtent {
    std::uint32_t width{1280};
    std::uint32_t height{720};
};

struct RenderConfiguration {
    RenderExtent extent{};
    std::uint32_t samples_per_pixel{1};
    std::uint32_t maximum_path_depth{4};
    std::uint32_t tile_edge_length{16};
    std::uint64_t seed{0};
    std::string xpu_device_id;
};

[[nodiscard]] core::Status validate_render_configuration(const RenderConfiguration& configuration);

} // namespace blackframe::renderer
