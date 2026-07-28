#include <Blackframe/Renderer/RenderConfiguration.hpp>
#include <limits>

namespace blackframe::renderer {
namespace {

constexpr auto maximum_dimension = std::uint32_t{1U << 20U};
constexpr auto maximum_pixel_count =
    static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max());
constexpr auto maximum_samples_per_pixel = std::uint32_t{1U << 24U};
constexpr auto maximum_path_depth = std::uint32_t{4096U};
constexpr auto maximum_tile_edge_length = std::uint32_t{4096U};
constexpr auto maximum_sample_work_items = std::uint64_t{1ULL << 48U};
constexpr auto maximum_transport_work_items = std::uint64_t{1ULL << 56U};

[[nodiscard]] constexpr auto product_exceeds(const std::uint64_t left, const std::uint64_t right,
                                             const std::uint64_t limit) noexcept -> bool {
    return left != 0 && right > limit / left;
}

[[nodiscard]] auto resource_limit_error(const char* message) -> core::Status {
    return std::unexpected(core::Error{
        .code = core::StatusCode::resource_exhausted,
        .message = message,
    });
}

} // namespace

core::Status validate_render_configuration(const RenderConfiguration& configuration) {
    if (configuration.extent.width == 0 || configuration.extent.height == 0) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "Render width and height must be greater than zero.",
        });
    }

    if (configuration.extent.width > maximum_dimension ||
        configuration.extent.height > maximum_dimension) {
        return resource_limit_error("Render dimensions exceed the configured safety limit.");
    }

    const auto pixel_count = static_cast<std::uint64_t>(configuration.extent.width) *
                             static_cast<std::uint64_t>(configuration.extent.height);
    if (pixel_count > maximum_pixel_count) {
        return resource_limit_error("The requested image contains too many pixels.");
    }

    if (configuration.samples_per_pixel == 0 || configuration.maximum_path_depth == 0 ||
        configuration.tile_edge_length == 0) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "Samples, path depth, and tile edge length must be greater than zero.",
        });
    }

    if (configuration.samples_per_pixel > maximum_samples_per_pixel) {
        return resource_limit_error("Samples per pixel exceed the configured safety limit.");
    }
    if (configuration.maximum_path_depth > maximum_path_depth) {
        return resource_limit_error("Path depth exceeds the configured safety limit.");
    }
    if (configuration.tile_edge_length > maximum_tile_edge_length) {
        return resource_limit_error("Tile dimensions exceed the configured safety limit.");
    }

    // Bound every currently knowable work product before a backend can use it for allocation or
    // dispatch sizing. Backend-owned byte sizes still require their own checked multiplication.
    if (product_exceeds(pixel_count, configuration.samples_per_pixel, maximum_sample_work_items)) {
        return resource_limit_error("The requested sample workload exceeds the safety limit.");
    }

    const auto sample_work_items =
        pixel_count * static_cast<std::uint64_t>(configuration.samples_per_pixel);
    if (product_exceeds(sample_work_items, configuration.maximum_path_depth,
                        maximum_transport_work_items)) {
        return resource_limit_error("The requested transport workload exceeds the safety limit.");
    }

    if (product_exceeds(configuration.tile_edge_length, configuration.tile_edge_length,
                        maximum_pixel_count)) {
        return resource_limit_error("The requested tile contains too many pixels.");
    }

    return {};
}

} // namespace blackframe::renderer
