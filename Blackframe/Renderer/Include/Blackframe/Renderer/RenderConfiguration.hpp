#pragma once

#include <Blackframe/Core/Status.hpp>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace blackframe::renderer {

inline constexpr std::uint32_t CurrentRenderConfigurationSchemaVersion = 1;

enum class RenderConfigurationValueKind {
    unsigned_integer,
    string,
};

struct RenderConfigurationFieldSchema {
    std::string_view key;
    RenderConfigurationValueKind value_kind;
    bool required;
    std::uint64_t minimum_unsigned_value;
    std::uint64_t maximum_unsigned_value;
    std::size_t maximum_string_size;
};

struct RenderConfigurationSchema {
    std::uint32_t version;
    bool allows_unknown_keys;
    std::span<const RenderConfigurationFieldSchema> fields;
};

struct RenderExtent {
    std::uint32_t width{1280};
    std::uint32_t height{720};
};

[[nodiscard]] core::Status validate_render_extent(RenderExtent extent);

struct RenderConfiguration {
    std::uint32_t schema_version{CurrentRenderConfigurationSchemaVersion};
    RenderExtent extent{};
    std::uint32_t samples_per_pixel{1};
    // Global workload safety ceiling; it is not mapped to per-category PathDepthLimits.
    std::uint32_t maximum_path_depth{4};
    std::uint32_t tile_edge_length{16};
    std::uint64_t seed{0};
    std::string xpu_device_id;
};

[[nodiscard]] RenderConfigurationSchema render_configuration_schema() noexcept;

// Serialized configurations enter the renderer only through this function. It
// rejects malformed JSON, unknown or duplicate keys, unsupported schema
// versions, and invalid values before returning the typed configuration.
[[nodiscard]] core::Result<RenderConfiguration>
parse_and_validate_render_configuration(std::string_view encoded_configuration);

[[nodiscard]] core::Status validate_render_configuration(const RenderConfiguration& configuration);

} // namespace blackframe::renderer
