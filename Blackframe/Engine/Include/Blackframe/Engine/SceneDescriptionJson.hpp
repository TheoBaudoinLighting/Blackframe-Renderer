#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Engine/SceneDescription.hpp>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>

namespace blackframe::engine {

inline constexpr std::string_view SceneDescriptionJsonSchemaName{"blackframe.scene"};
inline constexpr std::uint32_t CurrentSceneDescriptionJsonSchemaVersion = 1U;

// Parsing and writing are both constrained by caller-selected limits that cannot exceed the hard
// format ceilings below. Smaller limits are useful for trusted workload profiles and deterministic
// allocation-failure tests; zero never means unlimited.
struct SceneDescriptionJsonLimits final {
    std::uint64_t maximum_encoded_bytes{64ULL << 20U};
    std::uint32_t maximum_nesting_depth{64U};
    std::uint64_t maximum_total_values{64'000'000ULL};
    std::uint64_t maximum_total_string_bytes{16ULL << 20U};
    std::uint32_t maximum_records_per_registry{1'000'000U};
    std::uint64_t maximum_mesh_vertices{4'000'000ULL};
    std::uint64_t maximum_mesh_triangles{4'000'000ULL};
    std::uint64_t maximum_image_scalar_values{16'000'000ULL};
    // Logical retained payload bytes. Container growth is additionally bounded by the record,
    // value, string, mesh, and image limits above; allocator overhead is intentionally not part of
    // the portable wire contract.
    std::uint64_t maximum_decoded_bytes{512ULL << 20U};

    [[nodiscard]] constexpr bool
    operator==(const SceneDescriptionJsonLimits&) const noexcept = default;
};

inline constexpr auto MaximumSceneDescriptionJsonLimits = SceneDescriptionJsonLimits{
    .maximum_encoded_bytes = 256ULL << 20U,
    .maximum_nesting_depth = 128U,
    .maximum_total_values = 128'000'000ULL,
    .maximum_total_string_bytes = 64ULL << 20U,
    .maximum_records_per_registry = 4'000'000U,
    .maximum_mesh_vertices = 16'000'000ULL,
    .maximum_mesh_triangles = 16'000'000ULL,
    .maximum_image_scalar_values = 64'000'000ULL,
    .maximum_decoded_bytes = 2ULL << 30U,
};

// The native format has exactly one public typed boundary: SceneDescription. It never serializes a
// FrameScene, acceleration structure, backend handle, queue, device pointer, or native C++ layout.
// All encoded numeric values are finite; serialization explicitly rejects a typed description
// containing NaN or infinity instead of inventing a string sentinel or clamping it.
[[nodiscard]] core::Result<std::string>
serialize_scene_description_json(const SceneDescription& description,
                                 SceneDescriptionJsonLimits limits = {});

[[nodiscard]] core::Result<SceneDescription>
deserialize_scene_description_json(std::string_view encoded_scene,
                                   SceneDescriptionJsonLimits limits = {});

static_assert(std::is_standard_layout_v<SceneDescriptionJsonLimits>);
static_assert(std::is_trivially_copyable_v<SceneDescriptionJsonLimits>);

} // namespace blackframe::engine
