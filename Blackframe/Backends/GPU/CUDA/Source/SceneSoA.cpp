#include <Blackframe/Backends/GPU/CUDA/SceneSoA.hpp>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <limits>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace blackframe::engine {
namespace {

namespace column = xpu::shared::scene_soa_column;
using xpu::shared::SceneSoaHeader;

static_assert(std::endian::native == std::endian::little,
              "CUDA scene serialization requires a little-endian host.");
static_assert(std::numeric_limits<float>::is_iec559);
static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));
static_assert(renderer::TransportSpectrumSampleCount == xpu::shared::SceneSoaSpectrumLaneCount);
static_assert(std::variant_size_v<ScenePunctualLight> == 3U);
static_assert(std::is_same_v<std::variant_alternative_t<0, ScenePunctualLight>, ScenePointLight>);
static_assert(
    std::is_same_v<std::variant_alternative_t<1, ScenePunctualLight>, SceneDirectionalLight>);
static_assert(std::is_same_v<std::variant_alternative_t<2, ScenePunctualLight>, SceneSpotLight>);

struct SerializedScene final {
    SceneSoaHeader header{};
    std::vector<std::uint8_t> bytes;
};

[[nodiscard]] core::Error scene_soa_error(const core::StatusCode code, std::string message) {
    return core::Error{.code = code, .message = std::move(message)};
}

[[nodiscard]] core::Error cuda_copy_error(const cudaError_t status, const std::size_t byte_count) {
    return scene_soa_error(xpu::cuda::cuda_memory_status_code(static_cast<std::int32_t>(status)),
                           "CUDA scene upload failed for " + std::to_string(byte_count) +
                               " bytes: " + cudaGetErrorName(status) + " (" +
                               cudaGetErrorString(status) + ").");
}

[[nodiscard]] bool add_overflows(const std::uint64_t left, const std::uint64_t right) noexcept {
    return left > std::numeric_limits<std::uint64_t>::max() - right;
}

[[nodiscard]] core::Status prepare_layout(SceneSoaHeader& header, const std::size_t host_max_size,
                                          const xpu::cuda::DeviceMemoryBudget budget) {
    auto cursor = xpu::shared::scene_soa_align_up(sizeof(SceneSoaHeader));
    for (auto column_index = std::uint32_t{0}; column_index < column::count; ++column_index) {
        auto& descriptor = header.columns[column_index];
        descriptor.element_count = xpu::shared::scene_soa_column_count(header, column_index);
        descriptor.element_size = xpu::shared::scene_soa_column_element_size(column_index);
        if (descriptor.element_count == 0U) {
            continue;
        }
        if (cursor > std::numeric_limits<std::uint64_t>::max() -
                         (xpu::shared::SceneSoaColumnAlignment - 1U)) {
            return std::unexpected(scene_soa_error(core::StatusCode::resource_exhausted,
                                                   "CUDA scene column alignment overflowed."));
        }
        cursor = xpu::shared::scene_soa_align_up(cursor);
        if (descriptor.element_count >
            std::numeric_limits<std::uint64_t>::max() / descriptor.element_size) {
            return std::unexpected(scene_soa_error(core::StatusCode::resource_exhausted,
                                                   "CUDA scene column byte count overflowed."));
        }
        const auto byte_count = descriptor.element_count * descriptor.element_size;
        if (add_overflows(cursor, byte_count)) {
            return std::unexpected(scene_soa_error(core::StatusCode::resource_exhausted,
                                                   "CUDA scene aggregate byte count overflowed."));
        }
        descriptor.offset_bytes = cursor;
        cursor += byte_count;
    }
    header.total_size_bytes = cursor;
    if (cursor > budget.maximum_bytes) {
        return std::unexpected(scene_soa_error(
            core::StatusCode::resource_exhausted,
            "CUDA scene exceeds its explicit device-memory budget before serialization."));
    }
    if (cursor > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
        cursor > static_cast<std::uint64_t>(host_max_size)) {
        return std::unexpected(
            scene_soa_error(core::StatusCode::resource_exhausted,
                            "CUDA scene exceeds the addressable host serialization range."));
    }
    if (xpu::shared::validate_scene_soa_header(header) !=
        xpu::shared::SceneSoaHeaderValidationStatus::valid) {
        return std::unexpected(
            scene_soa_error(core::StatusCode::internal_error,
                            "CUDA scene layout preparation violated its frozen schema."));
    }
    return {};
}

template <typename Value, typename Visitor>
[[nodiscard]] core::Status append_column(SceneSoaHeader& header, std::vector<std::uint8_t>& bytes,
                                         std::uint32_t& next_expected_column,
                                         const std::uint32_t column_index,
                                         const std::uint64_t element_count, Visitor&& visit) {
    static_assert(std::is_trivially_copyable_v<Value>);
    if (column_index >= column::count || column_index != next_expected_column ||
        xpu::shared::scene_soa_column_count(header, column_index) != element_count ||
        xpu::shared::scene_soa_column_element_size(column_index) != sizeof(Value)) {
        return std::unexpected(scene_soa_error(
            core::StatusCode::internal_error,
            "CUDA scene serializer received a column inconsistent with its frozen schema."));
    }

    auto& descriptor = header.columns[column_index];
    ++next_expected_column;
    if (descriptor.element_count != element_count || descriptor.element_size != sizeof(Value)) {
        return std::unexpected(scene_soa_error(
            core::StatusCode::internal_error,
            "CUDA scene column disagrees with its prevalidated layout descriptor."));
    }
    if (element_count == 0U) {
        if (descriptor.offset_bytes != 0U) {
            return std::unexpected(
                scene_soa_error(core::StatusCode::internal_error,
                                "Empty CUDA scene column has a non-zero prevalidated offset."));
        }
        return {};
    }
    if (element_count > std::numeric_limits<std::uint64_t>::max() / sizeof(Value)) {
        return std::unexpected(scene_soa_error(core::StatusCode::resource_exhausted,
                                               "CUDA scene column byte count overflowed."));
    }
    const auto column_byte_count = element_count * sizeof(Value);
    const auto current_size = static_cast<std::uint64_t>(bytes.size());
    if (current_size >
        std::numeric_limits<std::uint64_t>::max() - (xpu::shared::SceneSoaColumnAlignment - 1U)) {
        return std::unexpected(scene_soa_error(core::StatusCode::resource_exhausted,
                                               "CUDA scene column alignment overflowed."));
    }
    const auto offset = xpu::shared::scene_soa_align_up(current_size);
    if (add_overflows(offset, column_byte_count) ||
        offset + column_byte_count > static_cast<std::uint64_t>(bytes.max_size())) {
        return std::unexpected(scene_soa_error(core::StatusCode::resource_exhausted,
                                               "CUDA scene column exceeds host container limits."));
    }

    const auto offset_size = static_cast<std::size_t>(offset);
    const auto end_size = static_cast<std::size_t>(offset + column_byte_count);
    bytes.resize(offset_size, std::uint8_t{0});
    bytes.resize(end_size, std::uint8_t{0});
    if (descriptor.offset_bytes != offset) {
        return std::unexpected(
            scene_soa_error(core::StatusCode::internal_error,
                            "CUDA scene column placement diverged from its prevalidated layout."));
    }

    auto emitted_count = std::uint64_t{0};
    auto emitted_too_many = false;
    auto emit = [&](const Value value) {
        if (emitted_count < element_count) {
            const auto element_offset =
                offset_size + static_cast<std::size_t>(emitted_count) * sizeof(Value);
            std::memcpy(bytes.data() + element_offset, &value, sizeof(Value));
        } else {
            emitted_too_many = true;
        }
        ++emitted_count;
    };
    std::forward<Visitor>(visit)(emit);
    if (emitted_too_many || emitted_count != element_count) {
        return std::unexpected(scene_soa_error(
            core::StatusCode::internal_error,
            "CUDA scene serializer emitted a column with an inconsistent element count."));
    }
    return {};
}

[[nodiscard]] float component(const renderer::Point3 value, const std::uint32_t index) noexcept {
    switch (index) {
    case 0U:
        return value.x;
    case 1U:
        return value.y;
    default:
        return value.z;
    }
}

[[nodiscard]] float component(const renderer::Vector3 value, const std::uint32_t index) noexcept {
    switch (index) {
    case 0U:
        return value.x;
    case 1U:
        return value.y;
    default:
        return value.z;
    }
}

[[nodiscard]] float component(const renderer::Normal3 value, const std::uint32_t index) noexcept {
    switch (index) {
    case 0U:
        return value.x;
    case 1U:
        return value.y;
    default:
        return value.z;
    }
}

[[nodiscard]] float component(const renderer::Point2 value, const std::uint32_t index) noexcept {
    return index == 0U ? value.x : value.y;
}

[[nodiscard]] std::uint64_t normalized_hash(const std::vector<std::uint8_t>& bytes) noexcept {
    auto hash = xpu::shared::SceneSoaFnv1aOffsetBasis;
    constexpr auto hash_begin = xpu::shared::SceneSoaContentHashOffset;
    constexpr auto hash_end = hash_begin + sizeof(std::uint64_t);
    for (auto index = std::size_t{0}; index < bytes.size(); ++index) {
        const auto value = index >= hash_begin && index < hash_end ? std::uint8_t{0} : bytes[index];
        hash ^= value;
        hash *= xpu::shared::SceneSoaFnv1aPrime;
    }
    return hash;
}

[[nodiscard]] std::uint32_t punctual_kind(const ScenePunctualLight& light) noexcept {
    if (std::holds_alternative<ScenePointLight>(light)) {
        return static_cast<std::uint32_t>(xpu::shared::SceneSoaPunctualKind::point);
    }
    if (std::holds_alternative<SceneDirectionalLight>(light)) {
        return static_cast<std::uint32_t>(xpu::shared::SceneSoaPunctualKind::directional);
    }
    return static_cast<std::uint32_t>(xpu::shared::SceneSoaPunctualKind::spot);
}

[[nodiscard]] renderer::Point3 punctual_position(const ScenePunctualLight& light) noexcept {
    if (const auto* point = std::get_if<ScenePointLight>(&light)) {
        return point->position;
    }
    if (const auto* spot = std::get_if<SceneSpotLight>(&light)) {
        return spot->position;
    }
    return {};
}

[[nodiscard]] renderer::Vector3 punctual_position_error(const ScenePunctualLight& light) noexcept {
    if (const auto* point = std::get_if<ScenePointLight>(&light)) {
        return point->absolute_position_error;
    }
    if (const auto* spot = std::get_if<SceneSpotLight>(&light)) {
        return spot->absolute_position_error;
    }
    return {};
}

[[nodiscard]] renderer::Vector3 punctual_direction(const ScenePunctualLight& light) noexcept {
    if (const auto* directional = std::get_if<SceneDirectionalLight>(&light)) {
        return directional->propagation_direction;
    }
    if (const auto* spot = std::get_if<SceneSpotLight>(&light)) {
        return spot->emission_direction;
    }
    return {};
}

[[nodiscard]] float punctual_inner_angle(const ScenePunctualLight& light) noexcept {
    if (const auto* spot = std::get_if<SceneSpotLight>(&light)) {
        return spot->inner_half_angle_radians;
    }
    return 0.0F;
}

[[nodiscard]] float punctual_outer_angle(const ScenePunctualLight& light) noexcept {
    if (const auto* spot = std::get_if<SceneSpotLight>(&light)) {
        return spot->outer_half_angle_radians;
    }
    return 0.0F;
}

[[nodiscard]] renderer::TransportSpectrum
punctual_spectrum(const ScenePunctualLight& light) noexcept {
    if (const auto* point = std::get_if<ScenePointLight>(&light)) {
        return point->spectral_radiant_intensity;
    }
    if (const auto* directional = std::get_if<SceneDirectionalLight>(&light)) {
        return directional->spectral_irradiance;
    }
    return std::get<SceneSpotLight>(light).on_axis_spectral_radiant_intensity;
}

[[nodiscard]] core::Result<SerializedScene>
serialize_scene(const FrameScene& scene, const xpu::cuda::DeviceMemoryBudget budget) try {
    auto serialized = SerializedScene{};
    auto& header = serialized.header;
    const auto objects = scene.objects();
    const auto geometries = scene.geometries();
    const auto materials = scene.materials();
    const auto instances = scene.instances();
    const auto punctual_lights = scene.punctual_lights();
    const auto mesh_area_light_ids = scene.mesh_area_light_instance_ids();
    const auto& environment = scene.spectral_environment();

    header.magic = xpu::shared::SceneSoaMagic;
    header.abi_major = xpu::shared::SceneSoaAbiMajor;
    header.abi_minor = xpu::shared::SceneSoaAbiMinor;
    header.header_size = sizeof(SceneSoaHeader);
    header.column_count = column::count;
    header.hash_algorithm = xpu::shared::SceneSoaHashAlgorithmFnv1a64;
    header.object_count = objects.size();
    header.geometry_count = geometries.size();
    header.material_count = materials.size();
    header.instance_count = instances.size();
    header.punctual_light_count = punctual_lights.size();
    header.mesh_area_light_count = mesh_area_light_ids.size();
    header.environment_count = environment.has_value() ? 1U : 0U;

    for (const auto& geometry : geometries) {
        if (!geometry.mesh) {
            return std::unexpected(scene_soa_error(
                core::StatusCode::internal_error,
                "Closed frame scene contains a geometry without its validated mesh."));
        }
        if (add_overflows(header.vertex_count, geometry.mesh->positions().size()) ||
            add_overflows(header.triangle_count, geometry.mesh->triangles().size())) {
            return std::unexpected(
                scene_soa_error(core::StatusCode::resource_exhausted,
                                "CUDA scene aggregate mesh counts exceed the 64-bit schema."));
        }
        header.vertex_count += geometry.mesh->positions().size();
        header.triangle_count += geometry.mesh->triangles().size();
    }

    if (auto status = prepare_layout(header, serialized.bytes.max_size(), budget); !status) {
        return std::unexpected(std::move(status.error()));
    }

    auto geometry_vertex_offsets = std::vector<std::uint64_t>{};
    auto geometry_triangle_offsets = std::vector<std::uint64_t>{};
    geometry_vertex_offsets.reserve(geometries.size());
    geometry_triangle_offsets.reserve(geometries.size());
    auto vertex_offset = std::uint64_t{0};
    auto triangle_offset = std::uint64_t{0};
    for (const auto& geometry : geometries) {
        geometry_vertex_offsets.push_back(vertex_offset);
        geometry_triangle_offsets.push_back(triangle_offset);
        vertex_offset += geometry.mesh->positions().size();
        triangle_offset += geometry.mesh->triangles().size();
    }

    auto local_transforms = std::vector<const renderer::AffineTransform*>{};
    auto world_transforms = std::vector<const renderer::AffineTransform*>{};
    local_transforms.reserve(instances.size());
    world_transforms.reserve(instances.size());
    for (const auto& instance : instances) {
        const auto local = scene.local_transform(instance.id);
        const auto world = scene.world_transform(instance.id);
        if (!local || !world) {
            return std::unexpected(scene_soa_error(
                core::StatusCode::internal_error,
                "Closed frame scene lost a validated instance transform during serialization."));
        }
        local_transforms.push_back(&local->get());
        world_transforms.push_back(&world->get());
    }

    serialized.bytes.reserve(static_cast<std::size_t>(header.total_size_bytes));
    serialized.bytes.resize(sizeof(SceneSoaHeader), std::uint8_t{0});

#define BLACKFRAME_APPEND_SCENE_COLUMN(type, index, count, visitor)                                \
    if (auto status = append_column<type>(header, serialized.bytes, next_expected_column, index,   \
                                          count, visitor);                                         \
        !status) {                                                                                 \
        return std::unexpected(std::move(status.error()));                                         \
    }

    auto next_expected_column = std::uint32_t{0};

    BLACKFRAME_APPEND_SCENE_COLUMN(std::uint32_t, column::object_id, header.object_count,
                                   [&](auto&& emit) {
                                       for (const auto& object : objects) {
                                           emit(object.id.value);
                                       }
                                   });

    BLACKFRAME_APPEND_SCENE_COLUMN(std::uint32_t, column::geometry_id, header.geometry_count,
                                   [&](auto&& emit) {
                                       for (const auto& geometry : geometries) {
                                           emit(geometry.id.value);
                                       }
                                   });
    BLACKFRAME_APPEND_SCENE_COLUMN(std::uint64_t, column::geometry_vertex_offset,
                                   header.geometry_count, [&](auto&& emit) {
                                       for (const auto value : geometry_vertex_offsets) {
                                           emit(value);
                                       }
                                   });
    BLACKFRAME_APPEND_SCENE_COLUMN(
        std::uint64_t, column::geometry_vertex_count, header.geometry_count, [&](auto&& emit) {
            for (const auto& geometry : geometries) {
                emit(static_cast<std::uint64_t>(geometry.mesh->positions().size()));
            }
        });
    BLACKFRAME_APPEND_SCENE_COLUMN(std::uint64_t, column::geometry_triangle_offset,
                                   header.geometry_count, [&](auto&& emit) {
                                       for (const auto value : geometry_triangle_offsets) {
                                           emit(value);
                                       }
                                   });
    BLACKFRAME_APPEND_SCENE_COLUMN(
        std::uint64_t, column::geometry_triangle_count, header.geometry_count, [&](auto&& emit) {
            for (const auto& geometry : geometries) {
                emit(static_cast<std::uint64_t>(geometry.mesh->triangles().size()));
            }
        });

    for (auto coordinate = std::uint32_t{0}; coordinate < 3U; ++coordinate) {
        BLACKFRAME_APPEND_SCENE_COLUMN(float, column::position_x + coordinate, header.vertex_count,
                                       [&](auto&& emit) {
                                           for (const auto& geometry : geometries) {
                                               for (const auto value : geometry.mesh->positions()) {
                                                   emit(component(value, coordinate));
                                               }
                                           }
                                       });
    }
    for (auto coordinate = std::uint32_t{0}; coordinate < 3U; ++coordinate) {
        BLACKFRAME_APPEND_SCENE_COLUMN(float, column::normal_x + coordinate, header.vertex_count,
                                       [&](auto&& emit) {
                                           for (const auto& geometry : geometries) {
                                               for (const auto value : geometry.mesh->normals()) {
                                                   emit(component(value, coordinate));
                                               }
                                           }
                                       });
    }
    for (auto coordinate = std::uint32_t{0}; coordinate < 2U; ++coordinate) {
        BLACKFRAME_APPEND_SCENE_COLUMN(float, column::texture_coordinate_x + coordinate,
                                       header.vertex_count, [&](auto&& emit) {
                                           for (const auto& geometry : geometries) {
                                               for (const auto value :
                                                    geometry.mesh->texture_coordinates()) {
                                                   emit(component(value, coordinate));
                                               }
                                           }
                                       });
    }
    for (auto vertex = std::uint32_t{0}; vertex < 3U; ++vertex) {
        BLACKFRAME_APPEND_SCENE_COLUMN(std::uint32_t, column::triangle_vertex_0 + vertex,
                                       header.triangle_count, [&](auto&& emit) {
                                           for (const auto& geometry : geometries) {
                                               for (const auto& triangle :
                                                    geometry.mesh->triangles()) {
                                                   emit(triangle.vertices[vertex]);
                                               }
                                           }
                                       });
    }

    BLACKFRAME_APPEND_SCENE_COLUMN(std::uint32_t, column::material_id, header.material_count,
                                   [&](auto&& emit) {
                                       for (const auto& material : materials) {
                                           emit(material.id.value);
                                       }
                                   });
    BLACKFRAME_APPEND_SCENE_COLUMN(
        std::uint8_t, column::material_spectral_present, header.material_count, [&](auto&& emit) {
            for (const auto& material : materials) {
                emit(material.spectral ? std::uint8_t{1} : std::uint8_t{0});
            }
        });
    for (auto lane = std::uint32_t{0}; lane < xpu::shared::SceneSoaSpectrumLaneCount; ++lane) {
        BLACKFRAME_APPEND_SCENE_COLUMN(
            float, column::material_wavelength_nanometers + lane, header.material_count,
            [&](auto&& emit) {
                for (const auto& material : materials) {
                    emit(material.spectral ? material.spectral->wavelengths[lane].nanometers
                                           : 0.0F);
                }
            });
    }
    for (auto lane = std::uint32_t{0}; lane < xpu::shared::SceneSoaSpectrumLaneCount; ++lane) {
        BLACKFRAME_APPEND_SCENE_COLUMN(
            float, column::material_wavelength_pdf + lane, header.material_count, [&](auto&& emit) {
                for (const auto& material : materials) {
                    emit(material.spectral ? material.spectral->wavelengths[lane].probability.value
                                           : 0.0F);
                }
            });
    }
    for (auto lane = std::uint32_t{0}; lane < xpu::shared::SceneSoaSpectrumLaneCount; ++lane) {
        BLACKFRAME_APPEND_SCENE_COLUMN(
            std::uint8_t, column::material_wavelength_measure + lane, header.material_count,
            [&](auto&& emit) {
                for (const auto& material : materials) {
                    emit(material.spectral
                             ? static_cast<std::uint8_t>(
                                   material.spectral->wavelengths[lane].probability.measure)
                             : std::uint8_t{0});
                }
            });
    }
    for (auto lane = std::uint32_t{0}; lane < xpu::shared::SceneSoaSpectrumLaneCount; ++lane) {
        BLACKFRAME_APPEND_SCENE_COLUMN(
            float, column::material_reflectance + lane, header.material_count, [&](auto&& emit) {
                for (const auto& material : materials) {
                    emit(material.spectral ? material.spectral->reflectance[lane] : 0.0F);
                }
            });
    }
    for (auto lane = std::uint32_t{0}; lane < xpu::shared::SceneSoaSpectrumLaneCount; ++lane) {
        BLACKFRAME_APPEND_SCENE_COLUMN(
            float, column::material_emitted_radiance + lane, header.material_count,
            [&](auto&& emit) {
                for (const auto& material : materials) {
                    emit(material.spectral ? material.spectral->emitted_radiance[lane] : 0.0F);
                }
            });
    }

    BLACKFRAME_APPEND_SCENE_COLUMN(std::uint32_t, column::instance_id, header.instance_count,
                                   [&](auto&& emit) {
                                       for (const auto& instance : instances) {
                                           emit(instance.id.value);
                                       }
                                   });
    BLACKFRAME_APPEND_SCENE_COLUMN(
        std::uint8_t, column::instance_parent_present, header.instance_count, [&](auto&& emit) {
            for (const auto& instance : instances) {
                emit(instance.parent ? std::uint8_t{1} : std::uint8_t{0});
            }
        });
    BLACKFRAME_APPEND_SCENE_COLUMN(std::uint32_t, column::instance_parent_id, header.instance_count,
                                   [&](auto&& emit) {
                                       for (const auto& instance : instances) {
                                           emit(instance.parent ? instance.parent->value : 0U);
                                       }
                                   });
    BLACKFRAME_APPEND_SCENE_COLUMN(std::uint32_t, column::instance_object_id, header.instance_count,
                                   [&](auto&& emit) {
                                       for (const auto& instance : instances) {
                                           emit(instance.object.value);
                                       }
                                   });
    BLACKFRAME_APPEND_SCENE_COLUMN(std::uint32_t, column::instance_geometry_id,
                                   header.instance_count, [&](auto&& emit) {
                                       for (const auto& instance : instances) {
                                           emit(instance.geometry.value);
                                       }
                                   });
    BLACKFRAME_APPEND_SCENE_COLUMN(std::uint32_t, column::instance_material_id,
                                   header.instance_count, [&](auto&& emit) {
                                       for (const auto& instance : instances) {
                                           emit(instance.material.value);
                                       }
                                   });
    BLACKFRAME_APPEND_SCENE_COLUMN(std::uint32_t, column::instance_visibility_mask,
                                   header.instance_count, [&](auto&& emit) {
                                       for (const auto& instance : instances) {
                                           emit(instance.visibility_mask);
                                       }
                                   });

    for (auto element = std::uint32_t{0}; element < xpu::shared::SceneSoaMatrixElementCount;
         ++element) {
        BLACKFRAME_APPEND_SCENE_COLUMN(float, column::instance_local_to_parent + element,
                                       header.instance_count, [&](auto&& emit) {
                                           for (const auto& instance : instances) {
                                               emit(instance.local_to_parent.elements[element]);
                                           }
                                       });
    }
    for (auto element = std::uint32_t{0}; element < xpu::shared::SceneSoaMatrixElementCount;
         ++element) {
        BLACKFRAME_APPEND_SCENE_COLUMN(float, column::instance_parent_to_local + element,
                                       header.instance_count, [&](auto&& emit) {
                                           for (const auto* transform : local_transforms) {
                                               emit(transform->inverse_matrix().elements[element]);
                                           }
                                       });
    }
    for (auto element = std::uint32_t{0}; element < xpu::shared::SceneSoaMatrixElementCount;
         ++element) {
        BLACKFRAME_APPEND_SCENE_COLUMN(float, column::instance_local_to_world + element,
                                       header.instance_count, [&](auto&& emit) {
                                           for (const auto* transform : world_transforms) {
                                               emit(transform->matrix().elements[element]);
                                           }
                                       });
    }
    for (auto element = std::uint32_t{0}; element < xpu::shared::SceneSoaMatrixElementCount;
         ++element) {
        BLACKFRAME_APPEND_SCENE_COLUMN(float, column::instance_world_to_local + element,
                                       header.instance_count, [&](auto&& emit) {
                                           for (const auto* transform : world_transforms) {
                                               emit(transform->inverse_matrix().elements[element]);
                                           }
                                       });
    }

    BLACKFRAME_APPEND_SCENE_COLUMN(std::uint32_t, column::punctual_kind,
                                   header.punctual_light_count, [&](auto&& emit) {
                                       for (const auto& light : punctual_lights) {
                                           emit(punctual_kind(light));
                                       }
                                   });
    for (auto coordinate = std::uint32_t{0}; coordinate < 3U; ++coordinate) {
        BLACKFRAME_APPEND_SCENE_COLUMN(float, column::punctual_position_x + coordinate,
                                       header.punctual_light_count, [&](auto&& emit) {
                                           for (const auto& light : punctual_lights) {
                                               emit(
                                                   component(punctual_position(light), coordinate));
                                           }
                                       });
    }
    for (auto coordinate = std::uint32_t{0}; coordinate < 3U; ++coordinate) {
        BLACKFRAME_APPEND_SCENE_COLUMN(
            float, column::punctual_position_error_x + coordinate, header.punctual_light_count,
            [&](auto&& emit) {
                for (const auto& light : punctual_lights) {
                    emit(component(punctual_position_error(light), coordinate));
                }
            });
    }
    for (auto coordinate = std::uint32_t{0}; coordinate < 3U; ++coordinate) {
        BLACKFRAME_APPEND_SCENE_COLUMN(
            float, column::punctual_direction_x + coordinate, header.punctual_light_count,
            [&](auto&& emit) {
                for (const auto& light : punctual_lights) {
                    emit(component(punctual_direction(light), coordinate));
                }
            });
    }
    BLACKFRAME_APPEND_SCENE_COLUMN(float, column::punctual_inner_half_angle,
                                   header.punctual_light_count, [&](auto&& emit) {
                                       for (const auto& light : punctual_lights) {
                                           emit(punctual_inner_angle(light));
                                       }
                                   });
    BLACKFRAME_APPEND_SCENE_COLUMN(float, column::punctual_outer_half_angle,
                                   header.punctual_light_count, [&](auto&& emit) {
                                       for (const auto& light : punctual_lights) {
                                           emit(punctual_outer_angle(light));
                                       }
                                   });
    for (auto lane = std::uint32_t{0}; lane < xpu::shared::SceneSoaSpectrumLaneCount; ++lane) {
        BLACKFRAME_APPEND_SCENE_COLUMN(float, column::punctual_spectrum + lane,
                                       header.punctual_light_count, [&](auto&& emit) {
                                           for (const auto& light : punctual_lights) {
                                               emit(punctual_spectrum(light)[lane]);
                                           }
                                       });
    }

    BLACKFRAME_APPEND_SCENE_COLUMN(std::uint32_t, column::mesh_area_light_instance_id,
                                   header.mesh_area_light_count, [&](auto&& emit) {
                                       for (const auto id : mesh_area_light_ids) {
                                           emit(id.value);
                                       }
                                   });

    for (auto lane = std::uint32_t{0}; lane < xpu::shared::SceneSoaSpectrumLaneCount; ++lane) {
        BLACKFRAME_APPEND_SCENE_COLUMN(float, column::environment_wavelength_nanometers + lane,
                                       header.environment_count, [&](auto&& emit) {
                                           if (environment) {
                                               emit((*environment).wavelengths[lane].nanometers);
                                           }
                                       });
    }
    for (auto lane = std::uint32_t{0}; lane < xpu::shared::SceneSoaSpectrumLaneCount; ++lane) {
        BLACKFRAME_APPEND_SCENE_COLUMN(
            float, column::environment_wavelength_pdf + lane, header.environment_count,
            [&](auto&& emit) {
                if (environment) {
                    emit((*environment).wavelengths[lane].probability.value);
                }
            });
    }
    for (auto lane = std::uint32_t{0}; lane < xpu::shared::SceneSoaSpectrumLaneCount; ++lane) {
        BLACKFRAME_APPEND_SCENE_COLUMN(
            std::uint8_t, column::environment_wavelength_measure + lane, header.environment_count,
            [&](auto&& emit) {
                if (environment) {
                    emit(static_cast<std::uint8_t>(
                        (*environment).wavelengths[lane].probability.measure));
                }
            });
    }
    for (auto lane = std::uint32_t{0}; lane < xpu::shared::SceneSoaSpectrumLaneCount; ++lane) {
        BLACKFRAME_APPEND_SCENE_COLUMN(float, column::environment_radiance + lane,
                                       header.environment_count, [&](auto&& emit) {
                                           if (environment) {
                                               emit((*environment).radiance[lane]);
                                           }
                                       });
    }

#undef BLACKFRAME_APPEND_SCENE_COLUMN

    if (next_expected_column != column::count) {
        return std::unexpected(scene_soa_error(
            core::StatusCode::internal_error,
            "CUDA scene serializer did not emit every column in its frozen schema."));
    }
    if (serialized.bytes.size() != header.total_size_bytes ||
        xpu::shared::validate_scene_soa_header(header) !=
            xpu::shared::SceneSoaHeaderValidationStatus::valid) {
        return std::unexpected(scene_soa_error(
            core::StatusCode::internal_error,
            "CUDA scene serializer produced a header that violates its frozen schema."));
    }
    std::memcpy(serialized.bytes.data(), &header, sizeof(header));
    header.content_hash = normalized_hash(serialized.bytes);
    std::memcpy(serialized.bytes.data(), &header, sizeof(header));
    return serialized;
} catch (const std::bad_alloc&) {
    return std::unexpected(scene_soa_error(core::StatusCode::resource_exhausted,
                                           "CUDA scene serialization exhausted host memory."));
} catch (const std::length_error&) {
    return std::unexpected(
        scene_soa_error(core::StatusCode::resource_exhausted,
                        "CUDA scene serialization exceeded a host container length limit."));
}

} // namespace

CudaSceneSoA::CudaSceneSoA(SceneSoaHeader header,
                           xpu::cuda::DeviceBuffer<std::uint8_t> device_bytes) noexcept
    : header_(header), device_bytes_(std::move(device_bytes)) {}

CudaSceneSoA::CudaSceneSoA(CudaSceneSoA&& other) noexcept
    : header_(std::exchange(other.header_, SceneSoaHeader{})),
      device_bytes_(std::move(other.device_bytes_)) {}

core::Result<CudaSceneSoA> CudaSceneSoA::upload(const FrameScene& scene,
                                                const CudaSceneSoAUploadOptions options) {
    if (options.abi_major != xpu::shared::SceneSoaAbiMajor ||
        options.abi_minor != xpu::shared::SceneSoaAbiMinor) {
        return std::unexpected(
            scene_soa_error(core::StatusCode::incompatible,
                            "Requested CUDA scene SoA ABI version is not supported."));
    }
    auto serialized = serialize_scene(scene, options.device_memory_budget);
    if (!serialized) {
        return std::unexpected(std::move(serialized.error()));
    }
    if (serialized->bytes.size() != serialized->header.total_size_bytes) {
        return std::unexpected(scene_soa_error(
            core::StatusCode::internal_error,
            "CUDA scene serialization byte count diverged from its validated header."));
    }

    auto allocation = xpu::cuda::DeviceBuffer<std::uint8_t>::allocate(serialized->bytes.size(),
                                                                      options.device_memory_budget);
    if (!allocation) {
        return std::unexpected(std::move(allocation.error()));
    }
    auto device_bytes = std::move(*allocation);
    const auto copy_status = cudaMemcpy(device_bytes.data(), serialized->bytes.data(),
                                        serialized->bytes.size(), cudaMemcpyHostToDevice);
    if (copy_status != cudaSuccess) {
        return std::unexpected(cuda_copy_error(copy_status, serialized->bytes.size()));
    }
    return CudaSceneSoA{serialized->header, std::move(device_bytes)};
}

core::Status CudaSceneSoA::close() {
    auto status = device_bytes_.close();
    if (device_bytes_.empty()) {
        header_ = SceneSoaHeader{};
    }
    return status;
}

} // namespace blackframe::engine
