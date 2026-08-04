#include <Blackframe/Backends/GPU/CUDA/SceneSoA.hpp>
#include <Blackframe/XPU/CUDA/SceneSoaHash.hpp>
#include <Blackframe/XPU/Shared/SceneSoaAbi.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace blackframe::engine {
namespace {

namespace column = xpu::shared::scene_soa_column;
using core::StatusCode;
using xpu::shared::SceneSoaHeader;

inline constexpr std::uint64_t FrozenRichSceneHash = 0x11460E65A124D659ULL;

[[nodiscard]] testing::AssertionResult select_test_device() {
    int device_count = 0;
    const auto count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess) {
        return testing::AssertionFailure()
               << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_status);
    }
    if (device_count <= 0) {
        return testing::AssertionFailure() << "No CUDA device is available.";
    }
    const auto select_status = cudaSetDevice(0);
    if (select_status != cudaSuccess) {
        return testing::AssertionFailure()
               << "cudaSetDevice failed: " << cudaGetErrorString(select_status);
    }
    return testing::AssertionSuccess();
}

[[nodiscard]] renderer::Matrix4 affine_matrix(const renderer::Vector3 scale,
                                              const renderer::Vector3 translation) {
    auto matrix = renderer::identity_matrix<renderer::TransportScalar>();
    matrix(0, 0) = scale.x;
    matrix(1, 1) = scale.y;
    matrix(2, 2) = scale.z;
    matrix(0, 1) = -0.0F;
    matrix(0, 3) = translation.x;
    matrix(1, 3) = translation.y;
    matrix(2, 3) = translation.z;
    return matrix;
}

[[nodiscard]] std::shared_ptr<const TriangleMesh> make_mesh() {
    auto mesh = TriangleMesh::create(
        std::vector{
            renderer::Point3{.x = -0.0F, .y = 0.0F, .z = 0.0F},
            renderer::Point3{.x = 1.0F, .y = 0.0F, .z = 0.0F},
            renderer::Point3{.x = 1.0F, .y = 1.0F, .z = 0.0F},
            renderer::Point3{.x = 0.0F, .y = 1.0F, .z = 0.0F},
        },
        std::vector(4, renderer::Normal3{.x = 0.0F, .y = 0.0F, .z = 1.0F}),
        std::vector{
            renderer::Point2{.x = -0.0F, .y = 0.0F},
            renderer::Point2{.x = 1.0F, .y = 0.0F},
            renderer::Point2{.x = 1.0F, .y = 1.0F},
            renderer::Point2{.x = 0.0F, .y = 1.0F},
        },
        std::vector{
            TriangleVertexIndices{.vertices = {0U, 1U, 2U}},
            TriangleVertexIndices{.vertices = {0U, 2U, 3U}},
        });
    if (!mesh) {
        throw std::runtime_error{mesh.error().message};
    }
    return std::make_shared<const TriangleMesh>(std::move(*mesh));
}

[[nodiscard]] renderer::SampledWavelengths make_wavelengths() {
    const auto wavelengths = renderer::sample_uniform_visible_wavelengths(0.375F);
    if (!wavelengths) {
        throw std::runtime_error{wavelengths.error().message};
    }
    return *wavelengths;
}

[[nodiscard]] SceneSpectralMaterial make_material(const renderer::SampledWavelengths wavelengths,
                                                  const renderer::TransportSpectrum reflectance,
                                                  const renderer::TransportSpectrum emission) {
    return SceneSpectralMaterial{
        .wavelengths = wavelengths,
        .reflectance = reflectance,
        .emitted_radiance = emission,
    };
}

[[nodiscard]] FrameSceneDescription make_rich_description(const bool permute_sorted_domains,
                                                          const bool distinct_mesh_allocations) {
    constexpr auto maximum = std::numeric_limits<std::uint32_t>::max();
    const auto wavelengths = make_wavelengths();
    const auto first_mesh = make_mesh();
    const auto second_mesh = distinct_mesh_allocations ? make_mesh() : first_mesh;
    auto description =
        FrameSceneDescription{
            .objects =
                {
                    SceneObject{.id = {.value = maximum}},
                    SceneObject{.id = {.value = 0U}},
                },
            .geometries =
                {
                    SceneGeometry{.id = {.value = maximum}, .mesh = first_mesh},
                    SceneGeometry{.id = {.value = 0U}, .mesh = second_mesh},
                },
            .materials =
                {
                    SceneMaterial{
                        .id = {.value = maximum},
                        .spectral = make_material(
                            wavelengths,
                            renderer::TransportSpectrum{.values = {0.15F, 0.25F, 0.35F, 0.45F}},
                            renderer::TransportSpectrum{.values = {1.0F, 2.0F, 3.0F, 4.0F}}),
                    },
                    SceneMaterial{
                        .id = {.value = 0U},
                        .spectral = make_material(
                            wavelengths,
                            renderer::TransportSpectrum{.values = {0.65F, 0.55F, 0.45F, 0.35F}},
                            renderer::TransportSpectrum{}),
                    },
                },
            .instances =
                {
                    SceneInstance{
                        .id = {.value = 17U},
                        .parent = renderer::InstanceId{.value = maximum},
                        .object = {.value = 0U},
                        .geometry = {.value = 0U},
                        .material = {.value = 0U},
                        .local_to_parent = affine_matrix({.x = 2.0F, .y = 1.5F, .z = 1.0F},
                                                         {.x = 0.0F, .y = 1.0F, .z = 0.0F}),
                        .visibility_mask = 0x00FF00FFU,
                    },
                    SceneInstance{
                        .id = {.value = 0U},
                        .parent = renderer::InstanceId{.value = 17U},
                        .object = {.value = maximum},
                        .geometry = {.value = maximum},
                        .material = {.value = maximum},
                        .local_to_parent = affine_matrix({.x = 1.0F, .y = 1.0F, .z = 0.5F},
                                                         {.x = -2.0F, .y = 0.5F, .z = 3.0F}),
                        .visibility_mask = 0xA5A55A5AU,
                    },
                    SceneInstance{
                        .id = {.value = maximum},
                        .parent = std::nullopt,
                        .object = {.value = maximum},
                        .geometry = {.value = maximum},
                        .material = {.value = maximum},
                        .local_to_parent = affine_matrix({.x = 1.0F, .y = 2.0F, .z = 1.0F},
                                                         {.x = 4.0F, .y = -1.0F, .z = 2.0F}),
                        .visibility_mask = renderer::AllRayVisibility,
                    },
                },
            .punctual_lights =
                {
                    SceneSpotLight{
                        .position = {.x = -2.0F, .y = 1.0F, .z = 4.0F},
                        .absolute_position_error = {.x = 0.03F, .y = 0.02F, .z = 0.01F},
                        .emission_direction = {.x = 0.0F, .y = 0.0F, .z = -1.0F},
                        .inner_half_angle_radians = 0.25F,
                        .outer_half_angle_radians = 0.5F,
                        .on_axis_spectral_radiant_intensity =
                            renderer::TransportSpectrum{.values = {3.0F, 6.0F, 9.0F, 12.0F}},
                    },
                    ScenePointLight{
                        .position = {.x = 1.0F, .y = 2.0F, .z = 3.0F},
                        .absolute_position_error = {.x = 0.01F, .y = 0.02F, .z = 0.03F},
                        .spectral_radiant_intensity =
                            renderer::TransportSpectrum{.values = {1.0F, 2.0F, 3.0F, 4.0F}},
                    },
                    SceneDirectionalLight{
                        .propagation_direction = {.x = 0.0F, .y = 0.0F, .z = -1.0F},
                        .spectral_irradiance =
                            renderer::TransportSpectrum{.values = {2.0F, 4.0F, 6.0F, 8.0F}},
                    },
                },
            .spectral_environment =
                SceneSpectralEnvironment{
                    .wavelengths = wavelengths,
                    .radiance = renderer::TransportSpectrum{.values = {0.1F, 0.2F, 0.3F, 0.4F}},
                },
        };

    if (permute_sorted_domains) {
        std::ranges::reverse(description.objects);
        std::ranges::reverse(description.geometries);
        std::ranges::reverse(description.materials);
        std::ranges::reverse(description.instances);
    }
    return description;
}

[[nodiscard]] FrameSceneHandle make_rich_scene(const bool permute_sorted_domains = false,
                                               const bool distinct_mesh_allocations = false) {
    auto scene = FrameScene::create(
        make_rich_description(permute_sorted_domains, distinct_mesh_allocations));
    if (!scene) {
        throw std::runtime_error{scene.error().message};
    }
    return *scene;
}

[[nodiscard]] std::vector<std::uint8_t> download(const CudaSceneSoA& scene) {
    auto result = std::vector<std::uint8_t>(scene.size_bytes());
    const auto status =
        cudaMemcpy(result.data(), scene.device_data(), result.size(), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        throw std::runtime_error{cudaGetErrorString(status)};
    }
    return result;
}

[[nodiscard]] SceneSoaHeader read_header(const std::span<const std::uint8_t> bytes) {
    if (bytes.size() < sizeof(SceneSoaHeader)) {
        throw std::runtime_error{"Downloaded CUDA scene is smaller than its ABI header."};
    }
    auto header = SceneSoaHeader{};
    std::memcpy(&header, bytes.data(), sizeof(header));
    return header;
}

template <typename Value>
[[nodiscard]] std::vector<Value> read_column(const std::span<const std::uint8_t> bytes,
                                             const SceneSoaHeader& header,
                                             const std::uint32_t index) {
    static_assert(std::is_trivially_copyable_v<Value>);
    if (index >= column::count) {
        throw std::runtime_error{"CUDA scene column index is outside the frozen schema."};
    }
    const auto& descriptor = header.columns[index];
    if (descriptor.element_size != sizeof(Value) ||
        descriptor.element_count > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error{"CUDA scene column type disagrees with its descriptor."};
    }
    auto result = std::vector<Value>(static_cast<std::size_t>(descriptor.element_count));
    if (result.empty()) {
        if (descriptor.offset_bytes != 0U) {
            throw std::runtime_error{"Empty CUDA scene column has a non-zero offset."};
        }
        return result;
    }
    const auto byte_count = result.size() * sizeof(Value);
    if (descriptor.offset_bytes > bytes.size() ||
        byte_count > bytes.size() - static_cast<std::size_t>(descriptor.offset_bytes)) {
        throw std::runtime_error{"CUDA scene column exceeds the downloaded blob."};
    }
    std::memcpy(result.data(), bytes.data() + descriptor.offset_bytes, byte_count);
    return result;
}

template <typename Value>
void expect_column(const std::span<const std::uint8_t> bytes, const SceneSoaHeader& header,
                   const std::uint32_t index, const std::vector<Value>& expected) {
    const auto actual = read_column<Value>(bytes, header, index);
    ASSERT_EQ(actual.size(), expected.size()) << "column " << index;
    for (auto element = std::size_t{0}; element < actual.size(); ++element) {
        SCOPED_TRACE(testing::Message{} << "column " << index << ", element " << element);
        if constexpr (std::is_same_v<Value, float>) {
            EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[element]),
                      std::bit_cast<std::uint32_t>(expected[element]));
        } else {
            EXPECT_EQ(actual[element], expected[element]);
        }
    }
}

template <typename Value, typename Range, typename Projection>
[[nodiscard]] std::vector<Value> projected(const Range& range, Projection&& projection) {
    auto result = std::vector<Value>{};
    result.reserve(range.size());
    for (const auto& value : range) {
        result.push_back(std::forward<Projection>(projection)(value));
    }
    return result;
}

[[nodiscard]] std::uint64_t normalized_hash(const std::span<const std::uint8_t> bytes) {
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

[[nodiscard]] float point_component(const renderer::Point3 value, const std::uint32_t component) {
    return component == 0U ? value.x : component == 1U ? value.y : value.z;
}

[[nodiscard]] float vector_component(const renderer::Vector3 value, const std::uint32_t component) {
    return component == 0U ? value.x : component == 1U ? value.y : value.z;
}

[[nodiscard]] float normal_component(const renderer::Normal3 value, const std::uint32_t component) {
    return component == 0U ? value.x : component == 1U ? value.y : value.z;
}

[[nodiscard]] float point_component(const renderer::Point2 value, const std::uint32_t component) {
    return component == 0U ? value.x : value.y;
}

[[nodiscard]] renderer::Point3 expected_punctual_position(const ScenePunctualLight& light) {
    if (const auto* value = std::get_if<ScenePointLight>(&light)) {
        return value->position;
    }
    if (const auto* value = std::get_if<SceneSpotLight>(&light)) {
        return value->position;
    }
    return {};
}

[[nodiscard]] renderer::Vector3 expected_punctual_position_error(const ScenePunctualLight& light) {
    if (const auto* value = std::get_if<ScenePointLight>(&light)) {
        return value->absolute_position_error;
    }
    if (const auto* value = std::get_if<SceneSpotLight>(&light)) {
        return value->absolute_position_error;
    }
    return {};
}

[[nodiscard]] renderer::Vector3 expected_punctual_direction(const ScenePunctualLight& light) {
    if (const auto* value = std::get_if<SceneDirectionalLight>(&light)) {
        return value->propagation_direction;
    }
    if (const auto* value = std::get_if<SceneSpotLight>(&light)) {
        return value->emission_direction;
    }
    return {};
}

[[nodiscard]] renderer::TransportSpectrum
expected_punctual_spectrum(const ScenePunctualLight& light) {
    if (const auto* value = std::get_if<ScenePointLight>(&light)) {
        return value->spectral_radiant_intensity;
    }
    if (const auto* value = std::get_if<SceneDirectionalLight>(&light)) {
        return value->spectral_irradiance;
    }
    return std::get<SceneSpotLight>(light).on_axis_spectral_radiant_intensity;
}

TEST(CudaSceneSoAAbi, RejectsIncompatibleAndMalformedHeaders) {
    auto header = SceneSoaHeader{};
    header.magic = xpu::shared::SceneSoaMagic;
    header.abi_major = xpu::shared::SceneSoaAbiMajor;
    header.abi_minor = xpu::shared::SceneSoaAbiMinor;
    header.header_size = sizeof(SceneSoaHeader);
    header.column_count = column::count;
    header.hash_algorithm = xpu::shared::SceneSoaHashAlgorithmFnv1a64;
    header.total_size_bytes = sizeof(SceneSoaHeader);
    for (auto index = std::uint32_t{0}; index < column::count; ++index) {
        header.columns[index].element_size = xpu::shared::scene_soa_column_element_size(index);
    }
    EXPECT_EQ(xpu::shared::validate_scene_soa_header(header),
              xpu::shared::SceneSoaHeaderValidationStatus::valid);

    auto invalid = header;
    invalid.abi_major = 2U;
    EXPECT_EQ(xpu::shared::validate_scene_soa_header(invalid),
              xpu::shared::SceneSoaHeaderValidationStatus::incompatible_version);
    invalid = header;
    invalid.environment_count = 2U;
    EXPECT_EQ(xpu::shared::validate_scene_soa_header(invalid),
              xpu::shared::SceneSoaHeaderValidationStatus::invalid_environment_count);
    invalid = header;
    invalid.columns[column::object_id].reserved = 1U;
    EXPECT_EQ(xpu::shared::validate_scene_soa_header(invalid),
              xpu::shared::SceneSoaHeaderValidationStatus::invalid_column_descriptor);

    auto overflowing = header;
    overflowing.object_count = 4'611'686'018'427'387'006ULL;
    overflowing.geometry_count = 1U;
    overflowing.columns[column::object_id] = {
        .offset_bytes = sizeof(SceneSoaHeader),
        .element_count = overflowing.object_count,
        .element_size = sizeof(std::uint32_t),
    };
    overflowing.columns[column::geometry_id] = {
        .offset_bytes = 0U,
        .element_count = 1U,
        .element_size = sizeof(std::uint32_t),
    };
    EXPECT_EQ(xpu::shared::validate_scene_soa_header(overflowing),
              xpu::shared::SceneSoaHeaderValidationStatus::size_overflow);
}

TEST(CudaSceneSoAHash, RejectsInvalidLaunchArgumentsBeforeDispatch) {
    auto output = std::uint64_t{};
    EXPECT_EQ(blackframe_cuda_launch_scene_soa_hash(nullptr, sizeof(SceneSoaHeader), &output),
              static_cast<int>(cudaErrorInvalidValue));
    EXPECT_EQ(blackframe_cuda_launch_scene_soa_hash(reinterpret_cast<const std::uint8_t*>(&output),
                                                    sizeof(output), &output),
              static_cast<int>(cudaErrorInvalidValue));
    EXPECT_EQ(blackframe_cuda_launch_scene_soa_hash(reinterpret_cast<const std::uint8_t*>(&output),
                                                    sizeof(SceneSoaHeader), nullptr),
              static_cast<int>(cudaErrorInvalidValue));
}

TEST(CudaSceneSoA, MapsEveryCpuFieldToDeviceColumnsBitForBit) {
    ASSERT_TRUE(select_test_device());
    const auto cpu_scene = make_rich_scene();
    auto uploaded_result = CudaSceneSoA::upload(*cpu_scene);
    ASSERT_TRUE(uploaded_result) << uploaded_result.error().message;
    auto uploaded = std::move(*uploaded_result);
    const auto bytes = download(uploaded);
    const auto header = read_header(bytes);

    EXPECT_EQ(std::memcmp(&header, &uploaded.header(), sizeof(header)), 0);
    EXPECT_EQ(xpu::shared::validate_scene_soa_header(header),
              xpu::shared::SceneSoaHeaderValidationStatus::valid);
    EXPECT_EQ(header.total_size_bytes, bytes.size());
    EXPECT_EQ(header.content_hash, normalized_hash(bytes));

    const auto objects = cpu_scene->objects();
    const auto geometries = cpu_scene->geometries();
    const auto materials = cpu_scene->materials();
    const auto instances = cpu_scene->instances();
    const auto lights = cpu_scene->punctual_lights();
    const auto area_light_ids = cpu_scene->mesh_area_light_instance_ids();
    const auto& environment = cpu_scene->spectral_environment();

    EXPECT_EQ(header.object_count, objects.size());
    EXPECT_EQ(header.geometry_count, geometries.size());
    EXPECT_EQ(header.material_count, materials.size());
    EXPECT_EQ(header.instance_count, instances.size());
    EXPECT_EQ(header.punctual_light_count, lights.size());
    EXPECT_EQ(header.mesh_area_light_count, area_light_ids.size());
    EXPECT_EQ(header.environment_count, 1U);

    expect_column(
        bytes, header, column::object_id,
        projected<std::uint32_t>(objects, [](const auto& value) { return value.id.value; }));
    expect_column(
        bytes, header, column::geometry_id,
        projected<std::uint32_t>(geometries, [](const auto& value) { return value.id.value; }));

    auto vertex_offsets = std::vector<std::uint64_t>{};
    auto vertex_counts = std::vector<std::uint64_t>{};
    auto triangle_offsets = std::vector<std::uint64_t>{};
    auto triangle_counts = std::vector<std::uint64_t>{};
    auto positions = std::array<std::vector<float>, 3>{};
    auto normals = std::array<std::vector<float>, 3>{};
    auto texture_coordinates = std::array<std::vector<float>, 2>{};
    auto triangle_vertices = std::array<std::vector<std::uint32_t>, 3>{};
    auto vertex_offset = std::uint64_t{0};
    auto triangle_offset = std::uint64_t{0};
    for (const auto& geometry : geometries) {
        vertex_offsets.push_back(vertex_offset);
        triangle_offsets.push_back(triangle_offset);
        vertex_counts.push_back(geometry.mesh->positions().size());
        triangle_counts.push_back(geometry.mesh->triangles().size());
        vertex_offset += geometry.mesh->positions().size();
        triangle_offset += geometry.mesh->triangles().size();
        for (const auto value : geometry.mesh->positions()) {
            for (auto component = std::uint32_t{0}; component < 3U; ++component) {
                positions[component].push_back(point_component(value, component));
            }
        }
        for (const auto value : geometry.mesh->normals()) {
            for (auto component = std::uint32_t{0}; component < 3U; ++component) {
                normals[component].push_back(normal_component(value, component));
            }
        }
        for (const auto value : geometry.mesh->texture_coordinates()) {
            for (auto component = std::uint32_t{0}; component < 2U; ++component) {
                texture_coordinates[component].push_back(point_component(value, component));
            }
        }
        for (const auto& triangle : geometry.mesh->triangles()) {
            for (auto vertex = std::uint32_t{0}; vertex < 3U; ++vertex) {
                triangle_vertices[vertex].push_back(triangle.vertices[vertex]);
            }
        }
    }
    EXPECT_EQ(header.vertex_count, vertex_offset);
    EXPECT_EQ(header.triangle_count, triangle_offset);
    expect_column(bytes, header, column::geometry_vertex_offset, vertex_offsets);
    expect_column(bytes, header, column::geometry_vertex_count, vertex_counts);
    expect_column(bytes, header, column::geometry_triangle_offset, triangle_offsets);
    expect_column(bytes, header, column::geometry_triangle_count, triangle_counts);
    for (auto component = std::uint32_t{0}; component < 3U; ++component) {
        expect_column(bytes, header, column::position_x + component, positions[component]);
        expect_column(bytes, header, column::normal_x + component, normals[component]);
        expect_column(bytes, header, column::triangle_vertex_0 + component,
                      triangle_vertices[component]);
    }
    for (auto component = std::uint32_t{0}; component < 2U; ++component) {
        expect_column(bytes, header, column::texture_coordinate_x + component,
                      texture_coordinates[component]);
    }

    expect_column(
        bytes, header, column::material_id,
        projected<std::uint32_t>(materials, [](const auto& value) { return value.id.value; }));
    expect_column(bytes, header, column::material_spectral_present,
                  projected<std::uint8_t>(materials, [](const auto& value) {
                      return value.spectral ? std::uint8_t{1} : std::uint8_t{0};
                  }));
    for (auto lane = std::uint32_t{0}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        expect_column(bytes, header, column::material_wavelength_nanometers + lane,
                      projected<float>(materials, [lane](const auto& value) {
                          return value.spectral ? value.spectral->wavelengths[lane].nanometers
                                                : 0.0F;
                      }));
        expect_column(bytes, header, column::material_wavelength_pdf + lane,
                      projected<float>(materials, [lane](const auto& value) {
                          return value.spectral
                                     ? value.spectral->wavelengths[lane].probability.value
                                     : 0.0F;
                      }));
        expect_column(bytes, header, column::material_wavelength_measure + lane,
                      projected<std::uint8_t>(materials, [lane](const auto& value) {
                          return value.spectral
                                     ? static_cast<std::uint8_t>(
                                           value.spectral->wavelengths[lane].probability.measure)
                                     : std::uint8_t{0};
                      }));
        expect_column(bytes, header, column::material_reflectance + lane,
                      projected<float>(materials, [lane](const auto& value) {
                          return value.spectral ? value.spectral->reflectance[lane] : 0.0F;
                      }));
        expect_column(bytes, header, column::material_emitted_radiance + lane,
                      projected<float>(materials, [lane](const auto& value) {
                          return value.spectral ? value.spectral->emitted_radiance[lane] : 0.0F;
                      }));
    }

    expect_column(
        bytes, header, column::instance_id,
        projected<std::uint32_t>(instances, [](const auto& value) { return value.id.value; }));
    expect_column(bytes, header, column::instance_parent_present,
                  projected<std::uint8_t>(instances, [](const auto& value) {
                      return value.parent ? std::uint8_t{1} : std::uint8_t{0};
                  }));
    expect_column(bytes, header, column::instance_parent_id,
                  projected<std::uint32_t>(instances, [](const auto& value) {
                      return value.parent ? value.parent->value : 0U;
                  }));
    expect_column(
        bytes, header, column::instance_object_id,
        projected<std::uint32_t>(instances, [](const auto& value) { return value.object.value; }));
    expect_column(bytes, header, column::instance_geometry_id,
                  projected<std::uint32_t>(instances,
                                           [](const auto& value) { return value.geometry.value; }));
    expect_column(bytes, header, column::instance_material_id,
                  projected<std::uint32_t>(instances,
                                           [](const auto& value) { return value.material.value; }));
    expect_column(bytes, header, column::instance_visibility_mask,
                  projected<std::uint32_t>(
                      instances, [](const auto& value) { return value.visibility_mask; }));
    for (auto element = std::uint32_t{0}; element < 16U; ++element) {
        auto local = std::vector<float>{};
        auto local_inverse = std::vector<float>{};
        auto world = std::vector<float>{};
        auto world_inverse = std::vector<float>{};
        for (const auto& instance : instances) {
            const auto local_transform = cpu_scene->local_transform(instance.id);
            const auto world_transform = cpu_scene->world_transform(instance.id);
            ASSERT_TRUE(local_transform);
            ASSERT_TRUE(world_transform);
            local.push_back(instance.local_to_parent.elements[element]);
            local_inverse.push_back(local_transform->get().inverse_matrix().elements[element]);
            world.push_back(world_transform->get().matrix().elements[element]);
            world_inverse.push_back(world_transform->get().inverse_matrix().elements[element]);
        }
        expect_column(bytes, header, column::instance_local_to_parent + element, local);
        expect_column(bytes, header, column::instance_parent_to_local + element, local_inverse);
        expect_column(bytes, header, column::instance_local_to_world + element, world);
        expect_column(bytes, header, column::instance_world_to_local + element, world_inverse);
    }

    expect_column(bytes, header, column::punctual_kind,
                  projected<std::uint32_t>(lights, [](const auto& light) {
                      return static_cast<std::uint32_t>(light.index());
                  }));
    for (auto component = std::uint32_t{0}; component < 3U; ++component) {
        expect_column(bytes, header, column::punctual_position_x + component,
                      projected<float>(lights, [component](const auto& light) {
                          return point_component(expected_punctual_position(light), component);
                      }));
        expect_column(bytes, header, column::punctual_position_error_x + component,
                      projected<float>(lights, [component](const auto& light) {
                          return vector_component(expected_punctual_position_error(light),
                                                  component);
                      }));
        expect_column(bytes, header, column::punctual_direction_x + component,
                      projected<float>(lights, [component](const auto& light) {
                          return vector_component(expected_punctual_direction(light), component);
                      }));
    }
    expect_column(bytes, header, column::punctual_inner_half_angle,
                  projected<float>(lights, [](const auto& light) {
                      const auto* spot = std::get_if<SceneSpotLight>(&light);
                      return spot != nullptr ? spot->inner_half_angle_radians : 0.0F;
                  }));
    expect_column(bytes, header, column::punctual_outer_half_angle,
                  projected<float>(lights, [](const auto& light) {
                      const auto* spot = std::get_if<SceneSpotLight>(&light);
                      return spot != nullptr ? spot->outer_half_angle_radians : 0.0F;
                  }));
    for (auto lane = std::uint32_t{0}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        expect_column(bytes, header, column::punctual_spectrum + lane,
                      projected<float>(lights, [lane](const auto& light) {
                          return expected_punctual_spectrum(light)[lane];
                      }));
    }

    expect_column(
        bytes, header, column::mesh_area_light_instance_id,
        projected<std::uint32_t>(area_light_ids, [](const auto value) { return value.value; }));
    ASSERT_TRUE(environment);
    for (auto lane = std::uint32_t{0}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        expect_column(bytes, header, column::environment_wavelength_nanometers + lane,
                      std::vector{environment->wavelengths[lane].nanometers});
        expect_column(bytes, header, column::environment_wavelength_pdf + lane,
                      std::vector{environment->wavelengths[lane].probability.value});
        expect_column(bytes, header, column::environment_wavelength_measure + lane,
                      std::vector{static_cast<std::uint8_t>(
                          environment->wavelengths[lane].probability.measure)});
        expect_column(bytes, header, column::environment_radiance + lane,
                      std::vector{environment->radiance[lane]});
    }
}

TEST(CudaSceneSoA, DeviceHashMatchesHostHashAndFrozenValue) {
    ASSERT_TRUE(select_test_device());
    const auto cpu_scene = make_rich_scene();
    auto uploaded_result = CudaSceneSoA::upload(*cpu_scene);
    ASSERT_TRUE(uploaded_result) << uploaded_result.error().message;
    auto uploaded = std::move(*uploaded_result);

    auto output_result = xpu::cuda::DeviceBuffer<std::uint64_t>::allocate(1U);
    ASSERT_TRUE(output_result) << output_result.error().message;
    auto output = std::move(*output_result);
    const auto launch_status = blackframe_cuda_launch_scene_soa_hash(
        uploaded.device_data(), uploaded.size_bytes(), output.data());
    ASSERT_EQ(launch_status, static_cast<int>(cudaSuccess))
        << cudaGetErrorString(static_cast<cudaError_t>(launch_status));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto device_hash = std::uint64_t{0};
    ASSERT_EQ(cudaMemcpy(&device_hash, output.data(), sizeof(device_hash), cudaMemcpyDeviceToHost),
              cudaSuccess);
    const auto bytes = download(uploaded);
    EXPECT_EQ(device_hash, normalized_hash(bytes));
    EXPECT_EQ(device_hash, uploaded.header().content_hash);
    EXPECT_EQ(device_hash, FrozenRichSceneHash);
}

TEST(CudaSceneSoA, CanonicalBytesIgnoreInsertionAndAllocationIdentity) {
    ASSERT_TRUE(select_test_device());
    const auto first_scene = make_rich_scene(false, false);
    const auto second_scene = make_rich_scene(true, true);
    auto first_result = CudaSceneSoA::upload(*first_scene);
    auto second_result = CudaSceneSoA::upload(*second_scene);
    ASSERT_TRUE(first_result) << first_result.error().message;
    ASSERT_TRUE(second_result) << second_result.error().message;
    auto first = std::move(*first_result);
    auto second = std::move(*second_result);

    EXPECT_EQ(first.header().content_hash, second.header().content_hash);
    EXPECT_EQ(download(first), download(second));
}

TEST(CudaSceneSoA, PunctualRegistryOrderRemainsSemantic) {
    ASSERT_TRUE(select_test_device());
    auto first_description = make_rich_description(false, false);
    auto second_description = make_rich_description(true, true);
    std::ranges::reverse(second_description.punctual_lights);
    const auto first_scene_result = FrameScene::create(std::move(first_description));
    const auto second_scene_result = FrameScene::create(std::move(second_description));
    ASSERT_TRUE(first_scene_result) << first_scene_result.error().message;
    ASSERT_TRUE(second_scene_result) << second_scene_result.error().message;
    auto first_result = CudaSceneSoA::upload(**first_scene_result);
    auto second_result = CudaSceneSoA::upload(**second_scene_result);
    ASSERT_TRUE(first_result) << first_result.error().message;
    ASSERT_TRUE(second_result) << second_result.error().message;
    EXPECT_NE(first_result->header().content_hash, second_result->header().content_hash);
}

TEST(CudaSceneSoA, SerializesAnEmptySceneAndRejectsAnInsufficientBudget) {
    ASSERT_TRUE(select_test_device());
    const auto empty_scene_result = FrameScene::create(FrameSceneDescription{});
    ASSERT_TRUE(empty_scene_result) << empty_scene_result.error().message;
    auto empty_upload_result = CudaSceneSoA::upload(**empty_scene_result);
    ASSERT_TRUE(empty_upload_result) << empty_upload_result.error().message;
    auto empty_upload = std::move(*empty_upload_result);
    EXPECT_EQ(empty_upload.size_bytes(), sizeof(SceneSoaHeader));
    EXPECT_EQ(empty_upload.header().object_count, 0U);
    EXPECT_EQ(xpu::shared::validate_scene_soa_header(empty_upload.header()),
              xpu::shared::SceneSoaHeaderValidationStatus::valid);
    for (const auto& descriptor : empty_upload.header().columns) {
        EXPECT_EQ(descriptor.offset_bytes, 0U);
        EXPECT_EQ(descriptor.element_count, 0U);
        EXPECT_NE(descriptor.element_size, 0U);
    }

    const auto rich_scene = make_rich_scene();
    const auto rejected = CudaSceneSoA::upload(
        *rich_scene,
        CudaSceneSoAUploadOptions{
            .device_memory_budget =
                xpu::cuda::DeviceMemoryBudget{.maximum_bytes = sizeof(SceneSoaHeader) - 1U},
        });
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, StatusCode::resource_exhausted);
    EXPECT_NE(rejected.error().message.find("explicit device-memory budget"), std::string::npos);

    const auto incompatible =
        CudaSceneSoA::upload(*rich_scene, CudaSceneSoAUploadOptions{.abi_major = 2U});
    ASSERT_FALSE(incompatible);
    EXPECT_EQ(incompatible.error().code, StatusCode::incompatible);
}

TEST(CudaSceneSoA, EncodesAbsentSpectralDataWithoutInventingAnEnvironment) {
    ASSERT_TRUE(select_test_device());
    constexpr auto maximum = std::numeric_limits<std::uint32_t>::max();
    auto description = FrameSceneDescription{
        .objects = {SceneObject{.id = {.value = 0U}}},
        .geometries = {SceneGeometry{.id = {.value = maximum}, .mesh = make_mesh()}},
        .materials = {SceneMaterial{.id = {.value = 0U}, .spectral = std::nullopt}},
        .instances = {SceneInstance{
            .id = {.value = maximum},
            .parent = std::nullopt,
            .object = {.value = 0U},
            .geometry = {.value = maximum},
            .material = {.value = 0U},
            .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
        }},
    };
    const auto cpu_scene_result = FrameScene::create(std::move(description));
    ASSERT_TRUE(cpu_scene_result) << cpu_scene_result.error().message;
    auto upload_result = CudaSceneSoA::upload(**cpu_scene_result);
    ASSERT_TRUE(upload_result) << upload_result.error().message;
    auto upload = std::move(*upload_result);
    const auto bytes = download(upload);
    const auto header = read_header(bytes);

    EXPECT_EQ(header.environment_count, 0U);
    EXPECT_EQ(header.punctual_light_count, 0U);
    EXPECT_EQ(header.mesh_area_light_count, 0U);
    expect_column(bytes, header, column::material_spectral_present, std::vector<std::uint8_t>{0U});
    for (auto lane = std::uint32_t{0}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        expect_column(bytes, header, column::material_wavelength_nanometers + lane,
                      std::vector{0.0F});
        expect_column(bytes, header, column::material_wavelength_pdf + lane, std::vector{0.0F});
        expect_column(bytes, header, column::material_wavelength_measure + lane,
                      std::vector<std::uint8_t>{0U});
        expect_column(bytes, header, column::material_reflectance + lane, std::vector{0.0F});
        expect_column(bytes, header, column::material_emitted_radiance + lane, std::vector{0.0F});
        EXPECT_TRUE(
            read_column<float>(bytes, header, column::environment_wavelength_nanometers + lane)
                .empty());
        EXPECT_TRUE(read_column<float>(bytes, header, column::environment_radiance + lane).empty());
    }
}

TEST(CudaSceneSoA, MoveTransfersTheDeviceSnapshotExactlyOnce) {
    ASSERT_TRUE(select_test_device());
    const auto cpu_scene = make_rich_scene();
    auto uploaded_result = CudaSceneSoA::upload(*cpu_scene);
    ASSERT_TRUE(uploaded_result) << uploaded_result.error().message;
    auto source = std::move(*uploaded_result);
    const auto expected_hash = source.header().content_hash;
    CudaSceneSoA moved{std::move(source)};

    EXPECT_FALSE(source);
    EXPECT_TRUE(source.empty());
    EXPECT_EQ(source.header().magic, 0U);
    EXPECT_TRUE(moved);
    EXPECT_EQ(moved.header().content_hash, expected_hash);
    const auto close_status = moved.close();
    ASSERT_TRUE(close_status) << close_status.error().message;
    EXPECT_FALSE(moved);
    EXPECT_EQ(moved.header().magic, 0U);
}

} // namespace
} // namespace blackframe::engine
