#include <Blackframe/Engine/SceneDescriptionJson.hpp>
#include <Blackframe/Renderer/HostImage.hpp>
#include <Blackframe/Renderer/HostImageMipChain.hpp>
#include <Blackframe/Renderer/MatrixOperations.hpp>
#include <Blackframe/Renderer/WavelengthSampling.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

constexpr std::string_view MinimalSceneJson = R"json({
  "schema": "blackframe.scene",
  "schema_version": 1,
  "active": {"film": 1, "camera": 2, "render_options": 3},
  "films": [
    {"id": 1, "extent": [8, 4], "crop": [0, 0, 8, 4], "accumulation": "float32"}
  ],
  "cameras": [
    {"id": 2, "film": 1, "model": {
      "type": "pinhole",
      "origin": [0, 0, 0],
      "normal": [0, 0, 1],
      "tangent": [1, 0, 0],
      "bitangent": [0, 1, 0],
      "vertical_fov_radians": 1,
      "t_min": 0,
      "t_max": 100,
      "visibility_mask": 4294967295,
      "current_medium": 0
    }}
  ],
  "render_options": [
    {
      "id": 3,
      "film": 1,
      "samples_per_pixel": 2,
      "maximum_path_depth": 4,
      "tile_edge_length": 4,
      "seed": 18446744073709551615,
      "pixel_jitter": "center",
      "mis_heuristic": "power",
      "light_sampling_strategy": "uniform",
      "depth_limits": {
        "diffuse": 4,
        "glossy": 4,
        "specular": 4,
        "transmission": 4,
        "volume": 4
      },
      "russian_roulette": {"mode": "disabled"}
    }
  ],
  "constant_textures": [],
  "host_image_textures": [],
  "objects": [],
  "geometries": [],
  "materials": [],
  "instances": [],
  "lights": []
})json";

constexpr std::string_view CanonicalMinimalSceneJson =
    R"json({"schema":"blackframe.scene","schema_version":1,)json"
    R"json("active":{"film":1,"camera":2,"render_options":3},)json"
    R"json("films":[{"id":1,"extent":[8,4],"crop":[0,0,8,4],"accumulation":"float32"}],)json"
    R"json("cameras":[{"id":2,"film":1,"model":{"type":"pinhole","origin":[0,0,0],)json"
    R"json("normal":[0,0,1],"tangent":[1,0,0],"bitangent":[0,1,0],)json"
    R"json("vertical_fov_radians":1,"t_min":0,"t_max":100,"visibility_mask":4294967295,)json"
    R"json("current_medium":0}}],"render_options":[{"id":3,"film":1,"samples_per_pixel":2,)json"
    R"json("maximum_path_depth":4,"tile_edge_length":4,"seed":18446744073709551615,)json"
    R"json("pixel_jitter":"center","mis_heuristic":"power","light_sampling_strategy":"uniform",)json"
    R"json("depth_limits":{"diffuse":4,"glossy":4,"specular":4,"transmission":4,"volume":4},)json"
    R"json("russian_roulette":{"mode":"disabled"}}],"constant_textures":[],)json"
    R"json("host_image_textures":[],"objects":[],"geometries":[],"materials":[],)json"
    R"json("instances":[],"lights":[]})json";

[[nodiscard]] std::filesystem::path snapshot_source_path() {
#if defined(_WIN32)
    return std::filesystem::path{R"(C:\Blackframe\fixtures\surface-data.exr)"};
#else
    return std::filesystem::path{"/Blackframe/fixtures/surface-data.exr"};
#endif
}

template <typename Result> [[nodiscard]] auto take_or_throw(Result&& result) {
    if (!result) {
        throw std::runtime_error{result.error().message};
    }
    return std::move(*result);
}

[[nodiscard]] renderer::TransportSpectrum spectrum(const float value) {
    auto result = renderer::TransportSpectrum{};
    result.values.fill(value);
    return result;
}

[[nodiscard]] renderer::OrthonormalFrame camera_frame() {
    return take_or_throw(renderer::OrthonormalFrame::from_normal_and_tangent(
        renderer::Normal3{.x = 0.2F, .y = 0.3F, .z = 1.0F},
        renderer::Vector3{.x = 1.0F, .y = -0.1F, .z = -0.17F}));
}

[[nodiscard]] std::shared_ptr<const TriangleMesh> two_triangle_mesh() {
    auto mesh = TriangleMesh::create(
        {
            renderer::Point3{},
            renderer::Point3{.x = 1.0F},
            renderer::Point3{.x = 1.0F, .y = 1.0F},
            renderer::Point3{.y = 1.0F},
        },
        std::vector(4U, renderer::Normal3{.z = 1.0F}),
        {
            renderer::Point2{},
            renderer::Point2{.x = 1.0F},
            renderer::Point2{.x = 1.0F, .y = 1.0F},
            renderer::Point2{.y = 1.0F},
        },
        {
            TriangleVertexIndices{.vertices = {0U, 1U, 2U}},
            TriangleVertexIndices{.vertices = {0U, 2U, 3U}},
        });
    return std::make_shared<const TriangleMesh>(take_or_throw(std::move(mesh)));
}

[[nodiscard]] renderer::HostImageMipChainHandle surface_image() {
    auto pixels = std::vector<renderer::TransportScalar>{
        0.25F, 0.50F, 1.00F, 0.10F, 0.50F, 0.50F, 1.00F, 0.20F,
        0.75F, 0.50F, 1.00F, 0.30F, 1.00F, 0.50F, 1.00F, 0.40F,
    };
    auto image = renderer::HostImage::create_snapshot(renderer::HostImageSnapshotDescription{
        .source_path = snapshot_source_path(),
        .format_name = "openexr",
        .source_color_space = renderer::TextureColorSpace::data,
        .storage_color_space = renderer::TextureColorSpace::data,
        .origin_x = -2,
        .origin_y = 3,
        .width = 2U,
        .height = 2U,
        .channel_names = {"R", "G", "B", "height"},
        .pixels = std::move(pixels),
    });
    return take_or_throw(renderer::HostImageMipChain::generate(take_or_throw(std::move(image))));
}

[[nodiscard]] renderer::HostImageMipChainHandle single_texel_surface_image() {
    auto image = renderer::HostImage::create_snapshot(renderer::HostImageSnapshotDescription{
        .source_path = snapshot_source_path(),
        .format_name = "openexr",
        .source_color_space = renderer::TextureColorSpace::data,
        .storage_color_space = renderer::TextureColorSpace::data,
        .width = 1U,
        .height = 1U,
        .channel_names = {"R", "G", "B", "height"},
        .pixels = {0.5F, 0.5F, 1.0F, 0.25F},
    });
    return take_or_throw(renderer::HostImageMipChain::generate(take_or_throw(std::move(image))));
}

[[nodiscard]] SceneClosureMixture closure_mixture(const renderer::ClosureKind kind) {
    auto closures = renderer::ClosureSet{};
    auto status = renderer::ClosureAppendStatus::invalid_payload;
    const auto weight = spectrum(0.5F);
    switch (kind) {
    case renderer::ClosureKind::lambertian_reflection:
        status = closures.append_lambertian_reflection(weight);
        break;
    case renderer::ClosureKind::rough_diffuse_reflection:
        status = closures.append_rough_diffuse_reflection(weight, 0.35F);
        break;
    case renderer::ClosureKind::rough_conductor_reflection:
        status = closures.append_rough_conductor_reflection(weight, spectrum(1.5F), spectrum(2.0F),
                                                            0.2F, 0.4F);
        break;
    case renderer::ClosureKind::rough_dielectric:
        status = closures.append_rough_dielectric(weight, 1.0F, 1.5F, 0.15F, 0.45F);
        break;
    case renderer::ClosureKind::specular_reflection:
        status = closures.append_specular_reflection(weight);
        break;
    case renderer::ClosureKind::specular_transmission:
        status = closures.append_specular_transmission(weight, 1.0F, 1.5F);
        break;
    case renderer::ClosureKind::none:
        break;
    }
    if (status != renderer::ClosureAppendStatus::appended) {
        throw std::runtime_error{"Cannot construct the scene JSON closure fixture."};
    }
    const auto probabilities = std::array<renderer::TransportScalar, 1>{1.0F};
    return take_or_throw(SceneClosureMixture::create(
        closures, probabilities, SceneClosureFrameMode::surface_tangent, 0.125F));
}

[[nodiscard]] SceneConstantTexture constant_float(const renderer::TextureId id) {
    return SceneConstantTexture{
        .id = id,
        .texture = take_or_throw(renderer::ConstantFloatTexture::create(-0.0F)),
    };
}

[[nodiscard]] SceneConstantTexture constant_color(const renderer::TextureId id) {
    return SceneConstantTexture{
        .id = id,
        .texture = take_or_throw(renderer::ConstantColorTexture::create(
            renderer::LinearRGB{.red = 0.25F, .green = 0.5F, .blue = 0.75F})),
    };
}

[[nodiscard]] SceneConstantTexture constant_spectrum(const renderer::TextureId id) {
    return SceneConstantTexture{
        .id = id,
        .texture = take_or_throw(renderer::ConstantSpectrumTexture::create(
            renderer::TransportSpectrum{.values = {0.1F, 0.2F, 0.3F, 0.4F}})),
    };
}

[[nodiscard]] SceneDescriptionInput complete_input(const bool infinite_camera_range = false) {
    constexpr auto active_film = renderer::FilmId{.value = 2U};
    constexpr auto secondary_film = renderer::FilmId{.value = 1U};
    constexpr auto active_camera = renderer::CameraId{.value = 4U};
    constexpr auto secondary_camera = renderer::CameraId{.value = 3U};
    constexpr auto active_options = renderer::RenderOptionsId{.value = 6U};
    constexpr auto secondary_options = renderer::RenderOptionsId{.value = 5U};
    constexpr auto host_texture = renderer::TextureId{.value = 13U};
    constexpr auto first_object = renderer::ObjectId{.value = 20U};
    constexpr auto second_object = renderer::ObjectId{.value = 21U};
    constexpr auto geometry = renderer::GeometryId{.value = 30U};
    constexpr auto root_instance = renderer::InstanceId{.value = 50U};
    constexpr auto child_instance = renderer::InstanceId{.value = 51U};

    const auto wavelengths = take_or_throw(renderer::sample_uniform_visible_wavelengths(0.375F));
    const auto frame = camera_frame();
    const auto mesh = two_triangle_mesh();
    const auto image = surface_image();
    const auto roulette =
        take_or_throw(renderer::RussianRoulettePolicy::create_enabled(3U, 0.1F, 0.9F));

    constexpr auto kinds = std::array{
        renderer::ClosureKind::lambertian_reflection,
        renderer::ClosureKind::rough_diffuse_reflection,
        renderer::ClosureKind::rough_conductor_reflection,
        renderer::ClosureKind::rough_dielectric,
        renderer::ClosureKind::specular_reflection,
        renderer::ClosureKind::specular_transmission,
    };
    auto materials = std::vector<SceneMaterial>{};
    materials.reserve(kinds.size());
    for (auto index = std::size_t{}; index < kinds.size(); ++index) {
        auto material = SceneSpectralMaterial{
            .wavelengths = wavelengths,
            .closure_mixture = closure_mixture(kinds[index]),
            .emitted_radiance = index == 0U ? spectrum(0.025F) : spectrum(0.0F),
        };
        if (index == 0U) {
            material.normal_map = SceneNormalMapBinding{
                .texture = host_texture,
                .red_channel = 0U,
                .green_channel = 1U,
                .blue_channel = 2U,
                .y_convention = renderer::TangentSpaceNormalYConvention::negative_v,
                .u_wrap = renderer::TextureWrapMode::mirror,
                .v_wrap = renderer::TextureWrapMode::clamp,
                .ewa_limits = {.maximum_anisotropy = 8U, .maximum_texel_visits = 1'024U},
            };
            material.bump_map = SceneBumpMapBinding{
                .texture = host_texture,
                .channel = 3U,
                .scale = -0.125F,
                .u_wrap = renderer::TextureWrapMode::black,
                .v_wrap = renderer::TextureWrapMode::repeat,
                .ewa_limits = {.maximum_anisotropy = 4U, .maximum_texel_visits = 512U},
            };
        }
        materials.push_back(SceneMaterial{
            .id = renderer::MaterialId{.value = static_cast<std::uint32_t>(40U + index)},
            .spectral = material,
        });
    }

    return SceneDescriptionInput{
        .active_film = active_film,
        .active_camera = active_camera,
        .active_render_options = active_options,
        .films =
            {
                SceneFilmDescription{
                    .id = active_film,
                    .extent = {.width = 32U, .height = 16U},
                    .crop = {.minimum_x = 2U, .minimum_y = 1U, .maximum_x = 30U, .maximum_y = 15U},
                    .accumulation_precision = renderer::AccumulationPrecision::float64,
                },
                SceneFilmDescription{
                    .id = secondary_film,
                    .extent = {.width = 8U, .height = 4U},
                    .crop = {.maximum_x = 8U, .maximum_y = 4U},
                    .accumulation_precision = renderer::AccumulationPrecision::float32,
                },
            },
        .cameras =
            {
                SceneCameraDescription{
                    .id = active_camera,
                    .film = active_film,
                    .model =
                        ScenePinholeCameraDescription{
                            .origin = {.x = 1.0F, .y = 2.0F, .z = -3.0F},
                            .orientation = frame,
                            .vertical_field_of_view_radians = 0.9F,
                            .t_min = 0.01F,
                            .t_max =
                                infinite_camera_range
                                    ? std::numeric_limits<renderer::TransportScalar>::infinity()
                                    : 1'000.0F,
                            .visibility_mask = 0x00FF'00FFU,
                            .current_medium = renderer::VacuumMedium,
                        },
                },
                SceneCameraDescription{
                    .id = secondary_camera,
                    .film = secondary_film,
                    .model =
                        ScenePinholeCameraDescription{
                            .origin = {},
                            .orientation = frame,
                            .vertical_field_of_view_radians = 1.0F,
                            .t_min = 0.0F,
                            .t_max = 10.0F,
                        },
                },
            },
        .render_options =
            {
                SceneRenderOptionsDescription{
                    .id = active_options,
                    .film = active_film,
                    .options =
                        SceneRenderOptions{
                            .samples_per_pixel = 8U,
                            .maximum_path_depth = 8U,
                            .tile_edge_length = 8U,
                            .seed = std::numeric_limits<std::uint64_t>::max(),
                            .pixel_jitter = renderer::PixelJitterMode::center,
                            .mis_heuristic = renderer::MisHeuristic::balance,
                            .light_sampling_strategy =
                                renderer::LightSamplingStrategy::spatial_tree,
                            .depth_limits = {.diffuse = 2U,
                                             .glossy = 3U,
                                             .specular = 4U,
                                             .transmission = 5U,
                                             .volume = 1U},
                            .roulette_policy = roulette,
                        },
                },
                SceneRenderOptionsDescription{
                    .id = secondary_options,
                    .film = secondary_film,
                    .options = {},
                },
            },
        .constant_textures =
            {
                constant_spectrum(renderer::TextureId{.value = 12U}),
                constant_float(renderer::TextureId{.value = 10U}),
                constant_color(renderer::TextureId{.value = 11U}),
            },
        .host_image_textures = {{.id = host_texture, .image = image}},
        .objects = {{.id = second_object}, {.id = first_object}},
        .geometries = {{.id = geometry, .mesh = mesh}},
        .materials = std::move(materials),
        .instances =
            {
                SceneInstance{
                    .id = child_instance,
                    .parent = root_instance,
                    .object = second_object,
                    .geometry = geometry,
                    .material = renderer::MaterialId{.value = 41U},
                    .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
                    .visibility_mask = 0x0000'FFFFU,
                },
                SceneInstance{
                    .id = root_instance,
                    .parent = std::nullopt,
                    .object = first_object,
                    .geometry = geometry,
                    .material = renderer::MaterialId{.value = 40U},
                    .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
                },
            },
        .lights =
            {
                SceneLightDescription{
                    .id = renderer::LightId{.value = 63U},
                    .light =
                        SceneSpectralEnvironment{
                            .wavelengths = wavelengths,
                            .radiance = spectrum(0.1F),
                        },
                },
                SceneLightDescription{
                    .id = renderer::LightId{.value = 60U},
                    .light =
                        ScenePointLight{
                            .position = {.x = 1.0F, .y = 2.0F, .z = 3.0F},
                            .absolute_position_error = {.x = 0.01F, .y = 0.02F, .z = 0.03F},
                            .spectral_radiant_intensity = spectrum(2.0F),
                        },
                },
                SceneLightDescription{
                    .id = renderer::LightId{.value = 61U},
                    .light =
                        SceneDirectionalLight{
                            .propagation_direction = {.x = 0.0F, .y = -1.0F, .z = 0.0F},
                            .spectral_irradiance = spectrum(0.75F),
                        },
                },
                SceneLightDescription{
                    .id = renderer::LightId{.value = 62U},
                    .light =
                        SceneSpotLight{
                            .position = {.x = -1.0F, .y = 2.0F, .z = 1.0F},
                            .absolute_position_error = {.x = 0.01F, .y = 0.01F, .z = 0.01F},
                            .emission_direction = {.x = 0.0F, .y = -1.0F, .z = 0.0F},
                            .inner_half_angle_radians = 0.25F,
                            .outer_half_angle_radians = 0.5F,
                            .on_axis_spectral_radiant_intensity = spectrum(3.0F),
                        },
                },
            },
    };
}

[[nodiscard]] SceneDescription complete_scene(const bool infinite_camera_range = false) {
    return take_or_throw(SceneDescription::create(complete_input(infinite_camera_range)));
}

[[nodiscard]] std::string replace_once(std::string source, const std::string_view needle,
                                       const std::string_view replacement) {
    const auto offset = source.find(needle);
    if (offset == std::string::npos) {
        throw std::runtime_error{"The scene JSON test mutation anchor was not found."};
    }
    source.replace(offset, needle.size(), replacement);
    return source;
}

[[nodiscard]] std::string replace_scalar_after(std::string source, const std::string_view section,
                                               const std::string_view key,
                                               const std::string_view replacement) {
    const auto section_offset = source.find(section);
    if (section_offset == std::string::npos) {
        throw std::runtime_error{"The scene JSON section mutation anchor was not found."};
    }
    const auto key_offset = source.find(key, section_offset);
    if (key_offset == std::string::npos) {
        throw std::runtime_error{"The scene JSON key mutation anchor was not found."};
    }
    auto value_begin = source.find(':', key_offset + key.size());
    if (value_begin == std::string::npos) {
        throw std::runtime_error{"The scene JSON mutation has no value separator."};
    }
    ++value_begin;
    while (value_begin < source.size() &&
           (source[value_begin] == ' ' || source[value_begin] == '\t' ||
            source[value_begin] == '\r' || source[value_begin] == '\n')) {
        ++value_begin;
    }
    auto value_end = value_begin;
    while (value_end < source.size() && source[value_end] != ',' && source[value_end] != '}' &&
           source[value_end] != ']') {
        ++value_end;
    }
    source.replace(value_begin, value_end - value_begin, replacement);
    return source;
}

void expect_failure(const core::Result<SceneDescription>& result, const core::StatusCode code) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, code) << result.error().message;
    EXPECT_FALSE(result.error().message.empty());
}

void expect_write_failure(const core::Result<std::string>& result, const core::StatusCode code) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, code) << result.error().message;
    EXPECT_FALSE(result.error().message.empty());
}

template <typename Value>
void expect_dynamic_limit_boundary(const SceneDescription& scene, const std::string_view encoded,
                                   Value SceneDescriptionJsonLimits::* const member) {
    const auto find_minimum = [&](const bool writing) {
        auto lower = Value{1};
        auto upper = SceneDescriptionJsonLimits{}.*member;
        while (lower < upper) {
            const auto middle = static_cast<Value>(lower + (upper - lower) / Value{2});
            auto limits = SceneDescriptionJsonLimits{};
            limits.*member = middle;
            const auto succeeds =
                writing ? serialize_scene_description_json(scene, limits).has_value()
                        : deserialize_scene_description_json(encoded, limits).has_value();
            if (succeeds) {
                upper = middle;
            } else {
                lower = static_cast<Value>(middle + Value{1});
            }
        }
        return lower;
    };

    for (const auto writing : {false, true}) {
        const auto minimum = find_minimum(writing);
        ASSERT_GT(minimum, Value{1});
        auto limits = SceneDescriptionJsonLimits{};
        limits.*member = minimum;
        if (writing) {
            ASSERT_TRUE(serialize_scene_description_json(scene, limits).has_value());
            limits.*member = static_cast<Value>(minimum - Value{1});
            expect_write_failure(serialize_scene_description_json(scene, limits),
                                 core::StatusCode::resource_exhausted);
        } else {
            ASSERT_TRUE(deserialize_scene_description_json(encoded, limits).has_value());
            limits.*member = static_cast<Value>(minimum - Value{1});
            expect_failure(deserialize_scene_description_json(encoded, limits),
                           core::StatusCode::resource_exhausted);
        }
    }
}

template <typename Value>
void expect_equal_span(const std::span<const Value> left, const std::span<const Value> right) {
    ASSERT_EQ(left.size(), right.size());
    EXPECT_TRUE(std::ranges::equal(left, right));
}

void expect_meshes_equal(const SceneDescription& expected, const SceneDescription& actual) {
    ASSERT_EQ(expected.geometries().size(), actual.geometries().size());
    for (auto index = std::size_t{}; index < expected.geometries().size(); ++index) {
        const auto& left = expected.geometries()[index];
        const auto& right = actual.geometries()[index];
        ASSERT_NE(left.mesh, nullptr);
        ASSERT_NE(right.mesh, nullptr);
        EXPECT_EQ(left.id, right.id);
        expect_equal_span(left.mesh->positions(), right.mesh->positions());
        expect_equal_span(left.mesh->normals(), right.mesh->normals());
        expect_equal_span(left.mesh->texture_coordinates(), right.mesh->texture_coordinates());
        expect_equal_span(left.mesh->triangles(), right.mesh->triangles());
    }
}

void expect_host_images_equal(const SceneDescription& expected, const SceneDescription& actual) {
    ASSERT_EQ(expected.host_image_textures().size(), actual.host_image_textures().size());
    for (auto index = std::size_t{}; index < expected.host_image_textures().size(); ++index) {
        const auto& left = expected.host_image_textures()[index];
        const auto& right = actual.host_image_textures()[index];
        ASSERT_NE(left.image, nullptr);
        ASSERT_NE(right.image, nullptr);
        EXPECT_EQ(left.id, right.id);
        const auto left_source = left.image->source_image();
        const auto right_source = right.image->source_image();
        ASSERT_NE(left_source, nullptr);
        ASSERT_NE(right_source, nullptr);
        EXPECT_EQ(left_source->source_path(), right_source->source_path());
        EXPECT_EQ(left_source->format_name(), right_source->format_name());
        EXPECT_EQ(left_source->source_color_space(), right_source->source_color_space());
        EXPECT_EQ(left_source->storage_color_space(), right_source->storage_color_space());
        EXPECT_EQ(left_source->origin_x(), right_source->origin_x());
        EXPECT_EQ(left_source->origin_y(), right_source->origin_y());
        EXPECT_EQ(left_source->width(), right_source->width());
        EXPECT_EQ(left_source->height(), right_source->height());
        expect_equal_span(left_source->channel_names(), right_source->channel_names());
        expect_equal_span(left_source->pixels(), right_source->pixels());
        EXPECT_EQ(left.image->level_count(), right.image->level_count());
        EXPECT_EQ(left.image->generated_pixel_bytes(), right.image->generated_pixel_bytes());
        for (auto level = std::uint32_t{}; level < left.image->level_count(); ++level) {
            const auto left_level = left.image->level(level);
            const auto right_level = right.image->level(level);
            ASSERT_TRUE(left_level.has_value()) << left_level.error().message;
            ASSERT_TRUE(right_level.has_value()) << right_level.error().message;
            EXPECT_EQ((*left_level)->source_path(), (*right_level)->source_path());
            EXPECT_EQ((*left_level)->format_name(), (*right_level)->format_name());
            EXPECT_EQ((*left_level)->source_color_space(), (*right_level)->source_color_space());
            EXPECT_EQ((*left_level)->storage_color_space(), (*right_level)->storage_color_space());
            EXPECT_EQ((*left_level)->origin_x(), (*right_level)->origin_x());
            EXPECT_EQ((*left_level)->origin_y(), (*right_level)->origin_y());
            EXPECT_EQ((*left_level)->width(), (*right_level)->width());
            EXPECT_EQ((*left_level)->height(), (*right_level)->height());
            expect_equal_span((*left_level)->channel_names(), (*right_level)->channel_names());
            expect_equal_span((*left_level)->pixels(), (*right_level)->pixels());
        }
    }
}

TEST(SceneDescriptionJsonTest, AcceptsTheMinimalClosedVersionOneCorpus) {
    const auto parsed = deserialize_scene_description_json(MinimalSceneJson);

    ASSERT_TRUE(parsed.has_value()) << parsed.error().message;
    EXPECT_EQ(parsed->active_film_id().value, 1U);
    EXPECT_EQ(parsed->active_camera_id().value, 2U);
    EXPECT_EQ(parsed->active_render_options_id().value, 3U);
    EXPECT_EQ(parsed->active_render_options().options.seed,
              std::numeric_limits<std::uint64_t>::max());
    EXPECT_TRUE(parsed->constant_textures().empty());
    EXPECT_TRUE(parsed->host_image_textures().empty());
    EXPECT_TRUE(parsed->geometries().empty());
    EXPECT_TRUE(parsed->materials().empty());
    EXPECT_TRUE(parsed->lights().empty());

    const auto canonical = serialize_scene_description_json(*parsed);
    ASSERT_TRUE(canonical.has_value()) << canonical.error().message;
    EXPECT_EQ(*canonical, CanonicalMinimalSceneJson);
    const auto reparsed = deserialize_scene_description_json(*canonical);
    ASSERT_TRUE(reparsed.has_value()) << reparsed.error().message;
    const auto repeated = serialize_scene_description_json(*reparsed);
    ASSERT_TRUE(repeated.has_value()) << repeated.error().message;
    EXPECT_EQ(*repeated, *canonical);
}

TEST(SceneDescriptionJsonTest, RoundTripsEveryVersionOneDomainCanonically) {
    const auto expected = complete_scene();
    const auto first_encoding = serialize_scene_description_json(expected);
    ASSERT_TRUE(first_encoding.has_value()) << first_encoding.error().message;

    const auto decoded = deserialize_scene_description_json(*first_encoding);
    ASSERT_TRUE(decoded.has_value()) << decoded.error().message;
    const auto second_encoding = serialize_scene_description_json(*decoded);
    ASSERT_TRUE(second_encoding.has_value()) << second_encoding.error().message;

    EXPECT_EQ(*second_encoding, *first_encoding);
    EXPECT_EQ(decoded->active_film_id(), expected.active_film_id());
    EXPECT_EQ(decoded->active_camera_id(), expected.active_camera_id());
    EXPECT_EQ(decoded->active_render_options_id(), expected.active_render_options_id());
    expect_equal_span(expected.films(), decoded->films());
    expect_equal_span(expected.render_options(), decoded->render_options());
    expect_equal_span(expected.constant_textures(), decoded->constant_textures());
    expect_equal_span(expected.objects(), decoded->objects());
    expect_equal_span(expected.materials(), decoded->materials());
    expect_equal_span(expected.instances(), decoded->instances());
    expect_equal_span(expected.lights(), decoded->lights());
    expect_meshes_equal(expected, *decoded);
    expect_host_images_equal(expected, *decoded);

    ASSERT_EQ(decoded->cameras().size(), expected.cameras().size());
    for (auto index = std::size_t{}; index < expected.cameras().size(); ++index) {
        const auto& left = std::get<ScenePinholeCameraDescription>(expected.cameras()[index].model);
        const auto& right =
            std::get<ScenePinholeCameraDescription>(decoded->cameras()[index].model);
        EXPECT_EQ(expected.cameras()[index].id, decoded->cameras()[index].id);
        EXPECT_EQ(expected.cameras()[index].film, decoded->cameras()[index].film);
        EXPECT_EQ(left.origin, right.origin);
        EXPECT_EQ(left.orientation.normal(), right.orientation.normal());
        EXPECT_EQ(left.orientation.tangent(), right.orientation.tangent());
        EXPECT_EQ(left.orientation.bitangent(), right.orientation.bitangent());
        EXPECT_EQ(left.vertical_field_of_view_radians, right.vertical_field_of_view_radians);
        EXPECT_EQ(left.t_min, right.t_min);
        EXPECT_EQ(left.t_max, right.t_max);
        EXPECT_EQ(left.visibility_mask, right.visibility_mask);
        EXPECT_EQ(left.current_medium, right.current_medium);
    }
}

TEST(SceneDescriptionJsonTest, AcceptsArbitraryMemberOrder) {
    const auto scene = complete_scene();
    const auto canonical = take_or_throw(serialize_scene_description_json(scene));
    const auto reordered = replace_once(canonical, R"json({"id":10,"type":"float","value":-0})json",
                                        R"json({"value":-0,"id":10,"type":"float"})json");

    const auto decoded = deserialize_scene_description_json(reordered);
    ASSERT_TRUE(decoded.has_value()) << decoded.error().message;
    const auto repeated = serialize_scene_description_json(*decoded);
    ASSERT_TRUE(repeated.has_value()) << repeated.error().message;
    EXPECT_EQ(*repeated, canonical);
}

TEST(SceneDescriptionJsonTest, RoundTripsAnExplicitNullMaterialOutsideSpectralTransport) {
    const auto minimal = take_or_throw(deserialize_scene_description_json(MinimalSceneJson));
    auto input = SceneDescriptionInput{
        .active_film = minimal.active_film_id(),
        .active_camera = minimal.active_camera_id(),
        .active_render_options = minimal.active_render_options_id(),
        .films = {minimal.films().begin(), minimal.films().end()},
        .cameras = {minimal.cameras().begin(), minimal.cameras().end()},
        .render_options = {minimal.render_options().begin(), minimal.render_options().end()},
        .constant_textures = {},
        .host_image_textures = {},
        .objects = {},
        .geometries = {},
        .materials = {{.id = renderer::MaterialId{.value = 90U}, .spectral = std::nullopt}},
        .instances = {},
        .lights = {},
    };
    const auto scene = take_or_throw(SceneDescription::create(std::move(input)));
    const auto canonical = take_or_throw(serialize_scene_description_json(scene));
    EXPECT_NE(canonical.find(R"json("spectral":null)json"), std::string::npos);

    const auto decoded = deserialize_scene_description_json(canonical);
    ASSERT_TRUE(decoded.has_value()) << decoded.error().message;
    ASSERT_EQ(decoded->materials().size(), 1U);
    EXPECT_FALSE(decoded->materials().front().spectral.has_value());
    EXPECT_EQ(take_or_throw(serialize_scene_description_json(*decoded)), canonical);
}

TEST(SceneDescriptionJsonTest, RejectsMalformedUnknownAndDuplicateMembersWithoutFallback) {
    const auto valid = std::string{MinimalSceneJson};
    const auto cases = std::array{
        std::string{},
        std::string{"[]"},
        std::string{"{} trailing"},
        std::string{"{\"schema\":\"blackframe.scene\""},
        replace_once(valid, "  \"instances\": [],\n  \"lights\": []\n", "  \"instances\": []\n"),
        replace_once(valid, R"("constant_textures": [])", R"("constant_textures": {})"),
        replace_once(valid, R"("origin": [0, 0, 0])", R"("origin": [0, 0])"),
        replace_once(valid, R"("lights": [])", R"("lights": [], "unknown_root": 0)"),
        replace_once(valid, R"("film": 1, "camera")", R"("film": 1, "mystery": 0, "camera")"),
        replace_once(valid, R"("schema": "blackframe.scene")",
                     R"("schema":"blackframe.scene","\u0073chema":"blackframe.scene")"),
        replace_once(valid, R"("film": 1, "camera")", R"("film":1,"f\u0069lm":1,"camera")"),
        replace_once(valid, R"("type": "pinhole")", R"("type":"unknown_camera")"),
        replace_once(valid, R"("pixel_jitter": "center")", R"("pixel_jitter":"unknown_jitter")"),
        replace_once(valid, R"("russian_roulette": {"mode": "disabled"})",
                     R"("russian_roulette":{"mode":"disabled","first_eligible_depth":3})"),
    };

    for (const auto& encoded : cases) {
        SCOPED_TRACE(encoded);
        expect_failure(deserialize_scene_description_json(encoded),
                       core::StatusCode::invalid_argument);
    }

    const auto trailing_comma = replace_once(valid, "  \"lights\": []\n}", "  \"lights\": [],\n}");
    const auto invalid_escape =
        replace_once(valid, "blackframe.scene", R"json(blackframe\qscene)json");
    const auto isolated_surrogate =
        replace_once(valid, R"json("blackframe.scene")json", R"json("\uD800")json");
    auto malformed_utf8 = valid;
    const auto schema_offset = malformed_utf8.find("blackframe.scene");
    ASSERT_NE(schema_offset, std::string::npos);
    malformed_utf8[schema_offset] = static_cast<char>(0xC0U);
    for (const auto& encoded :
         {trailing_comma, invalid_escape, isolated_surrogate, malformed_utf8}) {
        SCOPED_TRACE(encoded);
        expect_failure(deserialize_scene_description_json(encoded),
                       core::StatusCode::invalid_argument);
    }
}

TEST(SceneDescriptionJsonTest, RejectsUnknownTextureClosureAndLightVariants) {
    const auto valid = take_or_throw(serialize_scene_description_json(complete_scene()));
    const auto cases = std::array{
        replace_once(valid, R"("type":"float")", R"("type":"unknown_texture")"),
        replace_once(valid, R"("type":"lambertian_reflection")", R"("type":"unknown_closure")"),
        replace_once(valid, R"("type":"point")", R"("type":"unknown_light")"),
        replace_once(valid, R"("frame_mode":"surface_tangent")", R"("frame_mode":"unknown_frame")"),
        replace_once(valid, R"("y_convention":"negative_v")",
                     R"("y_convention":"unknown_normal_convention")"),
        replace_once(valid, R"("u_wrap":"mirror")", R"("u_wrap":"unknown_wrap")"),
    };

    for (const auto& encoded : cases) {
        SCOPED_TRACE(encoded);
        expect_failure(deserialize_scene_description_json(encoded),
                       core::StatusCode::invalid_argument);
    }
}

TEST(SceneDescriptionJsonTest, RejectsNonFiniteAndOverflowingNumericTokens) {
    const auto valid = std::string{MinimalSceneJson};
    const auto cases = std::array{
        replace_once(valid, R"("vertical_fov_radians": 1)", R"("vertical_fov_radians": NaN)"),
        replace_once(valid, R"("vertical_fov_radians": 1)", R"("vertical_fov_radians": Infinity)"),
        replace_once(valid, R"("vertical_fov_radians": 1)", R"("vertical_fov_radians": -Infinity)"),
        replace_once(valid, R"("vertical_fov_radians": 1)", R"("vertical_fov_radians": 1e9999)"),
        replace_once(valid, R"("vertical_fov_radians": 1)", R"("vertical_fov_radians": 1e-9999)"),
        replace_once(valid, R"("active": {"film": 1)", R"("active": {"film": 4294967296)"),
        replace_once(valid, R"("seed": 18446744073709551615)", R"("seed": 18446744073709551616)"),
        replace_once(valid, R"("samples_per_pixel": 2)", R"("samples_per_pixel": 02)"),
        replace_once(valid, R"("samples_per_pixel": 2)", R"("samples_per_pixel": +2)"),
        replace_once(valid, R"("samples_per_pixel": 2)", R"("samples_per_pixel": 2.0)"),
    };

    for (const auto& encoded : cases) {
        SCOPED_TRACE(encoded);
        expect_failure(deserialize_scene_description_json(encoded),
                       core::StatusCode::invalid_argument);
    }
}

TEST(SceneDescriptionJsonTest, RejectsIncompatibleSchemaNamesAndVersions) {
    const auto valid = std::string{MinimalSceneJson};
    const auto cases = std::array{
        replace_once(valid, R"("schema": "blackframe.scene")", R"("schema":"another.scene")"),
        replace_once(valid, R"("schema_version": 1)", R"("schema_version":0)"),
        replace_once(valid, R"("schema_version": 1)", R"("schema_version":2)"),
    };

    for (const auto& encoded : cases) {
        SCOPED_TRACE(encoded);
        expect_failure(deserialize_scene_description_json(encoded), core::StatusCode::incompatible);
    }
}

TEST(SceneDescriptionJsonTest, EnforcesEncodedAndStructuredAllocationBoundaries) {
    const auto scene = complete_scene();
    const auto encoded = serialize_scene_description_json(scene);
    ASSERT_TRUE(encoded.has_value()) << encoded.error().message;

    auto limits = SceneDescriptionJsonLimits{};
    limits.maximum_encoded_bytes = encoded->size();
    ASSERT_TRUE(deserialize_scene_description_json(*encoded, limits).has_value());
    ASSERT_TRUE(serialize_scene_description_json(scene, limits).has_value());
    --limits.maximum_encoded_bytes;
    expect_failure(deserialize_scene_description_json(*encoded, limits),
                   core::StatusCode::resource_exhausted);
    expect_write_failure(serialize_scene_description_json(scene, limits),
                         core::StatusCode::resource_exhausted);

    limits = {};
    limits.maximum_records_per_registry = 6U;
    ASSERT_TRUE(deserialize_scene_description_json(*encoded, limits).has_value());
    ASSERT_TRUE(serialize_scene_description_json(scene, limits).has_value());
    limits.maximum_records_per_registry = 5U;
    expect_failure(deserialize_scene_description_json(*encoded, limits),
                   core::StatusCode::resource_exhausted);
    expect_write_failure(serialize_scene_description_json(scene, limits),
                         core::StatusCode::resource_exhausted);

    limits = {};
    limits.maximum_mesh_vertices = 4U;
    ASSERT_TRUE(deserialize_scene_description_json(*encoded, limits).has_value());
    ASSERT_TRUE(serialize_scene_description_json(scene, limits).has_value());
    limits.maximum_mesh_vertices = 3U;
    expect_failure(deserialize_scene_description_json(*encoded, limits),
                   core::StatusCode::resource_exhausted);
    expect_write_failure(serialize_scene_description_json(scene, limits),
                         core::StatusCode::resource_exhausted);

    limits = {};
    limits.maximum_mesh_triangles = 2U;
    ASSERT_TRUE(deserialize_scene_description_json(*encoded, limits).has_value());
    ASSERT_TRUE(serialize_scene_description_json(scene, limits).has_value());
    limits.maximum_mesh_triangles = 1U;
    expect_failure(deserialize_scene_description_json(*encoded, limits),
                   core::StatusCode::resource_exhausted);
    expect_write_failure(serialize_scene_description_json(scene, limits),
                         core::StatusCode::resource_exhausted);

    limits = {};
    limits.maximum_image_scalar_values = 16U;
    ASSERT_TRUE(deserialize_scene_description_json(*encoded, limits).has_value());
    ASSERT_TRUE(serialize_scene_description_json(scene, limits).has_value());
    limits.maximum_image_scalar_values = 15U;
    expect_failure(deserialize_scene_description_json(*encoded, limits),
                   core::StatusCode::resource_exhausted);
    expect_write_failure(serialize_scene_description_json(scene, limits),
                         core::StatusCode::resource_exhausted);

    limits = {};
    limits.maximum_decoded_bytes = 1U;
    expect_failure(deserialize_scene_description_json(*encoded, limits),
                   core::StatusCode::resource_exhausted);
    expect_write_failure(serialize_scene_description_json(scene, limits),
                         core::StatusCode::resource_exhausted);
    expect_dynamic_limit_boundary(scene, *encoded,
                                  &SceneDescriptionJsonLimits::maximum_decoded_bytes);

    const auto minimal = take_or_throw(deserialize_scene_description_json(MinimalSceneJson));
    const auto minimal_encoded = take_or_throw(serialize_scene_description_json(minimal));
    expect_dynamic_limit_boundary(minimal, minimal_encoded,
                                  &SceneDescriptionJsonLimits::maximum_nesting_depth);
    expect_dynamic_limit_boundary(minimal, minimal_encoded,
                                  &SceneDescriptionJsonLimits::maximum_total_values);
    expect_dynamic_limit_boundary(minimal, minimal_encoded,
                                  &SceneDescriptionJsonLimits::maximum_total_string_bytes);

    limits = MaximumSceneDescriptionJsonLimits;
    ++limits.maximum_nesting_depth;
    expect_failure(deserialize_scene_description_json(minimal_encoded, limits),
                   core::StatusCode::invalid_argument);
    expect_write_failure(serialize_scene_description_json(minimal, limits),
                         core::StatusCode::invalid_argument);
}

TEST(SceneDescriptionJsonTest, AcceptsAnExactDecodedBudgetForAnImageWithoutGeneratedMips) {
    auto input = complete_input();
    input.host_image_textures.front().image = single_texel_surface_image();
    const auto scene = take_or_throw(SceneDescription::create(std::move(input)));
    const auto encoded = take_or_throw(serialize_scene_description_json(scene));

    expect_dynamic_limit_boundary(scene, encoded,
                                  &SceneDescriptionJsonLimits::maximum_decoded_bytes);
}

TEST(SceneDescriptionJsonTest, RejectsDanglingReferencesAndCyclesAfterDecoding) {
    const auto encoded = take_or_throw(serialize_scene_description_json(complete_scene()));
    const auto dangling = replace_scalar_after(encoded, R"("instances")", R"("object")", "999");
    const auto cycle = replace_scalar_after(encoded, R"("instances")", R"("parent")", "50");

    expect_failure(deserialize_scene_description_json(dangling),
                   core::StatusCode::invalid_argument);
    expect_failure(deserialize_scene_description_json(cycle), core::StatusCode::invalid_argument);
}

TEST(SceneDescriptionJsonTest, RefusesToSerializeNonFiniteTypedValues) {
    const auto scene = complete_scene(true);
    const auto encoded = serialize_scene_description_json(scene);

    ASSERT_FALSE(encoded.has_value());
    EXPECT_EQ(encoded.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(encoded.error().message.empty());
}

TEST(SceneDescriptionJsonTest, RefusesToSerializeNonUtf8TypedMetadata) {
    auto invalid_format = std::string{static_cast<char>(0xC0U)};
    auto image = renderer::HostImage::create_snapshot(renderer::HostImageSnapshotDescription{
        .source_path = snapshot_source_path(),
        .format_name = std::move(invalid_format),
        .source_color_space = renderer::TextureColorSpace::data,
        .storage_color_space = renderer::TextureColorSpace::data,
        .width = 1U,
        .height = 1U,
        .channel_names = {"R", "G", "B", "height"},
        .pixels = {0.5F, 0.5F, 1.0F, 0.25F},
    });
    auto input = complete_input();
    input.host_image_textures.front().image =
        take_or_throw(renderer::HostImageMipChain::generate(take_or_throw(std::move(image))));
    const auto scene = take_or_throw(SceneDescription::create(std::move(input)));
    const auto encoded = serialize_scene_description_json(scene);

    ASSERT_FALSE(encoded.has_value());
    EXPECT_EQ(encoded.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(encoded.error().message.empty());
}

} // namespace
} // namespace blackframe::engine
