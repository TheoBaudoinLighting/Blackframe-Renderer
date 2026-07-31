#include "../../../Renderer/CornellDiffuseImageRenderer.hpp"

#include <Blackframe/Backends/CPU/Embree/AccelBackend.hpp>
#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Engine/SceneBsdfOnlyPathLoop.hpp>
#include <Blackframe/Engine/TriangleMesh.hpp>
#include <Blackframe/Renderer/Cie1931Sensor.hpp>
#include <Blackframe/Renderer/Color.hpp>
#include <Blackframe/Renderer/DisplayPsnr.hpp>
#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/IndependentSampler.hpp>
#include <Blackframe/Renderer/LinearMetrics.hpp>
#include <Blackframe/Renderer/MatrixTypes.hpp>
#include <Blackframe/Renderer/PathDepthLimits.hpp>
#include <Blackframe/Renderer/PathState.hpp>
#include <Blackframe/Renderer/RussianRoulette.hpp>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

namespace cornell = renderer::cornell_test;

inline constexpr auto EvaluationSamplesPerPixel = std::uint32_t{4};
inline constexpr auto MaximumExpectedMse = renderer::ReferenceScalar{1.0e-10};
inline constexpr auto MaximumExpectedRmse = renderer::ReferenceScalar{1.0e-5};
inline constexpr auto MinimumExpectedPsnr = renderer::ReferenceScalar{80};

[[nodiscard]] core::Error integration_error(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

class CountingAccelBackend final : public AccelBackend {
  public:
    explicit CountingAccelBackend(std::unique_ptr<AccelBackend> backend)
        : AccelBackend{backend->frame_scene()}, backend_{std::move(backend)} {}

    [[nodiscard]] AccelBackendKind kind() const noexcept override {
        return backend_->kind();
    }

    [[nodiscard]] core::Result<std::optional<AccelHit>>
    closest_hit(const renderer::Ray& ray) const override {
        ++closest_hit_count_;
        return backend_->closest_hit(ray);
    }

    [[nodiscard]] core::Result<bool> occluded(const renderer::Ray& ray) const override {
        return backend_->occluded(ray);
    }

    [[nodiscard]] core::Status rebuild(FrameSceneHandle scene) override {
        const auto retained_scene = scene;
        auto status = backend_->rebuild(std::move(scene));
        if (status) {
            publish_rebuild(retained_scene);
        }
        return status;
    }

    [[nodiscard]] core::Status refit(FrameSceneHandle scene) override {
        const auto retained_scene = scene;
        auto status = backend_->refit(std::move(scene));
        if (status) {
            publish_refit(retained_scene);
        }
        return status;
    }

    [[nodiscard]] AccelBuildStatistics build_statistics() const noexcept override {
        return backend_->build_statistics();
    }

    [[nodiscard]] std::uint64_t closest_hit_count() const noexcept {
        return closest_hit_count_;
    }

  private:
    std::unique_ptr<AccelBackend> backend_;
    mutable std::uint64_t closest_hit_count_{};
};

[[nodiscard]] std::string metric_text(const renderer::ReferenceScalar value) {
    auto buffer = std::array<char, 64>{};
    const auto [end, error] = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::general,
        std::numeric_limits<renderer::ReferenceScalar>::max_digits10);
    if (error != std::errc{}) {
        return "unrepresentable";
    }
    return std::string{buffer.data(), static_cast<std::size_t>(end - buffer.data())};
}

[[nodiscard]] core::Result<std::shared_ptr<const TriangleMesh>>
make_quad_mesh(const cornell::SurfaceFor<renderer::TransportScalar>& first,
               const cornell::SurfaceFor<renderer::TransportScalar>& second) {
    const auto& first_vertices = first.triangle().vertices();
    const auto& second_vertices = second.triangle().vertices();
    if (first_vertices[0] != second_vertices[0] || first_vertices[2] != second_vertices[1] ||
        first.triangle().geometric_normal() != second.triangle().geometric_normal()) {
        return std::unexpected(
            integration_error("The closed Cornell quad triangles do not share one planar edge."));
    }
    if (first.reflection().reflectance() != second.reflection().reflectance() ||
        first.emission().radiance() != second.emission().radiance() ||
        first.wavelengths() != second.wavelengths() ||
        first.visibility_mask() != second.visibility_mask()) {
        return std::unexpected(
            integration_error("The closed Cornell quad triangles do not share one material."));
    }

    const auto normal = first.triangle().geometric_normal();
    auto mesh = TriangleMesh::create(
        {
            first_vertices[0],
            first_vertices[1],
            first_vertices[2],
            second_vertices[2],
        },
        {normal, normal, normal, normal},
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
    if (!mesh) {
        return std::unexpected(mesh.error());
    }
    return std::make_shared<const TriangleMesh>(std::move(*mesh));
}

[[nodiscard]] core::Result<FrameSceneHandle>
make_cornell_frame_scene(const cornell::CornellImageScene<renderer::TransportScalar>& fixture) {
    constexpr auto triangles_per_quad = std::size_t{2};
    constexpr auto quad_count = std::size_t{5};
    if (fixture.surfaces.size() != triangles_per_quad * quad_count) {
        return std::unexpected(
            integration_error("The closed Cornell fixture no longer contains five quads."));
    }

    auto description = FrameSceneDescription{};
    description.objects.reserve(quad_count);
    description.geometries.reserve(quad_count);
    description.materials.reserve(quad_count);
    description.instances.reserve(quad_count);
    description.spectral_environment = SceneSpectralEnvironment{
        .wavelengths = fixture.wavelengths,
        .radiance = fixture.environment.environment().radiance(),
    };

    for (auto quad_index = std::size_t{0}; quad_index < quad_count; ++quad_index) {
        const auto& first = fixture.surfaces[quad_index * triangles_per_quad];
        const auto& second = fixture.surfaces[quad_index * triangles_per_quad + 1U];
        auto mesh = make_quad_mesh(first, second);
        if (!mesh) {
            return std::unexpected(mesh.error());
        }

        const auto index = static_cast<std::uint32_t>(quad_index);
        const auto object = renderer::ObjectId{.value = 100U + index};
        const auto geometry = renderer::GeometryId{.value = 200U + index};
        const auto material = renderer::MaterialId{.value = 300U + index};
        const auto instance = renderer::InstanceId{.value = 400U + index};
        description.objects.push_back(SceneObject{.id = object});
        description.geometries.push_back(SceneGeometry{
            .id = geometry,
            .mesh = std::move(*mesh),
        });
        description.materials.push_back(SceneMaterial{
            .id = material,
            .spectral =
                SceneSpectralMaterial{
                    .wavelengths = first.wavelengths(),
                    .reflectance = first.reflection().reflectance(),
                    .emitted_radiance = first.emission().radiance(),
                },
        });
        description.instances.push_back(SceneInstance{
            .id = instance,
            .parent = std::nullopt,
            .object = object,
            .geometry = geometry,
            .material = material,
            .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
            .visibility_mask = first.visibility_mask(),
        });
    }
    return FrameScene::create(std::move(description));
}

[[nodiscard]] core::Result<renderer::LinearRGB>
render_scene_sample(const cornell::CornellImageScene<renderer::TransportScalar>& fixture,
                    const AccelBackend& acceleration, const std::uint32_t pixel_x,
                    const std::uint32_t pixel_y, const std::uint64_t sample_index) {
    const auto index = renderer::PixelSampleIndex{
        .pixel_x = pixel_x,
        .pixel_y = pixel_y,
        .sample_index = sample_index,
        .seed = cornell::CornellEvaluationSeed,
    };
    const auto ray = fixture.camera.generate_primary_ray(index, renderer::PixelJitterMode::uniform,
                                                         renderer::TransportScalar{0});
    if (!ray) {
        return std::unexpected(ray.error());
    }
    const auto state =
        renderer::PathState::create_initial(fixture.wavelengths, renderer::VacuumMedium);
    if (!state) {
        return std::unexpected(state.error());
    }
    const auto stream = renderer::IndependentSampler{cornell::CornellEvaluationSeed}.make_stream(
        pixel_x, pixel_y, sample_index);
    const auto traced = trace_scene_bsdf_only(
        *ray, *state, stream, acceleration,
        renderer::PathDepthLimits{.diffuse = cornell::CornellMaximumDiffuseDepth},
        renderer::RussianRoulettePolicy::disabled());
    if (!traced) {
        return std::unexpected(traced.error());
    }
    const auto xyz = renderer::cie_1931_spectrum_to_xyz(traced->state.accumulated_radiance(),
                                                        fixture.wavelengths);
    if (!xyz) {
        return std::unexpected(xyz.error());
    }
    return renderer::xyz_to_linear_rgb(*xyz);
}

[[nodiscard]] core::Result<renderer::Film>
render_cornell_frame_scene(const cornell::CornellImageScene<renderer::TransportScalar>& fixture,
                           const AccelBackend& acceleration) {
    auto film = renderer::Film::create(cornell::Cornell64Specification.extent);
    if (!film) {
        return std::unexpected(film.error());
    }

    for (auto pixel_y = std::uint32_t{0}; pixel_y < cornell::Cornell64Specification.extent.height;
         ++pixel_y) {
        for (auto pixel_x = std::uint32_t{0};
             pixel_x < cornell::Cornell64Specification.extent.width; ++pixel_x) {
            for (auto sample_index = std::uint64_t{0}; sample_index < EvaluationSamplesPerPixel;
                 ++sample_index) {
                const auto sample =
                    render_scene_sample(fixture, acceleration, pixel_x, pixel_y, sample_index);
                if (!sample) {
                    return std::unexpected(sample.error());
                }
                const auto accumulated =
                    film->add_sample(pixel_x, pixel_y, *sample, renderer::TransportScalar{1});
                if (!accumulated) {
                    return std::unexpected(accumulated.error());
                }
            }
        }
    }
    return std::move(*film);
}

template <renderer::AccumulationPrecision Precision>
void expect_complete_film(const renderer::FilmT<Precision>& film) {
    EXPECT_EQ(film.extent().width, cornell::Cornell64Specification.extent.width);
    EXPECT_EQ(film.extent().height, cornell::Cornell64Specification.extent.height);
    EXPECT_EQ(film.crop(), renderer::full_film_crop(cornell::Cornell64Specification.extent));
    EXPECT_EQ(film.pixel_count(),
              static_cast<std::size_t>(cornell::Cornell64Specification.extent.width) *
                  cornell::Cornell64Specification.extent.height);
    for (auto pixel_y = std::uint32_t{0}; pixel_y < cornell::Cornell64Specification.extent.height;
         ++pixel_y) {
        for (auto pixel_x = std::uint32_t{0};
             pixel_x < cornell::Cornell64Specification.extent.width; ++pixel_x) {
            const auto pixel = film.pixel(pixel_x, pixel_y);
            ASSERT_TRUE(pixel.has_value()) << pixel_x << ", " << pixel_y;
            EXPECT_EQ(pixel->sample_count, EvaluationSamplesPerPixel);
            EXPECT_DOUBLE_EQ(static_cast<double>(pixel->weight_sum),
                             static_cast<double>(EvaluationSamplesPerPixel));
            const auto resolved = film.resolved_pixel(pixel_x, pixel_y);
            ASSERT_TRUE(resolved.has_value()) << pixel_x << ", " << pixel_y;
            EXPECT_TRUE(std::isfinite(resolved->red));
            EXPECT_TRUE(std::isfinite(resolved->green));
            EXPECT_TRUE(std::isfinite(resolved->blue));
        }
    }
}

TEST(CornellFrameSceneIntegrationTest, MatchesTheIndependentScalarOracleThroughEmbree) {
    const auto fixture = cornell::make_cornell_image_scene<renderer::TransportScalar>(
        cornell::Cornell64Specification);
    ASSERT_TRUE(fixture.has_value()) << (fixture.has_value() ? "" : fixture.error().message);
    const auto scene = make_cornell_frame_scene(*fixture);
    ASSERT_TRUE(scene.has_value()) << (scene.has_value() ? "" : scene.error().message);
    EXPECT_EQ((*scene)->objects().size(), 5U);
    EXPECT_EQ((*scene)->geometries().size(), 5U);
    EXPECT_EQ((*scene)->materials().size(), 5U);
    EXPECT_EQ((*scene)->instances().size(), 5U);

    auto embree = create_embree_accel_backend(*scene);
    ASSERT_TRUE(embree.has_value()) << (embree.has_value() ? "" : embree.error().message);
    ASSERT_NE(*embree, nullptr);
    auto acceleration = CountingAccelBackend{std::move(*embree)};
    EXPECT_EQ(acceleration.kind(), AccelBackendKind::embree);
    EXPECT_EQ(acceleration.frame_scene(), *scene);

    const auto rendered = render_cornell_frame_scene(*fixture, acceleration);
    ASSERT_TRUE(rendered.has_value()) << (rendered.has_value() ? "" : rendered.error().message);
    expect_complete_film(*rendered);
    const auto primary_ray_count =
        static_cast<std::uint64_t>(cornell::Cornell64Specification.extent.width) *
        cornell::Cornell64Specification.extent.height * EvaluationSamplesPerPixel;
    EXPECT_GT(acceleration.closest_hit_count(), primary_ray_count);

    const auto oracle = cornell::render_cornell_image<renderer::ReferenceScalar>(
        cornell::Cornell64Specification, EvaluationSamplesPerPixel, cornell::CornellEvaluationSeed,
        1U);
    ASSERT_TRUE(oracle.has_value()) << (oracle.has_value() ? "" : oracle.error().message);
    expect_complete_film(*oracle);

    const auto linear_metrics = renderer::compute_linear_metrics(*rendered, *oracle);
    const auto display_psnr = renderer::compute_display_psnr(*rendered, *oracle);
    ASSERT_TRUE(linear_metrics.has_value())
        << (linear_metrics.has_value() ? "" : linear_metrics.error().message);
    ASSERT_TRUE(display_psnr.has_value())
        << (display_psnr.has_value() ? "" : display_psnr.error().message);

    testing::Test::RecordProperty("backend", "cpu_embree");
    testing::Test::RecordProperty("resolution", "64x64");
    testing::Test::RecordProperty("samples_per_pixel", static_cast<int>(EvaluationSamplesPerPixel));
    testing::Test::RecordProperty("closest_hit_queries",
                                  std::to_string(acceleration.closest_hit_count()));
    testing::Test::RecordProperty("mse_linear", metric_text(linear_metrics->mse));
    testing::Test::RecordProperty("rmse_linear", metric_text(linear_metrics->rmse));
    testing::Test::RecordProperty("psnr_display", metric_text(display_psnr->psnr));

    EXPECT_LE(linear_metrics->mse, MaximumExpectedMse);
    EXPECT_LE(linear_metrics->rmse, MaximumExpectedRmse);
    EXPECT_TRUE(std::isinf(display_psnr->psnr) || display_psnr->psnr >= MinimumExpectedPsnr);
}

} // namespace
} // namespace blackframe::engine
