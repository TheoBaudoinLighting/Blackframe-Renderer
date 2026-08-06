#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Engine/SceneBsdfOnlyPathLoop.hpp>
#include <Blackframe/Engine/SceneSurfaceInteraction.hpp>
#include <Blackframe/Renderer/IndependentSampler.hpp>
#include <Blackframe/Renderer/LocalFrame.hpp>
#include <Blackframe/Renderer/MatrixOperations.hpp>
#include <Blackframe/Renderer/SampleDimensionMap.hpp>
#include <Blackframe/Renderer/SamplingMappings.hpp>
#include <Blackframe/Renderer/WavelengthSampling.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

[[nodiscard]] renderer::SampledWavelengths make_wavelengths(const float sample = 0.125F) {
    return renderer::sample_uniform_visible_wavelengths(sample).value();
}

[[nodiscard]] SceneClosureMixture
require_lambertian_scene_closure(const renderer::TransportSpectrum reflectance) {
    return SceneClosureMixture::create_lambertian(reflectance).value();
}

[[nodiscard]] std::shared_ptr<const TriangleMesh>
make_surface_mesh(const renderer::Normal3 shading_normal = {.z = 1.0F}) {
    auto mesh = TriangleMesh::create(
        {
            renderer::Point3{.x = 0.0F, .y = 0.0F, .z = 0.0F},
            renderer::Point3{.x = 2.0F, .y = 0.0F, .z = 0.0F},
            renderer::Point3{.x = 0.0F, .y = 2.0F, .z = 0.0F},
        },
        {
            shading_normal,
            shading_normal,
            shading_normal,
        },
        {
            renderer::Point2{.x = 0.0F, .y = 0.0F},
            renderer::Point2{.x = 1.0F, .y = 0.0F},
            renderer::Point2{.x = 0.0F, .y = 1.0F},
        },
        {
            TriangleVertexIndices{.vertices = {0U, 1U, 2U}},
        });
    return std::make_shared<const TriangleMesh>(std::move(mesh).value());
}

[[nodiscard]] FrameSceneDescription make_spectral_scene_description() {
    const auto wavelengths = make_wavelengths();
    return FrameSceneDescription{
        .objects = {SceneObject{.id = {.value = 41U}}},
        .geometries =
            {
                SceneGeometry{
                    .id = {.value = 42U},
                    .mesh = make_surface_mesh(),
                },
            },
        .materials =
            {
                SceneMaterial{
                    .id = {.value = 43U},
                    .spectral =
                        SceneSpectralMaterial{
                            .wavelengths = wavelengths,
                            .closure_mixture = require_lambertian_scene_closure(
                                {.values = {0.2F, 0.4F, 0.6F, 0.8F}}),
                            .emitted_radiance = {.values = {1.0F, 2.0F, 3.0F, 4.0F}},
                        },
                },
            },
        .instances =
            {
                SceneInstance{
                    .id = {.value = 44U},
                    .parent = std::nullopt,
                    .object = {.value = 41U},
                    .geometry = {.value = 42U},
                    .material = {.value = 43U},
                    .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
                    .visibility_mask = renderer::AllRayVisibility,
                },
            },
        .spectral_environment =
            SceneSpectralEnvironment{
                .wavelengths = wavelengths,
                .radiance = {.values = {0.01F, 0.02F, 0.03F, 0.04F}},
            },
    };
}

[[nodiscard]] FrameSceneHandle make_spectral_scene() {
    return FrameScene::create(make_spectral_scene_description()).value();
}

[[nodiscard]] renderer::Ray make_surface_ray() {
    return renderer::Ray::create(renderer::Point3{.x = 0.5F, .y = 1.0F, .z = -2.0F},
                                 renderer::Vector3{.x = 0.0F, .y = 0.0F, .z = 1.0F}, 0.0F, 8.0F,
                                 0.375F, renderer::AllRayVisibility, renderer::VacuumMedium)
        .value();
}

[[nodiscard]] AccelHit make_surface_hit() {
    return AccelHit{
        .object = {.value = 41U},
        .triangle =
            {
                .parameter = 2.0F,
                .position = {.x = 0.5F, .y = 1.0F, .z = 0.0F},
                .geometric_normal = {.x = 0.0F, .y = 0.0F, .z = 1.0F},
                .barycentrics =
                    {
                        .vertex0 = 0.25F,
                        .vertex1 = 0.25F,
                        .vertex2 = 0.5F,
                    },
            },
        .identifiers =
            {
                .instance = {.value = 44U},
                .geometry = {.value = 42U},
                .primitive = {.value = 0U},
                .material = {.value = 43U},
            },
    };
}

class FixedHitBackend final : public AccelBackend {
  public:
    FixedHitBackend(FrameSceneHandle scene, std::optional<AccelHit> hit)
        : AccelBackend{std::move(scene)}, hit_{std::move(hit)} {}

    [[nodiscard]] AccelBackendKind kind() const noexcept override {
        return AccelBackendKind::analytic_reference;
    }

    [[nodiscard]] core::Result<std::optional<AccelHit>>
    closest_hit(const renderer::Ray&) const override {
        return hit_;
    }

    [[nodiscard]] core::Result<bool> occluded(const renderer::Ray&) const override {
        return hit_.has_value();
    }

    [[nodiscard]] core::Status rebuild(FrameSceneHandle scene) override {
        publish_rebuild(std::move(scene));
        return {};
    }

    [[nodiscard]] core::Status refit(FrameSceneHandle scene) override {
        publish_refit(std::move(scene));
        return {};
    }

  private:
    std::optional<AccelHit> hit_;
};

[[nodiscard]] core::Result<ResolvedSceneSurface>
resolve_fixed_hit(const FrameSceneHandle& scene, const AccelHit& hit, const renderer::Ray& ray) {
    auto acceleration = FixedHitBackend{scene, hit};
    auto resolved = resolve_scene_surface(acceleration, ray);
    if (!resolved) {
        return std::unexpected(resolved.error());
    }
    if (!*resolved) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::internal_error,
            .message = "A fixed test hit was unexpectedly resolved as a miss.",
        });
    }
    return std::move(**resolved);
}

template <typename Result> void expect_error(const Result& result, const core::StatusCode code) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, code);
    EXPECT_FALSE(result.error().message.empty());
}

void expect_point_near(const renderer::Point3 actual, const renderer::Point3 expected,
                       const float tolerance = 1.0e-6F) {
    EXPECT_NEAR(actual.x, expected.x, tolerance);
    EXPECT_NEAR(actual.y, expected.y, tolerance);
    EXPECT_NEAR(actual.z, expected.z, tolerance);
}

void expect_vector_near(const renderer::Vector3 actual, const renderer::Vector3 expected,
                        const float tolerance = 1.0e-6F) {
    EXPECT_NEAR(actual.x, expected.x, tolerance);
    EXPECT_NEAR(actual.y, expected.y, tolerance);
    EXPECT_NEAR(actual.z, expected.z, tolerance);
}

void expect_normal_near(const renderer::Normal3 actual, const renderer::Normal3 expected,
                        const float tolerance = 1.0e-6F) {
    EXPECT_NEAR(actual.x, expected.x, tolerance);
    EXPECT_NEAR(actual.y, expected.y, tolerance);
    EXPECT_NEAR(actual.z, expected.z, tolerance);
}

TEST(SpectralFrameSceneTest, ClosesACompleteSpectralSnapshotWithoutLosingSceneData) {
    const auto description = make_spectral_scene_description();
    const auto expected_environment = *description.spectral_environment;
    const auto expected_material = *description.materials.front().spectral;

    const auto result = FrameScene::create(description);

    ASSERT_TRUE(result.has_value()) << result.error().message;
    const auto& scene = **result;
    ASSERT_TRUE(scene.spectral_environment().has_value());
    EXPECT_EQ(*scene.spectral_environment(), expected_environment);
    const auto material = scene.material(renderer::MaterialId{.value = 43U});
    ASSERT_TRUE(material.has_value()) << material.error().message;
    ASSERT_TRUE(material->get().spectral.has_value());
    EXPECT_EQ(*material->get().spectral, expected_material);
}

TEST(SpectralFrameSceneTest, RejectsIncompleteSpectralSnapshotsWithoutImplicitBlackData) {
    {
        auto description = make_spectral_scene_description();
        description.spectral_environment.reset();
        expect_error(FrameScene::create(std::move(description)),
                     core::StatusCode::invalid_argument);
    }
    {
        auto description = make_spectral_scene_description();
        description.materials.front().spectral.reset();
        expect_error(FrameScene::create(std::move(description)),
                     core::StatusCode::invalid_argument);
    }
}

TEST(SpectralFrameSceneTest, RejectsInvalidSpectralPacketsAndClosureValues) {
    {
        auto description = make_spectral_scene_description();
        description.spectral_environment->wavelengths[0].probability.value = 0.0F;
        expect_error(FrameScene::create(std::move(description)),
                     core::StatusCode::invalid_argument);
    }
    {
        auto description = make_spectral_scene_description();
        description.materials.front().spectral->wavelengths = make_wavelengths(0.5F);
        expect_error(FrameScene::create(std::move(description)),
                     core::StatusCode::invalid_argument);
    }
    {
        auto description = make_spectral_scene_description();
        description.materials.front().spectral->closure_mixture.component_probabilities[1U] = 0.25F;
        expect_error(FrameScene::create(std::move(description)),
                     core::StatusCode::invalid_argument);
    }
    {
        auto description = make_spectral_scene_description();
        description.materials.front().spectral->emitted_radiance[1] = -0.01F;
        expect_error(FrameScene::create(std::move(description)),
                     core::StatusCode::invalid_argument);
    }
    {
        auto description = make_spectral_scene_description();
        description.spectral_environment->radiance[0] =
            std::numeric_limits<renderer::TransportScalar>::infinity();
        expect_error(FrameScene::create(std::move(description)),
                     core::StatusCode::invalid_argument);
    }
}

TEST(SceneSurfaceInteractionTest, ResolvesGeometryIdentifiersTimeAndSpectralClosures) {
    const auto scene = make_spectral_scene();
    const auto ray = make_surface_ray();
    const auto hit = make_surface_hit();

    const auto result = resolve_fixed_hit(scene, hit, ray);

    ASSERT_TRUE(result.has_value()) << result.error().message;
    const auto& interaction = result->interaction;
    expect_point_near(interaction.position(), renderer::Point3{.x = 0.5F, .y = 1.0F, .z = 0.0F});
    expect_normal_near(interaction.geometric_normal(),
                       renderer::Normal3{.x = 0.0F, .y = 0.0F, .z = 1.0F});
    expect_normal_near(interaction.shading_normal(),
                       renderer::Normal3{.x = 0.0F, .y = 0.0F, .z = 1.0F});
    EXPECT_NEAR(interaction.uv().x, 0.25F, 1.0e-6F);
    EXPECT_NEAR(interaction.uv().y, 0.5F, 1.0e-6F);
    expect_vector_near(interaction.dpdu(), renderer::Vector3{.x = 2.0F, .y = 0.0F, .z = 0.0F});
    expect_vector_near(interaction.dpdv(), renderer::Vector3{.x = 0.0F, .y = 2.0F, .z = 0.0F});
    EXPECT_EQ(interaction.identifiers(), hit.identifiers);
    EXPECT_EQ(interaction.time(), ray.time());
    const auto closures = result->closures.closure_set().closures();
    ASSERT_EQ(closures.size(), 1U);
    EXPECT_EQ(closures.front().kind, renderer::ClosureKind::lambertian_reflection);
    EXPECT_EQ(closures.front().weight,
              (renderer::TransportSpectrum{.values = {0.2F, 0.4F, 0.6F, 0.8F}}));
    ASSERT_EQ(result->closures.component_probabilities().size(), 1U);
    EXPECT_EQ(result->closures.component_probabilities().front(), 1.0F);
    EXPECT_EQ(result->emission.radiance(),
              (renderer::TransportSpectrum{.values = {1.0F, 2.0F, 3.0F, 4.0F}}));
    EXPECT_TRUE(std::isfinite(result->position_error.x));
    EXPECT_TRUE(std::isfinite(result->position_error.y));
    EXPECT_TRUE(std::isfinite(result->position_error.z));
    EXPECT_GT(result->position_error.x, 0.0F);
    EXPECT_GT(result->position_error.y, 0.0F);
    EXPECT_GT(result->position_error.z, 0.0F);
}

TEST(SceneSurfaceInteractionTest, OrientsShadingNormalsUnderMirroredInstances) {
    auto description = make_spectral_scene_description();
    description.instances.front().local_to_parent =
        renderer::AffineTransform::scale(renderer::Vector3{.x = -1.0F, .y = 1.0F, .z = 1.0F})
            .value()
            .matrix();
    const auto scene = FrameScene::create(std::move(description));
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    const auto ray = renderer::Ray::create(renderer::Point3{.x = -0.5F, .y = 1.0F, .z = -2.0F},
                                           renderer::Vector3{.z = 1.0F}, 0.0F, 8.0F, 0.25F,
                                           renderer::AllRayVisibility, renderer::VacuumMedium);
    ASSERT_TRUE(ray.has_value()) << ray.error().message;
    auto hit = make_surface_hit();
    hit.triangle.position.x = -0.5F;
    hit.triangle.geometric_normal.z = -1.0F;

    const auto resolved = resolve_fixed_hit(*scene, hit, *ray);

    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;
    expect_normal_near(resolved->interaction.geometric_normal(), renderer::Normal3{.z = -1.0F});
    expect_normal_near(resolved->interaction.shading_normal(), renderer::Normal3{.z = -1.0F});
}

TEST(SceneSurfaceInteractionTest, NormalizesNormalsAfterExtremeAnisotropicScale) {
    auto description = make_spectral_scene_description();
    description.instances.front().local_to_parent =
        renderer::AffineTransform::scale(renderer::Vector3{.x = 1.0F, .y = 1.0F, .z = 1.0e-30F})
            .value()
            .matrix();
    const auto scene = FrameScene::create(std::move(description));
    ASSERT_TRUE(scene.has_value()) << scene.error().message;

    const auto resolved = resolve_fixed_hit(*scene, make_surface_hit(), make_surface_ray());

    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;
    expect_normal_near(resolved->interaction.shading_normal(), renderer::Normal3{.z = 1.0F});
}

TEST(SceneSurfaceInteractionTest, BoundsTransformAndTraversalPositionError) {
    auto description = make_spectral_scene_description();
    description.instances.front().local_to_parent =
        renderer::AffineTransform::translation(renderer::Vector3{.x = 100000.0F}).value().matrix();
    const auto scene = FrameScene::create(std::move(description));
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    const auto ray = renderer::Ray::create(renderer::Point3{.x = 100000.5F, .y = 1.0F, .z = -2.0F},
                                           renderer::Vector3{.z = 1.0F}, 0.0F, 8.0F, 0.25F,
                                           renderer::AllRayVisibility, renderer::VacuumMedium);
    ASSERT_TRUE(ray.has_value()) << ray.error().message;
    auto hit = make_surface_hit();
    hit.triangle.position.x = 100000.5F;

    const auto resolved = resolve_fixed_hit(*scene, hit, *ray);

    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;
    const auto position = resolved->interaction.position();
    EXPECT_LE(std::abs(position.x - hit.triangle.position.x), resolved->position_error.x);
    EXPECT_LE(std::abs(position.y - hit.triangle.position.y), resolved->position_error.y);
    EXPECT_LE(std::abs(position.z - hit.triangle.position.z), resolved->position_error.z);
    EXPECT_GT(resolved->position_error.x, 0.01F);
}

TEST(SceneSurfaceInteractionTest, ResolvesAgainstTheSnapshotPublishedByRefit) {
    auto description = make_spectral_scene_description();
    const auto original_scene = FrameScene::create(description);
    ASSERT_TRUE(original_scene.has_value()) << original_scene.error().message;
    auto acceleration = create_analytic_accel_backend(*original_scene);
    ASSERT_TRUE(acceleration.has_value()) << acceleration.error().message;

    description.instances.front().local_to_parent =
        renderer::AffineTransform::translation(renderer::Vector3{.z = 1.0F}).value().matrix();
    const auto translated_scene = FrameScene::create(std::move(description));
    ASSERT_TRUE(translated_scene.has_value()) << translated_scene.error().message;
    const auto refit = (*acceleration)->refit(*translated_scene);
    ASSERT_TRUE(refit.has_value()) << refit.error().message;
    const auto ray = renderer::Ray::create(renderer::Point3{.x = 0.5F, .y = 1.0F, .z = -2.0F},
                                           renderer::Vector3{.z = 1.0F}, 0.0F, 8.0F, 0.5F,
                                           renderer::AllRayVisibility, renderer::VacuumMedium);
    ASSERT_TRUE(ray.has_value()) << ray.error().message;

    const auto resolved = resolve_scene_surface(**acceleration, *ray);

    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;
    ASSERT_TRUE(resolved->has_value());
    EXPECT_EQ((*acceleration)->frame_scene(), *translated_scene);
    expect_point_near((**resolved).interaction.position(),
                      renderer::Point3{.x = 0.5F, .y = 1.0F, .z = 1.0F});
}

TEST(SceneSurfaceInteractionTest, RejectsHitsWhoseIdentifiersDoNotBelongToTheSnapshot) {
    const auto scene = make_spectral_scene();
    const auto ray = make_surface_ray();

    {
        auto hit = make_surface_hit();
        hit.identifiers.instance = renderer::InstanceId{.value = 404U};
        expect_error(resolve_fixed_hit(scene, hit, ray), core::StatusCode::not_found);
    }
    {
        auto hit = make_surface_hit();
        hit.object = renderer::ObjectId{.value = 404U};
        expect_error(resolve_fixed_hit(scene, hit, ray), core::StatusCode::incompatible);
    }
    {
        auto hit = make_surface_hit();
        hit.identifiers.geometry = renderer::GeometryId{.value = 404U};
        expect_error(resolve_fixed_hit(scene, hit, ray), core::StatusCode::incompatible);
    }
    {
        auto hit = make_surface_hit();
        hit.identifiers.material = renderer::MaterialId{.value = 404U};
        expect_error(resolve_fixed_hit(scene, hit, ray), core::StatusCode::incompatible);
    }
}

TEST(SceneSurfaceInteractionTest, RejectsUnknownPrimitivesWithoutClampingTheIdentifier) {
    const auto scene = make_spectral_scene();
    const auto ray = make_surface_ray();
    auto hit = make_surface_hit();
    hit.identifiers.primitive = renderer::PrimitiveId{.value = 1U};

    expect_error(resolve_fixed_hit(scene, hit, ray), core::StatusCode::incompatible);
}

TEST(SceneSurfaceInteractionTest, RejectsInvalidBarycentricsWithoutRenormalizingThem) {
    const auto scene = make_spectral_scene();
    const auto ray = make_surface_ray();

    {
        auto hit = make_surface_hit();
        hit.triangle.barycentrics.vertex0 = -0.01F;
        hit.triangle.barycentrics.vertex1 = 0.51F;
        expect_error(resolve_fixed_hit(scene, hit, ray), core::StatusCode::invalid_argument);
    }
    {
        auto hit = make_surface_hit();
        hit.triangle.barycentrics.vertex2 = 0.4F;
        expect_error(resolve_fixed_hit(scene, hit, ray), core::StatusCode::invalid_argument);
    }
    {
        auto hit = make_surface_hit();
        hit.triangle.barycentrics.vertex1 =
            std::numeric_limits<renderer::TransportScalar>::quiet_NaN();
        expect_error(resolve_fixed_hit(scene, hit, ray), core::StatusCode::invalid_argument);
    }
}

TEST(SceneSurfaceInteractionTest, RejectsHitParametersOutsideTheSourceRay) {
    const auto scene = make_spectral_scene();
    const auto ray = make_surface_ray();
    auto hit = make_surface_hit();
    hit.triangle.parameter = 9.0F;

    expect_error(resolve_fixed_hit(scene, hit, ray), core::StatusCode::invalid_argument);
}

TEST(SceneBsdfOnlyPathLoopTest, RejectsGeometryOnlyScenesWithoutAnImplicitEnvironment) {
    auto description = make_spectral_scene_description();
    description.spectral_environment.reset();
    description.materials.front().spectral.reset();
    const auto scene = FrameScene::create(std::move(description));
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    const auto acceleration = create_analytic_accel_backend(*scene);
    ASSERT_TRUE(acceleration.has_value()) << acceleration.error().message;
    const auto state =
        renderer::PathState::create_initial(make_wavelengths(), renderer::VacuumMedium);
    ASSERT_TRUE(state.has_value()) << state.error().message;
    const auto stream = renderer::IndependentSampler{17U}.make_stream(0U, 0U, 0U);

    const auto traced = trace_scene_bsdf_only(make_surface_ray(), *state, stream, **acceleration,
                                              renderer::PathDepthLimits{.diffuse = 1U},
                                              renderer::RussianRoulettePolicy::disabled());

    expect_error(traced, core::StatusCode::unavailable);
}

TEST(SceneBsdfOnlyPathLoopTest, RejectsAPathResolvedAtDifferentWavelengths) {
    const auto scene = make_spectral_scene();
    const auto acceleration = create_analytic_accel_backend(scene);
    ASSERT_TRUE(acceleration.has_value()) << acceleration.error().message;
    const auto state =
        renderer::PathState::create_initial(make_wavelengths(0.5F), renderer::VacuumMedium);
    ASSERT_TRUE(state.has_value()) << state.error().message;
    const auto stream = renderer::IndependentSampler{17U}.make_stream(0U, 0U, 0U);

    const auto traced = trace_scene_bsdf_only(make_surface_ray(), *state, stream, **acceleration,
                                              renderer::PathDepthLimits{.diffuse = 1U},
                                              renderer::RussianRoulettePolicy::disabled());

    expect_error(traced, core::StatusCode::incompatible);
}

TEST(SceneBsdfOnlyPathLoopTest, SamplesTheShadingFrameAndKeepsGeometricSupport) {
    constexpr auto geometric_normal = renderer::Normal3{.z = 1.0F};
    constexpr auto shading_normal = renderer::Normal3{.x = 0.6F, .z = 0.8F};
    auto description = make_spectral_scene_description();
    description.geometries.front().mesh = make_surface_mesh(shading_normal);
    const auto scene = FrameScene::create(std::move(description));
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    const auto acceleration = create_analytic_accel_backend(*scene);
    ASSERT_TRUE(acceleration.has_value()) << acceleration.error().message;
    const auto state =
        renderer::PathState::create_initial(make_wavelengths(), renderer::VacuumMedium);
    ASSERT_TRUE(state.has_value()) << state.error().message;
    const auto ray = renderer::Ray::create(renderer::Point3{.x = 0.5F, .y = 1.0F, .z = 2.0F},
                                           renderer::Vector3{.z = -1.0F}, 0.0F, 8.0F, 0.375F,
                                           renderer::AllRayVisibility, renderer::VacuumMedium);
    ASSERT_TRUE(ray.has_value()) << ray.error().message;

    const auto dimensions = renderer::sample_dimensions_for_bounce(0U);
    const auto shading_frame = renderer::OrthonormalFrame::from_normal(shading_normal);
    ASSERT_TRUE(dimensions.has_value()) << dimensions.error().message;
    ASSERT_TRUE(shading_frame.has_value()) << shading_frame.error().message;
    const auto sampler = renderer::IndependentSampler{0xD1B54A32D192ED03ULL};
    auto selected_stream = std::optional<renderer::SampleStream>{};
    auto expected_direction = std::optional<renderer::Vector3>{};
    for (auto sample = std::uint64_t{}; sample < 128U; ++sample) {
        const auto stream = sampler.make_stream(0U, 0U, sample);
        const auto local = renderer::map_cosine_hemisphere(renderer::Point2{
            .x = stream.sample_1d(dimensions->bsdf_u),
            .y = stream.sample_1d(dimensions->bsdf_v),
        });
        ASSERT_TRUE(local.has_value()) << local.error().message;
        const auto world = renderer::normalized(shading_frame->to_world(*local));
        ASSERT_TRUE(world.has_value()) << world.error().message;
        if (renderer::dot(geometric_normal, *world) > 0.2F) {
            selected_stream = stream;
            expected_direction = *world;
            break;
        }
    }
    ASSERT_TRUE(selected_stream.has_value());
    ASSERT_TRUE(expected_direction.has_value());

    const auto traced = trace_scene_bsdf_only(*ray, *state, *selected_stream, **acceleration,
                                              renderer::PathDepthLimits{.diffuse = 1U},
                                              renderer::RussianRoulettePolicy::disabled());
    ASSERT_TRUE(traced.has_value()) << traced.error().message;
    EXPECT_EQ(traced->termination, renderer::BsdfOnlyPathTermination::escaped_environment);
    EXPECT_EQ(traced->state.depth(), 1U);
    expect_vector_near(traced->terminal_ray.direction(), *expected_direction, 2.0e-6F);
    EXPECT_GT(renderer::dot(geometric_normal, traced->terminal_ray.direction()), 0.0F);
    EXPECT_GT(renderer::dot(shading_normal, traced->terminal_ray.direction()), 0.0F);
}

TEST(SceneBsdfOnlyPathLoopTest, ZeroDepthLimitDisablesAClosureInsideAContinuousMixture) {
    const auto make_description = [](const bool include_disabled_diffuse) {
        auto description = make_spectral_scene_description();
        auto closures = renderer::ClosureSet{};
        if (include_disabled_diffuse) {
            EXPECT_EQ(closures.append_lambertian_reflection(
                          renderer::TransportSpectrum{.values = {0.8F, 0.8F, 0.8F, 0.8F}}),
                      renderer::ClosureAppendStatus::appended);
        }
        EXPECT_EQ(closures.append_rough_conductor_reflection(
                      renderer::TransportSpectrum{.values = {0.85F, 0.85F, 0.85F, 0.85F}},
                      renderer::TransportSpectrum{.values = {0.25F, 0.45F, 0.75F, 1.10F}},
                      renderer::TransportSpectrum{.values = {3.2F, 2.7F, 2.2F, 1.8F}}, 0.3F, 0.3F),
                  renderer::ClosureAppendStatus::appended);

        const auto mixture =
            include_disabled_diffuse
                ? SceneClosureMixture::create(closures,
                                              std::array<renderer::TransportScalar, 2U>{0.5F, 0.5F})
                : SceneClosureMixture::create(closures,
                                              std::array<renderer::TransportScalar, 1U>{1.0F});
        EXPECT_TRUE(mixture.has_value()) << (mixture ? "" : mixture.error().message);
        if (mixture) {
            description.materials.front().spectral->closure_mixture = *mixture;
        }
        description.materials.front().spectral->emitted_radiance = {};
        return description;
    };

    const auto glossy_scene = FrameScene::create(make_description(false));
    const auto mixed_scene = FrameScene::create(make_description(true));
    ASSERT_TRUE(glossy_scene.has_value()) << glossy_scene.error().message;
    ASSERT_TRUE(mixed_scene.has_value()) << mixed_scene.error().message;
    const auto glossy_acceleration = create_analytic_accel_backend(*glossy_scene);
    const auto mixed_acceleration = create_analytic_accel_backend(*mixed_scene);
    ASSERT_TRUE(glossy_acceleration.has_value()) << glossy_acceleration.error().message;
    ASSERT_TRUE(mixed_acceleration.has_value()) << mixed_acceleration.error().message;

    const auto state =
        renderer::PathState::create_initial(make_wavelengths(), renderer::VacuumMedium);
    const auto dimensions = renderer::sample_dimensions_for_bounce(0U);
    ASSERT_TRUE(state.has_value()) << state.error().message;
    ASSERT_TRUE(dimensions.has_value()) << dimensions.error().message;
    const auto ray = renderer::Ray::create(renderer::Point3{.x = 0.5F, .y = 1.0F, .z = 2.0F},
                                           renderer::Vector3{.z = -1.0F}, 0.0F, 8.0F, 0.375F,
                                           renderer::AllRayVisibility, renderer::VacuumMedium);
    ASSERT_TRUE(ray.has_value()) << ray.error().message;

    const auto sampler = renderer::IndependentSampler{0x94D049BB133111EBULL};
    auto selected_stream = std::optional<renderer::SampleStream>{};
    for (auto sample = std::uint64_t{}; sample < 128U; ++sample) {
        const auto stream = sampler.make_stream(0U, 0U, sample);
        if (stream.sample_1d(dimensions->bsdf_component) <= 0.5F) {
            continue;
        }
        const auto candidate = trace_scene_bsdf_only(*ray, *state, stream, **glossy_acceleration,
                                                     renderer::PathDepthLimits{.glossy = 1U},
                                                     renderer::RussianRoulettePolicy::disabled());
        if (candidate &&
            candidate->termination == renderer::BsdfOnlyPathTermination::escaped_environment) {
            selected_stream = stream;
            break;
        }
    }
    ASSERT_TRUE(selected_stream.has_value());

    constexpr auto limits = renderer::PathDepthLimits{.glossy = 1U};
    const auto glossy = trace_scene_bsdf_only(*ray, *state, *selected_stream, **glossy_acceleration,
                                              limits, renderer::RussianRoulettePolicy::disabled());
    const auto mixed = trace_scene_bsdf_only(*ray, *state, *selected_stream, **mixed_acceleration,
                                             limits, renderer::RussianRoulettePolicy::disabled());
    ASSERT_TRUE(glossy.has_value()) << glossy.error().message;
    ASSERT_TRUE(mixed.has_value()) << mixed.error().message;
    ASSERT_EQ(glossy->termination, renderer::BsdfOnlyPathTermination::escaped_environment);
    ASSERT_EQ(mixed->termination, renderer::BsdfOnlyPathTermination::escaped_environment);
    EXPECT_EQ(glossy->state.depth_counters(), (renderer::PathDepthCounters{.glossy = 1U}));
    EXPECT_EQ(mixed->state.depth_counters(), glossy->state.depth_counters());
    expect_vector_near(mixed->terminal_ray.direction(), glossy->terminal_ray.direction());
    for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        EXPECT_NEAR(mixed->state.accumulated_radiance()[lane],
                    glossy->state.accumulated_radiance()[lane], 1.0e-6F);
    }
}

} // namespace
} // namespace blackframe::engine
