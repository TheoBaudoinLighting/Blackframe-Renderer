#include <Blackframe/Engine/SceneVisibility.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace blackframe::engine {
namespace {

class RecordingOcclusionBackend final : public AccelBackend {
  public:
    explicit RecordingOcclusionBackend(
        core::Result<bool> result,
        const renderer::RayMask blocker_mask = renderer::AllRayVisibility)
        : AccelBackend{FrameSceneHandle{}}, result_{std::move(result)},
          blocker_mask_{blocker_mask} {}

    [[nodiscard]] AccelBackendKind kind() const noexcept override {
        return AccelBackendKind::analytic_reference;
    }

    [[nodiscard]] core::Result<std::optional<AccelHit>>
    closest_hit(const renderer::Ray&) const override {
        return std::optional<AccelHit>{};
    }

    [[nodiscard]] core::Result<bool> occluded(const renderer::Ray& ray) const override {
        ++occlusion_calls_;
        observed_mask_ = ray.mask();
        if (!result_) {
            return std::unexpected(result_.error());
        }
        return *result_ && (ray.mask() & blocker_mask_) != 0U;
    }

    [[nodiscard]] core::Status rebuild(FrameSceneHandle) override {
        return std::unexpected(core::Error{
            .code = core::StatusCode::unavailable,
            .message = "Synthetic visibility backend does not rebuild.",
        });
    }

    [[nodiscard]] core::Status refit(FrameSceneHandle) override {
        return std::unexpected(core::Error{
            .code = core::StatusCode::unavailable,
            .message = "Synthetic visibility backend does not refit.",
        });
    }

    [[nodiscard]] std::size_t occlusion_calls() const noexcept {
        return occlusion_calls_;
    }

    [[nodiscard]] renderer::RayMask observed_mask() const noexcept {
        return observed_mask_;
    }

  private:
    core::Result<bool> result_;
    renderer::RayMask blocker_mask_;
    mutable std::size_t occlusion_calls_{};
    mutable renderer::RayMask observed_mask_{};
};

[[nodiscard]] renderer::Ray
make_visibility_ray(const renderer::RayMask mask = renderer::AllRayVisibility,
                    const renderer::MediumId medium = renderer::VacuumMedium) {
    return renderer::Ray::create(renderer::Point3{.x = 1.0F, .y = 2.0F, .z = 3.0F},
                                 renderer::Vector3{.x = 0.0F, .y = 0.0F, .z = 1.0F}, 0.0F,
                                 std::numeric_limits<renderer::TransportScalar>::infinity(), 0.25F,
                                 mask, medium)
        .value();
}

TEST(SceneVisibilityTest, ReturnsUnitTransmittanceForAnUnoccludedVacuumRay) {
    auto backend = RecordingOcclusionBackend{false};

    const auto transmittance = trace_vacuum_visibility(backend, make_visibility_ray());

    ASSERT_TRUE(transmittance.has_value()) << transmittance.error().message;
    EXPECT_EQ(*transmittance, (renderer::TransportSpectrum{.values = {1.0F, 1.0F, 1.0F, 1.0F}}));
    EXPECT_EQ(backend.occlusion_calls(), 1U);
}

TEST(SceneVisibilityTest, ReturnsZeroTransmittanceForAnOccludedVacuumRay) {
    auto backend = RecordingOcclusionBackend{true};

    const auto transmittance = trace_vacuum_visibility(backend, make_visibility_ray());

    ASSERT_TRUE(transmittance.has_value()) << transmittance.error().message;
    EXPECT_EQ(*transmittance, renderer::TransportSpectrum{});
    EXPECT_EQ(backend.occlusion_calls(), 1U);
}

TEST(SceneVisibilityTest, PreservesVisibilityMasksIncludingTheZeroMask) {
    constexpr auto blocker_mask = renderer::RayMask{1U << 7U};
    auto backend = RecordingOcclusionBackend{true, blocker_mask};

    const auto unrelated =
        trace_vacuum_visibility(backend, make_visibility_ray(renderer::RayMask{1U << 3U}));
    ASSERT_TRUE(unrelated.has_value()) << unrelated.error().message;
    EXPECT_EQ(*unrelated, (renderer::TransportSpectrum{.values = {1.0F, 1.0F, 1.0F, 1.0F}}));
    EXPECT_EQ(backend.observed_mask(), renderer::RayMask{1U << 3U});

    const auto matching = trace_vacuum_visibility(backend, make_visibility_ray(blocker_mask));
    ASSERT_TRUE(matching.has_value()) << matching.error().message;
    EXPECT_EQ(*matching, renderer::TransportSpectrum{});
    EXPECT_EQ(backend.observed_mask(), blocker_mask);

    const auto zero = trace_vacuum_visibility(backend, make_visibility_ray(renderer::RayMask{}));
    ASSERT_TRUE(zero.has_value()) << zero.error().message;
    EXPECT_EQ(*zero, (renderer::TransportSpectrum{.values = {1.0F, 1.0F, 1.0F, 1.0F}}));
    EXPECT_EQ(backend.observed_mask(), renderer::RayMask{});
    EXPECT_EQ(backend.occlusion_calls(), 3U);
}

TEST(SceneVisibilityTest, RejectsNonVacuumBeforeTraversal) {
    auto backend = RecordingOcclusionBackend{true};

    const auto transmittance = trace_vacuum_visibility(
        backend, make_visibility_ray(renderer::AllRayVisibility, renderer::MediumId{.value = 17U}));

    ASSERT_FALSE(transmittance.has_value());
    EXPECT_EQ(transmittance.error().code, core::StatusCode::unavailable);
    EXPECT_EQ(transmittance.error().message,
              "Scene visibility supports vacuum transmittance only.");
    EXPECT_EQ(backend.occlusion_calls(), 0U);
}

TEST(SceneVisibilityTest, PropagatesBackendErrorsWithoutChangingTheirDiagnostics) {
    const auto expected = core::Error{
        .code = core::StatusCode::platform_error,
        .message = "Synthetic acceleration traversal failed.",
    };
    auto backend = RecordingOcclusionBackend{std::unexpected(expected)};

    const auto transmittance = trace_vacuum_visibility(backend, make_visibility_ray());

    ASSERT_FALSE(transmittance.has_value());
    EXPECT_EQ(transmittance.error().code, expected.code);
    EXPECT_EQ(transmittance.error().message, expected.message);
    EXPECT_EQ(backend.occlusion_calls(), 1U);
}

} // namespace
} // namespace blackframe::engine
