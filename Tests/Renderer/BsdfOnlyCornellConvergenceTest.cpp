#include <Blackframe/Renderer/BsdfOnlyPathLoop.hpp>
#include <Blackframe/Renderer/IndependentSampler.hpp>
#include <Blackframe/Renderer/WavelengthSampling.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace blackframe::renderer {
namespace {

template <SpectrumScalar Scalar>
using SpectrumFor = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;

template <SpectrumScalar Scalar> using SurfaceFor = BsdfOnlyTriangleSurfaceT<Scalar>;

template <SpectrumScalar Scalar> using WavelengthsFor = SampledWavelengthsT<Scalar>;

template <SpectrumScalar Scalar>
[[nodiscard]] SpectrumFor<Scalar>
spectrum(const std::array<Scalar, TransportSpectrumSampleCount> values) {
    return SpectrumFor<Scalar>{.values = values};
}

template <SpectrumScalar Scalar>
[[nodiscard]] SpectrumFor<Scalar> constant_spectrum(const Scalar value) {
    auto result = SpectrumFor<Scalar>{};
    result.values.fill(value);
    return result;
}

template <SpectrumScalar Scalar> [[nodiscard]] WavelengthsFor<Scalar> fixed_wavelengths() {
    const auto stream = IndependentSamplerT<Scalar>{0}.make_stream(0, 0, 0);
    return sample_visible_wavelengths(stream);
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<SurfaceFor<Scalar>>
make_surface(const Point3T<Scalar> vertex0, const Point3T<Scalar> vertex1,
             const Point3T<Scalar> vertex2, const SpectrumFor<Scalar>& reflectance,
             const SpectrumFor<Scalar>& radiance, const WavelengthsFor<Scalar>& wavelengths) {
    const auto triangle = TriangleT<Scalar>::create(vertex0, vertex1, vertex2);
    if (!triangle.has_value()) {
        return std::unexpected(triangle.error());
    }
    const auto reflection = LambertianReflectionT<Scalar>::create(reflectance);
    if (!reflection.has_value()) {
        return std::unexpected(reflection.error());
    }
    const auto emission = OneSidedSurfaceEmissionT<Scalar>::create(radiance);
    if (!emission.has_value()) {
        return std::unexpected(emission.error());
    }
    return SurfaceFor<Scalar>{*triangle, *reflection, *emission, wavelengths, AllRayVisibility};
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Status
append_quad(std::vector<SurfaceFor<Scalar>>& surfaces, const Point3T<Scalar> vertex0,
            const Point3T<Scalar> vertex1, const Point3T<Scalar> vertex2,
            const Point3T<Scalar> vertex3, const SpectrumFor<Scalar>& reflectance,
            const SpectrumFor<Scalar>& radiance, const WavelengthsFor<Scalar>& wavelengths) {
    const auto first = make_surface(vertex0, vertex1, vertex2, reflectance, radiance, wavelengths);
    if (!first.has_value()) {
        return std::unexpected(first.error());
    }
    const auto second = make_surface(vertex0, vertex2, vertex3, reflectance, radiance, wavelengths);
    if (!second.has_value()) {
        return std::unexpected(second.error());
    }
    surfaces.push_back(*first);
    surfaces.push_back(*second);
    return {};
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<std::vector<SurfaceFor<Scalar>>>
make_cornell_surfaces(const WavelengthsFor<Scalar>& wavelengths) {
    const auto white = constant_spectrum(Scalar{0.75});
    const auto left_tint =
        spectrum<Scalar>({Scalar{0.75}, Scalar{0.25}, Scalar{0.25}, Scalar{0.2}});
    const auto right_tint =
        spectrum<Scalar>({Scalar{0.2}, Scalar{0.75}, Scalar{0.25}, Scalar{0.2}});
    const auto black = SpectrumFor<Scalar>{};
    const auto ceiling_radiance = spectrum<Scalar>({Scalar{8}, Scalar{6}, Scalar{4}, Scalar{2}});

    auto surfaces = std::vector<SurfaceFor<Scalar>>{};
    surfaces.reserve(10);
    for (const auto& status : std::array{
             // Back wall, +Z.
             append_quad(surfaces, Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{-1}},
                         Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{-1}},
                         Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{1}},
                         Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{1}}, white, black,
                         wavelengths),
             // Floor, +Y.
             append_quad(surfaces, Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{-1}},
                         Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{-1}, .z = Scalar{2}},
                         Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{-1}, .z = Scalar{2}},
                         Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{-1}}, white, black,
                         wavelengths),
             // Entire ceiling emitter, -Y. Its large area keeps this BSDF-only regression bounded.
             append_quad(surfaces, Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{1}},
                         Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{1}},
                         Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{1}, .z = Scalar{2}},
                         Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{1}, .z = Scalar{2}}, black,
                         ceiling_radiance, wavelengths),
             // Left wall, +X.
             append_quad(surfaces, Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{-1}},
                         Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{1}},
                         Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{1}, .z = Scalar{2}},
                         Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{-1}, .z = Scalar{2}},
                         left_tint, black, wavelengths),
             // Right wall, -X.
             append_quad(surfaces, Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{-1}},
                         Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{-1}, .z = Scalar{2}},
                         Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{1}, .z = Scalar{2}},
                         Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{1}}, right_tint, black,
                         wavelengths),
         }) {
        if (!status.has_value()) {
            return std::unexpected(status.error());
        }
    }
    return surfaces;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<RayT<Scalar>> make_probe_ray(const Vector3T<Scalar> direction) {
    return RayT<Scalar>::create(Point3T<Scalar>{.z = Scalar{3}}, direction, Scalar{0},
                                std::numeric_limits<Scalar>::infinity(), Scalar{0},
                                AllRayVisibility, VacuumMedium);
}

template <SpectrumScalar Scalar> [[nodiscard]] std::array<Vector3T<Scalar>, 3> probe_directions() {
    return {
        Vector3T<Scalar>{.z = Scalar{-3}},
        Vector3T<Scalar>{.y = Scalar{-1}, .z = Scalar{-2}},
        Vector3T<Scalar>{.x = Scalar{-1}, .z = Scalar{-2}},
    };
}

template <SpectrumScalar Scalar> struct CornellEstimateT final {
    std::array<SpectrumFor<Scalar>, 3> probe_radiance{};
    std::uint64_t first_bounce_light_paths{};
    std::uint64_t multiple_bounce_light_paths{};
};

template <SpectrumScalar Scalar>
[[nodiscard]] bool has_radiance(const SpectrumFor<Scalar>& radiance) {
    return std::ranges::any_of(radiance.values,
                               [](const Scalar lane) { return lane != Scalar{0}; });
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<CornellEstimateT<Scalar>>
estimate_cornell(const std::vector<SurfaceFor<Scalar>>& surfaces,
                 const WavelengthsFor<Scalar>& wavelengths, const std::uint64_t seed,
                 const std::uint32_t sample_count) {
    const auto constant_environment = ConstantEnvironmentT<Scalar>::create(SpectrumFor<Scalar>{});
    if (!constant_environment.has_value()) {
        return std::unexpected(constant_environment.error());
    }
    const auto environment = BsdfOnlyEnvironmentT<Scalar>{*constant_environment, wavelengths};

    auto estimate = CornellEstimateT<Scalar>{};
    const auto sampler = IndependentSamplerT<Scalar>{seed};
    const auto directions = probe_directions<Scalar>();
    for (auto probe = std::size_t{0}; probe < directions.size(); ++probe) {
        const auto ray = make_probe_ray(directions[probe]);
        if (!ray.has_value()) {
            return std::unexpected(ray.error());
        }

        for (auto sample_index = std::uint64_t{0}; sample_index < sample_count; ++sample_index) {
            const auto stream =
                sampler.make_stream(static_cast<std::uint32_t>(probe), 0, sample_index);
            const auto state = PathStateT<Scalar>::create_initial(wavelengths, VacuumMedium);
            if (!state.has_value()) {
                return std::unexpected(state.error());
            }
            const auto traced =
                trace_bsdf_only(*ray, *state, stream, std::span<const SurfaceFor<Scalar>>{surfaces},
                                environment, PathDepthLimits{.diffuse = 3});
            if (!traced.has_value()) {
                return std::unexpected(traced.error());
            }
            const auto& counters = traced->state.depth_counters();
            if (counters.diffuse != traced->state.depth() || counters.glossy != 0 ||
                counters.specular != 0 || counters.transmission != 0 || counters.volume != 0) {
                return std::unexpected(core::Error{
                    .code = core::StatusCode::invalid_argument,
                    .message = "Cornell BSDF-only depth counters are inconsistent.",
                });
            }

            const auto& radiance = traced->state.accumulated_radiance();
            for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
                estimate.probe_radiance[probe][lane] += radiance[lane];
            }
            if (has_radiance(radiance)) {
                if (traced->state.depth() <= 2) {
                    ++estimate.first_bounce_light_paths;
                } else {
                    ++estimate.multiple_bounce_light_paths;
                }
            }
        }

        for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
            estimate.probe_radiance[probe][lane] /= static_cast<Scalar>(sample_count);
        }
    }
    return estimate;
}

template <SpectrumScalar Scalar, std::size_t Count>
[[nodiscard]] std::array<SpectrumFor<Scalar>, 3>
mean_estimates(const std::array<CornellEstimateT<Scalar>, Count>& estimates) {
    auto mean = std::array<SpectrumFor<Scalar>, 3>{};
    for (const auto& estimate : estimates) {
        for (auto probe = std::size_t{0}; probe < mean.size(); ++probe) {
            for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
                mean[probe][lane] += estimate.probe_radiance[probe][lane];
            }
        }
    }
    for (auto& probe : mean) {
        for (auto& lane : probe.values) {
            lane /= static_cast<Scalar>(Count);
        }
    }
    return mean;
}

[[nodiscard]] std::array<ReferenceSpectrum, 3>
to_reference(const std::array<TransportSpectrum, 3>& transport) {
    auto result = std::array<ReferenceSpectrum, 3>{};
    for (auto probe = std::size_t{0}; probe < result.size(); ++probe) {
        for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
            result[probe][lane] = static_cast<ReferenceScalar>(transport[probe][lane]);
        }
    }
    return result;
}

[[nodiscard]] ReferenceScalar relative_rmse(const std::array<ReferenceSpectrum, 3>& estimate,
                                            const std::array<ReferenceSpectrum, 3>& reference) {
    auto squared_error = ReferenceScalar{0};
    auto squared_reference = ReferenceScalar{0};
    for (auto probe = std::size_t{0}; probe < estimate.size(); ++probe) {
        for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
            const auto difference = estimate[probe][lane] - reference[probe][lane];
            squared_error = std::fma(difference, difference, squared_error);
            squared_reference =
                std::fma(reference[probe][lane], reference[probe][lane], squared_reference);
        }
    }
    return std::sqrt(squared_error / squared_reference);
}

[[nodiscard]] ReferenceScalar median(std::array<ReferenceScalar, 8> values) {
    std::ranges::sort(values);
    return (values[3] + values[4]) * 0.5;
}

[[nodiscard]] core::Result<std::size_t>
closest_surface(const ReferenceRay& ray,
                const std::span<const ReferenceBsdfOnlyTriangleSurface> surfaces) {
    auto nearest_parameter = std::numeric_limits<ReferenceScalar>::infinity();
    auto nearest_index = std::optional<std::size_t>{};
    for (auto index = std::size_t{0}; index < surfaces.size(); ++index) {
        const auto intersection = surfaces[index].triangle().intersect(ray);
        if (!intersection.has_value()) {
            return std::unexpected(intersection.error());
        }
        if (intersection->has_value() && (**intersection).parameter < nearest_parameter) {
            nearest_parameter = (**intersection).parameter;
            nearest_index = index;
        }
    }
    if (!nearest_index.has_value()) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::not_found,
            .message = "A Cornell validation probe missed its expected diffuse surface.",
        });
    }
    return *nearest_index;
}

TEST(BsdfOnlyCornellConvergenceTest, ConvergesAcrossSeedsWithoutNee) {
    const auto reference_wavelengths = fixed_wavelengths<ReferenceScalar>();
    const auto reference_surfaces = make_cornell_surfaces(reference_wavelengths);
    ASSERT_TRUE(reference_surfaces.has_value());
    ASSERT_EQ(reference_surfaces->size(), 10U);

    constexpr auto expected_surface_pairs =
        std::array{std::array{0U, 1U}, std::array{2U, 3U}, std::array{6U, 7U}};
    const auto directions = probe_directions<ReferenceScalar>();
    for (auto probe = std::size_t{0}; probe < directions.size(); ++probe) {
        const auto ray = make_probe_ray(directions[probe]);
        ASSERT_TRUE(ray.has_value());
        const auto closest = closest_surface(*ray, *reference_surfaces);
        ASSERT_TRUE(closest.has_value());
        EXPECT_TRUE(*closest == expected_surface_pairs[probe][0] ||
                    *closest == expected_surface_pairs[probe][1]);
    }

    const auto reference_a =
        estimate_cornell(*reference_surfaces, reference_wavelengths, 0x243F6A8885A308D3ULL, 4096);
    const auto reference_b =
        estimate_cornell(*reference_surfaces, reference_wavelengths, 0x13198A2E03707344ULL, 4096);
    ASSERT_TRUE(reference_a.has_value())
        << (reference_a.has_value() ? "" : reference_a.error().message);
    ASSERT_TRUE(reference_b.has_value())
        << (reference_b.has_value() ? "" : reference_b.error().message);
    const auto reference_parts = std::array{*reference_a, *reference_b};
    const auto reference = mean_estimates(reference_parts);

    for (const auto& probe : reference) {
        auto energy = ReferenceScalar{0};
        for (const auto lane : probe.values) {
            ASSERT_TRUE(std::isfinite(lane));
            ASSERT_GE(lane, 0.0);
            energy += lane;
        }
        EXPECT_GT(energy, 0.0);
    }
    EXPECT_GT(reference_a->first_bounce_light_paths + reference_b->first_bounce_light_paths, 0U);
    EXPECT_GT(reference_a->multiple_bounce_light_paths + reference_b->multiple_bounce_light_paths,
              0U);
    EXPECT_LT(relative_rmse(reference_a->probe_radiance, reference_b->probe_radiance), 0.2);

    constexpr auto evaluation_seeds = std::array{
        0xA4093822299F31D0ULL, 0x082EFA98EC4E6C89ULL, 0x452821E638D01377ULL, 0xBE5466CF34E90C6CULL,
        0xC0AC29B7C97C50DDULL, 0x3F84D5B5B5470917ULL, 0x9216D5D98979FB1BULL, 0xD1310BA698DFB5ACULL,
    };
    auto errors_32 = std::array<ReferenceScalar, evaluation_seeds.size()>{};
    auto errors_128 = std::array<ReferenceScalar, evaluation_seeds.size()>{};
    auto estimates_128 = std::array<CornellEstimateT<ReferenceScalar>, evaluation_seeds.size()>{};
    for (auto index = std::size_t{0}; index < evaluation_seeds.size(); ++index) {
        const auto estimate_32 = estimate_cornell(*reference_surfaces, reference_wavelengths,
                                                  evaluation_seeds[index], 32);
        const auto estimate_128 = estimate_cornell(*reference_surfaces, reference_wavelengths,
                                                   evaluation_seeds[index], 128);
        ASSERT_TRUE(estimate_32.has_value());
        ASSERT_TRUE(estimate_128.has_value());
        errors_32[index] = relative_rmse(estimate_32->probe_radiance, reference);
        errors_128[index] = relative_rmse(estimate_128->probe_radiance, reference);
        estimates_128[index] = *estimate_128;
    }
    EXPECT_LT(median(errors_128), median(errors_32) * 0.72);
    EXPECT_LT(relative_rmse(mean_estimates(estimates_128), reference), 0.12);

    const auto replay =
        estimate_cornell(*reference_surfaces, reference_wavelengths, evaluation_seeds[0], 128);
    ASSERT_TRUE(replay.has_value());
    EXPECT_EQ(replay->probe_radiance, estimates_128[0].probe_radiance);
    EXPECT_EQ(replay->first_bounce_light_paths, estimates_128[0].first_bounce_light_paths);
    EXPECT_EQ(replay->multiple_bounce_light_paths, estimates_128[0].multiple_bounce_light_paths);

    const auto transport_wavelengths = fixed_wavelengths<TransportScalar>();
    const auto transport_surfaces = make_cornell_surfaces(transport_wavelengths);
    ASSERT_TRUE(transport_surfaces.has_value());
    auto transport_estimates =
        std::array<CornellEstimateT<TransportScalar>, evaluation_seeds.size()>{};
    for (auto index = std::size_t{0}; index < evaluation_seeds.size(); ++index) {
        const auto estimate = estimate_cornell(*transport_surfaces, transport_wavelengths,
                                               evaluation_seeds[index], 128);
        ASSERT_TRUE(estimate.has_value());
        transport_estimates[index] = *estimate;
    }
    EXPECT_LT(relative_rmse(to_reference(mean_estimates(transport_estimates)), reference), 0.15);
}

} // namespace
} // namespace blackframe::renderer
