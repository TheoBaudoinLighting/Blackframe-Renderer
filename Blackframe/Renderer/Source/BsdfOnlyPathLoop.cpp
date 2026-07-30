#include <Blackframe/Renderer/BsdfOnlyPathLoop.hpp>
#include <Blackframe/Renderer/LocalFrame.hpp>
#include <Blackframe/Renderer/RayOriginOffset.hpp>
#include <Blackframe/Renderer/SampleDimensionMap.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <utility>

namespace blackframe::renderer {
namespace {

[[nodiscard]] core::Error path_loop_error(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

template <SpectrumScalar Scalar>
using PathSpectrum = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;

template <SpectrumScalar Scalar>
[[nodiscard]] bool finite_non_negative(const PathSpectrum<Scalar>& spectrum) noexcept {
    for (const auto value : spectrum.values) {
        if (!std::isfinite(value) || value < Scalar{0}) {
            return false;
        }
    }
    return true;
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool zero_spectrum(const PathSpectrum<Scalar>& spectrum) noexcept {
    for (const auto value : spectrum.values) {
        if (value != Scalar{0}) {
            return false;
        }
    }
    return true;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<PathSpectrum<Scalar>> checked_product(const PathSpectrum<Scalar>& left,
                                                                 const PathSpectrum<Scalar>& right,
                                                                 const char* const error_message) {
    auto result = PathSpectrum<Scalar>{};
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        result[lane] = left[lane] * right[lane];
        if (!std::isfinite(result[lane]) ||
            (left[lane] != Scalar{0} && right[lane] != Scalar{0} && result[lane] == Scalar{0})) {
            return std::unexpected(path_loop_error(error_message));
        }
    }
    return result;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<PathSpectrum<Scalar>>
accumulate_emission(const PathSpectrum<Scalar>& accumulated, const PathSpectrum<Scalar>& throughput,
                    const PathSpectrum<Scalar>& emitted_radiance) {
    const auto contribution =
        checked_product(throughput, emitted_radiance,
                        "BSDF-only emitted-radiance multiplication is not representable.");
    if (!contribution.has_value()) {
        return std::unexpected(contribution.error());
    }

    auto result = PathSpectrum<Scalar>{};
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        result[lane] = accumulated[lane] + (*contribution)[lane];
        if (!std::isfinite(result[lane])) {
            return std::unexpected(
                path_loop_error("BSDF-only radiance accumulation is not representable."));
        }
    }
    return result;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<Vector3T<Scalar>>
robust_unit_direction(const Vector3T<Scalar> direction) {
    if (!std::isfinite(direction.x) || !std::isfinite(direction.y) || !std::isfinite(direction.z)) {
        return std::unexpected(path_loop_error("A BSDF-only path direction must remain finite."));
    }

    const auto maximum_component =
        std::max({std::abs(direction.x), std::abs(direction.y), std::abs(direction.z)});
    if (maximum_component == Scalar{0}) {
        return std::unexpected(path_loop_error("A BSDF-only path direction must remain non-zero."));
    }

    const auto scaled = direction / maximum_component;
    const auto magnitude = std::sqrt(length_squared(scaled));
    const auto result = scaled / magnitude;
    if (!std::isfinite(result.x) || !std::isfinite(result.y) || !std::isfinite(result.z)) {
        return std::unexpected(
            path_loop_error("BSDF-only path normalization is not representable."));
    }
    return result;
}

template <SpectrumScalar Scalar> struct ResolvedTriangleHit final {
    const BsdfOnlyTriangleSurfaceT<Scalar>* surface;
    TriangleHitT<Scalar> hit;
};

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<std::optional<ResolvedTriangleHit<Scalar>>>
closest_hit(const RayT<Scalar>& ray,
            const std::span<const BsdfOnlyTriangleSurfaceT<Scalar>> surfaces) {
    auto nearest = std::optional<ResolvedTriangleHit<Scalar>>{};
    for (const auto& surface : surfaces) {
        if ((ray.mask() & surface.visibility_mask()) == 0) {
            continue;
        }

        const auto intersection = surface.triangle().intersect(ray);
        if (!intersection.has_value()) {
            return std::unexpected(intersection.error());
        }
        if (intersection->has_value() &&
            (!nearest.has_value() || (**intersection).parameter < nearest->hit.parameter)) {
            nearest = ResolvedTriangleHit<Scalar>{
                .surface = &surface,
                .hit = **intersection,
            };
        }
    }
    return nearest;
}

template <SpectrumScalar Scalar> struct TrianglePositionWithError final {
    Point3T<Scalar> point;
    Vector3T<Scalar> absolute_error;
};

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<TrianglePositionWithError<Scalar>>
triangle_position_with_error(const ResolvedTriangleHit<Scalar>& resolved) {
    constexpr auto epsilon = std::numeric_limits<Scalar>::epsilon();
    constexpr auto gamma7 = (Scalar{7} * epsilon) / (Scalar{1} - Scalar{7} * epsilon);

    const auto& vertices = resolved.surface->triangle().vertices();
    const auto barycentrics = std::array{
        resolved.hit.barycentrics.vertex0,
        resolved.hit.barycentrics.vertex1,
        resolved.hit.barycentrics.vertex2,
    };
    const auto vertex_coordinates = std::array{
        std::array{vertices[0].x, vertices[0].y, vertices[0].z},
        std::array{vertices[1].x, vertices[1].y, vertices[1].z},
        std::array{vertices[2].x, vertices[2].y, vertices[2].z},
    };
    auto triangle_scale = Scalar{0};
    for (auto axis = std::size_t{0}; axis < std::size_t{3}; ++axis) {
        triangle_scale = std::max(
            {triangle_scale, std::abs(vertex_coordinates[1][axis] - vertex_coordinates[0][axis]),
             std::abs(vertex_coordinates[2][axis] - vertex_coordinates[0][axis])});
    }
    const auto representable_floor = epsilon * triangle_scale;
    if (!std::isfinite(triangle_scale) || !std::isfinite(representable_floor) ||
        !(representable_floor > Scalar{0})) {
        return std::unexpected(
            path_loop_error("The BSDF-only triangle position-error scale is not representable."));
    }

    auto point_components = std::array<Scalar, 3>{};
    auto error_components = std::array<Scalar, 3>{};
    for (auto axis = std::size_t{0}; axis < point_components.size(); ++axis) {
        const auto products = std::array{barycentrics[0] * vertex_coordinates[0][axis],
                                         barycentrics[1] * vertex_coordinates[1][axis],
                                         barycentrics[2] * vertex_coordinates[2][axis]};
        point_components[axis] =
            std::fma(barycentrics[0], vertex_coordinates[0][axis],
                     std::fma(barycentrics[1], vertex_coordinates[1][axis], products[2]));

        auto interpolation_magnitude = Scalar{0};
        for (const auto product : products) {
            interpolation_magnitude += std::abs(product);
        }
        // Exact zero coordinates still need a normal-range displacement for the watertight
        // predicate. The floor comes from this triangle's edge scale and the active format.
        error_components[axis] = std::max(gamma7 * interpolation_magnitude, representable_floor);
        if (!std::isfinite(point_components[axis]) || !std::isfinite(interpolation_magnitude) ||
            !std::isfinite(error_components[axis]) || error_components[axis] < Scalar{0}) {
            return std::unexpected(path_loop_error(
                "The BSDF-only triangle position reconstruction is not representable."));
        }
    }

    return TrianglePositionWithError<Scalar>{
        .point =
            {
                .x = point_components[0],
                .y = point_components[1],
                .z = point_components[2],
            },
        .absolute_error =
            {
                .x = error_components[0],
                .y = error_components[1],
                .z = error_components[2],
            },
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<BsdfOnlyPathResultT<Scalar>>
make_result(const PathSpectrum<Scalar>& beta, const PathSpectrum<Scalar>& radiance,
            const std::uint32_t depth, const Scalar eta_scale,
            const SampledWavelengthsT<Scalar>& wavelengths, const PathDeltaFlags delta_flags,
            const MediumId current_medium, const PathDepthLimits& depth_limits,
            const PathDepthCounters& depth_counters, const RayT<Scalar>& terminal_ray,
            const BsdfOnlyPathTermination termination, const ScatteringLobe blocked_depth_limits) {
    const auto depth_status = validate_path_depth_state(depth_limits, depth_counters, depth);
    if (!depth_status.has_value()) {
        return std::unexpected(depth_status.error());
    }
    const auto state = PathStateT<Scalar>::create(beta, radiance, depth_counters, eta_scale,
                                                  wavelengths, delta_flags, current_medium);
    if (!state.has_value()) {
        return std::unexpected(state.error());
    }
    return BsdfOnlyPathResultT<Scalar>{
        .state = *state,
        .terminal_ray = terminal_ray,
        .termination = termination,
        .blocked_depth_limits = blocked_depth_limits,
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<BsdfOnlyPathResultT<Scalar>>
trace_bsdf_only_impl(const RayT<Scalar>& initial_ray, const PathStateT<Scalar>& initial_state,
                     const SampleStreamT<Scalar>& sample_stream,
                     const std::span<const BsdfOnlyTriangleSurfaceT<Scalar>> surfaces,
                     const BsdfOnlyEnvironmentT<Scalar>& environment,
                     const PathDepthLimits& depth_limits) {
    if (initial_ray.current_medium() != initial_state.current_medium()) {
        return std::unexpected(path_loop_error(
            "The BSDF-only ray and path state must carry the same current medium."));
    }
    if (!finite_non_negative(initial_state.beta())) {
        return std::unexpected(
            path_loop_error("BSDF-only path throughput must be finite and non-negative."));
    }
    const auto initial_depth_status = validate_path_depth_state(
        depth_limits, initial_state.depth_counters(), initial_state.depth());
    if (!initial_depth_status.has_value()) {
        return std::unexpected(initial_depth_status.error());
    }
    if (environment.wavelengths() != initial_state.wavelengths()) {
        return std::unexpected(
            path_loop_error("The BSDF-only environment was not resolved at the path wavelengths."));
    }
    for (const auto& surface : surfaces) {
        if (surface.wavelengths() != initial_state.wavelengths()) {
            return std::unexpected(
                path_loop_error("A BSDF-only surface was not resolved at the path wavelengths."));
        }
    }

    auto beta = initial_state.beta();
    auto radiance = initial_state.accumulated_radiance();
    auto depth = initial_state.depth();
    auto depth_counters = initial_state.depth_counters();
    const auto eta_scale = initial_state.eta_scale();
    const auto wavelengths = initial_state.wavelengths();
    auto delta_flags = initial_state.delta_flags();
    const auto current_medium = initial_state.current_medium();
    auto ray = initial_ray;

    const auto finish = [&](const BsdfOnlyPathTermination termination,
                            const ScatteringLobe blocked_depth_limits)
        -> core::Result<BsdfOnlyPathResultT<Scalar>> {
        return make_result(beta, radiance, depth, eta_scale, wavelengths, delta_flags,
                           current_medium, depth_limits, depth_counters, ray, termination,
                           blocked_depth_limits);
    };

    while (true) {
        const auto resolved = closest_hit(ray, surfaces);
        if (!resolved.has_value()) {
            return std::unexpected(resolved.error());
        }

        if (!resolved->has_value()) {
            const auto emitted = environment.environment().eval(ray.direction());
            if (!emitted.has_value()) {
                return std::unexpected(emitted.error());
            }
            const auto accumulated = accumulate_emission(radiance, beta, *emitted);
            if (!accumulated.has_value()) {
                return std::unexpected(accumulated.error());
            }
            radiance = *accumulated;
            return finish(BsdfOnlyPathTermination::escaped_environment, ScatteringLobe::none);
        }

        const auto& surface_hit = **resolved;
        const auto emitted = surface_hit.surface->emission().eval(surface_hit.hit.geometric_normal,
                                                                  -ray.direction());
        if (!emitted.has_value()) {
            return std::unexpected(emitted.error());
        }
        const auto accumulated = accumulate_emission(radiance, beta, *emitted);
        if (!accumulated.has_value()) {
            return std::unexpected(accumulated.error());
        }
        radiance = *accumulated;

        constexpr auto diffuse_reflection = ScatteringLobe::diffuse | ScatteringLobe::reflection;
        const auto depth_event =
            evaluate_path_depth_event(depth_limits, depth_counters, diffuse_reflection);
        if (!depth_event.has_value()) {
            return std::unexpected(depth_event.error());
        }
        if (!depth_event->accepted()) {
            return finish(BsdfOnlyPathTermination::depth_limit, depth_event->blocked_limits);
        }
        if (zero_spectrum(beta)) {
            return finish(BsdfOnlyPathTermination::zero_throughput, ScatteringLobe::none);
        }

        const auto frame = OrthonormalFrameT<Scalar>::from_normal(surface_hit.hit.geometric_normal);
        if (!frame.has_value()) {
            return std::unexpected(frame.error());
        }
        const auto outgoing_world = robust_unit_direction(-ray.direction());
        if (!outgoing_world.has_value()) {
            return std::unexpected(outgoing_world.error());
        }
        const auto outgoing_local = frame->to_local(*outgoing_world);

        const auto dimensions = sample_dimensions_for_bounce(depth);
        if (!dimensions.has_value()) {
            return std::unexpected(dimensions.error());
        }
        const auto canonical_sample = Point2T<Scalar>{
            .x = sample_stream.sample_1d(dimensions->bsdf_u),
            .y = sample_stream.sample_1d(dimensions->bsdf_v),
        };
        const auto sampled =
            surface_hit.surface->reflection().sample(outgoing_local, canonical_sample);
        if (!sampled.has_value()) {
            return std::unexpected(sampled.error());
        }
        if (!sampled->has_value()) {
            return finish(BsdfOnlyPathTermination::outside_bsdf_support, ScatteringLobe::none);
        }

        const auto& bsdf_sample = **sampled;
        if (!finite_non_negative(bsdf_sample.value) ||
            bsdf_sample.probability.measure != ProbabilityMeasure::solid_angle ||
            !std::isfinite(bsdf_sample.probability.value) ||
            !(bsdf_sample.probability.value > Scalar{0}) ||
            !(bsdf_sample.incoming_local.z > Scalar{0})) {
            return std::unexpected(
                path_loop_error("The Lambertian BSDF returned an invalid continuation sample."));
        }

        // For the cosine-sampled Lambertian closure, f * cos(theta) / pdf cancels exactly to rho.
        // Applying that identity avoids a representable rho disappearing through rho / pi first.
        const auto updated_beta =
            checked_product(beta, surface_hit.surface->reflection().reflectance(),
                            "BSDF-only Lambertian throughput is not representable.");
        if (!updated_beta.has_value()) {
            return std::unexpected(updated_beta.error());
        }

        const auto incoming_world =
            robust_unit_direction(frame->to_world(bsdf_sample.incoming_local));
        if (!incoming_world.has_value()) {
            return std::unexpected(incoming_world.error());
        }
        if (!(dot(surface_hit.hit.geometric_normal, *incoming_world) > Scalar{0})) {
            return std::unexpected(
                path_loop_error("A Lambertian continuation left the geometric-normal support."));
        }

        const auto next_depth = path_depth_total(depth_event->counters);
        if (!next_depth.has_value()) {
            return std::unexpected(next_depth.error());
        }
        constexpr auto next_delta_flags = PathDeltaFlags::any_non_delta_bounces;

        const auto position = triangle_position_with_error(surface_hit);
        if (!position.has_value()) {
            return std::unexpected(position.error());
        }
        const auto origin = offset_ray_origin(position->point, position->absolute_error,
                                              surface_hit.hit.geometric_normal, *incoming_world);
        if (!origin.has_value()) {
            return std::unexpected(origin.error());
        }
        if (*origin == position->point) {
            return std::unexpected(path_loop_error(
                "The derived triangle error did not move the continuation-ray origin."));
        }

        const auto next_ray = RayT<Scalar>::create(*origin, *incoming_world, Scalar{0},
                                                   std::numeric_limits<Scalar>::infinity(),
                                                   ray.time(), ray.mask(), ray.current_medium());
        if (!next_ray.has_value()) {
            return std::unexpected(next_ray.error());
        }

        beta = *updated_beta;
        depth = *next_depth;
        depth_counters = depth_event->counters;
        delta_flags = next_delta_flags;
        ray = *next_ray;
        if (zero_spectrum(beta)) {
            return finish(BsdfOnlyPathTermination::zero_throughput, ScatteringLobe::none);
        }
    }
}

} // namespace

core::Result<BsdfOnlyPathResult>
trace_bsdf_only(const Ray& initial_ray, const PathState& initial_state,
                const SampleStream& sample_stream,
                const std::span<const BsdfOnlyTriangleSurface> surfaces,
                const BsdfOnlyEnvironment& environment, const PathDepthLimits& depth_limits) {
    return trace_bsdf_only_impl(initial_ray, initial_state, sample_stream, surfaces, environment,
                                depth_limits);
}

core::Result<ReferenceBsdfOnlyPathResult>
trace_bsdf_only(const ReferenceRay& initial_ray, const ReferencePathState& initial_state,
                const ReferenceSampleStream& sample_stream,
                const std::span<const ReferenceBsdfOnlyTriangleSurface> surfaces,
                const ReferenceBsdfOnlyEnvironment& environment,
                const PathDepthLimits& depth_limits) {
    return trace_bsdf_only_impl(initial_ray, initial_state, sample_stream, surfaces, environment,
                                depth_limits);
}

} // namespace blackframe::renderer
