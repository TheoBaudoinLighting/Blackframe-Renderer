#include <Blackframe/Renderer/BsdfOnlyPathLoop.hpp>
#include <Blackframe/Renderer/Detail/BsdfOnlyPathLoop.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>

namespace blackframe::renderer {
namespace {

template <SpectrumScalar Scalar> struct LinearSurfaceHit final {
    const BsdfOnlyTriangleSurfaceT<Scalar>* surface;
    TriangleHitT<Scalar> triangle_hit;
    Point3T<Scalar> reconstructed_position;
    Vector3T<Scalar> absolute_error;

    [[nodiscard]] constexpr const Normal3T<Scalar>& geometric_normal() const noexcept {
        return triangle_hit.geometric_normal;
    }

    [[nodiscard]] constexpr const Normal3T<Scalar>& shading_normal() const noexcept {
        return triangle_hit.geometric_normal;
    }

    [[nodiscard]] constexpr const LambertianReflectionT<Scalar>& reflection() const noexcept {
        return surface->reflection();
    }

    [[nodiscard]] constexpr const OneSidedSurfaceEmissionT<Scalar>& emission() const noexcept {
        return surface->emission();
    }

    [[nodiscard]] constexpr const Point3T<Scalar>& position() const noexcept {
        return reconstructed_position;
    }

    [[nodiscard]] constexpr const Vector3T<Scalar>& position_error() const noexcept {
        return absolute_error;
    }
};

template <SpectrumScalar Scalar> struct LinearHitCandidate final {
    const BsdfOnlyTriangleSurfaceT<Scalar>* surface;
    TriangleHitT<Scalar> hit;
};

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<std::pair<Point3T<Scalar>, Vector3T<Scalar>>>
triangle_position_with_error(const LinearHitCandidate<Scalar>& resolved) {
    constexpr auto epsilon = std::numeric_limits<Scalar>::epsilon();
    constexpr auto gamma7 = (Scalar{7} * epsilon) / (Scalar{1} - Scalar{7} * epsilon);
    const auto& vertices = resolved.surface->triangle().vertices();
    const auto barycentrics =
        std::array{resolved.hit.barycentrics.vertex0, resolved.hit.barycentrics.vertex1,
                   resolved.hit.barycentrics.vertex2};
    const auto coordinates = std::array{
        std::array{vertices[0].x, vertices[0].y, vertices[0].z},
        std::array{vertices[1].x, vertices[1].y, vertices[1].z},
        std::array{vertices[2].x, vertices[2].y, vertices[2].z},
    };

    auto triangle_scale = Scalar{0};
    for (auto axis = std::size_t{0}; axis < 3U; ++axis) {
        triangle_scale =
            std::max({triangle_scale, std::abs(coordinates[1][axis] - coordinates[0][axis]),
                      std::abs(coordinates[2][axis] - coordinates[0][axis])});
    }
    const auto representable_floor = epsilon * triangle_scale;
    if (!std::isfinite(triangle_scale) || !std::isfinite(representable_floor) ||
        !(representable_floor > Scalar{0})) {
        return std::unexpected(bsdf_only_path_loop_detail::path_loop_error(
            "The BSDF-only triangle position-error scale is not representable."));
    }

    auto point = std::array<Scalar, 3>{};
    auto error = std::array<Scalar, 3>{};
    for (auto axis = std::size_t{0}; axis < 3U; ++axis) {
        const auto products = std::array{barycentrics[0] * coordinates[0][axis],
                                         barycentrics[1] * coordinates[1][axis],
                                         barycentrics[2] * coordinates[2][axis]};
        point[axis] = std::fma(barycentrics[0], coordinates[0][axis],
                               std::fma(barycentrics[1], coordinates[1][axis], products[2]));
        const auto interpolation_magnitude =
            std::abs(products[0]) + std::abs(products[1]) + std::abs(products[2]);
        error[axis] = std::max(gamma7 * interpolation_magnitude, representable_floor);
        if (!std::isfinite(point[axis]) || !std::isfinite(interpolation_magnitude) ||
            !std::isfinite(error[axis]) || error[axis] < Scalar{0}) {
            return std::unexpected(bsdf_only_path_loop_detail::path_loop_error(
                "The BSDF-only triangle position reconstruction is not representable."));
        }
    }

    return std::pair{
        Point3T<Scalar>{.x = point[0], .y = point[1], .z = point[2]},
        Vector3T<Scalar>{.x = error[0], .y = error[1], .z = error[2]},
    };
}

template <SpectrumScalar Scalar> class LinearSurfaceQuery final {
  public:
    explicit LinearSurfaceQuery(const std::span<const BsdfOnlyTriangleSurfaceT<Scalar>> surfaces)
        : surfaces_{surfaces} {}

    [[nodiscard]] core::Status validate(const SampledWavelengthsT<Scalar>& wavelengths) const {
        for (const auto& surface : surfaces_) {
            if (surface.wavelengths() != wavelengths) {
                return std::unexpected(bsdf_only_path_loop_detail::path_loop_error(
                    "A BSDF-only surface was not resolved at the path wavelengths."));
            }
        }
        return {};
    }

    [[nodiscard]] core::Result<std::optional<LinearSurfaceHit<Scalar>>>
    closest_hit(const RayT<Scalar>& ray) const {
        auto nearest = std::optional<LinearHitCandidate<Scalar>>{};
        for (const auto& surface : surfaces_) {
            if ((ray.mask() & surface.visibility_mask()) == 0) {
                continue;
            }
            const auto intersection = surface.triangle().intersect(ray);
            if (!intersection) {
                return std::unexpected(intersection.error());
            }
            if (*intersection &&
                (!nearest || (**intersection).parameter < nearest->hit.parameter)) {
                nearest = LinearHitCandidate<Scalar>{
                    .surface = &surface,
                    .hit = **intersection,
                };
            }
        }
        if (!nearest) {
            return std::optional<LinearSurfaceHit<Scalar>>{};
        }

        const auto position = triangle_position_with_error(*nearest);
        if (!position) {
            return std::unexpected(position.error());
        }
        return std::optional<LinearSurfaceHit<Scalar>>{LinearSurfaceHit<Scalar>{
            .surface = nearest->surface,
            .triangle_hit = nearest->hit,
            .reconstructed_position = position->first,
            .absolute_error = position->second,
        }};
    }

  private:
    std::span<const BsdfOnlyTriangleSurfaceT<Scalar>> surfaces_;
};

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<BsdfOnlyPathResultT<Scalar>>
trace_linear_bsdf_only(const RayT<Scalar>& initial_ray, const PathStateT<Scalar>& initial_state,
                       const SampleStreamT<Scalar>& sample_stream,
                       const std::span<const BsdfOnlyTriangleSurfaceT<Scalar>> surfaces,
                       const BsdfOnlyEnvironmentT<Scalar>& environment,
                       const PathDepthLimits& depth_limits,
                       const RussianRoulettePolicyT<Scalar>& roulette_policy) {
    auto query = LinearSurfaceQuery<Scalar>{surfaces};
    return bsdf_only_path_loop_detail::trace_bsdf_only_with_query(initial_ray, initial_state,
                                                                  sample_stream, query, environment,
                                                                  depth_limits, roulette_policy);
}

} // namespace

core::Result<BsdfOnlyPathResult> trace_bsdf_only(
    const Ray& initial_ray, const PathState& initial_state, const SampleStream& sample_stream,
    const std::span<const BsdfOnlyTriangleSurface> surfaces, const BsdfOnlyEnvironment& environment,
    const PathDepthLimits& depth_limits, const RussianRoulettePolicy& roulette_policy) {
    return trace_linear_bsdf_only(initial_ray, initial_state, sample_stream, surfaces, environment,
                                  depth_limits, roulette_policy);
}

core::Result<ReferenceBsdfOnlyPathResult>
trace_bsdf_only(const ReferenceRay& initial_ray, const ReferencePathState& initial_state,
                const ReferenceSampleStream& sample_stream,
                const std::span<const ReferenceBsdfOnlyTriangleSurface> surfaces,
                const ReferenceBsdfOnlyEnvironment& environment,
                const PathDepthLimits& depth_limits,
                const ReferenceRussianRoulettePolicy& roulette_policy) {
    return trace_linear_bsdf_only(initial_ray, initial_state, sample_stream, surfaces, environment,
                                  depth_limits, roulette_policy);
}

} // namespace blackframe::renderer
