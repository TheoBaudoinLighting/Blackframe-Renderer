#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <Blackframe/Renderer/TransportConventions.hpp>
#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
#include <numbers>
#include <optional>
#include <type_traits>

namespace blackframe::renderer {

template <GeometryScalar Scalar>
using GgxProbabilityDensityT = std::conditional_t<std::same_as<Scalar, TransportScalar>,
                                                  ProbabilityDensity, ReferenceProbabilityDensity>;

template <GeometryScalar Scalar> struct GgxVisibleNormalSampleT final {
    Normal3T<Scalar> microfacet_normal{};
    GgxProbabilityDensityT<Scalar> probability{
        .value = Scalar{0},
        .measure = ProbabilityMeasure::solid_angle,
    };
};

using GgxVisibleNormalSample = GgxVisibleNormalSampleT<TransportScalar>;
using ReferenceGgxVisibleNormalSample = GgxVisibleNormalSampleT<ReferenceScalar>;

namespace ggx_microfacet_detail {

[[nodiscard]] inline core::Error invalid_ggx(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] bool unit_direction(const Vector3T<Scalar> direction) noexcept {
    if (!std::isfinite(direction.x) || !std::isfinite(direction.y) || !std::isfinite(direction.z)) {
        return false;
    }
    const auto squared_length = std::fma(
        direction.x, direction.x, std::fma(direction.y, direction.y, direction.z * direction.z));
    constexpr auto tolerance = std::numeric_limits<Scalar>::epsilon() * Scalar{128};
    return std::isfinite(squared_length) && std::abs(squared_length - Scalar{1}) <= tolerance;
}

template <GeometryScalar Scalar>
[[nodiscard]] bool unit_normal(const Normal3T<Scalar> normal) noexcept {
    return unit_direction(Vector3T<Scalar>{.x = normal.x, .y = normal.y, .z = normal.z});
}

template <GeometryScalar Scalar>
[[nodiscard]] bool unit_square(const Point2T<Scalar> sample) noexcept {
    return std::isfinite(sample.x) && sample.x >= Scalar{0} && sample.x < Scalar{1} &&
           std::isfinite(sample.y) && sample.y >= Scalar{0} && sample.y < Scalar{1};
}

template <GeometryScalar Scalar>
[[nodiscard]] bool representable_alpha_pair(const Scalar alpha_x, const Scalar alpha_y) noexcept {
    if (!std::isfinite(alpha_x) || !(alpha_x > Scalar{0}) || !std::isfinite(alpha_y) ||
        !(alpha_y > Scalar{0})) {
        return false;
    }

    const auto square_root_limit = std::sqrt(std::numeric_limits<Scalar>::max());
    const auto square_root_pi = std::sqrt(std::numbers::pi_v<Scalar>);
    const auto maximum_alpha = square_root_limit * square_root_pi;
    const auto minimum_alpha = Scalar{1} / maximum_alpha;
    if (alpha_x < minimum_alpha || alpha_x > maximum_alpha || alpha_y < minimum_alpha ||
        alpha_y > maximum_alpha) {
        return false;
    }

    // Preserve the complete historical isotropic domain, including its two exact endpoints.
    if (alpha_x == alpha_y) {
        return true;
    }

    // For a unit normal, q = hypot(mx/alpha_x, my/alpha_y, mz) lies between the
    // smallest and largest of {1/alpha_x, 1/alpha_y, 1}. Since
    // D = 1 / (pi * alpha_x * alpha_y * q^4), these two extrema prove that every
    // orientation of the continuous distribution remains representable. Logs avoid making the
    // validation itself overflow for otherwise valid scalar inputs.
    using EnvelopeScalar = long double;
    const auto wide_alpha_x = static_cast<EnvelopeScalar>(alpha_x);
    const auto wide_alpha_y = static_cast<EnvelopeScalar>(alpha_y);
    const auto inverse_alpha_x = EnvelopeScalar{1} / wide_alpha_x;
    const auto inverse_alpha_y = EnvelopeScalar{1} / wide_alpha_y;
    const auto minimum_q = std::min({EnvelopeScalar{1}, inverse_alpha_x, inverse_alpha_y});
    const auto maximum_q = std::max({EnvelopeScalar{1}, inverse_alpha_x, inverse_alpha_y});
    const auto log_scale = std::log(std::numbers::inv_pi_v<EnvelopeScalar>) -
                           std::log(wide_alpha_x) - std::log(wide_alpha_y);
    const auto maximum_log_density = log_scale - EnvelopeScalar{4} * std::log(minimum_q);
    const auto minimum_log_density = log_scale - EnvelopeScalar{4} * std::log(maximum_q);
    const auto scalar_maximum_log =
        std::log(static_cast<EnvelopeScalar>(std::numeric_limits<Scalar>::max()));
    const auto scalar_minimum_log =
        std::log(static_cast<EnvelopeScalar>(std::numeric_limits<Scalar>::denorm_min()));
    return std::isfinite(maximum_log_density) && std::isfinite(minimum_log_density) &&
           maximum_log_density <= scalar_maximum_log && minimum_log_density >= scalar_minimum_log;
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Vector3T<Scalar>> normalize_vector(const Vector3T<Scalar> value) {
    const auto length = std::hypot(std::hypot(value.x, value.y), value.z);
    if (!std::isfinite(length) || !(length > Scalar{0})) {
        return std::unexpected(invalid_ggx("A GGX VNDF intermediate vector is not normalizable."));
    }
    const auto result = Vector3T<Scalar>{
        .x = value.x / length,
        .y = value.y / length,
        .z = value.z / length,
    };
    if (!unit_direction(result)) {
        return std::unexpected(
            invalid_ggx("A GGX VNDF intermediate direction is not representable."));
    }
    return result;
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Normal3T<Scalar>> normalize_normal(const Normal3T<Scalar> value) {
    const auto length = std::hypot(std::hypot(value.x, value.y), value.z);
    if (!std::isfinite(length) || !(length > Scalar{0})) {
        return std::unexpected(invalid_ggx("A GGX visible microfacet normal is not normalizable."));
    }
    const auto result = Normal3T<Scalar>{
        .x = value.x / length,
        .y = value.y / length,
        .z = value.z / length,
    };
    if (!unit_normal(result)) {
        return std::unexpected(
            invalid_ggx("A GGX visible microfacet normal is not representable."));
    }
    return result;
}

template <GeometryScalar Scalar> struct SmithTerms final {
    Scalar normal{};
    Scalar tangent{};
    Scalar radius{};
    Scalar normal_over_radius{};
};

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<SmithTerms<Scalar>> smith_terms(const Scalar alpha,
                                                           const Vector3T<Scalar> direction) {
    const auto sine = std::hypot(direction.x, direction.y);
    const auto inverse_alpha = Scalar{1} / alpha;
    const auto normal = alpha <= Scalar{1} ? direction.z : direction.z * inverse_alpha;
    const auto tangent = alpha <= Scalar{1} ? alpha * sine : sine;
    const auto radius = std::hypot(normal, tangent);
    if (!std::isfinite(normal) || !std::isfinite(tangent) || !std::isfinite(radius) ||
        !(normal > Scalar{0}) || !(radius > Scalar{0})) {
        return std::unexpected(
            invalid_ggx("GGX Smith geometry is not representable for the supplied direction."));
    }
    const auto normal_over_radius = normal / radius;
    if (!std::isfinite(normal_over_radius) || !(normal_over_radius > Scalar{0}) ||
        normal_over_radius > Scalar{1}) {
        return std::unexpected(
            invalid_ggx("GGX Smith geometry is not representable for the supplied direction."));
    }
    return SmithTerms<Scalar>{
        .normal = normal,
        .tangent = tangent,
        .radius = radius,
        .normal_over_radius = normal_over_radius,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<SmithTerms<Scalar>>
smith_terms(const Scalar alpha_x, const Scalar alpha_y, const Vector3T<Scalar> direction) {
    const auto common_scale = std::max({Scalar{1}, alpha_x, alpha_y});
    const auto normal = direction.z / common_scale;
    const auto tangent_x = (alpha_x / common_scale) * direction.x;
    const auto tangent_y = (alpha_y / common_scale) * direction.y;
    const auto tangent = std::hypot(tangent_x, tangent_y);
    const auto radius = std::hypot(normal, tangent);
    if (!std::isfinite(normal) || !std::isfinite(tangent_x) || !std::isfinite(tangent_y) ||
        !std::isfinite(tangent) || !std::isfinite(radius) || !(normal > Scalar{0}) ||
        !(radius > Scalar{0})) {
        return std::unexpected(
            invalid_ggx("GGX Smith geometry is not representable for the supplied direction."));
    }
    const auto normal_over_radius = normal / radius;
    if (!std::isfinite(normal_over_radius) || !(normal_over_radius > Scalar{0}) ||
        normal_over_radius > Scalar{1}) {
        return std::unexpected(
            invalid_ggx("GGX Smith geometry is not representable for the supplied direction."));
    }
    return SmithTerms<Scalar>{
        .normal = normal,
        .tangent = tangent,
        .radius = radius,
        .normal_over_radius = normal_over_radius,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Scalar> nonnegative_roundoff(const Scalar value, const Scalar scale) {
    if (value >= Scalar{0}) {
        return value;
    }
    // The two disk reconstructions use at most sixteen rounded elementary operations. Only their
    // forward-error envelope is projected back onto the exact non-negative domain; anything
    // farther outside it remains an explicit error and can never select a different algorithm.
    const auto bound =
        std::numeric_limits<Scalar>::epsilon() * Scalar{16} * std::max(Scalar{1}, std::abs(scale));
    if (-value <= bound) {
        return Scalar{0};
    }
    return std::unexpected(
        invalid_ggx("A GGX VNDF square-root argument is negative beyond round-off."));
}

} // namespace ggx_microfacet_detail

// Anisotropic Trowbridge--Reitz (GGX) microfacet distribution. Alpha X and Y are mathematical
// slope widths along the caller-supplied local tangent and bitangent, not perceptual roughness, and
// are never remapped or clamped. Tangential rotation belongs to that local frame rather than this
// distribution. A zero width changes the distribution from solid angle to a Dirac mass and
// therefore belongs to a separate delta lobe. D is normalized in projected solid angle: integral
// D(wm) * wm.z d_omega equals one. Directions and microfacet normals use the local +Z convention
// and are never face-forwarded implicitly.
template <GeometryScalar Scalar> class GgxMicrofacetT final {
  public:
    using probability_density_type = GgxProbabilityDensityT<Scalar>;
    using sample_type = GgxVisibleNormalSampleT<Scalar>;

    [[nodiscard]] static core::Result<GgxMicrofacetT> create(const Scalar alpha) {
        if (!std::isfinite(alpha) || !(alpha > Scalar{0})) {
            return std::unexpected(ggx_microfacet_detail::invalid_ggx(
                "GGX alpha must be finite and strictly positive."));
        }

        // The extrema of D on the closed upper hemisphere are alpha^2/pi and
        // 1/(pi*alpha^2). This derived envelope rejects only continuous distributions whose
        // density cannot be represented by Scalar; it is not a roughness epsilon.
        const auto square_root_limit = std::sqrt(std::numeric_limits<Scalar>::max());
        const auto square_root_pi = std::sqrt(std::numbers::pi_v<Scalar>);
        const auto maximum_alpha = square_root_limit * square_root_pi;
        const auto minimum_alpha = Scalar{1} / maximum_alpha;
        if (alpha < minimum_alpha || alpha > maximum_alpha) {
            return std::unexpected(ggx_microfacet_detail::invalid_ggx(
                "The continuous GGX distribution is not representable for this alpha."));
        }
        return GgxMicrofacetT{alpha, alpha};
    }

    [[nodiscard]] static core::Result<GgxMicrofacetT> create(const Scalar alpha_x,
                                                             const Scalar alpha_y) {
        if (!std::isfinite(alpha_x) || !(alpha_x > Scalar{0}) || !std::isfinite(alpha_y) ||
            !(alpha_y > Scalar{0})) {
            return std::unexpected(ggx_microfacet_detail::invalid_ggx(
                "GGX alpha X and alpha Y must be finite and strictly positive."));
        }
        if (!ggx_microfacet_detail::representable_alpha_pair(alpha_x, alpha_y)) {
            return std::unexpected(ggx_microfacet_detail::invalid_ggx(
                "The continuous anisotropic GGX distribution is not representable for these "
                "widths."));
        }
        return GgxMicrofacetT{alpha_x, alpha_y};
    }

    // Legacy isotropic accessor. An anisotropic instance returns its X-axis width; new code should
    // use alpha_x() and alpha_y() explicitly.
    [[nodiscard]] constexpr Scalar alpha() const noexcept {
        return alpha_x_;
    }

    [[nodiscard]] constexpr Scalar alpha_x() const noexcept {
        return alpha_x_;
    }

    [[nodiscard]] constexpr Scalar alpha_y() const noexcept {
        return alpha_y_;
    }

    // Returns D(wm), not a probability density by itself. Exact tangent and lower-hemisphere
    // normals are outside the engine's open +Z microfacet support and return zero.
    [[nodiscard]] core::Result<Scalar>
    normal_distribution(const Normal3T<Scalar> microfacet_normal) const {
        if (!ggx_microfacet_detail::unit_normal(microfacet_normal)) {
            return std::unexpected(ggx_microfacet_detail::invalid_ggx(
                "GGX microfacet normals must be finite unit normals."));
        }
        if (!(microfacet_normal.z > Scalar{0})) {
            return Scalar{0};
        }

        auto denominator_root = Scalar{};
        auto density_root = Scalar{};
        if (alpha_x_ == alpha_y_) {
            const auto radial = std::hypot(microfacet_normal.x, microfacet_normal.y);
            if (alpha_x_ <= Scalar{1}) {
                denominator_root = std::hypot(radial, alpha_x_ * microfacet_normal.z);
                density_root = (alpha_x_ / denominator_root) / denominator_root;
            } else {
                const auto inverse_alpha = Scalar{1} / alpha_x_;
                denominator_root = std::hypot(radial * inverse_alpha, microfacet_normal.z);
                density_root = (inverse_alpha / denominator_root) / denominator_root;
            }
            density_root *= std::sqrt(std::numbers::inv_pi_v<Scalar>);
        } else {
            denominator_root = std::hypot(microfacet_normal.x / alpha_x_,
                                          microfacet_normal.y / alpha_y_, microfacet_normal.z);
            const auto normalization_root = std::sqrt(std::numbers::inv_pi_v<Scalar>) /
                                            (std::sqrt(alpha_x_) * std::sqrt(alpha_y_));
            density_root = (normalization_root / denominator_root) / denominator_root;
        }
        const auto value = density_root * density_root;
        if (!std::isfinite(denominator_root) || !(denominator_root > Scalar{0}) ||
            !std::isfinite(value) || !(value > Scalar{0})) {
            return std::unexpected(ggx_microfacet_detail::invalid_ggx(
                "The GGX normal distribution value is not representable."));
        }
        return value;
    }

    // Lambda is defined on the open +Z direction domain. At exact grazing its mathematical value
    // is infinite, so the finite engine API reports an explicit domain error rather than a clamp.
    [[nodiscard]] core::Result<Scalar> smith_lambda(const Vector3T<Scalar> direction) const {
        if (!ggx_microfacet_detail::unit_direction(direction)) {
            return std::unexpected(ggx_microfacet_detail::invalid_ggx(
                "GGX Smith directions must be finite unit vectors."));
        }
        if (!(direction.z > Scalar{0})) {
            return std::unexpected(ggx_microfacet_detail::invalid_ggx(
                "GGX Lambda requires a direction in the open +Z hemisphere."));
        }

        const auto terms = alpha_x_ == alpha_y_
                               ? ggx_microfacet_detail::smith_terms(alpha_x_, direction)
                               : ggx_microfacet_detail::smith_terms(alpha_x_, alpha_y_, direction);
        if (!terms) {
            return std::unexpected(terms.error());
        }
        if (terms->tangent == Scalar{0}) {
            return Scalar{0};
        }

        auto value = Scalar{};
        if (terms->tangent <= terms->normal) {
            const auto slope = terms->tangent / terms->normal;
            const auto root = std::hypot(Scalar{1}, slope);
            value = (slope * Scalar{0.5}) * (slope / (root + Scalar{1}));
        } else {
            value =
                (Scalar{1} - terms->normal_over_radius) / (Scalar{2} * terms->normal_over_radius);
        }
        if (!std::isfinite(value) || !(value > Scalar{0})) {
            return std::unexpected(ggx_microfacet_detail::invalid_ggx(
                "GGX Lambda is not representable for the supplied direction."));
        }
        return value;
    }

    [[nodiscard]] core::Result<Scalar> smith_g1(const Vector3T<Scalar> direction) const {
        if (!ggx_microfacet_detail::unit_direction(direction)) {
            return std::unexpected(ggx_microfacet_detail::invalid_ggx(
                "GGX Smith directions must be finite unit vectors."));
        }
        if (!(direction.z > Scalar{0})) {
            return Scalar{0};
        }
        const auto terms = alpha_x_ == alpha_y_
                               ? ggx_microfacet_detail::smith_terms(alpha_x_, direction)
                               : ggx_microfacet_detail::smith_terms(alpha_x_, alpha_y_, direction);
        if (!terms) {
            return std::unexpected(terms.error());
        }
        const auto value =
            Scalar{2} * terms->normal_over_radius / (Scalar{1} + terms->normal_over_radius);
        if (!std::isfinite(value) || !(value > Scalar{0}) || value > Scalar{1}) {
            return std::unexpected(ggx_microfacet_detail::invalid_ggx(
                "GGX Smith G1 is not representable for the supplied direction."));
        }
        return value;
    }

    // Height-correlated Smith G2. This is deliberately not the separable product G1(wo)*G1(wi).
    [[nodiscard]] core::Result<Scalar> smith_g2(const Vector3T<Scalar> outgoing_local,
                                                const Vector3T<Scalar> incoming_local) const {
        if (!ggx_microfacet_detail::unit_direction(outgoing_local) ||
            !ggx_microfacet_detail::unit_direction(incoming_local)) {
            return std::unexpected(ggx_microfacet_detail::invalid_ggx(
                "GGX Smith directions must be finite unit vectors."));
        }
        if (!(outgoing_local.z > Scalar{0}) || !(incoming_local.z > Scalar{0})) {
            return Scalar{0};
        }
        const auto outgoing_terms =
            alpha_x_ == alpha_y_
                ? ggx_microfacet_detail::smith_terms(alpha_x_, outgoing_local)
                : ggx_microfacet_detail::smith_terms(alpha_x_, alpha_y_, outgoing_local);
        if (!outgoing_terms) {
            return std::unexpected(outgoing_terms.error());
        }
        const auto incoming_terms =
            alpha_x_ == alpha_y_
                ? ggx_microfacet_detail::smith_terms(alpha_x_, incoming_local)
                : ggx_microfacet_detail::smith_terms(alpha_x_, alpha_y_, incoming_local);
        if (!incoming_terms) {
            return std::unexpected(incoming_terms.error());
        }

        const auto smaller =
            std::min(outgoing_terms->normal_over_radius, incoming_terms->normal_over_radius);
        const auto larger =
            std::max(outgoing_terms->normal_over_radius, incoming_terms->normal_over_radius);
        const auto value = Scalar{2} * smaller / (Scalar{1} + smaller / larger);
        if (!std::isfinite(value) || !(value > Scalar{0}) || value > Scalar{1}) {
            return std::unexpected(ggx_microfacet_detail::invalid_ggx(
                "GGX Smith G2 is not representable for the supplied directions."));
        }
        return value;
    }

    // Conditional density p(wm | wo) of visible microfacet normals in solid angle. This is not a
    // reflected-direction PDF; a future reflection closure must still apply its Jacobian.
    [[nodiscard]] core::Result<probability_density_type>
    visible_normal_pdf(const Vector3T<Scalar> outgoing_local,
                       const Normal3T<Scalar> microfacet_normal) const {
        if (!ggx_microfacet_detail::unit_direction(outgoing_local) ||
            !ggx_microfacet_detail::unit_normal(microfacet_normal)) {
            return std::unexpected(ggx_microfacet_detail::invalid_ggx(
                "GGX VNDF queries require finite unit directions and normals."));
        }
        auto result = probability_density_type{
            .value = Scalar{0},
            .measure = ProbabilityMeasure::solid_angle,
        };
        const auto visible_dot = dot(outgoing_local, microfacet_normal);
        if (!(outgoing_local.z > Scalar{0}) || !(microfacet_normal.z > Scalar{0}) ||
            !(visible_dot > Scalar{0})) {
            return result;
        }

        const auto distribution = normal_distribution(microfacet_normal);
        if (!distribution) {
            return std::unexpected(distribution.error());
        }
        // Evaluate z + hypot(z, alphaX*x, alphaY*y) directly. Unlike Lambda or G1, this finite
        // VNDF denominator can remain representable when their normal/radius ratio does not, so
        // the PDF must not inherit an avoidable intermediate underflow.
        const auto physical_radius =
            alpha_x_ == alpha_y_
                ? std::hypot(outgoing_local.z,
                             alpha_x_ * std::hypot(outgoing_local.x, outgoing_local.y))
                : std::hypot(outgoing_local.z, alpha_x_ * outgoing_local.x,
                             alpha_y_ * outgoing_local.y);
        const auto physical_denominator = outgoing_local.z + physical_radius;
        const auto visible_factor = Scalar{2} * visible_dot / physical_denominator;
        result.value = *distribution * visible_factor;
        if (!std::isfinite(physical_denominator) || !(physical_denominator > Scalar{0}) ||
            !std::isfinite(result.value) || !(result.value > Scalar{0})) {
            return std::unexpected(ggx_microfacet_detail::invalid_ggx(
                "The GGX visible-normal PDF is not representable."));
        }
        return result;
    }

    [[nodiscard]] core::Result<std::optional<sample_type>>
    sample_visible_normal(const Vector3T<Scalar> outgoing_local,
                          const Point2T<Scalar> canonical_sample) const {
        if (!ggx_microfacet_detail::unit_direction(outgoing_local)) {
            return std::unexpected(ggx_microfacet_detail::invalid_ggx(
                "GGX VNDF sampling requires a finite unit outgoing direction."));
        }
        if (!ggx_microfacet_detail::unit_square(canonical_sample)) {
            return std::unexpected(ggx_microfacet_detail::invalid_ggx(
                "GGX VNDF sampling requires a finite sample in the half-open unit square [0, "
                "1)."));
        }
        if (!(outgoing_local.z > Scalar{0})) {
            return std::optional<sample_type>{};
        }

        const auto isotropic = alpha_x_ == alpha_y_;
        const auto inverse_alpha = Scalar{1} / alpha_x_;
        const auto common_scale = std::max({Scalar{1}, alpha_x_, alpha_y_});
        const auto scaled_alpha_x = alpha_x_ / common_scale;
        const auto scaled_alpha_y = alpha_y_ / common_scale;
        auto stretched = Vector3T<Scalar>{};
        if (isotropic) {
            stretched = alpha_x_ <= Scalar{1}
                            ? Vector3T<Scalar>{.x = alpha_x_ * outgoing_local.x,
                                               .y = alpha_x_ * outgoing_local.y,
                                               .z = outgoing_local.z}
                            : Vector3T<Scalar>{.x = outgoing_local.x,
                                               .y = outgoing_local.y,
                                               .z = outgoing_local.z * inverse_alpha};
        } else {
            stretched = Vector3T<Scalar>{
                .x = scaled_alpha_x * outgoing_local.x,
                .y = scaled_alpha_y * outgoing_local.y,
                .z = outgoing_local.z / common_scale,
            };
        }
        if ((isotropic && alpha_x_ <= Scalar{1} &&
             ((outgoing_local.x != Scalar{0} && stretched.x == Scalar{0}) ||
              (outgoing_local.y != Scalar{0} && stretched.y == Scalar{0}))) ||
            (isotropic && alpha_x_ > Scalar{1} && stretched.z == Scalar{0}) ||
            (!isotropic && ((outgoing_local.x != Scalar{0} && stretched.x == Scalar{0}) ||
                            (outgoing_local.y != Scalar{0} && stretched.y == Scalar{0}) ||
                            stretched.z == Scalar{0}))) {
            return std::unexpected(ggx_microfacet_detail::invalid_ggx(
                "The GGX VNDF view stretch is not representable."));
        }
        const auto hemisphere_view = ggx_microfacet_detail::normalize_vector(stretched);
        if (!hemisphere_view) {
            return std::unexpected(hemisphere_view.error());
        }

        const auto tangent_length = std::hypot(hemisphere_view->x, hemisphere_view->y);
        const auto tangent = tangent_length > Scalar{0}
                                 ? Vector3T<Scalar>{.x = -hemisphere_view->y / tangent_length,
                                                    .y = hemisphere_view->x / tangent_length}
                                 : Vector3T<Scalar>{.x = Scalar{1}};
        const auto bitangent = cross(*hemisphere_view, tangent);

        const auto disk_radius = std::sqrt(canonical_sample.x);
        const auto azimuth = Scalar{2} * std::numbers::pi_v<Scalar> * canonical_sample.y;
        const auto disk_x = disk_radius * std::cos(azimuth);
        const auto initial_disk_y = disk_radius * std::sin(azimuth);
        const auto disk_y_limit_squared = (Scalar{1} - disk_x) * (Scalar{1} + disk_x);
        const auto checked_disk_y_limit_squared = ggx_microfacet_detail::nonnegative_roundoff(
            disk_y_limit_squared, Scalar{1} + std::abs(disk_x));
        if (!checked_disk_y_limit_squared) {
            return std::unexpected(checked_disk_y_limit_squared.error());
        }
        const auto disk_y_limit = std::sqrt(*checked_disk_y_limit_squared);
        const auto projection = Scalar{0.5} * (Scalar{1} + hemisphere_view->z);
        const auto disk_y =
            std::fma(projection, initial_disk_y, (Scalar{1} - projection) * disk_y_limit);

        const auto hemisphere_z_squared = std::fma(-disk_y, disk_y, *checked_disk_y_limit_squared);
        const auto checked_hemisphere_z_squared = ggx_microfacet_detail::nonnegative_roundoff(
            hemisphere_z_squared, *checked_disk_y_limit_squared + std::abs(disk_y * disk_y));
        if (!checked_hemisphere_z_squared) {
            return std::unexpected(checked_hemisphere_z_squared.error());
        }
        const auto hemisphere_z = std::sqrt(*checked_hemisphere_z_squared);
        const auto hemisphere_normal = Vector3T<Scalar>{
            .x = std::fma(disk_x, tangent.x,
                          std::fma(disk_y, bitangent.x, hemisphere_z * hemisphere_view->x)),
            .y = std::fma(disk_x, tangent.y,
                          std::fma(disk_y, bitangent.y, hemisphere_z * hemisphere_view->y)),
            .z = std::fma(disk_x, tangent.z,
                          std::fma(disk_y, bitangent.z, hemisphere_z * hemisphere_view->z)),
        };
        if (!(hemisphere_normal.z > Scalar{0})) {
            return std::unexpected(ggx_microfacet_detail::invalid_ggx(
                "The GGX VNDF hemisphere sample is outside its open support."));
        }

        auto unstretched = Normal3T<Scalar>{};
        if (isotropic) {
            unstretched = alpha_x_ <= Scalar{1}
                              ? Normal3T<Scalar>{.x = alpha_x_ * hemisphere_normal.x,
                                                 .y = alpha_x_ * hemisphere_normal.y,
                                                 .z = hemisphere_normal.z}
                              : Normal3T<Scalar>{.x = hemisphere_normal.x,
                                                 .y = hemisphere_normal.y,
                                                 .z = hemisphere_normal.z * inverse_alpha};
        } else {
            unstretched = Normal3T<Scalar>{
                .x = scaled_alpha_x * hemisphere_normal.x,
                .y = scaled_alpha_y * hemisphere_normal.y,
                .z = hemisphere_normal.z / common_scale,
            };
        }
        if ((isotropic && alpha_x_ <= Scalar{1} &&
             ((hemisphere_normal.x != Scalar{0} && unstretched.x == Scalar{0}) ||
              (hemisphere_normal.y != Scalar{0} && unstretched.y == Scalar{0}))) ||
            (isotropic && alpha_x_ > Scalar{1} && unstretched.z == Scalar{0}) ||
            (!isotropic && ((hemisphere_normal.x != Scalar{0} && unstretched.x == Scalar{0}) ||
                            (hemisphere_normal.y != Scalar{0} && unstretched.y == Scalar{0}) ||
                            unstretched.z == Scalar{0}))) {
            return std::unexpected(ggx_microfacet_detail::invalid_ggx(
                "The GGX visible-normal unstretch is not representable."));
        }
        const auto microfacet_normal = ggx_microfacet_detail::normalize_normal(unstretched);
        if (!microfacet_normal) {
            return std::unexpected(microfacet_normal.error());
        }
        if (!(microfacet_normal->z > Scalar{0}) ||
            !(dot(outgoing_local, *microfacet_normal) > Scalar{0})) {
            return std::unexpected(ggx_microfacet_detail::invalid_ggx(
                "The GGX VNDF sample is outside its visible support."));
        }

        const auto probability = visible_normal_pdf(outgoing_local, *microfacet_normal);
        if (!probability) {
            return std::unexpected(probability.error());
        }
        return std::optional<sample_type>{sample_type{
            .microfacet_normal = *microfacet_normal,
            .probability = *probability,
        }};
    }

  private:
    constexpr GgxMicrofacetT(const Scalar alpha_x, const Scalar alpha_y) noexcept
        : alpha_x_{alpha_x}, alpha_y_{alpha_y} {}

    Scalar alpha_x_{};
    Scalar alpha_y_{};
};

using GgxMicrofacet = GgxMicrofacetT<TransportScalar>;
using ReferenceGgxMicrofacet = GgxMicrofacetT<ReferenceScalar>;

static_assert(std::is_standard_layout_v<GgxVisibleNormalSample>);
static_assert(std::is_trivially_copyable_v<GgxVisibleNormalSample>);
static_assert(std::is_standard_layout_v<ReferenceGgxVisibleNormalSample>);
static_assert(std::is_trivially_copyable_v<ReferenceGgxVisibleNormalSample>);
static_assert(std::is_standard_layout_v<GgxMicrofacet>);
static_assert(std::is_trivially_copyable_v<GgxMicrofacet>);
static_assert(std::is_standard_layout_v<ReferenceGgxMicrofacet>);
static_assert(std::is_trivially_copyable_v<ReferenceGgxMicrofacet>);

} // namespace blackframe::renderer
