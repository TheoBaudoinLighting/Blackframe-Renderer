#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace blackframe::renderer {

struct InstanceId final {
    std::uint32_t value{};

    [[nodiscard]] constexpr bool operator==(const InstanceId&) const noexcept = default;
};

struct GeometryId final {
    std::uint32_t value{};

    [[nodiscard]] constexpr bool operator==(const GeometryId&) const noexcept = default;
};

struct PrimitiveId final {
    std::uint32_t value{};

    [[nodiscard]] constexpr bool operator==(const PrimitiveId&) const noexcept = default;
};

struct MaterialId final {
    std::uint32_t value{};

    [[nodiscard]] constexpr bool operator==(const MaterialId&) const noexcept = default;
};

struct SurfaceIdentifiers final {
    InstanceId instance{};
    GeometryId geometry{};
    PrimitiveId primitive{};
    MaterialId material{};

    [[nodiscard]] constexpr bool operator==(const SurfaceIdentifiers&) const noexcept = default;
};

namespace surface_interaction_detail {

[[nodiscard]] inline core::Error surface_interaction_error(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

template <GeometryScalar Scalar> [[nodiscard]] bool finite(const Point2T<Scalar> value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y);
}

template <GeometryScalar Scalar> [[nodiscard]] bool finite(const Point3T<Scalar> value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

template <GeometryScalar Scalar> [[nodiscard]] bool finite(const Vector3T<Scalar> value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

template <GeometryScalar Scalar> [[nodiscard]] bool finite(const Normal3T<Scalar> value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

template <GeometryScalar Scalar>
[[nodiscard]] Scalar fused_squared_length(const Normal3T<Scalar> value) noexcept {
    return std::fma(value.x, value.x, std::fma(value.y, value.y, value.z * value.z));
}

template <GeometryScalar Scalar>
[[nodiscard]] Scalar fused_dot(const Normal3T<Scalar> left, const Normal3T<Scalar> right) noexcept {
    return std::fma(left.x, right.x, std::fma(left.y, right.y, left.z * right.z));
}

template <GeometryScalar Scalar>
[[nodiscard]] bool unit_normal(const Normal3T<Scalar> value) noexcept {
    constexpr auto tolerance = std::numeric_limits<Scalar>::epsilon() * Scalar{128};
    const auto squared_length = fused_squared_length(value);
    return finite(value) && std::isfinite(squared_length) &&
           std::abs(squared_length - Scalar{1}) <= tolerance;
}

template <GeometryScalar Scalar>
[[nodiscard]] bool tangent_to_normal(const Vector3T<Scalar> value,
                                     const Normal3T<Scalar> normal) noexcept {
    const auto maximum_component =
        std::max({std::abs(value.x), std::abs(value.y), std::abs(value.z)});
    if (maximum_component == Scalar{0}) {
        return true;
    }

    const auto scaled = value / maximum_component;
    const auto scaled_length =
        std::sqrt(std::fma(scaled.x, scaled.x, std::fma(scaled.y, scaled.y, scaled.z * scaled.z)));
    const auto alignment =
        std::fma(scaled.x, normal.x, std::fma(scaled.y, normal.y, scaled.z * normal.z)) /
        scaled_length;
    constexpr auto tolerance = std::numeric_limits<Scalar>::epsilon() * Scalar{128};
    return std::isfinite(scaled_length) && scaled_length > Scalar{0} && std::isfinite(alignment) &&
           std::abs(alignment) <= tolerance;
}

} // namespace surface_interaction_detail

// Surface differentials are stored exactly as supplied. In particular, a UV
// singularity may have zero or collinear derivatives; no replacement frame is
// synthesized here.
template <GeometryScalar Scalar> class SurfaceInteractionT final {
  public:
    [[nodiscard]] static core::Result<SurfaceInteractionT>
    create(const Point3T<Scalar> position, const Normal3T<Scalar> geometric_normal,
           const Normal3T<Scalar> shading_normal, const Point2T<Scalar> uv,
           const Vector3T<Scalar> dpdu, const Vector3T<Scalar> dpdv,
           const SurfaceIdentifiers identifiers, const Scalar time) {
        if (!surface_interaction_detail::finite(position)) {
            return std::unexpected(surface_interaction_detail::surface_interaction_error(
                "A surface interaction position must be finite."));
        }
        if (!surface_interaction_detail::unit_normal(geometric_normal)) {
            return std::unexpected(surface_interaction_detail::surface_interaction_error(
                "A surface interaction geometric normal must be finite and unit length."));
        }
        if (!surface_interaction_detail::unit_normal(shading_normal)) {
            return std::unexpected(surface_interaction_detail::surface_interaction_error(
                "A surface interaction shading normal must be finite and unit length."));
        }
        const auto normal_alignment =
            surface_interaction_detail::fused_dot(geometric_normal, shading_normal);
        if (!std::isfinite(normal_alignment) || normal_alignment <= Scalar{0}) {
            return std::unexpected(surface_interaction_detail::surface_interaction_error(
                "Surface normals must lie in the same open hemisphere."));
        }
        if (!surface_interaction_detail::finite(uv)) {
            return std::unexpected(surface_interaction_detail::surface_interaction_error(
                "Surface interaction UV coordinates must be finite."));
        }
        if (!surface_interaction_detail::finite(dpdu) ||
            !surface_interaction_detail::finite(dpdv)) {
            return std::unexpected(surface_interaction_detail::surface_interaction_error(
                "Surface interaction derivatives must be finite."));
        }
        if (!surface_interaction_detail::tangent_to_normal(dpdu, geometric_normal) ||
            !surface_interaction_detail::tangent_to_normal(dpdv, geometric_normal)) {
            return std::unexpected(surface_interaction_detail::surface_interaction_error(
                "Surface interaction derivatives must be tangent to the geometric normal."));
        }
        if (!std::isfinite(time)) {
            return std::unexpected(surface_interaction_detail::surface_interaction_error(
                "A surface interaction time must be finite."));
        }

        return SurfaceInteractionT{
            position, geometric_normal, shading_normal, uv, dpdu, dpdv, identifiers, time};
    }

    [[nodiscard]] constexpr const Point3T<Scalar>& position() const noexcept {
        return position_;
    }

    [[nodiscard]] constexpr const Normal3T<Scalar>& geometric_normal() const noexcept {
        return geometric_normal_;
    }

    [[nodiscard]] constexpr const Normal3T<Scalar>& shading_normal() const noexcept {
        return shading_normal_;
    }

    [[nodiscard]] constexpr const Point2T<Scalar>& uv() const noexcept {
        return uv_;
    }

    [[nodiscard]] constexpr const Vector3T<Scalar>& dpdu() const noexcept {
        return dpdu_;
    }

    [[nodiscard]] constexpr const Vector3T<Scalar>& dpdv() const noexcept {
        return dpdv_;
    }

    [[nodiscard]] constexpr const SurfaceIdentifiers& identifiers() const noexcept {
        return identifiers_;
    }

    [[nodiscard]] constexpr Scalar time() const noexcept {
        return time_;
    }

  private:
    constexpr SurfaceInteractionT(const Point3T<Scalar> position,
                                  const Normal3T<Scalar> geometric_normal,
                                  const Normal3T<Scalar> shading_normal, const Point2T<Scalar> uv,
                                  const Vector3T<Scalar> dpdu, const Vector3T<Scalar> dpdv,
                                  const SurfaceIdentifiers identifiers, const Scalar time) noexcept
        : position_{position}, geometric_normal_{geometric_normal}, shading_normal_{shading_normal},
          uv_{uv}, dpdu_{dpdu}, dpdv_{dpdv}, identifiers_{identifiers}, time_{time} {}

    Point3T<Scalar> position_;
    Normal3T<Scalar> geometric_normal_;
    Normal3T<Scalar> shading_normal_;
    Point2T<Scalar> uv_;
    Vector3T<Scalar> dpdu_;
    Vector3T<Scalar> dpdv_;
    SurfaceIdentifiers identifiers_;
    Scalar time_;
};

using SurfaceInteraction = SurfaceInteractionT<TransportScalar>;
using ReferenceSurfaceInteraction = SurfaceInteractionT<ReferenceScalar>;

static_assert(!std::is_same_v<InstanceId, GeometryId>);
static_assert(!std::is_same_v<InstanceId, PrimitiveId>);
static_assert(!std::is_same_v<InstanceId, MaterialId>);
static_assert(!std::is_same_v<GeometryId, PrimitiveId>);
static_assert(!std::is_same_v<GeometryId, MaterialId>);
static_assert(!std::is_same_v<PrimitiveId, MaterialId>);
static_assert(sizeof(InstanceId) == sizeof(std::uint32_t));
static_assert(sizeof(GeometryId) == sizeof(std::uint32_t));
static_assert(sizeof(PrimitiveId) == sizeof(std::uint32_t));
static_assert(sizeof(MaterialId) == sizeof(std::uint32_t));
static_assert(std::is_standard_layout_v<SurfaceIdentifiers>);
static_assert(std::is_trivially_copyable_v<SurfaceIdentifiers>);
static_assert(std::is_standard_layout_v<SurfaceInteraction>);
static_assert(std::is_trivially_copyable_v<SurfaceInteraction>);
static_assert(std::is_standard_layout_v<ReferenceSurfaceInteraction>);
static_assert(std::is_trivially_copyable_v<ReferenceSurfaceInteraction>);

} // namespace blackframe::renderer
