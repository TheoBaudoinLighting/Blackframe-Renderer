#pragma once

#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <type_traits>

namespace blackframe::renderer {

// Signed, unwrapped UV derivatives with respect to a one-pixel raster displacement.
// A zero footprint is a valid value and is never used to encode missing derivatives.
template <GeometryScalar Scalar> struct TextureCoordinateDifferentialsT final {
    Scalar dudx{};
    Scalar dvdx{};
    Scalar dudy{};
    Scalar dvdy{};

    [[nodiscard]] constexpr bool
    operator==(const TextureCoordinateDifferentialsT&) const noexcept = default;
};

using TextureCoordinateDifferentials = TextureCoordinateDifferentialsT<TransportScalar>;
using ReferenceTextureCoordinateDifferentials = TextureCoordinateDifferentialsT<ReferenceScalar>;

static_assert(sizeof(TextureCoordinateDifferentials) == 4U * sizeof(TransportScalar));
static_assert(sizeof(ReferenceTextureCoordinateDifferentials) == 4U * sizeof(ReferenceScalar));
static_assert(std::is_standard_layout_v<TextureCoordinateDifferentials>);
static_assert(std::is_standard_layout_v<ReferenceTextureCoordinateDifferentials>);
static_assert(std::is_trivially_copyable_v<TextureCoordinateDifferentials>);
static_assert(std::is_trivially_copyable_v<ReferenceTextureCoordinateDifferentials>);

} // namespace blackframe::renderer
