#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/HostImageFilter.hpp>
#include <Blackframe/Renderer/SurfaceInteraction.hpp>
#include <Blackframe/Renderer/TextureCoordinateDifferentials.hpp>
#include <Blackframe/Renderer/TextureWrap.hpp>
#include <cstdint>
#include <type_traits>

namespace blackframe::renderer {

class HostImageMipChain;

enum class TangentSpaceNormalYConvention : std::uint32_t {
    positive_v = 0U,
    negative_v = 1U,
};

[[nodiscard]] constexpr bool is_valid_tangent_space_normal_y_convention(
    const TangentSpaceNormalYConvention convention) noexcept {
    switch (convention) {
    case TangentSpaceNormalYConvention::positive_v:
    case TangentSpaceNormalYConvention::negative_v:
        return true;
    }
    return false;
}

struct HostNormalMapChannels final {
    std::uint32_t x{0U};
    std::uint32_t y{1U};
    std::uint32_t z{2U};

    [[nodiscard]] constexpr bool operator==(const HostNormalMapChannels&) const noexcept = default;
};

struct HostNormalMapOptions final {
    HostNormalMapChannels channels{};
    TangentSpaceNormalYConvention y_convention{TangentSpaceNormalYConvention::positive_v};
    TextureWrapMode u_wrap{TextureWrapMode::repeat};
    TextureWrapMode v_wrap{TextureWrapMode::repeat};
    HostImageEwaLimits ewa_limits{};

    [[nodiscard]] constexpr bool operator==(const HostNormalMapOptions&) const noexcept = default;
};

template <GeometryScalar Scalar> struct HostBumpMapOptionsT final {
    std::uint32_t height_channel{0U};
    TextureWrapMode u_wrap{TextureWrapMode::repeat};
    TextureWrapMode v_wrap{TextureWrapMode::repeat};
    HostImageEwaLimits ewa_limits{};
    Scalar displacement_scale{Scalar{1}};

    [[nodiscard]] constexpr bool operator==(const HostBumpMapOptionsT&) const noexcept = default;
};

using HostBumpMapOptions = HostBumpMapOptionsT<TransportScalar>;
using ReferenceHostBumpMapOptions = HostBumpMapOptionsT<ReferenceScalar>;

// Normal and height maps require a data-tagged mip chain and a full-rank supplied EWA footprint.
// Color tags, channels, derivatives, and missing flat normals are never guessed. The tangent frame
// is derived from dpdu and the handedness of dpdv; positive_v maps encoded +Y toward projected
// dpdv, while negative_v reverses that component. Bump gradients perturb tangents projected onto
// the current shading-normal plane and are oriented consistently with that normal. Degenerate or
// out-of-hemisphere results are errors. Neither operation mutates the geometric normal.
[[nodiscard]] core::Result<Normal3> evaluate_host_tangent_space_normal_map(
    const HostImageMipChain& mip_chain, const SurfaceInteraction& interaction,
    TextureCoordinateDifferentials differentials, HostNormalMapOptions options = {});

[[nodiscard]] core::Result<ReferenceNormal3> evaluate_host_tangent_space_normal_map(
    const HostImageMipChain& mip_chain, const ReferenceSurfaceInteraction& interaction,
    ReferenceTextureCoordinateDifferentials differentials, HostNormalMapOptions options = {});

[[nodiscard]] core::Result<Normal3> evaluate_host_filtered_bump_map(
    const HostImageMipChain& mip_chain, const SurfaceInteraction& interaction,
    TextureCoordinateDifferentials differentials, HostBumpMapOptions options = {});

[[nodiscard]] core::Result<ReferenceNormal3>
evaluate_host_filtered_bump_map(const HostImageMipChain& mip_chain,
                                const ReferenceSurfaceInteraction& interaction,
                                ReferenceTextureCoordinateDifferentials differentials,
                                ReferenceHostBumpMapOptions options = {});

static_assert(sizeof(TangentSpaceNormalYConvention) == sizeof(std::uint32_t));
static_assert(std::is_trivially_copyable_v<TangentSpaceNormalYConvention>);
static_assert(static_cast<std::uint32_t>(TangentSpaceNormalYConvention::positive_v) == 0U);
static_assert(static_cast<std::uint32_t>(TangentSpaceNormalYConvention::negative_v) == 1U);
static_assert(sizeof(HostNormalMapChannels) == 3U * sizeof(std::uint32_t));
static_assert(std::is_standard_layout_v<HostNormalMapChannels>);
static_assert(std::is_trivially_copyable_v<HostNormalMapChannels>);
static_assert(sizeof(HostNormalMapOptions) == 8U * sizeof(std::uint32_t));
static_assert(std::is_standard_layout_v<HostNormalMapOptions>);
static_assert(std::is_trivially_copyable_v<HostNormalMapOptions>);
static_assert(sizeof(HostBumpMapOptions) == 6U * sizeof(std::uint32_t));
static_assert(std::is_standard_layout_v<HostBumpMapOptions>);
static_assert(std::is_trivially_copyable_v<HostBumpMapOptions>);
static_assert(sizeof(ReferenceHostBumpMapOptions) == 4U * sizeof(std::uint64_t));
static_assert(std::is_standard_layout_v<ReferenceHostBumpMapOptions>);
static_assert(std::is_trivially_copyable_v<ReferenceHostBumpMapOptions>);

} // namespace blackframe::renderer
