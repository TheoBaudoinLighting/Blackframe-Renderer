#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/Color.hpp>
#include <Blackframe/Renderer/Spectrum.hpp>
#include <Blackframe/Renderer/TextureColorSpace.hpp>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace blackframe::renderer {

enum class ConstantTextureKind : std::uint32_t {
    float_value = 0U,
    color = 1U,
    spectrum = 2U,
};

namespace constant_texture_detail {

[[nodiscard]] constexpr bool same_transport_bits(const TransportScalar left,
                                                 const TransportScalar right) noexcept {
    return std::bit_cast<std::uint32_t>(left) == std::bit_cast<std::uint32_t>(right);
}

[[nodiscard]] inline core::Error invalid_constant_texture(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

} // namespace constant_texture_detail

// Constant textures preserve finite signed values exactly. Range and sign constraints belong to
// the material parameter that consumes the texture; this layer never clamps or changes color
// representation.
class ConstantFloatTexture final {
  public:
    constexpr ConstantFloatTexture() noexcept = default;

    [[nodiscard]] static core::Result<ConstantFloatTexture> create(const TransportScalar value) {
        if (!std::isfinite(value)) {
            return std::unexpected(constant_texture_detail::invalid_constant_texture(
                "A constant float texture requires a finite value."));
        }
        return ConstantFloatTexture{value};
    }

    [[nodiscard]] static constexpr ConstantTextureKind kind() noexcept {
        return ConstantTextureKind::float_value;
    }

    [[nodiscard]] constexpr TransportScalar value() const noexcept {
        return value_;
    }

    [[nodiscard]] constexpr bool operator==(const ConstantFloatTexture& other) const noexcept {
        return constant_texture_detail::same_transport_bits(value_, other.value_);
    }

  private:
    constexpr explicit ConstantFloatTexture(const TransportScalar value) noexcept : value_{value} {}

    TransportScalar value_{};
};

class ConstantColorTexture final {
  public:
    constexpr ConstantColorTexture() noexcept = default;

    [[nodiscard]] static core::Result<ConstantColorTexture> create(const LinearRGB value) {
        if (!std::isfinite(value.red) || !std::isfinite(value.green) ||
            !std::isfinite(value.blue)) {
            return std::unexpected(constant_texture_detail::invalid_constant_texture(
                "A constant color texture requires finite linear RGB components."));
        }
        return ConstantColorTexture{value};
    }

    [[nodiscard]] static constexpr ConstantTextureKind kind() noexcept {
        return ConstantTextureKind::color;
    }

    [[nodiscard]] static constexpr TextureColorSpace color_space() noexcept {
        return TextureWorkingColorSpace;
    }

    [[nodiscard]] constexpr LinearRGB value() const noexcept {
        return value_;
    }

    [[nodiscard]] constexpr bool operator==(const ConstantColorTexture& other) const noexcept {
        return constant_texture_detail::same_transport_bits(value_.red, other.value_.red) &&
               constant_texture_detail::same_transport_bits(value_.green, other.value_.green) &&
               constant_texture_detail::same_transport_bits(value_.blue, other.value_.blue);
    }

  private:
    constexpr explicit ConstantColorTexture(const LinearRGB value) noexcept : value_{value} {}

    LinearRGB value_{};
};

class ConstantSpectrumTexture final {
  public:
    constexpr ConstantSpectrumTexture() noexcept = default;

    [[nodiscard]] static core::Result<ConstantSpectrumTexture>
    create(const TransportSpectrum value) {
        for (const auto lane : value.values) {
            if (!std::isfinite(lane)) {
                return std::unexpected(constant_texture_detail::invalid_constant_texture(
                    "A constant spectrum texture requires every spectral lane to be finite."));
            }
        }
        return ConstantSpectrumTexture{value};
    }

    [[nodiscard]] static constexpr ConstantTextureKind kind() noexcept {
        return ConstantTextureKind::spectrum;
    }

    [[nodiscard]] constexpr TransportSpectrum value() const noexcept {
        return value_;
    }

    [[nodiscard]] constexpr bool operator==(const ConstantSpectrumTexture& other) const noexcept {
        for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
            if (!constant_texture_detail::same_transport_bits(value_[lane], other.value_[lane])) {
                return false;
            }
        }
        return true;
    }

  private:
    constexpr explicit ConstantSpectrumTexture(const TransportSpectrum value) noexcept
        : value_{value} {}

    TransportSpectrum value_{};
};

[[nodiscard]] constexpr ReferenceScalar
widen_constant_texture_value(const ConstantFloatTexture texture) noexcept {
    return static_cast<ReferenceScalar>(texture.value());
}

[[nodiscard]] constexpr ReferenceLinearRGB
widen_constant_texture_value(const ConstantColorTexture texture) noexcept {
    const auto value = texture.value();
    return {
        .red = static_cast<ReferenceScalar>(value.red),
        .green = static_cast<ReferenceScalar>(value.green),
        .blue = static_cast<ReferenceScalar>(value.blue),
    };
}

[[nodiscard]] constexpr ReferenceSpectrum
widen_constant_texture_value(const ConstantSpectrumTexture texture) noexcept {
    auto result = ReferenceSpectrum{};
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        result[lane] = static_cast<ReferenceScalar>(texture.value()[lane]);
    }
    return result;
}

static_assert(sizeof(ConstantTextureKind) == sizeof(std::uint32_t));
static_assert(static_cast<std::uint32_t>(ConstantTextureKind::float_value) == 0U);
static_assert(static_cast<std::uint32_t>(ConstantTextureKind::color) == 1U);
static_assert(static_cast<std::uint32_t>(ConstantTextureKind::spectrum) == 2U);

static_assert(std::is_standard_layout_v<ConstantFloatTexture>);
static_assert(std::is_trivially_copyable_v<ConstantFloatTexture>);
static_assert(sizeof(ConstantFloatTexture) == sizeof(TransportScalar));
static_assert(alignof(ConstantFloatTexture) == alignof(TransportScalar));
static_assert(std::is_standard_layout_v<ConstantColorTexture>);
static_assert(std::is_trivially_copyable_v<ConstantColorTexture>);
static_assert(sizeof(ConstantColorTexture) == sizeof(LinearRGB));
static_assert(alignof(ConstantColorTexture) == alignof(LinearRGB));
static_assert(std::is_standard_layout_v<ConstantSpectrumTexture>);
static_assert(std::is_trivially_copyable_v<ConstantSpectrumTexture>);
static_assert(sizeof(ConstantSpectrumTexture) == sizeof(TransportSpectrum));
static_assert(alignof(ConstantSpectrumTexture) == alignof(TransportSpectrum));

} // namespace blackframe::renderer
