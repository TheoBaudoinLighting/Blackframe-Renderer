#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace blackframe::xpu::shared {

inline constexpr std::uint16_t ConstantTextureAbiMajor = 1U;
inline constexpr std::uint16_t ConstantTextureAbiMinor = 0U;
inline constexpr std::uint32_t ConstantTextureValueCount = 4U;

enum class ConstantTextureKind : std::uint32_t {
    float_value = 0U,
    linear_rgb = 1U,
    sampled_spectrum = 2U,
};

enum class ConstantTextureEvaluationStatus : std::uint32_t {
    success = 0U,
    invalid_scene = 1U,
    invalid_request = 2U,
    unknown_texture = 3U,
    type_mismatch = 4U,
    invalid_record = 5U,
};

[[nodiscard]] constexpr bool
is_known_constant_texture_kind(const ConstantTextureKind kind) noexcept {
    switch (kind) {
    case ConstantTextureKind::float_value:
    case ConstantTextureKind::linear_rgb:
    case ConstantTextureKind::sampled_spectrum:
        return true;
    }
    return false;
}

struct alignas(16) ConstantTextureEvaluationRequest final {
    std::uint32_t texture_id{};
    ConstantTextureKind expected_kind{ConstantTextureKind::float_value};
    std::uint32_t reserved[2U]{};
};

// The status is authoritative. Successful results use values[0] for float, values[0..2] for
// scene-linear RGB, and all four lanes for a sampled spectrum. Every inactive value and reserved
// word is canonical zero; no consumer may reinterpret a type mismatch as a black texture.
struct alignas(16) ConstantTextureEvaluationResult final {
    float values[ConstantTextureValueCount]{};
    ConstantTextureKind kind{ConstantTextureKind::float_value};
    ConstantTextureEvaluationStatus status{ConstantTextureEvaluationStatus::invalid_request};
    std::uint32_t reserved[2U]{};
};

static_assert(sizeof(float) == 4U);
static_assert(sizeof(ConstantTextureKind) == sizeof(std::uint32_t));
static_assert(sizeof(ConstantTextureEvaluationStatus) == sizeof(std::uint32_t));
static_assert(static_cast<std::uint32_t>(ConstantTextureKind::float_value) == 0U);
static_assert(static_cast<std::uint32_t>(ConstantTextureKind::linear_rgb) == 1U);
static_assert(static_cast<std::uint32_t>(ConstantTextureKind::sampled_spectrum) == 2U);
static_assert(static_cast<std::uint32_t>(ConstantTextureEvaluationStatus::success) == 0U);
static_assert(static_cast<std::uint32_t>(ConstantTextureEvaluationStatus::invalid_scene) == 1U);
static_assert(static_cast<std::uint32_t>(ConstantTextureEvaluationStatus::invalid_request) == 2U);
static_assert(static_cast<std::uint32_t>(ConstantTextureEvaluationStatus::unknown_texture) == 3U);
static_assert(static_cast<std::uint32_t>(ConstantTextureEvaluationStatus::type_mismatch) == 4U);
static_assert(static_cast<std::uint32_t>(ConstantTextureEvaluationStatus::invalid_record) == 5U);
static_assert(std::is_standard_layout_v<ConstantTextureEvaluationRequest>);
static_assert(std::is_trivially_copyable_v<ConstantTextureEvaluationRequest>);
static_assert(std::is_trivially_destructible_v<ConstantTextureEvaluationRequest>);
static_assert(sizeof(ConstantTextureEvaluationRequest) == 16U);
static_assert(alignof(ConstantTextureEvaluationRequest) == 16U);
static_assert(offsetof(ConstantTextureEvaluationRequest, texture_id) == 0U);
static_assert(offsetof(ConstantTextureEvaluationRequest, expected_kind) == 4U);
static_assert(offsetof(ConstantTextureEvaluationRequest, reserved) == 8U);
static_assert(std::is_standard_layout_v<ConstantTextureEvaluationResult>);
static_assert(std::is_trivially_copyable_v<ConstantTextureEvaluationResult>);
static_assert(std::is_trivially_destructible_v<ConstantTextureEvaluationResult>);
static_assert(sizeof(ConstantTextureEvaluationResult) == 32U);
static_assert(alignof(ConstantTextureEvaluationResult) == 16U);
static_assert(offsetof(ConstantTextureEvaluationResult, values) == 0U);
static_assert(offsetof(ConstantTextureEvaluationResult, kind) == 16U);
static_assert(offsetof(ConstantTextureEvaluationResult, status) == 20U);
static_assert(offsetof(ConstantTextureEvaluationResult, reserved) == 24U);

} // namespace blackframe::xpu::shared
