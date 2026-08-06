#include <Blackframe/XPU/Shared/ConstantTextureAbi.hpp>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <type_traits>

namespace {

namespace shared = blackframe::xpu::shared;

TEST(HostDeviceConstantTextureAbi, FreezesKindsStatusesAndLayouts) {
    EXPECT_EQ(shared::ConstantTextureAbiMajor, 1U);
    EXPECT_EQ(shared::ConstantTextureAbiMinor, 0U);
    EXPECT_EQ(shared::ConstantTextureValueCount, 4U);
    EXPECT_EQ(static_cast<std::uint32_t>(shared::ConstantTextureKind::float_value), 0U);
    EXPECT_EQ(static_cast<std::uint32_t>(shared::ConstantTextureKind::linear_rgb), 1U);
    EXPECT_EQ(static_cast<std::uint32_t>(shared::ConstantTextureKind::sampled_spectrum), 2U);
    EXPECT_EQ(static_cast<std::uint32_t>(shared::ConstantTextureEvaluationStatus::success), 0U);
    EXPECT_EQ(static_cast<std::uint32_t>(shared::ConstantTextureEvaluationStatus::invalid_scene),
              1U);
    EXPECT_EQ(static_cast<std::uint32_t>(shared::ConstantTextureEvaluationStatus::invalid_request),
              2U);
    EXPECT_EQ(static_cast<std::uint32_t>(shared::ConstantTextureEvaluationStatus::unknown_texture),
              3U);
    EXPECT_EQ(static_cast<std::uint32_t>(shared::ConstantTextureEvaluationStatus::type_mismatch),
              4U);
    EXPECT_EQ(static_cast<std::uint32_t>(shared::ConstantTextureEvaluationStatus::invalid_record),
              5U);
    EXPECT_EQ(sizeof(shared::ConstantTextureEvaluationRequest), 16U);
    EXPECT_EQ(alignof(shared::ConstantTextureEvaluationRequest), 16U);
    EXPECT_EQ(offsetof(shared::ConstantTextureEvaluationRequest, expected_kind), 4U);
    EXPECT_EQ(sizeof(shared::ConstantTextureEvaluationResult), 32U);
    EXPECT_EQ(alignof(shared::ConstantTextureEvaluationResult), 16U);
    EXPECT_EQ(offsetof(shared::ConstantTextureEvaluationResult, status), 20U);
    EXPECT_TRUE(std::is_standard_layout_v<shared::ConstantTextureEvaluationRequest>);
    EXPECT_TRUE(std::is_trivially_copyable_v<shared::ConstantTextureEvaluationRequest>);
    EXPECT_TRUE(std::is_standard_layout_v<shared::ConstantTextureEvaluationResult>);
    EXPECT_TRUE(std::is_trivially_copyable_v<shared::ConstantTextureEvaluationResult>);
}

TEST(HostDeviceConstantTextureAbi, DefaultsKeepEveryReservedWordCanonical) {
    constexpr auto request = shared::ConstantTextureEvaluationRequest{};
    constexpr auto result = shared::ConstantTextureEvaluationResult{};

    EXPECT_EQ(request.reserved[0U], 0U);
    EXPECT_EQ(request.reserved[1U], 0U);
    EXPECT_EQ(result.status, shared::ConstantTextureEvaluationStatus::invalid_request);
    for (const auto value : result.values) {
        EXPECT_EQ(value, 0.0F);
    }
    EXPECT_EQ(result.reserved[0U], 0U);
    EXPECT_EQ(result.reserved[1U], 0U);
}

TEST(HostDeviceConstantTextureAbi, RejectsUnknownKindCodes) {
    EXPECT_TRUE(shared::is_known_constant_texture_kind(shared::ConstantTextureKind::float_value));
    EXPECT_TRUE(shared::is_known_constant_texture_kind(shared::ConstantTextureKind::linear_rgb));
    EXPECT_TRUE(
        shared::is_known_constant_texture_kind(shared::ConstantTextureKind::sampled_spectrum));
    EXPECT_FALSE(shared::is_known_constant_texture_kind(
        static_cast<shared::ConstantTextureKind>(0xFFFFFFFFU)));
}

} // namespace
