#include <Blackframe/Backends/GPU/CUDA/ConstantTextures.hpp>
#include <Blackframe/Engine/ConstantTextureEvaluation.hpp>
#include <Blackframe/XPU/CUDA/ConstantTextureKernel.hpp>
#include <Blackframe/XPU/CUDA/DeviceMemory.hpp>
#include <Blackframe/XPU/Shared/SceneSoaAbi.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace blackframe::engine {
namespace {

[[nodiscard]] testing::AssertionResult select_test_device() {
    auto device_count = int{};
    const auto count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess) {
        return testing::AssertionFailure()
               << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_status);
    }
    if (device_count <= 0) {
        return testing::AssertionFailure() << "No CUDA device is available.";
    }
    const auto select_status = cudaSetDevice(0);
    if (select_status != cudaSuccess) {
        return testing::AssertionFailure()
               << "cudaSetDevice failed: " << cudaGetErrorString(select_status);
    }
    return testing::AssertionSuccess();
}

template <class Texture, class Value> [[nodiscard]] Texture require_texture(const Value value) {
    auto texture = Texture::create(value);
    EXPECT_TRUE(texture.has_value()) << texture.error().message;
    return texture.value();
}

[[nodiscard]] FrameSceneHandle make_texture_scene() {
    auto scene = FrameScene::create(FrameSceneDescription{
        .constant_textures =
            {
                SceneConstantTexture{
                    .id = {.value = 29U},
                    .texture = require_texture<renderer::ConstantColorTexture>(renderer::LinearRGB{
                        .red = -0.5F,
                        .green = 0.25F,
                        .blue = 4.0F,
                    }),
                },
                SceneConstantTexture{
                    .id = {.value = 3U},
                    .texture = require_texture<renderer::ConstantFloatTexture>(-7.25F),
                },
                SceneConstantTexture{
                    .id = {.value = 71U},
                    .texture = require_texture<renderer::ConstantSpectrumTexture>(
                        renderer::TransportSpectrum{.values = {-2.0F, -0.0F, 0.75F, 16.0F}}),
                },
            },
        .objects = {},
        .geometries = {},
        .materials = {},
        .instances = {},
        .punctual_lights = {},
        .spectral_environment = std::nullopt,
    });
    EXPECT_TRUE(scene.has_value()) << scene.error().message;
    return scene.value();
}

void expect_same_bits(const float actual, const float expected) {
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual), std::bit_cast<std::uint32_t>(expected));
}

TEST(CudaConstantTextureParity, EvaluatesFloatColorAndSpectrumOnTheDeviceExactly) {
    ASSERT_TRUE(select_test_device());
    const auto cpu_scene = make_texture_scene();
    auto upload = CudaSceneSoA::upload(*cpu_scene);
    ASSERT_TRUE(upload.has_value()) << upload.error().message;

    constexpr auto float_ids =
        std::array{renderer::TextureId{.value = 3U}, renderer::TextureId{.value = 3U}};
    const auto cpu_float = evaluate_cpu_transport_constant_float_texture(*cpu_scene, float_ids[0U]);
    const auto scalar_float =
        evaluate_scalar_reference_constant_float_texture(*cpu_scene, float_ids[0U]);
    ASSERT_TRUE(cpu_float.has_value()) << cpu_float.error().message;
    ASSERT_TRUE(scalar_float.has_value()) << scalar_float.error().message;
    const auto floats = evaluate_cuda_constant_float_textures(*upload, float_ids);
    ASSERT_TRUE(floats.has_value()) << floats.error().message;
    ASSERT_EQ(floats->size(), float_ids.size());
    for (const auto value : *floats) {
        expect_same_bits(value, *cpu_float);
    }
    EXPECT_EQ(*scalar_float, static_cast<renderer::ReferenceScalar>(*cpu_float));

    constexpr auto color_ids = std::array{renderer::TextureId{.value = 29U}};
    const auto cpu_color = evaluate_cpu_transport_constant_color_texture(*cpu_scene, color_ids[0U]);
    const auto scalar_color =
        evaluate_scalar_reference_constant_color_texture(*cpu_scene, color_ids[0U]);
    ASSERT_TRUE(cpu_color.has_value()) << cpu_color.error().message;
    ASSERT_TRUE(scalar_color.has_value()) << scalar_color.error().message;
    const auto colors = evaluate_cuda_constant_color_textures(*upload, color_ids);
    ASSERT_TRUE(colors.has_value()) << colors.error().message;
    ASSERT_EQ(colors->size(), 1U);
    expect_same_bits(colors->front().red, cpu_color->red);
    expect_same_bits(colors->front().green, cpu_color->green);
    expect_same_bits(colors->front().blue, cpu_color->blue);
    EXPECT_EQ(scalar_color->red, static_cast<renderer::ReferenceScalar>(cpu_color->red));
    EXPECT_EQ(scalar_color->green, static_cast<renderer::ReferenceScalar>(cpu_color->green));
    EXPECT_EQ(scalar_color->blue, static_cast<renderer::ReferenceScalar>(cpu_color->blue));

    constexpr auto spectrum_ids = std::array{renderer::TextureId{.value = 71U}};
    const auto cpu_spectrum =
        evaluate_cpu_transport_constant_spectrum_texture(*cpu_scene, spectrum_ids[0U]);
    const auto scalar_spectrum =
        evaluate_scalar_reference_constant_spectrum_texture(*cpu_scene, spectrum_ids[0U]);
    ASSERT_TRUE(cpu_spectrum.has_value()) << cpu_spectrum.error().message;
    ASSERT_TRUE(scalar_spectrum.has_value()) << scalar_spectrum.error().message;
    const auto spectra = evaluate_cuda_constant_spectrum_textures(*upload, spectrum_ids);
    ASSERT_TRUE(spectra.has_value()) << spectra.error().message;
    ASSERT_EQ(spectra->size(), 1U);
    for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        expect_same_bits(spectra->front()[lane], (*cpu_spectrum)[lane]);
        EXPECT_EQ((*scalar_spectrum)[lane],
                  static_cast<renderer::ReferenceScalar>((*cpu_spectrum)[lane]));
    }
    EXPECT_TRUE(std::signbit(spectra->front()[1U]));
    EXPECT_TRUE(std::signbit((*cpu_spectrum)[1U]));
    EXPECT_TRUE(std::signbit((*scalar_spectrum)[1U]));
}

TEST(CudaConstantTextureParity, RejectsUnknownIdentifiersAndKindMismatchesWithoutFallback) {
    ASSERT_TRUE(select_test_device());
    const auto cpu_scene = make_texture_scene();
    auto upload = CudaSceneSoA::upload(*cpu_scene);
    ASSERT_TRUE(upload.has_value()) << upload.error().message;

    constexpr auto unknown_ids = std::array{renderer::TextureId{.value = 999U}};
    const auto unknown = evaluate_cuda_constant_float_textures(*upload, unknown_ids);
    ASSERT_FALSE(unknown.has_value());
    EXPECT_EQ(unknown.error().code, core::StatusCode::not_found);
    EXPECT_NE(unknown.error().message.find("not found"), std::string::npos);

    constexpr auto color_as_float = std::array{renderer::TextureId{.value = 29U}};
    const auto mismatch = evaluate_cuda_constant_float_textures(*upload, color_as_float);
    ASSERT_FALSE(mismatch.has_value());
    EXPECT_EQ(mismatch.error().code, core::StatusCode::incompatible);
    EXPECT_NE(mismatch.error().message.find("does not match"), std::string::npos);
}

TEST(CudaConstantTextureParity, HandlesZeroWorkAndEnforcesTheExplicitScratchBudget) {
    ASSERT_TRUE(select_test_device());
    const auto cpu_scene = make_texture_scene();
    auto upload = CudaSceneSoA::upload(*cpu_scene);
    ASSERT_TRUE(upload.has_value()) << upload.error().message;

    const auto empty = evaluate_cuda_constant_float_textures(*upload, {});
    ASSERT_TRUE(empty.has_value()) << empty.error().message;
    EXPECT_TRUE(empty->empty());
    EXPECT_EQ(blackframe_cuda_launch_constant_texture_evaluation(
                  upload->device_data(), upload->size_bytes(), nullptr, 0U, nullptr),
              static_cast<int>(cudaSuccess));

    constexpr auto ids = std::array{renderer::TextureId{.value = 3U}};
    constexpr auto required = sizeof(xpu::shared::ConstantTextureEvaluationRequest) +
                              sizeof(xpu::shared::ConstantTextureEvaluationResult);
    const auto rejected = evaluate_cuda_constant_float_textures(
        *upload, ids,
        CudaConstantTextureEvaluationOptions{
            .device_memory_budget = xpu::cuda::DeviceMemoryBudget{.maximum_bytes = required - 1U},
        });
    ASSERT_FALSE(rejected.has_value());
    EXPECT_EQ(rejected.error().code, core::StatusCode::resource_exhausted);
    EXPECT_NE(rejected.error().message.find("explicit device-memory budget"), std::string::npos);
}

TEST(CudaConstantTextureKernel, RejectsInvalidLaunchArgumentsBeforeDispatch) {
    EXPECT_EQ(blackframe_cuda_launch_constant_texture_evaluation(nullptr, 0U, nullptr, 0U, nullptr),
              static_cast<int>(cudaErrorInvalidValue));
}

TEST(CudaConstantTextureKernel, ReportsInvalidSceneFromAnIsolatedCorruptedSceneCopy) {
    ASSERT_TRUE(select_test_device());
    const auto cpu_scene = make_texture_scene();
    auto upload = CudaSceneSoA::upload(*cpu_scene);
    ASSERT_TRUE(upload.has_value()) << upload.error().message;

    auto host_scene_bytes = std::vector<std::uint8_t>(upload->size_bytes());
    ASSERT_EQ(cudaMemcpy(host_scene_bytes.data(), upload->device_data(), host_scene_bytes.size(),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    constexpr auto invalid_magic = std::uint64_t{};
    std::memcpy(host_scene_bytes.data() + offsetof(xpu::shared::SceneSoaHeader, magic),
                &invalid_magic, sizeof(invalid_magic));

    auto corrupted_scene = xpu::cuda::DeviceBuffer<std::uint8_t>::allocate(host_scene_bytes.size());
    ASSERT_TRUE(corrupted_scene.has_value()) << corrupted_scene.error().message;
    auto requests =
        xpu::cuda::DeviceBuffer<xpu::shared::ConstantTextureEvaluationRequest>::allocate(1U);
    ASSERT_TRUE(requests.has_value()) << requests.error().message;
    auto results =
        xpu::cuda::DeviceBuffer<xpu::shared::ConstantTextureEvaluationResult>::allocate(1U);
    ASSERT_TRUE(results.has_value()) << results.error().message;

    constexpr auto request = xpu::shared::ConstantTextureEvaluationRequest{
        .texture_id = 3U,
        .expected_kind = xpu::shared::ConstantTextureKind::float_value,
    };
    ASSERT_EQ(cudaMemcpy(corrupted_scene->data(), host_scene_bytes.data(), host_scene_bytes.size(),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(requests->data(), &request, sizeof(request), cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(blackframe_cuda_launch_constant_texture_evaluation(
                  corrupted_scene->data(), corrupted_scene->size_bytes(), requests->data(), 1U,
                  results->data()),
              static_cast<int>(cudaSuccess));

    auto result = xpu::shared::ConstantTextureEvaluationResult{};
    ASSERT_EQ(cudaMemcpy(&result, results->data(), sizeof(result), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(result.status, xpu::shared::ConstantTextureEvaluationStatus::invalid_scene);
    EXPECT_EQ(result.kind, xpu::shared::ConstantTextureKind::float_value);
    for (const auto value : result.values) {
        expect_same_bits(value, 0.0F);
    }
    EXPECT_EQ(result.reserved[0U], 0U);
    EXPECT_EQ(result.reserved[1U], 0U);
}

TEST(CudaConstantTextureKernel, ReportsInvalidRequestsWithoutDispatchFallback) {
    ASSERT_TRUE(select_test_device());
    const auto cpu_scene = make_texture_scene();
    auto upload = CudaSceneSoA::upload(*cpu_scene);
    ASSERT_TRUE(upload.has_value()) << upload.error().message;

    constexpr auto host_requests = std::array{
        xpu::shared::ConstantTextureEvaluationRequest{
            .texture_id = 3U,
            .expected_kind = static_cast<xpu::shared::ConstantTextureKind>(
                std::numeric_limits<std::uint32_t>::max()),
        },
        xpu::shared::ConstantTextureEvaluationRequest{
            .texture_id = 3U,
            .expected_kind = xpu::shared::ConstantTextureKind::float_value,
            .reserved = {1U, 0U},
        },
    };
    auto requests =
        xpu::cuda::DeviceBuffer<xpu::shared::ConstantTextureEvaluationRequest>::allocate(
            host_requests.size());
    ASSERT_TRUE(requests.has_value()) << requests.error().message;
    auto results = xpu::cuda::DeviceBuffer<xpu::shared::ConstantTextureEvaluationResult>::allocate(
        host_requests.size());
    ASSERT_TRUE(results.has_value()) << results.error().message;
    ASSERT_EQ(cudaMemcpy(requests->data(), host_requests.data(), sizeof(host_requests),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(blackframe_cuda_launch_constant_texture_evaluation(
                  upload->device_data(), upload->size_bytes(), requests->data(),
                  static_cast<std::uint32_t>(host_requests.size()), results->data()),
              static_cast<int>(cudaSuccess));

    auto host_results =
        std::array<xpu::shared::ConstantTextureEvaluationResult, host_requests.size()>{};
    ASSERT_EQ(cudaMemcpy(host_results.data(), results->data(), sizeof(host_results),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (const auto& result : host_results) {
        EXPECT_EQ(result.status, xpu::shared::ConstantTextureEvaluationStatus::invalid_request);
        EXPECT_EQ(result.kind, xpu::shared::ConstantTextureKind::float_value);
        for (const auto value : result.values) {
            expect_same_bits(value, 0.0F);
        }
        EXPECT_EQ(result.reserved[0U], 0U);
        EXPECT_EQ(result.reserved[1U], 0U);
    }
}

TEST(CudaConstantTextureKernel, ReportsInvalidRecordsFromAnIsolatedCorruptedSceneCopy) {
    ASSERT_TRUE(select_test_device());
    const auto cpu_scene = make_texture_scene();
    auto upload = CudaSceneSoA::upload(*cpu_scene);
    ASSERT_TRUE(upload.has_value()) << upload.error().message;

    auto host_scene_bytes = std::vector<std::uint8_t>(upload->size_bytes());
    ASSERT_EQ(cudaMemcpy(host_scene_bytes.data(), upload->device_data(), host_scene_bytes.size(),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    const auto& kind_column = upload->header().columns[xpu::shared::scene_soa_column::texture_kind];
    ASSERT_EQ(kind_column.element_count, upload->header().texture_count);
    ASSERT_GE(kind_column.element_count, 1U);
    ASSERT_LE(kind_column.offset_bytes + sizeof(std::uint32_t), host_scene_bytes.size());
    constexpr auto invalid_kind = std::numeric_limits<std::uint32_t>::max();
    std::memcpy(host_scene_bytes.data() + kind_column.offset_bytes, &invalid_kind,
                sizeof(invalid_kind));

    auto corrupted_scene = xpu::cuda::DeviceBuffer<std::uint8_t>::allocate(host_scene_bytes.size());
    ASSERT_TRUE(corrupted_scene.has_value()) << corrupted_scene.error().message;
    auto requests =
        xpu::cuda::DeviceBuffer<xpu::shared::ConstantTextureEvaluationRequest>::allocate(1U);
    ASSERT_TRUE(requests.has_value()) << requests.error().message;
    auto results =
        xpu::cuda::DeviceBuffer<xpu::shared::ConstantTextureEvaluationResult>::allocate(1U);
    ASSERT_TRUE(results.has_value()) << results.error().message;

    constexpr auto request = xpu::shared::ConstantTextureEvaluationRequest{
        .texture_id = 3U,
        .expected_kind = xpu::shared::ConstantTextureKind::float_value,
    };
    ASSERT_EQ(cudaMemcpy(corrupted_scene->data(), host_scene_bytes.data(), host_scene_bytes.size(),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(requests->data(), &request, sizeof(request), cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(blackframe_cuda_launch_constant_texture_evaluation(
                  corrupted_scene->data(), corrupted_scene->size_bytes(), requests->data(), 1U,
                  results->data()),
              static_cast<int>(cudaSuccess));

    auto result = xpu::shared::ConstantTextureEvaluationResult{};
    ASSERT_EQ(cudaMemcpy(&result, results->data(), sizeof(result), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(result.status, xpu::shared::ConstantTextureEvaluationStatus::invalid_record);
    EXPECT_EQ(result.kind, xpu::shared::ConstantTextureKind::float_value);
    for (const auto value : result.values) {
        expect_same_bits(value, 0.0F);
    }
    EXPECT_EQ(result.reserved[0U], 0U);
    EXPECT_EQ(result.reserved[1U], 0U);

    constexpr auto original_id = std::array{renderer::TextureId{.value = 3U}};
    const auto original_result = evaluate_cuda_constant_float_textures(*upload, original_id);
    ASSERT_TRUE(original_result.has_value()) << original_result.error().message;
    ASSERT_EQ(original_result->size(), 1U);
    expect_same_bits(original_result->front(), -7.25F);
}

} // namespace
} // namespace blackframe::engine
