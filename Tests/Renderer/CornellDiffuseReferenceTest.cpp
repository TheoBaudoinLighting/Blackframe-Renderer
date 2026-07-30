#include "CornellDiffuseReferenceImage.hpp"

#include <Blackframe/Renderer/DisplayPsnr.hpp>
#include <Blackframe/Renderer/LinearMetrics.hpp>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <string_view>

#if !defined(BLACKFRAME_CORNELL_REFERENCE_DIR) || !defined(BLACKFRAME_CORNELL_SCENE_SHA256) ||     \
    !defined(BLACKFRAME_CORNELL_SOURCE_BASE_COMMIT) || !defined(BLACKFRAME_CORNELL_VARIANT_64)
#error "Cornell reference test configuration is incomplete."
#endif

namespace blackframe::renderer::cornell_test {
namespace {

[[nodiscard]] constexpr const CornellImageSpecification& test_specification() noexcept {
#if BLACKFRAME_CORNELL_VARIANT_64
    return Cornell64Specification;
#else
    return Cornell256Specification;
#endif
}

[[nodiscard]] std::filesystem::path reference_path() {
    return std::filesystem::path{BLACKFRAME_CORNELL_REFERENCE_DIR} /
           test_specification().reference_filename;
}

[[nodiscard]] core::Result<LoadedCornellReference> load_reference() {
    return load_cornell_reference(reference_path(), test_specification(),
                                  BLACKFRAME_CORNELL_SCENE_SHA256,
                                  BLACKFRAME_CORNELL_SOURCE_BASE_COMMIT);
}

[[nodiscard]] std::string metric_text(const ReferenceScalar value) {
    auto buffer = std::array<char, 64>{};
    const auto [end, error] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                                            std::chars_format::general,
                                            std::numeric_limits<ReferenceScalar>::max_digits10);
    if (error != std::errc{}) {
        return "unrepresentable";
    }
    return std::string{buffer.data(), static_cast<std::size_t>(end - buffer.data())};
}

void expect_complete_reference(const Film& film) {
    const auto& specification = test_specification();
    EXPECT_EQ(film.extent().width, specification.extent.width);
    EXPECT_EQ(film.extent().height, specification.extent.height);
    EXPECT_EQ(film.crop(), full_film_crop(specification.extent));
    EXPECT_EQ(film.pixel_count(), static_cast<std::size_t>(specification.extent.width) *
                                      static_cast<std::size_t>(specification.extent.height));
    for (auto pixel_y = std::uint32_t{0}; pixel_y < specification.extent.height; ++pixel_y) {
        for (auto pixel_x = std::uint32_t{0}; pixel_x < specification.extent.width; ++pixel_x) {
            const auto pixel = film.pixel(pixel_x, pixel_y);
            ASSERT_TRUE(pixel.has_value()) << pixel_x << ", " << pixel_y;
            EXPECT_EQ(pixel->sample_count, 1U);
            EXPECT_FLOAT_EQ(pixel->weight_sum, 1.0F);
            const auto resolved = film.resolved_pixel(pixel_x, pixel_y);
            ASSERT_TRUE(resolved.has_value()) << pixel_x << ", " << pixel_y;
            EXPECT_TRUE(std::isfinite(resolved->red));
            EXPECT_TRUE(std::isfinite(resolved->green));
            EXPECT_TRUE(std::isfinite(resolved->blue));
        }
    }
}

void expect_transport_sample_count(const Film& film, const std::uint64_t samples_per_pixel) {
    const auto crop = film.crop();
    for (auto pixel_y = crop.minimum_y; pixel_y < crop.maximum_y; ++pixel_y) {
        for (auto pixel_x = crop.minimum_x; pixel_x < crop.maximum_x; ++pixel_x) {
            const auto pixel = film.pixel(pixel_x, pixel_y);
            ASSERT_TRUE(pixel.has_value()) << pixel_x << ", " << pixel_y;
            EXPECT_EQ(pixel->sample_count, samples_per_pixel);
            EXPECT_FLOAT_EQ(pixel->weight_sum, static_cast<TransportScalar>(samples_per_pixel));
        }
    }
}

TEST(CornellDiffuseReferenceTest, LoadsStrictHighSampleReferenceWithoutRegeneration) {
    const auto reference = load_reference();
    ASSERT_TRUE(reference.has_value()) << (reference.has_value() ? "" : reference.error().message);
    expect_complete_reference(reference->film);
    EXPECT_EQ(reference->metadata.scene, cornell_reference_scene_path(test_specification()));
    EXPECT_EQ(reference->metadata.seed, CornellReferenceSeed);
    EXPECT_EQ(reference->metadata.options, cornell_reference_options(test_specification()));
    EXPECT_EQ(reference->metadata.backend, CornellReferenceBackend);
    EXPECT_EQ(reference->metadata.capabilities, CornellReferenceCapabilities);
}

TEST(CornellDiffuseReferenceTest, MseDecreasesAsSamplesPerPixelIncrease) {
    const auto reference = load_reference();
    ASSERT_TRUE(reference.has_value()) << (reference.has_value() ? "" : reference.error().message);

    constexpr auto worker_count = std::uint32_t{8};
    const auto one_sample = render_cornell_image<TransportScalar>(
        test_specification(), 1, CornellEvaluationSeed, worker_count);
    const auto four_samples = render_cornell_image<TransportScalar>(
        test_specification(), 4, CornellEvaluationSeed, worker_count);
    ASSERT_TRUE(one_sample.has_value())
        << (one_sample.has_value() ? "" : one_sample.error().message);
    ASSERT_TRUE(four_samples.has_value())
        << (four_samples.has_value() ? "" : four_samples.error().message);
    expect_transport_sample_count(*one_sample, 1);
    expect_transport_sample_count(*four_samples, 4);

    const auto one_sample_metrics = compute_linear_metrics(*one_sample, reference->film);
    const auto four_sample_metrics = compute_linear_metrics(*four_samples, reference->film);
    const auto one_sample_psnr = compute_display_psnr(*one_sample, reference->film);
    const auto four_sample_psnr = compute_display_psnr(*four_samples, reference->film);
    ASSERT_TRUE(one_sample_metrics.has_value());
    ASSERT_TRUE(four_sample_metrics.has_value());
    ASSERT_TRUE(one_sample_psnr.has_value());
    ASSERT_TRUE(four_sample_psnr.has_value());

    testing::Test::RecordProperty("mse_1spp", metric_text(one_sample_metrics->mse));
    testing::Test::RecordProperty("mse_4spp", metric_text(four_sample_metrics->mse));
    testing::Test::RecordProperty("rmse_1spp", metric_text(one_sample_metrics->rmse));
    testing::Test::RecordProperty("rmse_4spp", metric_text(four_sample_metrics->rmse));
    testing::Test::RecordProperty("psnr_display_1spp", metric_text(one_sample_psnr->psnr));
    testing::Test::RecordProperty("psnr_display_4spp", metric_text(four_sample_psnr->psnr));

    EXPECT_TRUE(std::isfinite(one_sample_metrics->mse));
    EXPECT_TRUE(std::isfinite(four_sample_metrics->mse));
    EXPECT_GT(one_sample_metrics->mse, 0.0);
    EXPECT_GT(four_sample_metrics->mse, 0.0);
    EXPECT_LT(four_sample_metrics->mse, one_sample_metrics->mse);
    EXPECT_LE(four_sample_metrics->mse, one_sample_metrics->mse * 0.65);
    EXPECT_LT(four_sample_metrics->rmse, one_sample_metrics->rmse);
    EXPECT_GT(four_sample_psnr->psnr, one_sample_psnr->psnr);
}

TEST(CornellDiffuseReferenceTest, RejectsMissingReferenceWithoutFallback) {
    const auto missing = load_cornell_reference(
        reference_path().parent_path() / "absent" / test_specification().reference_filename,
        test_specification(), BLACKFRAME_CORNELL_SCENE_SHA256,
        BLACKFRAME_CORNELL_SOURCE_BASE_COMMIT);
    ASSERT_FALSE(missing.has_value());
    EXPECT_EQ(missing.error().code, core::StatusCode::not_found);
}

TEST(CornellDiffuseReferenceTest, RejectsMismatchedSourceBaseCommit) {
    const auto mismatched = load_cornell_reference(reference_path(), test_specification(),
                                                   BLACKFRAME_CORNELL_SCENE_SHA256,
                                                   "0000000000000000000000000000000000000000");
    ASSERT_FALSE(mismatched.has_value());
    EXPECT_EQ(mismatched.error().code, core::StatusCode::incompatible);
}

} // namespace
} // namespace blackframe::renderer::cornell_test
