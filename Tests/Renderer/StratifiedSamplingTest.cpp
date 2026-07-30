#include <Blackframe/Renderer/StratifiedSampling.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace blackframe::renderer {
namespace {

constexpr auto sampling_path = SampleStreamIndex{
    .pixel_x = 113,
    .pixel_y = 47,
    .sample_index = 91,
    .seed = 0xA5A5F00D12345678ULL,
};

template <GeometryScalar Scalar>
[[nodiscard]] std::optional<std::size_t> histogram_bin(const Scalar sample,
                                                       const std::size_t bin_count) {
    if (!(sample >= Scalar{0} && sample < Scalar{1})) {
        ADD_FAILURE() << "Sample " << sample << " is outside [0, 1).";
        return std::nullopt;
    }
    const auto scaled =
        static_cast<ReferenceScalar>(sample) * static_cast<ReferenceScalar>(bin_count);
    const auto bin = static_cast<std::size_t>(scaled);
    if (bin >= bin_count) {
        ADD_FAILURE() << "Sample " << sample << " maps outside the histogram.";
        return std::nullopt;
    }
    return bin;
}

template <GeometryScalar Scalar>
void expect_stratified_1d_coverage(const std::span<const Scalar> samples) {
    auto histogram = std::vector<std::uint32_t>(samples.size(), 0);
    for (std::size_t stratum = 0; stratum < samples.size(); ++stratum) {
        const auto bin = histogram_bin(samples[stratum], samples.size());
        ASSERT_TRUE(bin.has_value());
        EXPECT_EQ(*bin, stratum);
        ++histogram[*bin];
    }
    EXPECT_TRUE(std::ranges::all_of(histogram, [](const auto count) { return count == 1; }));
}

template <GeometryScalar Scalar>
void expect_stratified_2d_coverage(const std::span<const Point2T<Scalar>> samples,
                                   const std::size_t x_strata, const std::size_t y_strata) {
    auto histogram = std::vector<std::uint32_t>(x_strata * y_strata, 0);
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const auto expected_x = index % x_strata;
        const auto expected_y = index / x_strata;
        const auto x = histogram_bin(samples[index].x, x_strata);
        const auto y = histogram_bin(samples[index].y, y_strata);
        ASSERT_TRUE(x.has_value());
        ASSERT_TRUE(y.has_value());
        EXPECT_EQ(*x, expected_x);
        EXPECT_EQ(*y, expected_y);
        ++histogram[*y * x_strata + *x];
    }
    EXPECT_TRUE(std::ranges::all_of(histogram, [](const auto count) { return count == 1; }));
}

template <GeometryScalar Scalar>
void expect_latin_hypercube_coverage(const std::span<const Scalar> samples,
                                     const std::size_t sample_count,
                                     const std::size_t dimension_count) {
    std::vector<std::size_t> first_permutation(sample_count);
    bool found_distinct_permutation = false;

    for (std::size_t dimension = 0; dimension < dimension_count; ++dimension) {
        auto histogram = std::vector<std::uint32_t>(sample_count, 0);
        for (std::size_t sample = 0; sample < sample_count; ++sample) {
            const auto bin =
                histogram_bin(samples[sample * dimension_count + dimension], sample_count);
            ASSERT_TRUE(bin.has_value());
            ++histogram[*bin];
            if (dimension == 0) {
                first_permutation[sample] = *bin;
            } else if (*bin != first_permutation[sample]) {
                found_distinct_permutation = true;
            }
        }
        EXPECT_TRUE(std::ranges::all_of(histogram, [](const auto count) { return count == 1; }));
    }
    EXPECT_TRUE(found_distinct_permutation);
}

template <GeometryScalar Scalar>
void expect_jitter_sub_bin_coverage(const std::span<const Scalar> samples,
                                    const std::size_t stratum_count) {
    constexpr auto sub_bin_count = std::size_t{8};
    std::array<std::uint32_t, sub_bin_count> histogram{};

    for (const auto sample : samples) {
        const auto stratum = histogram_bin(sample, stratum_count);
        ASSERT_TRUE(stratum.has_value());
        const auto scaled =
            static_cast<ReferenceScalar>(sample) * static_cast<ReferenceScalar>(stratum_count);
        const auto fraction = scaled - static_cast<ReferenceScalar>(*stratum);
        ASSERT_GE(fraction, ReferenceScalar{0});
        ASSERT_LT(fraction, ReferenceScalar{1});
        const auto sub_bin =
            static_cast<std::size_t>(fraction * static_cast<ReferenceScalar>(sub_bin_count));
        ASSERT_LT(sub_bin, sub_bin_count);
        ++histogram[sub_bin];
    }

    const auto occupied =
        std::ranges::count_if(histogram, [](const auto count) { return count != 0; });
    EXPECT_GE(occupied, 6);
}

template <GeometryScalar Scalar>
void expect_stratified_2d_jitter(const std::span<const Point2T<Scalar>> samples,
                                 const std::size_t x_strata, const std::size_t y_strata) {
    std::vector<Scalar> x_samples;
    std::vector<Scalar> y_samples;
    x_samples.reserve(samples.size());
    y_samples.reserve(samples.size());
    for (const auto sample : samples) {
        x_samples.push_back(sample.x);
        y_samples.push_back(sample.y);
    }
    expect_jitter_sub_bin_coverage<Scalar>(x_samples, x_strata);
    expect_jitter_sub_bin_coverage<Scalar>(y_samples, y_strata);
}

TEST(StratifiedSamplingTest, CoversEveryOneDimensionalStratumInBothPrecisions) {
    constexpr auto stratum_count = std::size_t{4172};
    std::vector<TransportScalar> transport_samples(stratum_count);
    std::vector<ReferenceScalar> reference_samples(stratum_count);
    auto transport_rng = LocalPcg32{sampling_path, 10};
    auto reference_rng = LocalPcg32{sampling_path, 10};

    const auto transport_status =
        generate_stratified_1d(std::span{transport_samples}, transport_rng);
    const auto reference_status =
        generate_stratified_1d(std::span{reference_samples}, reference_rng);
    ASSERT_TRUE(transport_status.has_value());
    ASSERT_TRUE(reference_status.has_value());

    expect_stratified_1d_coverage<TransportScalar>(transport_samples);
    expect_stratified_1d_coverage<ReferenceScalar>(reference_samples);
    expect_jitter_sub_bin_coverage<TransportScalar>(transport_samples, stratum_count);
    expect_jitter_sub_bin_coverage<ReferenceScalar>(reference_samples, stratum_count);
    EXPECT_EQ(transport_rng.next_u64(), reference_rng.next_u64());
}

TEST(StratifiedSamplingTest, CoversEveryTwoDimensionalCellInBothPrecisions) {
    constexpr auto x_strata = std::size_t{17};
    constexpr auto y_strata = std::size_t{13};
    constexpr auto sample_count = x_strata * y_strata;
    std::vector<Point2> transport_samples(sample_count);
    std::vector<ReferencePoint2> reference_samples(sample_count);
    auto transport_rng = LocalPcg32{sampling_path, 11};
    auto reference_rng = LocalPcg32{sampling_path, 11};

    const auto transport_status =
        generate_stratified_2d(std::span{transport_samples}, x_strata, y_strata, transport_rng);
    const auto reference_status =
        generate_stratified_2d(std::span{reference_samples}, x_strata, y_strata, reference_rng);
    ASSERT_TRUE(transport_status.has_value());
    ASSERT_TRUE(reference_status.has_value());

    expect_stratified_2d_coverage<TransportScalar>(transport_samples, x_strata, y_strata);
    expect_stratified_2d_coverage<ReferenceScalar>(reference_samples, x_strata, y_strata);
    expect_stratified_2d_jitter<TransportScalar>(transport_samples, x_strata, y_strata);
    expect_stratified_2d_jitter<ReferenceScalar>(reference_samples, x_strata, y_strata);
    EXPECT_EQ(transport_rng.next_u64(), reference_rng.next_u64());
}

TEST(StratifiedSamplingTest, CoversEveryLatinHypercubeMarginalInBothPrecisions) {
    constexpr auto sample_count = std::size_t{257};
    constexpr auto dimension_count = std::size_t{7};
    constexpr auto value_count = sample_count * dimension_count;
    std::vector<TransportScalar> transport_samples(value_count);
    std::vector<ReferenceScalar> reference_samples(value_count);
    auto transport_rng = LocalPcg32{sampling_path, 12};
    auto reference_rng = LocalPcg32{sampling_path, 12};

    const auto transport_status = generate_latin_hypercube(
        std::span{transport_samples}, sample_count, dimension_count, transport_rng);
    const auto reference_status = generate_latin_hypercube(
        std::span{reference_samples}, sample_count, dimension_count, reference_rng);
    ASSERT_TRUE(transport_status.has_value());
    ASSERT_TRUE(reference_status.has_value());

    expect_latin_hypercube_coverage<TransportScalar>(transport_samples, sample_count,
                                                     dimension_count);
    expect_latin_hypercube_coverage<ReferenceScalar>(reference_samples, sample_count,
                                                     dimension_count);
    expect_jitter_sub_bin_coverage<TransportScalar>(transport_samples, sample_count);
    expect_jitter_sub_bin_coverage<ReferenceScalar>(reference_samples, sample_count);
    EXPECT_EQ(transport_rng.next_u64(), reference_rng.next_u64());
}

TEST(StratifiedSamplingTest, ReplaysEveryPatternAndSeparatesPathSeeds) {
    constexpr auto one_d_count = std::size_t{64};
    constexpr auto two_d_x = std::size_t{8};
    constexpr auto two_d_y = std::size_t{4};
    constexpr auto latin_samples = std::size_t{32};
    constexpr auto latin_dimensions = std::size_t{5};

    std::vector<TransportScalar> first_1d(one_d_count);
    std::vector<Point2> first_2d(two_d_x * two_d_y);
    std::vector<TransportScalar> first_latin(latin_samples * latin_dimensions);
    std::vector<TransportScalar> replay_1d(one_d_count);
    std::vector<Point2> replay_2d(two_d_x * two_d_y);
    std::vector<TransportScalar> replay_latin(latin_samples * latin_dimensions);
    auto first = LocalPcg32{sampling_path, 13};
    auto replay = LocalPcg32{sampling_path, 13};

    ASSERT_TRUE(generate_stratified_1d(std::span{first_1d}, first).has_value());
    ASSERT_TRUE(generate_stratified_2d(std::span{first_2d}, two_d_x, two_d_y, first).has_value());
    ASSERT_TRUE(
        generate_latin_hypercube(std::span{first_latin}, latin_samples, latin_dimensions, first)
            .has_value());
    ASSERT_TRUE(generate_stratified_1d(std::span{replay_1d}, replay).has_value());
    ASSERT_TRUE(generate_stratified_2d(std::span{replay_2d}, two_d_x, two_d_y, replay).has_value());
    ASSERT_TRUE(
        generate_latin_hypercube(std::span{replay_latin}, latin_samples, latin_dimensions, replay)
            .has_value());

    EXPECT_EQ(first_1d, replay_1d);
    EXPECT_EQ(first_2d, replay_2d);
    EXPECT_EQ(first_latin, replay_latin);
    EXPECT_EQ(first.next_u64(), replay.next_u64());

    auto changed_path = sampling_path;
    changed_path.seed ^= 0x8000000000000000ULL;
    auto changed_rng = LocalPcg32{changed_path, 13};
    std::vector<TransportScalar> changed_1d(one_d_count);
    ASSERT_TRUE(generate_stratified_1d(std::span{changed_1d}, changed_rng).has_value());
    EXPECT_NE(changed_1d, first_1d);
    expect_stratified_1d_coverage<TransportScalar>(changed_1d);
}

TEST(StratifiedSamplingTest, SupportsSingleStrataWithoutSpecialSubstitution) {
    std::array<TransportScalar, 1> one_d{};
    std::array<Point2, 1> two_d{};
    std::array<TransportScalar, 3> latin{};
    auto rng = LocalPcg32{sampling_path, 14};
    auto expected_rng = rng;
    const auto expected_1d = expected_rng.next_1d<TransportScalar>();
    const auto expected_2d_x = expected_rng.next_1d<TransportScalar>();
    const auto expected_2d_y = expected_rng.next_1d<TransportScalar>();
    const auto expected_latin = std::array{
        expected_rng.next_1d<TransportScalar>(),
        expected_rng.next_1d<TransportScalar>(),
        expected_rng.next_1d<TransportScalar>(),
    };

    ASSERT_TRUE(generate_stratified_1d(std::span{one_d}, rng).has_value());
    ASSERT_TRUE(generate_stratified_2d(std::span{two_d}, 1, 1, rng).has_value());
    ASSERT_TRUE(generate_latin_hypercube(std::span{latin}, 1, latin.size(), rng).has_value());

    EXPECT_EQ(one_d.front(), expected_1d);
    EXPECT_EQ(two_d.front().x, expected_2d_x);
    EXPECT_EQ(two_d.front().y, expected_2d_y);
    EXPECT_EQ(latin, expected_latin);
    EXPECT_EQ(rng.next_u64(), expected_rng.next_u64());
}

TEST(StratifiedSamplingTest, RejectsInvalidShapesBeforeOutputOrRngMutation) {
    constexpr auto sentinel = TransportScalar{-1};
    std::array<TransportScalar, 5> scalar_output{sentinel, sentinel, sentinel, sentinel, sentinel};
    std::array<Point2, 5> point_output{
        Point2{.x = sentinel, .y = sentinel}, Point2{.x = sentinel, .y = sentinel},
        Point2{.x = sentinel, .y = sentinel}, Point2{.x = sentinel, .y = sentinel},
        Point2{.x = sentinel, .y = sentinel},
    };
    const auto expected_scalars = scalar_output;
    const auto expected_points = point_output;
    auto rng = LocalPcg32{sampling_path, 15};
    auto untouched = rng;

    const auto empty_1d = generate_stratified_1d(std::span<TransportScalar>{}, rng);
    const auto zero_2d = generate_stratified_2d(std::span{point_output}, 0, 5, rng);
    const auto zero_y_2d = generate_stratified_2d(std::span{point_output}, 5, 0, rng);
    const auto short_2d = generate_stratified_2d(std::span{point_output}.first<3>(), 2, 2, rng);
    const auto long_2d = generate_stratified_2d(std::span{point_output}, 2, 2, rng);
    const auto overflow_2d = generate_stratified_2d(
        std::span<Point2>{}, std::numeric_limits<std::uint64_t>::max(), 2, rng);
    const auto precision_2d = generate_stratified_2d(
        std::span<Point2>{}, MaximumStratifiedSampleCount<TransportScalar> + 1, 1, rng);
    const auto y_precision_2d = generate_stratified_2d(
        std::span<Point2>{}, 1, MaximumStratifiedSampleCount<TransportScalar> + 1, rng);
    const auto zero_samples_latin = generate_latin_hypercube(std::span{scalar_output}, 0, 5, rng);
    const auto zero_latin = generate_latin_hypercube(std::span{scalar_output}, 5, 0, rng);
    const auto short_latin =
        generate_latin_hypercube(std::span{scalar_output}.first<4>(), 2, 3, rng);
    const auto long_latin = generate_latin_hypercube(std::span{scalar_output}, 2, 2, rng);
    const auto overflow_latin = generate_latin_hypercube(
        std::span<TransportScalar>{}, std::numeric_limits<std::uint64_t>::max(), 2, rng);
    const auto precision_latin = generate_latin_hypercube(
        std::span<TransportScalar>{}, MaximumStratifiedSampleCount<TransportScalar> + 1, 1, rng);

    for (const auto* const status :
         std::array{&empty_1d, &zero_2d, &zero_y_2d, &short_2d, &long_2d, &zero_samples_latin,
                    &zero_latin, &short_latin, &long_latin}) {
        ASSERT_FALSE(status->has_value());
        EXPECT_EQ(status->error().code, core::StatusCode::invalid_argument);
    }
    for (const auto* const status : std::array{&overflow_2d, &precision_2d, &y_precision_2d,
                                               &overflow_latin, &precision_latin}) {
        ASSERT_FALSE(status->has_value());
        EXPECT_EQ(status->error().code, core::StatusCode::resource_exhausted);
    }
    EXPECT_EQ(scalar_output, expected_scalars);
    EXPECT_EQ(point_output, expected_points);
    EXPECT_EQ(rng.next_u64(), untouched.next_u64());
}

} // namespace
} // namespace blackframe::renderer
