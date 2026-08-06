#include <Blackframe/Renderer/Detail/BsdfOnlyPathLoop.hpp>
#include <Blackframe/Renderer/Detail/DepthFilteredClosureMixture.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>

namespace blackframe::renderer {
namespace {

template <SpectrumScalar Scalar>
using SetFor =
    std::conditional_t<std::same_as<Scalar, TransportScalar>, ClosureSet, ReferenceClosureSet>;

template <SpectrumScalar Scalar>
using MixtureFor = std::conditional_t<std::same_as<Scalar, TransportScalar>, ClosureMixture,
                                      ReferenceClosureMixture>;

template <SpectrumScalar Scalar>
using FilteredFor =
    std::conditional_t<std::same_as<Scalar, TransportScalar>, detail::DepthFilteredClosureMixture,
                       detail::ReferenceDepthFilteredClosureMixture>;

template <SpectrumScalar Scalar>
[[nodiscard]] SampledSpectrum<TransportSpectrumSampleCount, Scalar>
constant_spectrum(const Scalar value) {
    auto spectrum = SampledSpectrum<TransportSpectrumSampleCount, Scalar>{};
    spectrum.values.fill(value);
    return spectrum;
}

template <SpectrumScalar Scalar>
void expect_spectrum_near(const SampledSpectrum<TransportSpectrumSampleCount, Scalar>& actual,
                          const SampledSpectrum<TransportSpectrumSampleCount, Scalar>& expected) {
    constexpr auto tolerance =
        std::same_as<Scalar, TransportScalar> ? ReferenceScalar{2.0e-5} : ReferenceScalar{5.0e-13};
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        EXPECT_NEAR(static_cast<ReferenceScalar>(actual[lane]),
                    static_cast<ReferenceScalar>(expected[lane]), tolerance);
    }
}

template <SpectrumScalar Scalar> void expect_continuous_record_filtering() {
    auto set = SetFor<Scalar>{};
    ASSERT_EQ(set.append_lambertian_reflection(constant_spectrum<Scalar>(Scalar{0.45})),
              ClosureAppendStatus::appended);
    ASSERT_EQ(set.append_rough_conductor_reflection(
                  constant_spectrum<Scalar>(Scalar{0.8}), constant_spectrum<Scalar>(Scalar{0.5}),
                  constant_spectrum<Scalar>(Scalar{2.5}), Scalar{0.35}),
              ClosureAppendStatus::appended);
    constexpr auto probabilities = std::array{Scalar{0.5}, Scalar{0.5}};
    const auto source = MixtureFor<Scalar>::create(std::move(set), probabilities);
    ASSERT_TRUE(source.has_value()) << source.error().message;

    const auto filtered = FilteredFor<Scalar>::create(
        *source, PathDepthLimits{.diffuse = 0U, .glossy = 1U}, PathDepthCounters{});
    ASSERT_TRUE(filtered.has_value()) << filtered.error().message;
    ASSERT_FALSE(filtered->source_empty());
    ASSERT_EQ(filtered->size(), 1U);
    EXPECT_EQ(filtered->source_closure_index(0U), 1U);
    EXPECT_EQ(filtered->active_closure(0U).kind, ClosureKind::rough_conductor_reflection);
    EXPECT_EQ(filtered->allowed_directions(0U), ScatteringLobe::reflection);
    ASSERT_EQ(filtered->component_probabilities().size(), 1U);
    EXPECT_EQ(filtered->component_probabilities().front(), Scalar{1});
    EXPECT_EQ(filtered->component_cdf()[0], Scalar{0});
    EXPECT_EQ(filtered->component_cdf()[1], Scalar{1});
    EXPECT_EQ(filtered->blocked_lobes(), ScatteringLobe::diffuse);

    auto conductor_set = SetFor<Scalar>{};
    ASSERT_EQ(conductor_set.append_rough_conductor_reflection(
                  constant_spectrum<Scalar>(Scalar{0.8}), constant_spectrum<Scalar>(Scalar{0.5}),
                  constant_spectrum<Scalar>(Scalar{2.5}), Scalar{0.35}),
              ClosureAppendStatus::appended);
    constexpr auto singleton_probability = std::array{Scalar{1}};
    const auto conductor =
        MixtureFor<Scalar>::create(std::move(conductor_set), singleton_probability);
    ASSERT_TRUE(conductor.has_value()) << conductor.error().message;

    constexpr auto outgoing = Vector3T<Scalar>{.z = Scalar{1}};
    constexpr auto canonical = Point2T<Scalar>{.x = Scalar{0.2}, .y = Scalar{0.65}};
    const auto expected =
        conductor->sample(outgoing, Scalar{0.4}, canonical, TransportMode::radiance);
    ASSERT_TRUE(expected.has_value()) << expected.error().message;
    ASSERT_TRUE(expected->has_value());
    for (const auto component_sample : std::array{Scalar{0.1}, Scalar{0.9}}) {
        const auto actual =
            filtered->sample(outgoing, component_sample, canonical, TransportMode::radiance);
        ASSERT_TRUE(actual.has_value()) << actual.error().message;
        ASSERT_TRUE(actual->has_value());
        EXPECT_EQ((**actual).selected_closure, 1U);
        EXPECT_EQ((**actual).selection_probability.value, Scalar{1});
        EXPECT_EQ((**actual).lobes, ScatteringLobe::glossy | ScatteringLobe::reflection);
        EXPECT_EQ((**actual).incoming_local, (**expected).incoming_local);
        EXPECT_EQ((**actual).probability.value, (**expected).probability.value);
        expect_spectrum_near((**actual).value, (**expected).value);
    }

    const auto actual_eval =
        filtered->eval(outgoing, (**expected).incoming_local, TransportMode::radiance);
    const auto expected_eval =
        conductor->eval(outgoing, (**expected).incoming_local, TransportMode::radiance);
    const auto actual_pdf =
        filtered->pdf(outgoing, (**expected).incoming_local, TransportMode::radiance);
    const auto expected_pdf =
        conductor->pdf(outgoing, (**expected).incoming_local, TransportMode::radiance);
    ASSERT_TRUE(actual_eval.has_value()) << actual_eval.error().message;
    ASSERT_TRUE(expected_eval.has_value()) << expected_eval.error().message;
    ASSERT_TRUE(actual_pdf.has_value()) << actual_pdf.error().message;
    ASSERT_TRUE(expected_pdf.has_value()) << expected_pdf.error().message;
    expect_spectrum_near(*actual_eval, *expected_eval);
    EXPECT_EQ(actual_pdf->value, expected_pdf->value);
}

TEST(DepthFilteredClosureMixtureTest,
     RemovesBlockedContinuousRecordsAndRenormalizesSelectionBeforeSampling) {
    expect_continuous_record_filtering<TransportScalar>();
    expect_continuous_record_filtering<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_rough_dielectric_branch_filtering() {
    auto set = SetFor<Scalar>{};
    ASSERT_EQ(set.append_rough_dielectric(constant_spectrum<Scalar>(Scalar{0.9}), Scalar{1},
                                          Scalar{1.5}, Scalar{0.35}),
              ClosureAppendStatus::appended);
    constexpr auto singleton_probability = std::array{Scalar{1}};
    const auto source = MixtureFor<Scalar>::create(std::move(set), singleton_probability);
    ASSERT_TRUE(source.has_value()) << source.error().message;

    const auto both = FilteredFor<Scalar>::create(
        *source, PathDepthLimits{.glossy = 1U, .transmission = 1U}, PathDepthCounters{});
    const auto reflection_only = FilteredFor<Scalar>::create(
        *source, PathDepthLimits{.glossy = 1U, .transmission = 0U}, PathDepthCounters{});
    const auto none = FilteredFor<Scalar>::create(
        *source, PathDepthLimits{.glossy = 0U, .transmission = 1U}, PathDepthCounters{});
    ASSERT_TRUE(both.has_value()) << both.error().message;
    ASSERT_TRUE(reflection_only.has_value()) << reflection_only.error().message;
    ASSERT_TRUE(none.has_value()) << none.error().message;
    ASSERT_EQ(both->size(), 1U);
    ASSERT_EQ(reflection_only->size(), 1U);
    EXPECT_TRUE(none->empty());
    EXPECT_FALSE(none->source_empty());
    EXPECT_EQ(none->blocked_lobes(), ScatteringLobe::glossy);
    EXPECT_EQ(both->allowed_directions(0U), ScatteringDirectionMask);
    EXPECT_EQ(reflection_only->allowed_directions(0U), ScatteringLobe::reflection);
    EXPECT_EQ(reflection_only->blocked_lobes(), ScatteringLobe::transmission);
    EXPECT_TRUE(
        std::ranges::equal(both->component_probabilities(), source->component_probabilities()));
    EXPECT_TRUE(std::ranges::equal(both->component_cdf(), source->component_cdf()));

    constexpr auto outgoing = Vector3T<Scalar>{.z = Scalar{1}};
    constexpr auto canonical = Point2T<Scalar>{.x = Scalar{0.08}, .y = Scalar{0.25}};
    const auto source_transmission =
        source->sample(outgoing, Scalar{0.9}, canonical, TransportMode::radiance);
    const auto both_transmission =
        both->sample(outgoing, Scalar{0.9}, canonical, TransportMode::radiance);
    ASSERT_TRUE(source_transmission.has_value()) << source_transmission.error().message;
    ASSERT_TRUE(source_transmission->has_value());
    ASSERT_TRUE(both_transmission.has_value()) << both_transmission.error().message;
    ASSERT_TRUE(both_transmission->has_value());
    ASSERT_EQ((**source_transmission).lobes, ScatteringLobe::glossy | ScatteringLobe::transmission);
    EXPECT_EQ((**both_transmission).incoming_local, (**source_transmission).incoming_local);
    EXPECT_EQ((**both_transmission).probability.value, (**source_transmission).probability.value);
    EXPECT_EQ((**both_transmission).eta_scale_multiplier,
              (**source_transmission).eta_scale_multiplier);

    const auto reflected =
        reflection_only->sample(outgoing, Scalar{0.9}, canonical, TransportMode::radiance);
    ASSERT_TRUE(reflected.has_value()) << reflected.error().message;
    ASSERT_TRUE(reflected->has_value());
    EXPECT_EQ((**reflected).lobes, ScatteringLobe::glossy | ScatteringLobe::reflection);
    EXPECT_EQ((**reflected).eta_scale_multiplier, Scalar{1});
    const auto source_reflection_pdf =
        source->pdf(outgoing, (**reflected).incoming_local, TransportMode::radiance);
    ASSERT_TRUE(source_reflection_pdf.has_value()) << source_reflection_pdf.error().message;
    EXPECT_GT((**reflected).probability.value, source_reflection_pdf->value);

    const auto removed_value = reflection_only->eval(
        outgoing, (**source_transmission).incoming_local, TransportMode::radiance);
    const auto removed_pdf = reflection_only->pdf(outgoing, (**source_transmission).incoming_local,
                                                  TransportMode::radiance);
    ASSERT_TRUE(removed_value.has_value()) << removed_value.error().message;
    ASSERT_TRUE(removed_pdf.has_value()) << removed_pdf.error().message;
    EXPECT_EQ(*removed_value, (SampledSpectrum<TransportSpectrumSampleCount, Scalar>{}));
    EXPECT_EQ(removed_pdf->value, Scalar{0});

    const auto model = closure_mixture_detail::rough_dielectric_from_record(
        source->closure_set().closures().front());
    ASSERT_TRUE(model.has_value()) << model.error().message;
    const auto forced_transmission = detail::rough_dielectric_sample_with_direction_mask(
        *model, outgoing, Scalar{0}, canonical, TransportMode::radiance,
        ScatteringLobe::transmission);
    ASSERT_TRUE(forced_transmission.has_value()) << forced_transmission.error().message;
    ASSERT_TRUE(forced_transmission->has_value());
    EXPECT_EQ((**forced_transmission).lobes, ScatteringLobe::glossy | ScatteringLobe::transmission);
}

TEST(DepthFilteredClosureMixtureTest,
     ConditionsRoughDielectricSamplingAndPdfOnTheRemainingDirections) {
    expect_rough_dielectric_branch_filtering<TransportScalar>();
    expect_rough_dielectric_branch_filtering<ReferenceScalar>();
}

TEST(DepthFilteredClosureMixtureTest, DistinguishesASourceAbsorberFromAFilteredDepthLimit) {
    const auto absorber = ClosureMixture::create(ClosureSet{}, std::span<const TransportScalar>{});
    ASSERT_TRUE(absorber.has_value()) << absorber.error().message;
    const auto empty_source = detail::DepthFilteredClosureMixture::create(
        *absorber, PathDepthLimits{}, PathDepthCounters{});
    ASSERT_TRUE(empty_source.has_value()) << empty_source.error().message;
    EXPECT_TRUE(empty_source->source_empty());
    EXPECT_TRUE(empty_source->empty());
    EXPECT_EQ(empty_source->blocked_lobes(), ScatteringLobe::none);

    auto set = ClosureSet{};
    ASSERT_EQ(set.append_lambertian_reflection(constant_spectrum(0.5F)),
              ClosureAppendStatus::appended);
    constexpr auto probability = std::array{1.0F};
    const auto source = ClosureMixture::create(std::move(set), probability);
    ASSERT_TRUE(source.has_value()) << source.error().message;
    const auto all_filtered = detail::DepthFilteredClosureMixture::create(
        *source, PathDepthLimits{}, PathDepthCounters{});
    ASSERT_TRUE(all_filtered.has_value()) << all_filtered.error().message;
    EXPECT_FALSE(all_filtered->source_empty());
    EXPECT_TRUE(all_filtered->empty());
    EXPECT_EQ(all_filtered->blocked_lobes(), ScatteringLobe::diffuse);
}

template <SpectrumScalar Scalar>
void expect_filtered_singleton_lambertian_preserves_subnormal_reflectance() {
    const auto reflectance_value = std::numeric_limits<Scalar>::denorm_min();
    ASSERT_EQ(std::fpclassify(reflectance_value), FP_SUBNORMAL);

    auto set = SetFor<Scalar>{};
    ASSERT_EQ(set.append_lambertian_reflection(constant_spectrum(reflectance_value)),
              ClosureAppendStatus::appended);
    ASSERT_EQ(set.append_rough_conductor_reflection(constant_spectrum(Scalar{0.8}),
                                                    constant_spectrum(Scalar{0.5}),
                                                    constant_spectrum(Scalar{2.5}), Scalar{0.35}),
              ClosureAppendStatus::appended);
    constexpr auto source_probabilities = std::array{Scalar{0.25}, Scalar{0.75}};
    const auto source = MixtureFor<Scalar>::create(std::move(set), source_probabilities);
    ASSERT_TRUE(source.has_value()) << source.error().message;

    const auto filtered = FilteredFor<Scalar>::create(
        *source, PathDepthLimits{.diffuse = 1U, .glossy = 0U}, PathDepthCounters{});
    ASSERT_TRUE(filtered.has_value()) << filtered.error().message;
    ASSERT_EQ(filtered->size(), 1U);
    EXPECT_EQ(filtered->source_closure_index(0U), 0U);
    EXPECT_EQ(filtered->active_closure(0U).kind, ClosureKind::lambertian_reflection);
    EXPECT_EQ(filtered->component_probabilities().front(), Scalar{1});

    constexpr auto outgoing = Vector3T<Scalar>{.z = Scalar{1}};
    constexpr auto direction_sample = Point2T<Scalar>{.x = Scalar{0.25}, .y = Scalar{0.5}};
    const auto sampled =
        filtered->sample(outgoing, Scalar{0.9}, direction_sample, TransportMode::radiance);
    ASSERT_TRUE(sampled.has_value()) << sampled.error().message;
    ASSERT_TRUE(sampled->has_value());
    EXPECT_EQ((**sampled).value, constant_spectrum<Scalar>(Scalar{0}));

    const auto beta_value = std::scalbn(Scalar{1}, std::numeric_limits<Scalar>::max_exponent - 8);
    const auto expected = beta_value * reflectance_value;
    ASSERT_TRUE(std::isnormal(expected));
    const auto beta = constant_spectrum(beta_value);
    const auto updated = bsdf_only_path_loop_detail::update_closure_throughput(
        beta, **sampled, Scalar{1}, *filtered);
    ASSERT_TRUE(updated.has_value()) << updated.error().message;
    for (const auto lane : updated->values) {
        EXPECT_EQ(lane, expected);
    }
}

TEST(DepthFilteredClosureMixtureTest,
     PreservesExactSubnormalLambertianProductAfterFilteringAMultiLobeSource) {
    expect_filtered_singleton_lambertian_preserves_subnormal_reflectance<TransportScalar>();
    expect_filtered_singleton_lambertian_preserves_subnormal_reflectance<ReferenceScalar>();
}

} // namespace
} // namespace blackframe::renderer
