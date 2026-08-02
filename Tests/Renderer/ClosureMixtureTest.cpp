#include <Blackframe/Renderer/ClosureMixture.hpp>
#include <Blackframe/Renderer/RoughConductorReflection.hpp>
#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <span>
#include <type_traits>
#include <utility>

namespace blackframe::renderer {
namespace {

template <SpectrumScalar Scalar>
using SetFor =
    std::conditional_t<std::same_as<Scalar, TransportScalar>, ClosureSet, ReferenceClosureSet>;

template <SpectrumScalar Scalar>
using MixtureFor = std::conditional_t<std::same_as<Scalar, TransportScalar>, ClosureMixture,
                                      ReferenceClosureMixture>;

template <SpectrumScalar Scalar>
using ReflectionFor = std::conditional_t<std::same_as<Scalar, TransportScalar>,
                                         LambertianReflection, ReferenceLambertianReflection>;

template <SpectrumScalar Scalar>
using RoughReflectionFor =
    std::conditional_t<std::same_as<Scalar, TransportScalar>, RoughDiffuseReflection,
                       ReferenceRoughDiffuseReflection>;

template <SpectrumScalar Scalar>
using RoughConductorFor =
    std::conditional_t<std::same_as<Scalar, TransportScalar>, RoughConductorReflection,
                       ReferenceRoughConductorReflection>;

template <SpectrumScalar Scalar>
using SpectrumFor = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;

template <SpectrumScalar Scalar> using DensityFor = ClosureProbabilityDensityT<Scalar>;

template <SpectrumScalar Scalar>
inline constexpr auto AnalyticTolerance = std::same_as<Scalar, TransportScalar> ? 8.0e-6 : 2.0e-13;

template <typename Result> void expect_invalid(const Result& result) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(result.error().message.empty());
}

template <SpectrumScalar Scalar>
[[nodiscard]] SpectrumFor<Scalar> constant_spectrum(const Scalar value) {
    auto spectrum = SpectrumFor<Scalar>{};
    spectrum.values.fill(value);
    return spectrum;
}

template <SpectrumScalar Scalar, std::size_t Count>
[[nodiscard]] SetFor<Scalar> make_set(const std::array<SpectrumFor<Scalar>, Count>& reflectances) {
    auto set = SetFor<Scalar>{};
    for (const auto& reflectance : reflectances) {
        EXPECT_EQ(set.append_lambertian_reflection(reflectance), ClosureAppendStatus::appended);
    }
    return set;
}

template <SpectrumScalar Scalar>
void expect_spectrum_near(const SpectrumFor<Scalar>& actual, const SpectrumFor<Scalar>& expected,
                          const double tolerance = AnalyticTolerance<Scalar>) {
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        EXPECT_NEAR(static_cast<double>(actual[lane]), static_cast<double>(expected[lane]),
                    tolerance);
    }
}

template <SpectrumScalar Scalar> void expect_distinct_pdf_sum() {
    const auto equal_probabilities = std::array{Scalar{0.5}, Scalar{0.5}};
    const auto quarter_probabilities = std::array{Scalar{0.25}, Scalar{0.75}};
    const auto conditionals = std::array{
        DensityFor<Scalar>{.value = Scalar{0.125}, .measure = ProbabilityMeasure::solid_angle},
        DensityFor<Scalar>{.value = Scalar{0.625}, .measure = ProbabilityMeasure::solid_angle},
    };

    const auto equal = mix_closure_probability_densities<Scalar>(equal_probabilities, conditionals);
    ASSERT_TRUE(equal.has_value());
    EXPECT_EQ(equal->value, Scalar{0.375});
    EXPECT_EQ(equal->measure, ProbabilityMeasure::solid_angle);

    const auto quarter =
        mix_closure_probability_densities<Scalar>(quarter_probabilities, conditionals);
    ASSERT_TRUE(quarter.has_value());
    EXPECT_EQ(quarter->value, Scalar{0.5});
    EXPECT_EQ(quarter->measure, ProbabilityMeasure::solid_angle);

    const auto singleton_probability = std::array{Scalar{1}};
    const auto singleton_conditional = std::array{DensityFor<Scalar>{
        .value = std::nextafter(Scalar{0.5}, Scalar{1}),
        .measure = ProbabilityMeasure::solid_angle,
    }};
    const auto singleton =
        mix_closure_probability_densities<Scalar>(singleton_probability, singleton_conditional);
    ASSERT_TRUE(singleton.has_value());
    EXPECT_EQ(singleton->value, singleton_conditional.front().value);
    EXPECT_EQ(singleton->measure, singleton_conditional.front().measure);

    expect_invalid(mix_closure_probability_densities<Scalar>(
        std::span<const Scalar>{}, std::span<const DensityFor<Scalar>>{}));
    expect_invalid(mix_closure_probability_densities<Scalar>(
        std::span<const Scalar>{equal_probabilities}.first(1U), conditionals));
    auto mixed_measures = conditionals;
    mixed_measures[1].measure = ProbabilityMeasure::discrete;
    expect_invalid(mix_closure_probability_densities<Scalar>(equal_probabilities, mixed_measures));
    for (const auto unsupported_measure : std::array{
             ProbabilityMeasure::area,
             ProbabilityMeasure::discrete,
             static_cast<ProbabilityMeasure>(0xffU),
         }) {
        auto unsupported = singleton_conditional;
        unsupported.front().measure = unsupported_measure;
        expect_invalid(
            mix_closure_probability_densities<Scalar>(singleton_probability, unsupported));
    }
    auto invalid_density = conditionals;
    invalid_density[0].value = std::numeric_limits<Scalar>::quiet_NaN();
    expect_invalid(mix_closure_probability_densities<Scalar>(equal_probabilities, invalid_density));
    invalid_density = conditionals;
    invalid_density[0].value = std::numeric_limits<Scalar>::infinity();
    expect_invalid(mix_closure_probability_densities<Scalar>(equal_probabilities, invalid_density));
    invalid_density = conditionals;
    invalid_density[0].value = -std::numeric_limits<Scalar>::denorm_min();
    expect_invalid(mix_closure_probability_densities<Scalar>(equal_probabilities, invalid_density));

    auto maximum_conditionals = conditionals;
    maximum_conditionals[0].value = std::numeric_limits<Scalar>::max();
    maximum_conditionals[1].value = std::numeric_limits<Scalar>::max();
    const auto maximum =
        mix_closure_probability_densities<Scalar>(equal_probabilities, maximum_conditionals);
    ASSERT_TRUE(maximum.has_value());
    EXPECT_EQ(maximum->value, std::numeric_limits<Scalar>::max());

    auto zero_conditionals = conditionals;
    zero_conditionals[0].value = Scalar{0};
    zero_conditionals[1].value = Scalar{0};
    const auto zero =
        mix_closure_probability_densities<Scalar>(equal_probabilities, zero_conditionals);
    ASSERT_TRUE(zero.has_value());
    EXPECT_EQ(zero->value, Scalar{0});
    EXPECT_EQ(zero->measure, ProbabilityMeasure::solid_angle);

    auto unrepresentable = conditionals;
    unrepresentable[0].value = std::numeric_limits<Scalar>::denorm_min();
    unrepresentable[1].value = Scalar{0};
    expect_invalid(mix_closure_probability_densities<Scalar>(equal_probabilities, unrepresentable));

    const auto too_many_probabilities = std::array<Scalar, MaximumClosureCount + 1U>{};
    const auto too_many_conditionals = std::array<DensityFor<Scalar>, MaximumClosureCount + 1U>{};
    expect_invalid(
        mix_closure_probability_densities<Scalar>(too_many_probabilities, too_many_conditionals));
}

TEST(ClosureMixtureTest, ComputesTheExplicitWeightedSumOfDistinctConditionalPdfs) {
    expect_distinct_pdf_sum<TransportScalar>();
    expect_distinct_pdf_sum<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_explicit_probability_validation() {
    const auto empty = MixtureFor<Scalar>::create(SetFor<Scalar>{}, std::span<const Scalar>{});
    ASSERT_TRUE(empty.has_value());
    EXPECT_TRUE(empty->closure_set().empty());
    EXPECT_TRUE(empty->component_probabilities().empty());
    EXPECT_TRUE(empty->component_cdf().empty());

    const auto set = make_set<Scalar>(std::array{
        constant_spectrum<Scalar>(Scalar{0.25}),
        constant_spectrum<Scalar>(Scalar{0.75}),
    });
    const auto explicit_probabilities = std::array{Scalar{0.25}, Scalar{0.75}};
    const auto valid = MixtureFor<Scalar>::create(set, explicit_probabilities);
    ASSERT_TRUE(valid.has_value());
    ASSERT_EQ(valid->component_probabilities().size(), explicit_probabilities.size());
    EXPECT_EQ(valid->component_probabilities()[0], explicit_probabilities[0]);
    EXPECT_EQ(valid->component_probabilities()[1], explicit_probabilities[1]);

    expect_invalid(MixtureFor<Scalar>::create(set, std::span<const Scalar>{}));
    for (const auto probabilities : std::array{
             std::array{Scalar{0}, Scalar{1}},
             std::array{-std::numeric_limits<Scalar>::denorm_min(), Scalar{1}},
             std::array{std::numeric_limits<Scalar>::quiet_NaN(), Scalar{1}},
             std::array{std::numeric_limits<Scalar>::infinity(), Scalar{1}},
             std::array{Scalar{0.25}, Scalar{0.25}},
             std::array{Scalar{0.75}, Scalar{0.5}},
             std::array{Scalar{1}, std::numeric_limits<Scalar>::denorm_min()},
             std::array{std::numeric_limits<Scalar>::epsilon() / Scalar{2}, Scalar{1}},
         }) {
        expect_invalid(MixtureFor<Scalar>::create(set, probabilities));
    }

    const auto non_dyadic_set = make_set<Scalar>(std::array{
        constant_spectrum<Scalar>(Scalar{0.1}),
        constant_spectrum<Scalar>(Scalar{0.2}),
        constant_spectrum<Scalar>(Scalar{0.3}),
    });
    const auto non_dyadic_probabilities = std::array{Scalar{0.1}, Scalar{0.2}, Scalar{0.7}};
    const auto non_dyadic = MixtureFor<Scalar>::create(non_dyadic_set, non_dyadic_probabilities);
    ASSERT_TRUE(non_dyadic.has_value());
    ASSERT_EQ(non_dyadic->component_cdf().size(), non_dyadic_probabilities.size() + 1U);
    EXPECT_EQ(non_dyadic->component_cdf().front(), Scalar{0});
    EXPECT_EQ(non_dyadic->component_cdf().back(), Scalar{1});

    const auto outgoing = Vector3T<Scalar>{.z = Scalar{1}};
    const auto direction_sample = Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}};
    auto canonical_total = Scalar{0};
    for (auto index = std::size_t{}; index < non_dyadic_probabilities.size(); ++index) {
        const auto effective_probability =
            non_dyadic->component_cdf()[index + 1U] - non_dyadic->component_cdf()[index];
        EXPECT_EQ(non_dyadic->component_probabilities()[index], effective_probability);
        canonical_total += non_dyadic->component_probabilities()[index];
        const auto sampled =
            non_dyadic->sample(outgoing, non_dyadic->component_cdf()[index], direction_sample);
        ASSERT_TRUE(sampled.has_value());
        ASSERT_TRUE(sampled->has_value());
        EXPECT_EQ((**sampled).selected_closure, index);
        EXPECT_EQ((**sampled).selection_probability.value, effective_probability);
    }
    EXPECT_EQ(canonical_total, Scalar{1});

    const auto sub_grid_probability = std::array{
        std::numeric_limits<Scalar>::denorm_min(),
        Scalar{0.5},
        Scalar{0.5},
    };
    expect_invalid(MixtureFor<Scalar>::create(non_dyadic_set, sub_grid_probability));

    const auto unit_conditionals = std::array{
        DensityFor<Scalar>{.value = Scalar{1}, .measure = ProbabilityMeasure::solid_angle},
        DensityFor<Scalar>{.value = Scalar{1}, .measure = ProbabilityMeasure::solid_angle},
        DensityFor<Scalar>{.value = Scalar{1}, .measure = ProbabilityMeasure::solid_angle},
    };
    const auto unit_sum =
        mix_closure_probability_densities<Scalar>(non_dyadic_probabilities, unit_conditionals);
    ASSERT_TRUE(unit_sum.has_value());
    EXPECT_EQ(unit_sum->value, Scalar{1});

    auto basis_conditionals = unit_conditionals;
    for (auto selected = std::size_t{}; selected < basis_conditionals.size(); ++selected) {
        for (auto index = std::size_t{}; index < basis_conditionals.size(); ++index) {
            basis_conditionals[index].value = index == selected ? Scalar{1} : Scalar{0};
        }
        const auto basis_sum =
            mix_closure_probability_densities<Scalar>(non_dyadic_probabilities, basis_conditionals);
        ASSERT_TRUE(basis_sum.has_value());
        EXPECT_EQ(basis_sum->value, non_dyadic->component_probabilities()[selected]);
    }

    const auto sampler_grid_probability =
        std::ldexp(Scalar{1}, -std::numeric_limits<Scalar>::digits);
    const auto minimum_supported =
        std::array{sampler_grid_probability, Scalar{1} - sampler_grid_probability};
    const auto minimum_supported_mixture = MixtureFor<Scalar>::create(set, minimum_supported);
    ASSERT_TRUE(minimum_supported_mixture.has_value());
    EXPECT_EQ(minimum_supported_mixture->component_probabilities()[0], sampler_grid_probability);
    EXPECT_EQ(minimum_supported_mixture->component_probabilities()[1],
              Scalar{1} - sampler_grid_probability);

    const auto outside_projection = std::array{Scalar{0.5}, Scalar{0.5} + sampler_grid_probability};
    expect_invalid(MixtureFor<Scalar>::create(set, outside_projection));

    const auto singleton_set = make_set<Scalar>(std::array{constant_spectrum<Scalar>(Scalar{0.5})});
    const auto non_unit_singleton = std::array{std::nextafter(Scalar{1}, Scalar{0})};
    expect_invalid(MixtureFor<Scalar>::create(singleton_set, non_unit_singleton));
}

TEST(ClosureMixtureTest, RequiresExplicitNormalizedDistinguishableProbabilities) {
    expect_explicit_probability_validation<TransportScalar>();
    expect_explicit_probability_validation<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_singleton_delegation() {
    const auto reflectance = SpectrumFor<Scalar>{
        .values = {Scalar{0.125}, Scalar{0.25}, Scalar{0.5}, Scalar{1}},
    };
    const auto set = make_set<Scalar>(std::array{reflectance});
    const auto probabilities = std::array{Scalar{1}};
    const auto mixture = MixtureFor<Scalar>::create(set, probabilities);
    const auto lambertian = ReflectionFor<Scalar>::create(reflectance);
    ASSERT_TRUE(mixture.has_value());
    ASSERT_TRUE(lambertian.has_value());

    const auto outgoing = Vector3T<Scalar>{.z = Scalar{1}};
    const auto incoming = Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
    const auto mixture_value = mixture->eval(outgoing, incoming);
    const auto lambertian_value = lambertian->eval(outgoing, incoming);
    const auto mixture_pdf = mixture->pdf(outgoing, incoming);
    const auto lambertian_pdf = lambertian->pdf(outgoing, incoming);
    ASSERT_TRUE(mixture_value.has_value());
    ASSERT_TRUE(lambertian_value.has_value());
    ASSERT_TRUE(mixture_pdf.has_value());
    ASSERT_TRUE(lambertian_pdf.has_value());
    EXPECT_EQ(*mixture_value, *lambertian_value);
    EXPECT_EQ(mixture_pdf->value, lambertian_pdf->value);
    EXPECT_EQ(mixture_pdf->measure, lambertian_pdf->measure);

    const auto direction_sample = Point2T<Scalar>{.x = Scalar{0.75}, .y = Scalar{0.5}};
    const auto mixture_sample = mixture->sample(outgoing, Scalar{0.875}, direction_sample);
    const auto lambertian_sample = lambertian->sample(outgoing, direction_sample);
    ASSERT_TRUE(mixture_sample.has_value());
    ASSERT_TRUE(mixture_sample->has_value());
    ASSERT_TRUE(lambertian_sample.has_value());
    ASSERT_TRUE(lambertian_sample->has_value());
    EXPECT_EQ((**mixture_sample).selected_closure, 0U);
    EXPECT_EQ((**mixture_sample).lobes, ScatteringLobe::diffuse | ScatteringLobe::reflection);
    EXPECT_EQ((**mixture_sample).selection_probability.value, Scalar{1});
    EXPECT_EQ((**mixture_sample).selection_probability.measure, ProbabilityMeasure::discrete);
    EXPECT_EQ((**mixture_sample).incoming_local, (**lambertian_sample).incoming_local);
    EXPECT_EQ((**mixture_sample).value, (**lambertian_sample).value);
    EXPECT_EQ((**mixture_sample).probability.value, (**lambertian_sample).probability.value);
    EXPECT_EQ((**mixture_sample).probability.measure, (**lambertian_sample).probability.measure);
}

TEST(ClosureMixtureTest, DelegatesASingletonWithoutApplyingSelectionTwice) {
    expect_singleton_delegation<TransportScalar>();
    expect_singleton_delegation<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_rough_diffuse_dispatch() {
    const auto rough_reflectance = SpectrumFor<Scalar>{
        .values = {Scalar{0.125}, Scalar{0.25}, Scalar{0.5}, Scalar{1}},
    };
    auto singleton_set = SetFor<Scalar>{};
    ASSERT_EQ(singleton_set.append_rough_diffuse_reflection(rough_reflectance, Scalar{0.75}),
              ClosureAppendStatus::appended);
    const auto singleton_probability = std::array{Scalar{1}};
    const auto singleton = MixtureFor<Scalar>::create(singleton_set, singleton_probability);
    const auto direct = RoughReflectionFor<Scalar>::create(rough_reflectance, Scalar{0.75});
    ASSERT_TRUE(singleton.has_value());
    ASSERT_TRUE(direct.has_value());

    const auto outgoing = Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
    const auto incoming = Vector3T<Scalar>{
        .x = Scalar{-0.3}, .y = Scalar{0.4}, .z = static_cast<Scalar>(std::sqrt(0.75L))};
    const auto singleton_value = singleton->eval(outgoing, incoming);
    const auto direct_value = direct->eval(outgoing, incoming);
    const auto singleton_pdf = singleton->pdf(outgoing, incoming);
    const auto direct_pdf = direct->pdf(outgoing, incoming);
    ASSERT_TRUE(singleton_value.has_value());
    ASSERT_TRUE(direct_value.has_value());
    ASSERT_TRUE(singleton_pdf.has_value());
    ASSERT_TRUE(direct_pdf.has_value());
    EXPECT_EQ(*singleton_value, *direct_value);
    EXPECT_EQ(singleton_pdf->value, direct_pdf->value);
    EXPECT_EQ(singleton_pdf->measure, direct_pdf->measure);

    const auto lambert_reflectance = constant_spectrum<Scalar>(Scalar{0.2});
    auto mixed_set = SetFor<Scalar>{};
    ASSERT_EQ(mixed_set.append_lambertian_reflection(lambert_reflectance),
              ClosureAppendStatus::appended);
    ASSERT_EQ(mixed_set.append_rough_diffuse_reflection(rough_reflectance, Scalar{0.75}),
              ClosureAppendStatus::appended);
    const auto probabilities = std::array{Scalar{0.25}, Scalar{0.75}};
    const auto mixture = MixtureFor<Scalar>::create(mixed_set, probabilities);
    const auto lambert = ReflectionFor<Scalar>::create(lambert_reflectance);
    ASSERT_TRUE(mixture.has_value());
    ASSERT_TRUE(lambert.has_value());

    const auto mixture_value = mixture->eval(outgoing, incoming);
    const auto lambert_value = lambert->eval(outgoing, incoming);
    ASSERT_TRUE(mixture_value.has_value());
    ASSERT_TRUE(lambert_value.has_value());
    auto expected_value = SpectrumFor<Scalar>{};
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        expected_value[lane] = (*lambert_value)[lane] + (*direct_value)[lane];
    }
    expect_spectrum_near(*mixture_value, expected_value);

    const auto canonical = Point2T<Scalar>{.x = Scalar{0.625}, .y = Scalar{0.25}};
    const auto selected_lambert = mixture->sample(outgoing, Scalar{0.125}, canonical);
    const auto selected_rough = mixture->sample(outgoing, Scalar{0.625}, canonical);
    ASSERT_TRUE(selected_lambert.has_value());
    ASSERT_TRUE(selected_lambert->has_value());
    ASSERT_TRUE(selected_rough.has_value());
    ASSERT_TRUE(selected_rough->has_value());
    EXPECT_EQ((**selected_lambert).selected_closure, 0U);
    EXPECT_EQ((**selected_lambert).lobes, ScatteringLobe::diffuse | ScatteringLobe::reflection);
    EXPECT_EQ((**selected_rough).selected_closure, 1U);
    EXPECT_EQ((**selected_rough).lobes, ScatteringLobe::diffuse | ScatteringLobe::reflection);
    EXPECT_EQ((**selected_lambert).selection_probability.value, Scalar{0.25});
    EXPECT_EQ((**selected_rough).selection_probability.value, Scalar{0.75});
    EXPECT_EQ((**selected_lambert).incoming_local, (**selected_rough).incoming_local);
    EXPECT_EQ((**selected_lambert).value, (**selected_rough).value);
    EXPECT_EQ((**selected_lambert).probability.value, (**selected_rough).probability.value);
}

TEST(ClosureMixtureTest, DispatchesRoughDiffuseWithoutAHiddenLambertianFallback) {
    expect_rough_diffuse_dispatch<TransportScalar>();
    expect_rough_diffuse_dispatch<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_rough_conductor_dispatch() {
    const auto coefficient = SpectrumFor<Scalar>{
        .values = {Scalar{0.25}, Scalar{0.5}, Scalar{0.75}, Scalar{1}},
    };
    const auto relative_eta = SpectrumFor<Scalar>{
        .values = {Scalar{0.2}, Scalar{0.5}, Scalar{1.5}, Scalar{2}},
    };
    const auto relative_k = SpectrumFor<Scalar>{
        .values = {Scalar{3}, Scalar{2}, Scalar{1}, Scalar{4}},
    };
    constexpr auto alpha = Scalar{0.5};
    auto singleton_set = SetFor<Scalar>{};
    ASSERT_EQ(singleton_set.append_rough_conductor_reflection(coefficient, relative_eta, relative_k,
                                                              alpha),
              ClosureAppendStatus::appended);
    const auto singleton_probability = std::array{Scalar{1}};
    const auto singleton = MixtureFor<Scalar>::create(singleton_set, singleton_probability);
    const auto direct =
        RoughConductorFor<Scalar>::create(coefficient, relative_eta, relative_k, alpha);
    ASSERT_TRUE(singleton.has_value());
    ASSERT_TRUE(direct.has_value());

    const auto outgoing = Vector3T<Scalar>{.z = Scalar{1}};
    const auto incoming = Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
    const auto singleton_value = singleton->eval(outgoing, incoming);
    const auto direct_value = direct->eval(outgoing, incoming);
    const auto singleton_pdf = singleton->pdf(outgoing, incoming);
    const auto direct_pdf = direct->pdf(outgoing, incoming);
    ASSERT_TRUE(singleton_value.has_value());
    ASSERT_TRUE(direct_value.has_value());
    ASSERT_TRUE(singleton_pdf.has_value());
    ASSERT_TRUE(direct_pdf.has_value());
    EXPECT_EQ(*singleton_value, *direct_value);
    EXPECT_EQ(singleton_pdf->value, direct_pdf->value);
    EXPECT_EQ(singleton_pdf->measure, direct_pdf->measure);

    const auto canonical = Point2T<Scalar>{.x = Scalar{0.2}, .y = Scalar{0.375}};
    const auto singleton_sample = singleton->sample(outgoing, Scalar{0.5}, canonical);
    const auto direct_sample = direct->sample(outgoing, canonical);
    ASSERT_TRUE(singleton_sample.has_value());
    ASSERT_TRUE(singleton_sample->has_value());
    ASSERT_TRUE(direct_sample.has_value());
    ASSERT_TRUE(direct_sample->has_value());
    EXPECT_EQ((**singleton_sample).selected_closure, 0U);
    EXPECT_EQ((**singleton_sample).lobes, ScatteringLobe::glossy | ScatteringLobe::reflection);
    EXPECT_EQ((**singleton_sample).selection_probability.value, Scalar{1});
    EXPECT_EQ((**singleton_sample).incoming_local, (**direct_sample).incoming_local);
    EXPECT_EQ((**singleton_sample).value, (**direct_sample).value);
    EXPECT_EQ((**singleton_sample).probability.value, (**direct_sample).probability.value);

    const auto lambert_reflectance = constant_spectrum<Scalar>(Scalar{0.2});
    auto mixed_set = SetFor<Scalar>{};
    ASSERT_EQ(mixed_set.append_lambertian_reflection(lambert_reflectance),
              ClosureAppendStatus::appended);
    ASSERT_EQ(
        mixed_set.append_rough_conductor_reflection(coefficient, relative_eta, relative_k, alpha),
        ClosureAppendStatus::appended);
    const auto probabilities = std::array{Scalar{0.25}, Scalar{0.75}};
    const auto mixture = MixtureFor<Scalar>::create(mixed_set, probabilities);
    const auto lambert = ReflectionFor<Scalar>::create(lambert_reflectance);
    ASSERT_TRUE(mixture.has_value());
    ASSERT_TRUE(lambert.has_value());

    const auto mixture_value = mixture->eval(outgoing, incoming);
    const auto lambert_value = lambert->eval(outgoing, incoming);
    const auto mixture_pdf = mixture->pdf(outgoing, incoming);
    const auto lambert_pdf = lambert->pdf(outgoing, incoming);
    ASSERT_TRUE(mixture_value.has_value());
    ASSERT_TRUE(lambert_value.has_value());
    ASSERT_TRUE(mixture_pdf.has_value());
    ASSERT_TRUE(lambert_pdf.has_value());
    auto expected_value = SpectrumFor<Scalar>{};
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        expected_value[lane] = (*lambert_value)[lane] + (*direct_value)[lane];
    }
    expect_spectrum_near(*mixture_value, expected_value);
    EXPECT_NEAR(
        static_cast<double>(mixture_pdf->value),
        static_cast<double>(Scalar{0.25} * lambert_pdf->value + Scalar{0.75} * direct_pdf->value),
        AnalyticTolerance<Scalar>);

    const auto selected_conductor = mixture->sample(outgoing, Scalar{0.625}, canonical);
    const auto lambert_sample = lambert->sample(outgoing, canonical);
    ASSERT_TRUE(selected_conductor.has_value());
    ASSERT_TRUE(selected_conductor->has_value());
    ASSERT_TRUE(lambert_sample.has_value());
    ASSERT_TRUE(lambert_sample->has_value());
    EXPECT_EQ((**selected_conductor).selected_closure, 1U);
    EXPECT_EQ((**selected_conductor).lobes, ScatteringLobe::glossy | ScatteringLobe::reflection);
    EXPECT_EQ((**selected_conductor).selection_probability.value, Scalar{0.75});
    EXPECT_EQ((**selected_conductor).incoming_local, (**direct_sample).incoming_local);
    EXPECT_NE((**selected_conductor).incoming_local, (**lambert_sample).incoming_local);
}

TEST(ClosureMixtureTest, DispatchesRoughConductorWithoutAHiddenDiffuseOrDeltaFallback) {
    expect_rough_conductor_dispatch<TransportScalar>();
    expect_rough_conductor_dispatch<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_two_component_mixture() {
    const auto first = SpectrumFor<Scalar>{
        .values = {Scalar{0.125}, Scalar{0.25}, Scalar{0.375}, Scalar{0.5}},
    };
    const auto second = SpectrumFor<Scalar>{
        .values = {Scalar{0.5}, Scalar{0.375}, Scalar{0.25}, Scalar{0.125}},
    };
    const auto expected_reflectance = constant_spectrum<Scalar>(Scalar{0.625});
    const auto set = make_set<Scalar>(std::array{first, second});
    const auto probabilities = std::array{Scalar{0.25}, Scalar{0.75}};
    const auto mixture = MixtureFor<Scalar>::create(set, probabilities);
    ASSERT_TRUE(mixture.has_value());

    const auto outgoing = Vector3T<Scalar>{.z = Scalar{1}};
    const auto incoming = Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
    const auto value = mixture->eval(outgoing, incoming);
    const auto probability = mixture->pdf(outgoing, incoming);
    ASSERT_TRUE(value.has_value());
    ASSERT_TRUE(probability.has_value());
    expect_spectrum_near(*value, expected_reflectance * std::numbers::inv_pi_v<Scalar>);
    EXPECT_NEAR(static_cast<double>(probability->value),
                static_cast<double>(incoming.z * std::numbers::inv_pi_v<Scalar>),
                AnalyticTolerance<Scalar>);
    EXPECT_EQ(probability->measure, ProbabilityMeasure::solid_angle);

    const auto direction_sample = Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}};
    const auto first_sample = mixture->sample(outgoing, Scalar{0.125}, direction_sample);
    const auto second_sample = mixture->sample(outgoing, Scalar{0.25}, direction_sample);
    ASSERT_TRUE(first_sample.has_value());
    ASSERT_TRUE(first_sample->has_value());
    ASSERT_TRUE(second_sample.has_value());
    ASSERT_TRUE(second_sample->has_value());
    EXPECT_EQ((**first_sample).selected_closure, 0U);
    EXPECT_EQ((**first_sample).selection_probability.value, Scalar{0.25});
    EXPECT_EQ((**second_sample).selected_closure, 1U);
    EXPECT_EQ((**second_sample).selection_probability.value, Scalar{0.75});
    EXPECT_EQ((**first_sample).incoming_local, (**second_sample).incoming_local);
    expect_spectrum_near((**first_sample).value, (**second_sample).value);
    EXPECT_EQ((**first_sample).probability.value, (**second_sample).probability.value);
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        const auto throughput = (**first_sample).value[lane] * (**first_sample).incoming_local.z /
                                (**first_sample).probability.value;
        EXPECT_NEAR(static_cast<double>(throughput),
                    static_cast<double>(expected_reflectance[lane]), AnalyticTolerance<Scalar>);
    }
}

TEST(ClosureMixtureTest, AddsPhysicalValuesAndWeightsEveryConditionalPdfExactlyOnce) {
    expect_two_component_mixture<TransportScalar>();
    expect_two_component_mixture<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_black_component_remains_selectable() {
    const auto set = make_set<Scalar>(std::array{
        constant_spectrum<Scalar>(Scalar{0}),
        constant_spectrum<Scalar>(Scalar{1}),
    });
    const auto probabilities = std::array{Scalar{0.5}, Scalar{0.5}};
    const auto mixture = MixtureFor<Scalar>::create(set, probabilities);
    ASSERT_TRUE(mixture.has_value());

    const auto outgoing = Vector3T<Scalar>{.z = Scalar{1}};
    const auto sampled = mixture->sample(outgoing, Scalar{0.25},
                                         Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}});
    ASSERT_TRUE(sampled.has_value());
    ASSERT_TRUE(sampled->has_value());
    EXPECT_EQ((**sampled).selected_closure, 0U);
    EXPECT_EQ((**sampled).selection_probability.value, Scalar{0.5});
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        const auto throughput =
            (**sampled).value[lane] * (**sampled).incoming_local.z / (**sampled).probability.value;
        EXPECT_NEAR(static_cast<double>(throughput), 1.0, AnalyticTolerance<Scalar>);
    }
}

TEST(ClosureMixtureTest, KeepsBlackComponentsInTheExplicitSamplingDistribution) {
    expect_black_component_remains_selectable<TransportScalar>();
    expect_black_component_remains_selectable<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_selection_boundaries_and_replay() {
    const auto set = make_set<Scalar>(std::array{
        constant_spectrum<Scalar>(Scalar{0.1}),
        constant_spectrum<Scalar>(Scalar{0.1}),
        constant_spectrum<Scalar>(Scalar{0.1}),
        constant_spectrum<Scalar>(Scalar{0.1}),
    });
    const auto probabilities = std::array{Scalar{0.25}, Scalar{0.25}, Scalar{0.25}, Scalar{0.25}};
    const auto mixture = MixtureFor<Scalar>::create(set, probabilities);
    ASSERT_TRUE(mixture.has_value());
    const auto outgoing = Vector3T<Scalar>{.z = Scalar{1}};
    const auto direction_sample = Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}};

    const auto boundary_cases = std::array{
        std::pair{Scalar{0}, 0U},    std::pair{std::nextafter(Scalar{0.25}, Scalar{0}), 0U},
        std::pair{Scalar{0.25}, 1U}, std::pair{std::nextafter(Scalar{0.5}, Scalar{0}), 1U},
        std::pair{Scalar{0.5}, 2U},  std::pair{std::nextafter(Scalar{0.75}, Scalar{0}), 2U},
        std::pair{Scalar{0.75}, 3U}, std::pair{std::nextafter(Scalar{1}, Scalar{0}), 3U},
    };
    for (const auto& [component_sample, expected_index] : boundary_cases) {
        const auto sampled = mixture->sample(outgoing, component_sample, direction_sample);
        ASSERT_TRUE(sampled.has_value());
        ASSERT_TRUE(sampled->has_value());
        EXPECT_EQ((**sampled).selected_closure, expected_index);
        EXPECT_EQ((**sampled).selection_probability.value, Scalar{0.25});
    }

    const auto replay_a = mixture->sample(outgoing, Scalar{0.625}, direction_sample);
    const auto replay_b = mixture->sample(outgoing, Scalar{0.625}, direction_sample);
    ASSERT_TRUE(replay_a.has_value());
    ASSERT_TRUE(replay_a->has_value());
    ASSERT_TRUE(replay_b.has_value());
    ASSERT_TRUE(replay_b->has_value());
    EXPECT_EQ((**replay_a).selected_closure, (**replay_b).selected_closure);
    EXPECT_EQ((**replay_a).incoming_local, (**replay_b).incoming_local);
    EXPECT_EQ((**replay_a).value, (**replay_b).value);
    EXPECT_EQ((**replay_a).probability.value, (**replay_b).probability.value);

    for (const auto invalid : std::array{
             -std::numeric_limits<Scalar>::denorm_min(),
             Scalar{1},
             std::numeric_limits<Scalar>::quiet_NaN(),
             std::numeric_limits<Scalar>::infinity(),
         }) {
        expect_invalid(mixture->sample(outgoing, invalid, direction_sample));
    }
}

TEST(ClosureMixtureTest, SelectsHalfOpenCdfIntervalsAndReplaysExactly) {
    expect_selection_boundaries_and_replay<TransportScalar>();
    expect_selection_boundaries_and_replay<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_full_capacity_is_not_clamped_or_averaged() {
    auto set = SetFor<Scalar>{};
    for (auto index = std::uint32_t{}; index < MaximumClosureCount; ++index) {
        ASSERT_EQ(set.append_lambertian_reflection(constant_spectrum<Scalar>(Scalar{1})),
                  ClosureAppendStatus::appended);
    }
    auto probabilities = std::array<Scalar, MaximumClosureCount>{};
    probabilities.fill(Scalar{0.125});
    const auto mixture = MixtureFor<Scalar>::create(set, probabilities);
    ASSERT_TRUE(mixture.has_value());

    const auto outgoing = Vector3T<Scalar>{.z = Scalar{1}};
    const auto incoming = Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
    const auto value = mixture->eval(outgoing, incoming);
    const auto probability = mixture->pdf(outgoing, incoming);
    ASSERT_TRUE(value.has_value());
    ASSERT_TRUE(probability.has_value());
    const auto expected_value = Scalar{8} * std::numbers::inv_pi_v<Scalar>;
    for (const auto lane : value->values) {
        EXPECT_NEAR(static_cast<double>(lane), static_cast<double>(expected_value),
                    AnalyticTolerance<Scalar>);
    }
    EXPECT_NEAR(static_cast<double>(probability->value),
                static_cast<double>(incoming.z * std::numbers::inv_pi_v<Scalar>),
                AnalyticTolerance<Scalar>);
}

TEST(ClosureMixtureTest, PreservesAllEightInlineClosuresWithoutClampingOrAveraging) {
    expect_full_capacity_is_not_clamped_or_averaged<TransportScalar>();
    expect_full_capacity_is_not_clamped_or_averaged<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_empty_and_invalid_inputs_are_explicit() {
    const auto mixture = MixtureFor<Scalar>::create(SetFor<Scalar>{}, std::span<const Scalar>{});
    ASSERT_TRUE(mixture.has_value());
    const auto normal = Vector3T<Scalar>{.z = Scalar{1}};
    const auto value = mixture->eval(normal, normal);
    const auto probability = mixture->pdf(normal, normal);
    const auto sampled =
        mixture->sample(normal, Scalar{0.5}, Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}});
    ASSERT_TRUE(value.has_value());
    ASSERT_TRUE(probability.has_value());
    ASSERT_TRUE(sampled.has_value());
    EXPECT_EQ(*value, SpectrumFor<Scalar>{});
    EXPECT_EQ(probability->value, Scalar{0});
    EXPECT_EQ(probability->measure, ProbabilityMeasure::solid_angle);
    EXPECT_FALSE(sampled->has_value());

    const auto invalid_direction = Vector3T<Scalar>{};
    expect_invalid(mixture->eval(invalid_direction, normal));
    expect_invalid(mixture->eval(normal, invalid_direction));
    expect_invalid(mixture->pdf(invalid_direction, normal));
    expect_invalid(mixture->pdf(normal, invalid_direction));
    expect_invalid(mixture->sample(invalid_direction, Scalar{0.5},
                                   Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}}));
    expect_invalid(
        mixture->sample(normal, Scalar{1}, Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}}));
    expect_invalid(
        mixture->sample(normal, Scalar{0.5}, Point2T<Scalar>{.x = Scalar{1}, .y = Scalar{0.5}}));
}

TEST(ClosureMixtureTest, TreatsAnEmptySetAsAbsorptionWithoutHidingInvalidInputs) {
    expect_empty_and_invalid_inputs_are_explicit<TransportScalar>();
    expect_empty_and_invalid_inputs_are_explicit<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_mixture_pdf_integrates_to_one() {
    const auto set = make_set<Scalar>(std::array{
        constant_spectrum<Scalar>(Scalar{0.1}),
        constant_spectrum<Scalar>(Scalar{0.2}),
        constant_spectrum<Scalar>(Scalar{0.3}),
    });
    const auto probabilities = std::array{Scalar{0.125}, Scalar{0.25}, Scalar{0.625}};
    const auto mixture = MixtureFor<Scalar>::create(set, probabilities);
    ASSERT_TRUE(mixture.has_value());
    const auto outgoing = Vector3T<Scalar>{.z = Scalar{1}};

    constexpr auto cosine_steps = std::size_t{128};
    constexpr auto azimuth_steps = std::size_t{16};
    constexpr auto delta_cosine = 2.0L / static_cast<long double>(cosine_steps);
    constexpr auto delta_azimuth =
        2.0L * std::numbers::pi_v<long double> / static_cast<long double>(azimuth_steps);
    auto integral = 0.0L;
    for (auto cosine_index = std::size_t{}; cosine_index < cosine_steps; ++cosine_index) {
        const auto cosine = -1.0L + (static_cast<long double>(cosine_index) + 0.5L) * delta_cosine;
        const auto radial = std::sqrt((1.0L - cosine) * (1.0L + cosine));
        for (auto azimuth_index = std::size_t{}; azimuth_index < azimuth_steps; ++azimuth_index) {
            const auto azimuth = (static_cast<long double>(azimuth_index) + 0.5L) * delta_azimuth;
            const auto incoming = Vector3T<Scalar>{
                .x = static_cast<Scalar>(radial * std::cos(azimuth)),
                .y = static_cast<Scalar>(radial * std::sin(azimuth)),
                .z = static_cast<Scalar>(cosine),
            };
            const auto probability = mixture->pdf(outgoing, incoming);
            ASSERT_TRUE(probability.has_value());
            EXPECT_EQ(probability->measure, ProbabilityMeasure::solid_angle);
            integral += static_cast<long double>(probability->value) * delta_cosine * delta_azimuth;
        }
    }
    const auto tolerance = std::same_as<Scalar, TransportScalar> ? 8.0e-6L : 2.0e-12L;
    EXPECT_NEAR(integral, 1.0L, tolerance);
}

TEST(ClosureMixtureTest, IntegratesTheCompleteMixturePdfToOneInBothPrecisions) {
    expect_mixture_pdf_integrates_to_one<TransportScalar>();
    expect_mixture_pdf_integrates_to_one<ReferenceScalar>();
}

template <typename Value, typename Field>
[[nodiscard]] Value overwrite_bytes(Value value, const std::size_t offset, const Field field) {
    auto bytes = std::bit_cast<std::array<std::byte, sizeof(Value)>>(value);
    const auto field_bytes = std::bit_cast<std::array<std::byte, sizeof(Field)>>(field);
    for (auto index = std::size_t{}; index < field_bytes.size(); ++index) {
        bytes[offset + index] = field_bytes[index];
    }
    return std::bit_cast<Value>(bytes);
}

template <SpectrumScalar Scalar> void expect_corrupt_records_are_rejected() {
    const auto valid_set = make_set<Scalar>(std::array{constant_spectrum<Scalar>(Scalar{0.5})});
    const auto probability = std::array{Scalar{1}};
    constexpr auto set_storage_offset = std::size_t{8};
    constexpr auto closure_weight_offset = std::size_t{8};
    constexpr auto parameter_offset =
        std::same_as<Scalar, TransportScalar> ? std::size_t{24} : std::size_t{40};

    const auto invalid_count = overwrite_bytes(valid_set, std::size_t{0}, MaximumClosureCount + 1U);
    const auto excessive_probabilities = std::array<Scalar, MaximumClosureCount + 1U>{};
    expect_invalid(MixtureFor<Scalar>::create(invalid_count, excessive_probabilities));

    const auto unknown_kind =
        overwrite_bytes(valid_set, set_storage_offset, static_cast<ClosureKind>(0xffffffffU));
    expect_invalid(MixtureFor<Scalar>::create(unknown_kind, probability));

    const auto inactive_kind = overwrite_bytes(valid_set, set_storage_offset, ClosureKind::none);
    expect_invalid(MixtureFor<Scalar>::create(inactive_kind, probability));

    const auto incompatible_lobes =
        overwrite_bytes(valid_set, set_storage_offset + std::size_t{4},
                        ScatteringLobe::glossy | ScatteringLobe::reflection);
    expect_invalid(MixtureFor<Scalar>::create(incompatible_lobes, probability));

    const auto nonzero_payload =
        overwrite_bytes(valid_set, set_storage_offset + parameter_offset, Scalar{1});
    expect_invalid(MixtureFor<Scalar>::create(nonzero_payload, probability));

    const auto invalid_weight =
        overwrite_bytes(valid_set, set_storage_offset + closure_weight_offset,
                        std::numeric_limits<Scalar>::quiet_NaN());
    expect_invalid(MixtureFor<Scalar>::create(invalid_weight, probability));

    auto rough_set = SetFor<Scalar>{};
    ASSERT_EQ(rough_set.append_rough_diffuse_reflection(constant_spectrum<Scalar>(Scalar{0.5}),
                                                        Scalar{0.75}),
              ClosureAppendStatus::appended);
    const auto rough_incompatible_lobes =
        overwrite_bytes(rough_set, set_storage_offset + std::size_t{4},
                        ScatteringLobe::glossy | ScatteringLobe::reflection);
    expect_invalid(MixtureFor<Scalar>::create(rough_incompatible_lobes, probability));

    const auto invalid_roughness =
        overwrite_bytes(rough_set, set_storage_offset + parameter_offset,
                        std::nextafter(Scalar{1}, std::numeric_limits<Scalar>::infinity()));
    expect_invalid(MixtureFor<Scalar>::create(invalid_roughness, probability));

    const auto rough_nonzero_reserved = overwrite_bytes(
        rough_set, set_storage_offset + parameter_offset + sizeof(Scalar), Scalar{1});
    expect_invalid(MixtureFor<Scalar>::create(rough_nonzero_reserved, probability));
}

TEST(ClosureMixtureTest, RejectsUnsupportedOrCorruptRecordsWithoutDispatchFallback) {
    expect_corrupt_records_are_rejected<TransportScalar>();
    expect_corrupt_records_are_rejected<ReferenceScalar>();
}

static_assert(std::is_standard_layout_v<ClosureMixture>);
static_assert(std::is_trivially_copyable_v<ClosureMixture>);
static_assert(std::is_trivially_destructible_v<ClosureMixture>);
static_assert(std::is_standard_layout_v<ReferenceClosureMixture>);
static_assert(std::is_trivially_copyable_v<ReferenceClosureMixture>);
static_assert(std::is_trivially_destructible_v<ReferenceClosureMixture>);
static_assert(!std::same_as<ClosureMixture, ReferenceClosureMixture>);

} // namespace
} // namespace blackframe::renderer
