#include <Blackframe/Renderer/ClosureSet.hpp>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>
#include <utility>

namespace blackframe::renderer {
namespace {

template <SpectrumScalar Scalar>
using ClosureSetFor =
    std::conditional_t<std::same_as<Scalar, TransportScalar>, ClosureSet, ReferenceClosureSet>;

template <SpectrumScalar Scalar>
[[nodiscard]] SampledSpectrum<TransportSpectrumSampleCount, Scalar>
constant_spectrum(const Scalar value) {
    auto spectrum = SampledSpectrum<TransportSpectrumSampleCount, Scalar>{};
    spectrum.values.fill(value);
    return spectrum;
}

template <typename Value> [[nodiscard]] auto object_bytes(const Value& value) {
    return std::bit_cast<std::array<std::byte, sizeof(Value)>>(value);
}

TEST(ClosureSetTest, FreezesTheInlineAbiForTransportAndReferencePrecision) {
    static_assert(MaximumClosureCount == 8U);
    static_assert(ClosureParameterScalarCount == 10U);
    static_assert(std::is_same_v<std::underlying_type_t<ClosureKind>, std::uint32_t>);
    static_assert(std::is_same_v<std::underlying_type_t<ClosureAppendStatus>, std::uint32_t>);
    static_assert(std::is_standard_layout_v<Closure>);
    static_assert(std::is_trivially_copyable_v<Closure>);
    static_assert(std::is_trivially_destructible_v<Closure>);
    static_assert(std::is_standard_layout_v<ReferenceClosure>);
    static_assert(std::is_trivially_copyable_v<ReferenceClosure>);
    static_assert(std::is_trivially_destructible_v<ReferenceClosure>);
    static_assert(std::is_standard_layout_v<ClosureSet>);
    static_assert(std::is_trivially_copyable_v<ClosureSet>);
    static_assert(std::is_trivially_destructible_v<ClosureSet>);
    static_assert(std::is_standard_layout_v<ReferenceClosureSet>);
    static_assert(std::is_trivially_copyable_v<ReferenceClosureSet>);
    static_assert(std::is_trivially_destructible_v<ReferenceClosureSet>);

    EXPECT_EQ(sizeof(ClosureKind), 4U);
    EXPECT_EQ(sizeof(ClosureAppendStatus), 4U);
    EXPECT_EQ(sizeof(Closure), 64U);
    EXPECT_EQ(alignof(Closure), 8U);
    EXPECT_EQ(offsetof(Closure, kind), 0U);
    EXPECT_EQ(offsetof(Closure, lobes), 4U);
    EXPECT_EQ(offsetof(Closure, weight), 8U);
    EXPECT_EQ(offsetof(Closure, parameters), 24U);
    EXPECT_EQ(sizeof(ReferenceClosure), 120U);
    EXPECT_EQ(alignof(ReferenceClosure), 8U);
    EXPECT_EQ(offsetof(ReferenceClosure, kind), 0U);
    EXPECT_EQ(offsetof(ReferenceClosure, lobes), 4U);
    EXPECT_EQ(offsetof(ReferenceClosure, weight), 8U);
    EXPECT_EQ(offsetof(ReferenceClosure, parameters), 40U);
    EXPECT_EQ(sizeof(ClosureSet), 520U);
    EXPECT_EQ(alignof(ClosureSet), 8U);
    EXPECT_EQ(closure_set_detail::ClosureSetLayoutProbe<TransportScalar>::active_count_offset, 0U);
    EXPECT_EQ(closure_set_detail::ClosureSetLayoutProbe<TransportScalar>::reserved_offset, 4U);
    EXPECT_EQ(closure_set_detail::ClosureSetLayoutProbe<TransportScalar>::closures_offset, 8U);
    EXPECT_EQ(sizeof(ReferenceClosureSet), 968U);
    EXPECT_EQ(alignof(ReferenceClosureSet), 8U);
    EXPECT_EQ(closure_set_detail::ClosureSetLayoutProbe<ReferenceScalar>::active_count_offset, 0U);
    EXPECT_EQ(closure_set_detail::ClosureSetLayoutProbe<ReferenceScalar>::reserved_offset, 4U);
    EXPECT_EQ(closure_set_detail::ClosureSetLayoutProbe<ReferenceScalar>::closures_offset, 8U);
    EXPECT_EQ(ClosureSet::capacity(), 8U);
    EXPECT_EQ(ReferenceClosureSet::capacity(), 8U);
}

TEST(ClosureSetTest, FreezesKindAndAppendStatusCodes) {
    EXPECT_EQ(static_cast<std::uint32_t>(ClosureKind::none), 0U);
    EXPECT_EQ(static_cast<std::uint32_t>(ClosureKind::lambertian_reflection), 1U);
    EXPECT_EQ(static_cast<std::uint32_t>(ClosureKind::rough_diffuse_reflection), 2U);
    EXPECT_EQ(static_cast<std::uint32_t>(ClosureKind::rough_conductor_reflection), 3U);
    EXPECT_TRUE(is_known_closure_kind(ClosureKind::none));
    EXPECT_TRUE(is_known_closure_kind(ClosureKind::lambertian_reflection));
    EXPECT_TRUE(is_known_closure_kind(ClosureKind::rough_diffuse_reflection));
    EXPECT_TRUE(is_known_closure_kind(ClosureKind::rough_conductor_reflection));
    EXPECT_FALSE(is_known_closure_kind(static_cast<ClosureKind>(4U)));
    EXPECT_FALSE(is_known_closure_kind(static_cast<ClosureKind>(0xffffffffU)));

    EXPECT_EQ(static_cast<std::uint32_t>(ClosureAppendStatus::appended), 0U);
    EXPECT_EQ(static_cast<std::uint32_t>(ClosureAppendStatus::invalid_payload), 1U);
    EXPECT_EQ(static_cast<std::uint32_t>(ClosureAppendStatus::capacity_exhausted), 2U);
}

TEST(ClosureSetTest, StartsAsCanonicalZeroedInlineStorage) {
    const auto transport = ClosureSet{};
    const auto reference = ReferenceClosureSet{};

    EXPECT_TRUE(transport.empty());
    EXPECT_FALSE(transport.full());
    EXPECT_EQ(transport.size(), 0U);
    EXPECT_TRUE(transport.closures().empty());
    EXPECT_TRUE(reference.empty());
    EXPECT_FALSE(reference.full());
    EXPECT_EQ(reference.size(), 0U);
    EXPECT_TRUE(reference.closures().empty());

    for (const auto byte : object_bytes(transport)) {
        EXPECT_EQ(byte, std::byte{0});
    }
    for (const auto byte : object_bytes(reference)) {
        EXPECT_EQ(byte, std::byte{0});
    }
}

template <SpectrumScalar Scalar> void expect_stable_capacity_and_order() {
    auto set = ClosureSetFor<Scalar>{};
    for (auto index = std::uint32_t{}; index < MaximumClosureCount; ++index) {
        const auto value =
            static_cast<Scalar>(index + 1U) / static_cast<Scalar>(MaximumClosureCount);
        EXPECT_EQ(set.append_lambertian_reflection(constant_spectrum(value)),
                  ClosureAppendStatus::appended);
        EXPECT_EQ(set.size(), index + 1U);
    }

    ASSERT_TRUE(set.full());
    ASSERT_EQ(set.closures().size(), MaximumClosureCount);
    for (auto index = std::uint32_t{}; index < MaximumClosureCount; ++index) {
        const auto expected =
            static_cast<Scalar>(index + 1U) / static_cast<Scalar>(MaximumClosureCount);
        const auto& closure = set.closures()[index];
        EXPECT_EQ(closure.kind, ClosureKind::lambertian_reflection);
        EXPECT_EQ(closure.lobes, ScatteringLobe::diffuse | ScatteringLobe::reflection);
        for (const auto value : closure.weight.values) {
            EXPECT_EQ(value, expected);
        }
        for (const auto parameter : closure.parameters) {
            EXPECT_EQ(parameter, Scalar{0});
        }
    }
}

TEST(ClosureSetTest, PreservesTheActivePrefixInBothPrecisions) {
    expect_stable_capacity_and_order<TransportScalar>();
    expect_stable_capacity_and_order<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_rough_diffuse_record() {
    auto set = ClosureSetFor<Scalar>{};
    const auto reflectance = constant_spectrum(Scalar{0.625});
    ASSERT_EQ(set.append_lambertian_reflection(constant_spectrum(Scalar{0.25})),
              ClosureAppendStatus::appended);
    ASSERT_EQ(set.append_rough_diffuse_reflection(reflectance, Scalar{0.75}),
              ClosureAppendStatus::appended);
    ASSERT_EQ(set.size(), 2U);

    const auto& closure = set.closures()[1];
    EXPECT_EQ(closure.kind, ClosureKind::rough_diffuse_reflection);
    EXPECT_EQ(closure.lobes, ScatteringLobe::diffuse | ScatteringLobe::reflection);
    EXPECT_EQ(closure.weight, reflectance);
    EXPECT_EQ(closure.parameters[0], Scalar{0.75});
    for (auto index = std::size_t{1}; index < closure.parameters.size(); ++index) {
        EXPECT_EQ(closure.parameters[index], Scalar{0});
    }
}

TEST(ClosureSetTest, StoresRoughDiffuseInTheExistingInlineAbi) {
    expect_rough_diffuse_record<TransportScalar>();
    expect_rough_diffuse_record<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_rough_conductor_record() {
    auto set = ClosureSetFor<Scalar>{};
    const auto coefficient = constant_spectrum(Scalar{0.625});
    const auto relative_eta = SampledSpectrum<TransportSpectrumSampleCount, Scalar>{
        .values = {Scalar{0.2}, Scalar{0.5}, Scalar{1.5}, Scalar{2}},
    };
    const auto relative_k = SampledSpectrum<TransportSpectrumSampleCount, Scalar>{
        .values = {Scalar{3}, Scalar{2}, Scalar{1}, Scalar{4}},
    };
    ASSERT_EQ(
        set.append_rough_conductor_reflection(coefficient, relative_eta, relative_k, Scalar{0.75}),
        ClosureAppendStatus::appended);
    ASSERT_EQ(set.size(), 1U);

    const auto& closure = set.closures().front();
    EXPECT_EQ(closure.kind, ClosureKind::rough_conductor_reflection);
    EXPECT_EQ(closure.lobes, ScatteringLobe::glossy | ScatteringLobe::reflection);
    EXPECT_EQ(closure.weight, coefficient);
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        EXPECT_EQ(closure.parameters[lane], relative_eta[lane]);
        EXPECT_EQ(closure.parameters[TransportSpectrumSampleCount + lane], relative_k[lane]);
    }
    EXPECT_EQ(closure.parameters[8], Scalar{0.75});
    EXPECT_EQ(closure.parameters[9], Scalar{0});
}

TEST(ClosureSetTest, StoresRoughConductorInTheExistingInlineAbi) {
    expect_rough_conductor_record<TransportScalar>();
    expect_rough_conductor_record<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_invalid_payloads_are_atomic() {
    auto set = ClosureSetFor<Scalar>{};
    ASSERT_EQ(set.append_lambertian_reflection(constant_spectrum(Scalar{0.25})),
              ClosureAppendStatus::appended);
    const auto original = object_bytes(set);

    const auto invalid_values = std::array{
        std::numeric_limits<Scalar>::quiet_NaN(),
        std::numeric_limits<Scalar>::infinity(),
        -std::numeric_limits<Scalar>::denorm_min(),
        std::nextafter(Scalar{1}, std::numeric_limits<Scalar>::infinity()),
    };
    for (const auto invalid : invalid_values) {
        auto reflectance = constant_spectrum(Scalar{0.5});
        reflectance[2] = invalid;
        EXPECT_EQ(set.append_lambertian_reflection(reflectance),
                  ClosureAppendStatus::invalid_payload);
        EXPECT_EQ(object_bytes(set), original);
    }
}

TEST(ClosureSetTest, RejectsMalformedLambertianPayloadsWithoutMutationOrClamping) {
    expect_invalid_payloads_are_atomic<TransportScalar>();
    expect_invalid_payloads_are_atomic<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_invalid_rough_diffuse_payloads_are_atomic() {
    auto set = ClosureSetFor<Scalar>{};
    ASSERT_EQ(set.append_rough_diffuse_reflection(constant_spectrum(Scalar{0.25}), Scalar{0.5}),
              ClosureAppendStatus::appended);
    const auto original = object_bytes(set);

    for (const auto invalid : std::array{
             std::numeric_limits<Scalar>::quiet_NaN(),
             std::numeric_limits<Scalar>::infinity(),
             -std::numeric_limits<Scalar>::denorm_min(),
             std::nextafter(Scalar{1}, std::numeric_limits<Scalar>::infinity()),
         }) {
        EXPECT_EQ(set.append_rough_diffuse_reflection(constant_spectrum(Scalar{0.5}), invalid),
                  ClosureAppendStatus::invalid_payload);
        EXPECT_EQ(object_bytes(set), original);

        auto malformed = constant_spectrum(Scalar{0.5});
        malformed[1] = invalid;
        EXPECT_EQ(set.append_rough_diffuse_reflection(malformed, Scalar{0.5}),
                  ClosureAppendStatus::invalid_payload);
        EXPECT_EQ(object_bytes(set), original);
    }
}

TEST(ClosureSetTest, RejectsMalformedRoughDiffusePayloadsAtomically) {
    expect_invalid_rough_diffuse_payloads_are_atomic<TransportScalar>();
    expect_invalid_rough_diffuse_payloads_are_atomic<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_invalid_rough_conductor_payloads_are_atomic() {
    auto set = ClosureSetFor<Scalar>{};
    const auto coefficient = constant_spectrum(Scalar{0.5});
    const auto relative_eta = constant_spectrum(Scalar{1.5});
    const auto relative_k = constant_spectrum(Scalar{2});
    ASSERT_EQ(
        set.append_rough_conductor_reflection(coefficient, relative_eta, relative_k, Scalar{0.5}),
        ClosureAppendStatus::appended);
    const auto original = object_bytes(set);
    const auto infinity = std::numeric_limits<Scalar>::infinity();

    auto malformed_coefficient = coefficient;
    malformed_coefficient[0] = std::nextafter(Scalar{1}, infinity);
    EXPECT_EQ(set.append_rough_conductor_reflection(malformed_coefficient, relative_eta, relative_k,
                                                    Scalar{0.5}),
              ClosureAppendStatus::invalid_payload);
    EXPECT_EQ(object_bytes(set), original);
    auto malformed_eta = relative_eta;
    malformed_eta[1] = Scalar{0};
    EXPECT_EQ(
        set.append_rough_conductor_reflection(coefficient, malformed_eta, relative_k, Scalar{0.5}),
        ClosureAppendStatus::invalid_payload);
    EXPECT_EQ(object_bytes(set), original);
    auto malformed_k = relative_k;
    malformed_k[2] = -std::numeric_limits<Scalar>::denorm_min();
    EXPECT_EQ(
        set.append_rough_conductor_reflection(coefficient, relative_eta, malformed_k, Scalar{0.5}),
        ClosureAppendStatus::invalid_payload);
    EXPECT_EQ(object_bytes(set), original);

    for (const auto invalid :
         std::array{std::numeric_limits<Scalar>::quiet_NaN(), infinity, -infinity}) {
        malformed_coefficient = coefficient;
        malformed_coefficient[0] = invalid;
        EXPECT_EQ(set.append_rough_conductor_reflection(malformed_coefficient, relative_eta,
                                                        relative_k, Scalar{0.5}),
                  ClosureAppendStatus::invalid_payload);
        malformed_eta = relative_eta;
        malformed_eta[1] = invalid;
        EXPECT_EQ(set.append_rough_conductor_reflection(coefficient, malformed_eta, relative_k,
                                                        Scalar{0.5}),
                  ClosureAppendStatus::invalid_payload);
        malformed_k = relative_k;
        malformed_k[2] = invalid;
        EXPECT_EQ(set.append_rough_conductor_reflection(coefficient, relative_eta, malformed_k,
                                                        Scalar{0.5}),
                  ClosureAppendStatus::invalid_payload);
        EXPECT_EQ(
            set.append_rough_conductor_reflection(coefficient, relative_eta, relative_k, invalid),
            ClosureAppendStatus::invalid_payload);
        EXPECT_EQ(object_bytes(set), original);
    }
    for (const auto invalid_alpha :
         std::array{Scalar{0}, Scalar{-0.0}, -std::numeric_limits<Scalar>::denorm_min()}) {
        EXPECT_EQ(set.append_rough_conductor_reflection(coefficient, relative_eta, relative_k,
                                                        invalid_alpha),
                  ClosureAppendStatus::invalid_payload);
        EXPECT_EQ(object_bytes(set), original);
    }
}

TEST(ClosureSetTest, RejectsMalformedRoughConductorPayloadsAtomically) {
    expect_invalid_rough_conductor_payloads_are_atomic<TransportScalar>();
    expect_invalid_rough_conductor_payloads_are_atomic<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_overflow_is_atomic() {
    auto set = ClosureSetFor<Scalar>{};
    for (auto index = std::uint32_t{}; index < MaximumClosureCount; ++index) {
        ASSERT_EQ(set.append_lambertian_reflection(constant_spectrum(
                      static_cast<Scalar>(index) / static_cast<Scalar>(MaximumClosureCount))),
                  ClosureAppendStatus::appended);
    }
    const auto full_set = object_bytes(set);
    EXPECT_EQ(set.append_lambertian_reflection(constant_spectrum(Scalar{1})),
              ClosureAppendStatus::capacity_exhausted);
    EXPECT_EQ(object_bytes(set), full_set);
    EXPECT_EQ(set.size(), MaximumClosureCount);
}

TEST(ClosureSetTest, RejectsTheNinthClosureWithoutEvictionOrOverwrite) {
    expect_overflow_is_atomic<TransportScalar>();
    expect_overflow_is_atomic<ReferenceScalar>();
}

TEST(ClosureSetTest, KeepsBlackAndDuplicateClosuresInsteadOfMergingOrDroppingThem) {
    auto set = ClosureSet{};
    const auto black = constant_spectrum(TransportScalar{0});
    EXPECT_EQ(set.append_lambertian_reflection(black), ClosureAppendStatus::appended);
    EXPECT_EQ(set.append_lambertian_reflection(black), ClosureAppendStatus::appended);
    ASSERT_EQ(set.size(), 2U);
    EXPECT_EQ(set.closures()[0].weight, black);
    EXPECT_EQ(set.closures()[1].weight, black);
}

static_assert(noexcept(
    std::declval<ClosureSet&>().append_lambertian_reflection(std::declval<TransportSpectrum>())));
static_assert(noexcept(std::declval<ReferenceClosureSet&>().append_lambertian_reflection(
    std::declval<ReferenceSpectrum>())));
static_assert(noexcept(std::declval<ClosureSet&>().append_rough_diffuse_reflection(
    std::declval<TransportSpectrum>(), std::declval<TransportScalar>())));
static_assert(noexcept(std::declval<ReferenceClosureSet&>().append_rough_diffuse_reflection(
    std::declval<ReferenceSpectrum>(), std::declval<ReferenceScalar>())));
static_assert(noexcept(std::declval<ClosureSet&>().append_rough_conductor_reflection(
    std::declval<TransportSpectrum>(), std::declval<TransportSpectrum>(),
    std::declval<TransportSpectrum>(), std::declval<TransportScalar>())));
static_assert(noexcept(std::declval<ReferenceClosureSet&>().append_rough_conductor_reflection(
    std::declval<ReferenceSpectrum>(), std::declval<ReferenceSpectrum>(),
    std::declval<ReferenceSpectrum>(), std::declval<ReferenceScalar>())));
static_assert(noexcept(std::declval<const ClosureSet&>().closures()));
static_assert(noexcept(std::declval<const ReferenceClosureSet&>().closures()));

} // namespace
} // namespace blackframe::renderer
