#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/Spectrum.hpp>
#include <Blackframe/Renderer/TransportConventions.hpp>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace blackframe::renderer {

enum class RussianRouletteMode : std::uint8_t {
    disabled,
    enabled,
};

enum class RussianRouletteOutcome : std::uint8_t {
    not_evaluated,
    survived,
    terminated,
};

template <SpectrumScalar Scalar>
using RussianRouletteProbabilityDensityT =
    std::conditional_t<std::same_as<Scalar, TransportScalar>, ProbabilityDensity,
                       ReferenceProbabilityDensity>;

// first_eligible_depth is the first completed scattering depth at which roulette is evaluated.
// The explicit probability bounds are part of the estimator policy, not repair fallbacks.
template <SpectrumScalar Scalar> class RussianRoulettePolicyT final {
  public:
    [[nodiscard]] static constexpr RussianRoulettePolicyT disabled() noexcept {
        return RussianRoulettePolicyT{
            RussianRouletteMode::disabled,
            0,
            Scalar{0},
            Scalar{0},
        };
    }

    [[nodiscard]] static core::Result<RussianRoulettePolicyT>
    create_enabled(const std::uint32_t first_eligible_depth,
                   const Scalar minimum_survival_probability,
                   const Scalar maximum_survival_probability) {
        if (first_eligible_depth == 0) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::invalid_argument,
                .message = "Russian roulette requires a strictly positive first eligible depth.",
            });
        }
        if (!std::isfinite(minimum_survival_probability) ||
            !std::isfinite(maximum_survival_probability) ||
            !(minimum_survival_probability > Scalar{0}) ||
            !(minimum_survival_probability < Scalar{1}) ||
            maximum_survival_probability < minimum_survival_probability ||
            maximum_survival_probability > Scalar{1}) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::invalid_argument,
                .message = "Russian roulette survival bounds must be finite and satisfy "
                           "0 < minimum < 1 and minimum <= maximum <= 1.",
            });
        }
        return RussianRoulettePolicyT{
            RussianRouletteMode::enabled,
            first_eligible_depth,
            minimum_survival_probability,
            maximum_survival_probability,
        };
    }

    [[nodiscard]] constexpr RussianRouletteMode mode() const noexcept {
        return mode_;
    }

    [[nodiscard]] constexpr bool is_enabled() const noexcept {
        return mode_ == RussianRouletteMode::enabled;
    }

    [[nodiscard]] constexpr std::uint32_t first_eligible_depth() const noexcept {
        return first_eligible_depth_;
    }

    [[nodiscard]] constexpr Scalar minimum_survival_probability() const noexcept {
        return minimum_survival_probability_;
    }

    [[nodiscard]] constexpr Scalar maximum_survival_probability() const noexcept {
        return maximum_survival_probability_;
    }

    [[nodiscard]] constexpr bool operator==(const RussianRoulettePolicyT&) const noexcept = default;

  private:
    constexpr RussianRoulettePolicyT(const RussianRouletteMode mode,
                                     const std::uint32_t first_eligible_depth,
                                     const Scalar minimum_survival_probability,
                                     const Scalar maximum_survival_probability) noexcept
        : mode_{mode}, first_eligible_depth_{first_eligible_depth},
          minimum_survival_probability_{minimum_survival_probability},
          maximum_survival_probability_{maximum_survival_probability} {}

    RussianRouletteMode mode_;
    std::uint32_t first_eligible_depth_;
    Scalar minimum_survival_probability_;
    Scalar maximum_survival_probability_;
};

using RussianRoulettePolicy = RussianRoulettePolicyT<TransportScalar>;
using ReferenceRussianRoulettePolicy = RussianRoulettePolicyT<ReferenceScalar>;

// A terminated decision retains the unscaled input throughput for terminal path diagnostics.
// Only a surviving decision returns throughput divided by its discrete survival probability.
template <SpectrumScalar Scalar> struct RussianRouletteResultT final {
    SampledSpectrum<TransportSpectrumSampleCount, Scalar> throughput{};
    RussianRouletteProbabilityDensityT<Scalar> survival_probability{
        .value = Scalar{1},
        .measure = ProbabilityMeasure::discrete,
    };
    RussianRouletteOutcome outcome{RussianRouletteOutcome::not_evaluated};
};

using RussianRouletteResult = RussianRouletteResultT<TransportScalar>;
using ReferenceRussianRouletteResult = RussianRouletteResultT<ReferenceScalar>;

[[nodiscard]] core::Status validate_russian_roulette_policy(const RussianRoulettePolicy& policy);

[[nodiscard]] core::Status
validate_russian_roulette_policy(const ReferenceRussianRoulettePolicy& policy);

[[nodiscard]] core::Result<RussianRouletteResult>
evaluate_russian_roulette(const TransportSpectrum& throughput, TransportScalar eta_scale,
                          std::uint32_t completed_depth, TransportScalar canonical_sample,
                          const RussianRoulettePolicy& policy);

[[nodiscard]] core::Result<ReferenceRussianRouletteResult>
evaluate_russian_roulette(const ReferenceSpectrum& throughput, ReferenceScalar eta_scale,
                          std::uint32_t completed_depth, ReferenceScalar canonical_sample,
                          const ReferenceRussianRoulettePolicy& policy);

static_assert(sizeof(RussianRouletteMode) == sizeof(std::uint8_t));
static_assert(sizeof(RussianRouletteOutcome) == sizeof(std::uint8_t));
static_assert(!std::is_default_constructible_v<RussianRoulettePolicy>);
static_assert(!std::is_default_constructible_v<ReferenceRussianRoulettePolicy>);
static_assert(std::is_standard_layout_v<RussianRoulettePolicy>);
static_assert(std::is_trivially_copyable_v<RussianRoulettePolicy>);
static_assert(std::is_standard_layout_v<ReferenceRussianRoulettePolicy>);
static_assert(std::is_trivially_copyable_v<ReferenceRussianRoulettePolicy>);
static_assert(std::is_standard_layout_v<RussianRouletteResult>);
static_assert(std::is_trivially_copyable_v<RussianRouletteResult>);
static_assert(std::is_standard_layout_v<ReferenceRussianRouletteResult>);
static_assert(std::is_trivially_copyable_v<ReferenceRussianRouletteResult>);

} // namespace blackframe::renderer
