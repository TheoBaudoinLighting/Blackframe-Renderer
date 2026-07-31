#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/Light.hpp>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace blackframe::renderer {

enum class LightSamplingStrategy : std::uint8_t {
    uniform,
    power_weighted,
};

template <SpectrumScalar Scalar> class LightSamplerT;

// The selection probability is discrete and remains separate from the selected light's
// conditional sample_li/pdf_li probability. A later joint estimator must multiply both terms.
template <SpectrumScalar Scalar> class LightSelectionProbabilityT final {
  public:
    [[nodiscard]] constexpr Scalar value() const noexcept {
        return value_;
    }

    [[nodiscard]] static constexpr ProbabilityMeasure measure() noexcept {
        return ProbabilityMeasure::discrete;
    }

    [[nodiscard]] constexpr LightProbabilityDensityT<Scalar> probability_density() const noexcept {
        return {
            .value = value_,
            .measure = ProbabilityMeasure::discrete,
        };
    }

    [[nodiscard]] constexpr bool
    operator==(const LightSelectionProbabilityT&) const noexcept = default;

  private:
    friend class LightSamplerT<Scalar>;

    constexpr explicit LightSelectionProbabilityT(const Scalar value) noexcept : value_{value} {}

    Scalar value_;
};

using LightSelectionProbability = LightSelectionProbabilityT<TransportScalar>;
using ReferenceLightSelectionProbability = LightSelectionProbabilityT<ReferenceScalar>;

// A slot is an index into an immutable heterogeneous light registry owned by the caller. It is not
// a scene identifier and is never narrowed from the registry size without validation.
template <SpectrumScalar Scalar> class LightSelectionT final {
  public:
    [[nodiscard]] constexpr std::uint32_t light_index() const noexcept {
        return light_index_;
    }

    [[nodiscard]] constexpr const LightSelectionProbabilityT<Scalar>& probability() const noexcept {
        return probability_;
    }

    [[nodiscard]] constexpr bool operator==(const LightSelectionT&) const noexcept = default;

  private:
    friend class LightSamplerT<Scalar>;

    constexpr LightSelectionT(const std::uint32_t light_index,
                              const LightSelectionProbabilityT<Scalar> probability) noexcept
        : light_index_{light_index}, probability_{probability} {}

    std::uint32_t light_index_;
    LightSelectionProbabilityT<Scalar> probability_;
};

using LightSelection = LightSelectionT<TransportScalar>;
using ReferenceLightSelection = LightSelectionT<ReferenceScalar>;

// LightSampler is immutable after construction and consumes only the caller-provided canonical
// light-selection dimension. The separate light_u/light_v dimensions remain available to the
// selected light model. Power-weighted construction consumes already evaluated four-lane spectral
// fluxes in stable registry order; it performs no hidden spectral conversion or light evaluation.
template <SpectrumScalar Scalar> class LightSamplerT final {
  public:
    LightSamplerT(const LightSamplerT&) = default;
    LightSamplerT(LightSamplerT&& other) noexcept;
    LightSamplerT& operator=(const LightSamplerT&) = delete;
    LightSamplerT& operator=(LightSamplerT&&) = delete;

    [[nodiscard]] static core::Result<LightSamplerT> create_uniform(std::size_t light_count);

    [[nodiscard]] static core::Result<LightSamplerT>
    create_power_weighted(std::span<const LightSpectrumT<Scalar>> spectral_powers);

    [[nodiscard]] core::Result<LightSelectionT<Scalar>> sample(Scalar canonical_sample) const;

    [[nodiscard]] core::Result<LightSelectionProbabilityT<Scalar>>
    probability(std::uint32_t light_index) const;

    [[nodiscard]] constexpr LightSamplingStrategy strategy() const noexcept {
        return strategy_;
    }

    [[nodiscard]] std::uint32_t light_count() const noexcept;

  private:
    LightSamplerT(LightSamplingStrategy strategy, std::vector<Scalar> cdf_boundaries) noexcept;

    LightSamplingStrategy strategy_;
    std::vector<Scalar> cdf_boundaries_;
};

using LightSampler = LightSamplerT<TransportScalar>;
using ReferenceLightSampler = LightSamplerT<ReferenceScalar>;

extern template class LightSamplerT<TransportScalar>;
extern template class LightSamplerT<ReferenceScalar>;

static_assert(sizeof(LightSamplingStrategy) == sizeof(std::uint8_t));
static_assert(!std::is_same_v<LightSampler, ReferenceLightSampler>);
static_assert(!std::is_default_constructible_v<LightSampler>);
static_assert(!std::is_default_constructible_v<ReferenceLightSampler>);
static_assert(std::is_standard_layout_v<LightSelectionProbability>);
static_assert(std::is_trivially_copyable_v<LightSelectionProbability>);
static_assert(std::is_standard_layout_v<ReferenceLightSelectionProbability>);
static_assert(std::is_trivially_copyable_v<ReferenceLightSelectionProbability>);
static_assert(std::is_standard_layout_v<LightSelection>);
static_assert(std::is_trivially_copyable_v<LightSelection>);
static_assert(std::is_standard_layout_v<ReferenceLightSelection>);
static_assert(std::is_trivially_copyable_v<ReferenceLightSelection>);

} // namespace blackframe::renderer
