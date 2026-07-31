#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/Light.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace blackframe::renderer {

enum class LightSamplingStrategy : std::uint8_t {
    uniform,
    power_weighted,
    spatial_tree,
};

// LightTreeInput is a construction record for the two model properties required by the spatial
// selection strategy. The sampler copies it, and its position in the input span is the stable
// heterogeneous-registry slot.
template <SpectrumScalar Scalar> struct LightTreeInputT final {
    Bounds3T<Scalar> bounds;
    LightSpectrumT<Scalar> spectral_power;
};

using LightTreeInput = LightTreeInputT<TransportScalar>;
using ReferenceLightTreeInput = LightTreeInputT<ReferenceScalar>;

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
// Spatial construction additionally copies conservative bounds into a deterministic median tree.
// At a query point, a finite node's scalar power is weighted by 1 / (1 + (d / s)^2), where d is
// distance to its AABB and s is the finite root's largest extent. Degenerate finite roots and
// canonically unbounded nodes retain unit spatial weight. This is a discrete variance heuristic,
// not a physical directional PDF; sample_li/pdf_li remain conditional on the selected model.
template <SpectrumScalar Scalar> class LightSamplerT final {
  public:
    LightSamplerT(const LightSamplerT&) = default;
    LightSamplerT(LightSamplerT&& other) noexcept;
    LightSamplerT& operator=(const LightSamplerT&) = delete;
    LightSamplerT& operator=(LightSamplerT&&) = delete;

    [[nodiscard]] static core::Result<LightSamplerT> create_uniform(std::size_t light_count);

    [[nodiscard]] static core::Result<LightSamplerT>
    create_power_weighted(std::span<const LightSpectrumT<Scalar>> spectral_powers);

    [[nodiscard]] static core::Result<LightSamplerT>
    create_spatial_tree(std::span<const LightTreeInputT<Scalar>> lights);

    [[nodiscard]] core::Result<LightSelectionT<Scalar>> sample(Scalar canonical_sample) const;

    [[nodiscard]] core::Result<LightSelectionT<Scalar>>
    sample(const LightSampleContextT<Scalar>& context, Scalar canonical_sample) const;

    [[nodiscard]] core::Result<LightSelectionProbabilityT<Scalar>>
    probability(std::uint32_t light_index) const;

    [[nodiscard]] core::Result<LightSelectionProbabilityT<Scalar>>
    probability(const LightSampleContextT<Scalar>& context, std::uint32_t light_index) const;

    [[nodiscard]] constexpr LightSamplingStrategy strategy() const noexcept {
        return strategy_;
    }

    [[nodiscard]] std::uint32_t light_count() const noexcept;

    [[nodiscard]] std::uint32_t tree_node_count() const noexcept;
    [[nodiscard]] constexpr std::uint32_t maximum_tree_depth() const noexcept {
        return maximum_tree_depth_;
    }
    [[nodiscard]] constexpr std::uint32_t finite_light_count() const noexcept {
        return finite_light_count_;
    }
    [[nodiscard]] constexpr std::uint32_t unbounded_light_count() const noexcept {
        return unbounded_light_count_;
    }

  private:
    enum class TreeNodeKind : std::uint8_t {
        internal,
        finite_leaf,
        unbounded_leaf,
    };

    struct TreeNode final {
        Bounds3T<Scalar> bounds;
        Scalar power_weight;
        std::uint32_t parent;
        std::uint32_t left_child;
        std::uint32_t right_child;
        std::uint32_t light_index;
        TreeNodeKind kind;
    };

    static constexpr auto InvalidTreeIndex = std::numeric_limits<std::uint32_t>::max();

    LightSamplerT(LightSamplingStrategy strategy, std::vector<Scalar> cdf_boundaries) noexcept;

    LightSamplerT(std::vector<Scalar> cdf_boundaries, std::vector<TreeNode> tree_nodes,
                  std::vector<std::uint32_t> light_to_node, Scalar finite_root_log_extent,
                  bool finite_root_has_extent, std::uint32_t tree_root,
                  std::uint32_t maximum_tree_depth, std::uint32_t finite_light_count,
                  std::uint32_t unbounded_light_count) noexcept;

    [[nodiscard]] core::Result<Scalar>
    tree_left_probability(const Point3T<Scalar>& reference_position,
                          std::uint32_t node_index) const;

    LightSamplingStrategy strategy_;
    std::vector<Scalar> cdf_boundaries_;
    std::vector<TreeNode> tree_nodes_;
    std::vector<std::uint32_t> light_to_node_;
    Scalar finite_root_log_extent_{};
    bool finite_root_has_extent_{};
    std::uint32_t tree_root_{InvalidTreeIndex};
    std::uint32_t maximum_tree_depth_{};
    std::uint32_t finite_light_count_{};
    std::uint32_t unbounded_light_count_{};
};

using LightSampler = LightSamplerT<TransportScalar>;
using ReferenceLightSampler = LightSamplerT<ReferenceScalar>;

extern template class LightSamplerT<TransportScalar>;
extern template class LightSamplerT<ReferenceScalar>;

static_assert(sizeof(LightSamplingStrategy) == sizeof(std::uint8_t));
static_assert(!std::is_same_v<LightSampler, ReferenceLightSampler>);
static_assert(!std::is_default_constructible_v<LightSampler>);
static_assert(!std::is_default_constructible_v<ReferenceLightSampler>);
static_assert(!std::is_same_v<LightTreeInput, ReferenceLightTreeInput>);
static_assert(std::is_standard_layout_v<LightSelectionProbability>);
static_assert(std::is_trivially_copyable_v<LightSelectionProbability>);
static_assert(std::is_standard_layout_v<ReferenceLightSelectionProbability>);
static_assert(std::is_trivially_copyable_v<ReferenceLightSelectionProbability>);
static_assert(std::is_standard_layout_v<LightSelection>);
static_assert(std::is_trivially_copyable_v<LightSelection>);
static_assert(std::is_standard_layout_v<ReferenceLightSelection>);
static_assert(std::is_trivially_copyable_v<ReferenceLightSelection>);

} // namespace blackframe::renderer
