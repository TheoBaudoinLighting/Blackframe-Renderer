#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/PathState.hpp>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace blackframe::renderer {

inline constexpr std::uint32_t CurrentPathStateSoASchemaVersion = 1U;

struct PathDepthCountersSoAColumns final {
    std::span<const std::uint32_t> diffuse;
    std::span<const std::uint32_t> glossy;
    std::span<const std::uint32_t> specular;
    std::span<const std::uint32_t> transmission;
    std::span<const std::uint32_t> volume;
};

// Version one stores every PathState field in a path-indexed column. Four-lane quantities use one
// column per spectral lane, while wavelength PDF measures remain explicit instead of being inferred
// during reconstruction. These spans are non-owning host views that expire with their PathStateSoA
// owner; they are not a wire format or a host/device ABI.
template <SpectrumScalar Scalar> struct PathStateSoAColumnsT final {
    std::uint32_t schema_version{};
    std::size_t path_count{};
    std::array<std::span<const Scalar>, TransportSpectrumSampleCount> beta;
    std::array<std::span<const Scalar>, TransportSpectrumSampleCount> accumulated_radiance;
    std::span<const std::uint32_t> depth;
    PathDepthCountersSoAColumns depth_counters;
    std::span<const Scalar> eta_scale;
    std::array<std::span<const Scalar>, TransportSpectrumSampleCount> wavelength_nanometers;
    std::array<std::span<const Scalar>, TransportSpectrumSampleCount> wavelength_pdf_values;
    std::array<std::span<const ProbabilityMeasure>, TransportSpectrumSampleCount>
        wavelength_pdf_measures;
    std::span<const PathDeltaFlags> delta_flags;
    std::span<const std::uint32_t> current_medium_values;
};

using PathStateSoAColumns = PathStateSoAColumnsT<TransportScalar>;
using ReferencePathStateSoAColumns = PathStateSoAColumnsT<ReferenceScalar>;

// PathStateSoA owns a schema-versioned structure of arrays. Construction always names the schema
// explicitly and copies already validated AoS states. Views returned by an owner cannot mutate its
// columns independently, so unequal column lengths and partially initialized owned paths are
// unrepresentable. at() reconstructs through PathState::create and also checks the stored depth
// cache; it never repairs invalid data.
template <SpectrumScalar Scalar> class PathStateSoAT final {
  public:
    using state_type = PathStateT<Scalar>;
    using columns_type = PathStateSoAColumnsT<Scalar>;

    PathStateSoAT(const PathStateSoAT&) = delete;
    PathStateSoAT(PathStateSoAT&& other) noexcept;
    PathStateSoAT& operator=(const PathStateSoAT&) = delete;
    PathStateSoAT& operator=(PathStateSoAT&&) = delete;
    ~PathStateSoAT() = default;

    [[nodiscard]] static core::Result<PathStateSoAT> from_aos(std::span<const state_type> states,
                                                              std::uint32_t schema_version);

    [[nodiscard]] constexpr std::uint32_t schema_version() const noexcept {
        return schema_version_;
    }

    [[nodiscard]] constexpr std::size_t size() const noexcept {
        return path_count_;
    }

    [[nodiscard]] constexpr bool empty() const noexcept {
        return path_count_ == 0U;
    }

    [[nodiscard]] columns_type columns() const& noexcept;
    [[nodiscard]] columns_type columns() && = delete;
    [[nodiscard]] columns_type columns() const&& = delete;
    [[nodiscard]] core::Result<state_type> at(std::size_t index) const;
    [[nodiscard]] core::Result<std::vector<state_type>> to_aos() const;

  private:
    PathStateSoAT() = default;
    void swap(PathStateSoAT& other) noexcept;

    std::uint32_t schema_version_{CurrentPathStateSoASchemaVersion};
    std::size_t path_count_{};
    std::array<std::vector<Scalar>, TransportSpectrumSampleCount> beta_;
    std::array<std::vector<Scalar>, TransportSpectrumSampleCount> accumulated_radiance_;
    std::vector<std::uint32_t> depth_;
    std::vector<std::uint32_t> diffuse_depth_;
    std::vector<std::uint32_t> glossy_depth_;
    std::vector<std::uint32_t> specular_depth_;
    std::vector<std::uint32_t> transmission_depth_;
    std::vector<std::uint32_t> volume_depth_;
    std::vector<Scalar> eta_scale_;
    std::array<std::vector<Scalar>, TransportSpectrumSampleCount> wavelength_nanometers_;
    std::array<std::vector<Scalar>, TransportSpectrumSampleCount> wavelength_pdf_values_;
    std::array<std::vector<ProbabilityMeasure>, TransportSpectrumSampleCount>
        wavelength_pdf_measures_;
    std::vector<PathDeltaFlags> delta_flags_;
    std::vector<std::uint32_t> current_medium_values_;
};

using PathStateSoA = PathStateSoAT<TransportScalar>;
using ReferencePathStateSoA = PathStateSoAT<ReferenceScalar>;

extern template class PathStateSoAT<TransportScalar>;
extern template class PathStateSoAT<ReferenceScalar>;

static_assert(TransportSpectrumSampleCount == 4U);
static_assert(!std::same_as<PathStateSoA, ReferencePathStateSoA>);
static_assert(!std::is_default_constructible_v<PathStateSoA>);
static_assert(!std::is_default_constructible_v<ReferencePathStateSoA>);
static_assert(std::is_move_constructible_v<PathStateSoA>);
static_assert(std::is_move_constructible_v<ReferencePathStateSoA>);
static_assert(std::is_nothrow_move_constructible_v<PathStateSoA>);
static_assert(std::is_nothrow_move_constructible_v<ReferencePathStateSoA>);
static_assert(!std::is_copy_constructible_v<PathStateSoA>);
static_assert(!std::is_copy_constructible_v<ReferencePathStateSoA>);

} // namespace blackframe::renderer
