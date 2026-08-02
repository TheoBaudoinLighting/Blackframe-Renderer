#include <Blackframe/Renderer/PathStateSoA.hpp>
#include <array>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace blackframe::renderer {
namespace {

static_assert(CurrentPathStateSoASchemaVersion == 1U);

[[nodiscard]] core::Error unsupported_schema(const std::uint32_t schema_version) {
    return core::Error{
        .code = core::StatusCode::incompatible,
        .message = "Unsupported path state SoA schema version " + std::to_string(schema_version) +
                   "; expected " + std::to_string(CurrentPathStateSoASchemaVersion) + ".",
    };
}

[[nodiscard]] core::Error exhausted_storage(const char* const message) {
    return core::Error{
        .code = core::StatusCode::resource_exhausted,
        .message = message,
    };
}

[[nodiscard]] core::Error invalid_index() {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = "Path state SoA index is outside the stored path domain.",
    };
}

[[nodiscard]] core::Error inconsistent_storage(std::string message) {
    return core::Error{
        .code = core::StatusCode::internal_error,
        .message = std::move(message),
    };
}

template <typename Value, std::size_t Count>
[[nodiscard]] std::array<std::span<const Value>, Count>
column_spans(const std::array<std::vector<Value>, Count>& columns) noexcept {
    auto result = std::array<std::span<const Value>, Count>{};
    for (auto index = std::size_t{}; index < Count; ++index) {
        result[index] = std::span<const Value>{columns[index]};
    }
    return result;
}

template <typename Value, std::size_t Count>
void resize_columns(std::array<std::vector<Value>, Count>& columns, const std::size_t path_count) {
    for (auto& column : columns) {
        column.resize(path_count);
    }
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool representable_path_count(const std::size_t count) {
    constexpr auto scalar_column_count = std::size_t{17};
    constexpr auto uint32_column_count = std::size_t{7};
    constexpr auto byte_column_count = std::size_t{5};
    constexpr auto bytes_per_path = scalar_column_count * sizeof(Scalar) +
                                    uint32_column_count * sizeof(std::uint32_t) +
                                    byte_column_count * sizeof(std::uint8_t);
    static_assert(bytes_per_path > 0U);
    if (count > std::numeric_limits<std::size_t>::max() / bytes_per_path) {
        return false;
    }
    return count <= std::vector<Scalar>{}.max_size() &&
           count <= std::vector<std::uint32_t>{}.max_size() &&
           count <= std::vector<ProbabilityMeasure>{}.max_size() &&
           count <= std::vector<PathDeltaFlags>{}.max_size();
}

} // namespace

template <SpectrumScalar Scalar>
PathStateSoAT<Scalar>::PathStateSoAT(PathStateSoAT&& other) noexcept {
    swap(other);
}

template <SpectrumScalar Scalar> void PathStateSoAT<Scalar>::swap(PathStateSoAT& other) noexcept {
    using std::swap;
    swap(schema_version_, other.schema_version_);
    swap(path_count_, other.path_count_);
    swap(beta_, other.beta_);
    swap(accumulated_radiance_, other.accumulated_radiance_);
    swap(depth_, other.depth_);
    swap(diffuse_depth_, other.diffuse_depth_);
    swap(glossy_depth_, other.glossy_depth_);
    swap(specular_depth_, other.specular_depth_);
    swap(transmission_depth_, other.transmission_depth_);
    swap(volume_depth_, other.volume_depth_);
    swap(eta_scale_, other.eta_scale_);
    swap(wavelength_nanometers_, other.wavelength_nanometers_);
    swap(wavelength_pdf_values_, other.wavelength_pdf_values_);
    swap(wavelength_pdf_measures_, other.wavelength_pdf_measures_);
    swap(delta_flags_, other.delta_flags_);
    swap(current_medium_values_, other.current_medium_values_);
}

template <SpectrumScalar Scalar>
core::Result<PathStateSoAT<Scalar>>
PathStateSoAT<Scalar>::from_aos(const std::span<const state_type> states,
                                const std::uint32_t schema_version) {
    if (schema_version != CurrentPathStateSoASchemaVersion) {
        return std::unexpected(unsupported_schema(schema_version));
    }
    if (!representable_path_count<Scalar>(states.size())) {
        return std::unexpected(
            exhausted_storage("Path state SoA storage size is not representable on this host."));
    }

    auto result = PathStateSoAT{};
    result.schema_version_ = schema_version;
    result.path_count_ = states.size();
    try {
        resize_columns(result.beta_, states.size());
        resize_columns(result.accumulated_radiance_, states.size());
        result.depth_.resize(states.size());
        result.diffuse_depth_.resize(states.size());
        result.glossy_depth_.resize(states.size());
        result.specular_depth_.resize(states.size());
        result.transmission_depth_.resize(states.size());
        result.volume_depth_.resize(states.size());
        result.eta_scale_.resize(states.size());
        resize_columns(result.wavelength_nanometers_, states.size());
        resize_columns(result.wavelength_pdf_values_, states.size());
        resize_columns(result.wavelength_pdf_measures_, states.size());
        result.delta_flags_.resize(states.size());
        result.current_medium_values_.resize(states.size());
    } catch (const std::bad_alloc&) {
        return std::unexpected(
            exhausted_storage("Path state SoA construction exhausted host memory."));
    } catch (const std::length_error&) {
        return std::unexpected(
            exhausted_storage("Path state SoA storage size exceeds host container limits."));
    }

    for (auto path = std::size_t{}; path < states.size(); ++path) {
        const auto& state = states[path];
        for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
            result.beta_[lane][path] = state.beta()[lane];
            result.accumulated_radiance_[lane][path] = state.accumulated_radiance()[lane];
            result.wavelength_nanometers_[lane][path] = state.wavelengths()[lane].nanometers;
            result.wavelength_pdf_values_[lane][path] = state.wavelengths()[lane].probability.value;
            result.wavelength_pdf_measures_[lane][path] =
                state.wavelengths()[lane].probability.measure;
        }
        const auto& counters = state.depth_counters();
        result.depth_[path] = state.depth();
        result.diffuse_depth_[path] = counters.diffuse;
        result.glossy_depth_[path] = counters.glossy;
        result.specular_depth_[path] = counters.specular;
        result.transmission_depth_[path] = counters.transmission;
        result.volume_depth_[path] = counters.volume;
        result.eta_scale_[path] = state.eta_scale();
        result.delta_flags_[path] = state.delta_flags();
        result.current_medium_values_[path] = state.current_medium().value;
    }
    return result;
}

template <SpectrumScalar Scalar>
typename PathStateSoAT<Scalar>::columns_type PathStateSoAT<Scalar>::columns() const& noexcept {
    return columns_type{
        .schema_version = schema_version_,
        .path_count = path_count_,
        .beta = column_spans(beta_),
        .accumulated_radiance = column_spans(accumulated_radiance_),
        .depth = depth_,
        .depth_counters =
            {
                .diffuse = diffuse_depth_,
                .glossy = glossy_depth_,
                .specular = specular_depth_,
                .transmission = transmission_depth_,
                .volume = volume_depth_,
            },
        .eta_scale = eta_scale_,
        .wavelength_nanometers = column_spans(wavelength_nanometers_),
        .wavelength_pdf_values = column_spans(wavelength_pdf_values_),
        .wavelength_pdf_measures = column_spans(wavelength_pdf_measures_),
        .delta_flags = delta_flags_,
        .current_medium_values = current_medium_values_,
    };
}

template <SpectrumScalar Scalar>
core::Result<typename PathStateSoAT<Scalar>::state_type>
PathStateSoAT<Scalar>::at(const std::size_t index) const {
    if (index >= path_count_) {
        return std::unexpected(invalid_index());
    }

    using spectrum_type = typename state_type::spectrum_type;
    using wavelengths_type = typename state_type::wavelengths_type;
    using wavelength_value_type = typename wavelengths_type::value_type;
    auto beta = spectrum_type{};
    auto radiance = spectrum_type{};
    auto wavelengths = wavelengths_type{};
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        beta[lane] = beta_[lane][index];
        radiance[lane] = accumulated_radiance_[lane][index];
        wavelengths[lane] = wavelength_value_type{
            .nanometers = wavelength_nanometers_[lane][index],
            .probability =
                {
                    .value = wavelength_pdf_values_[lane][index],
                    .measure = wavelength_pdf_measures_[lane][index],
                },
        };
    }
    const auto counters = PathDepthCounters{
        .diffuse = diffuse_depth_[index],
        .glossy = glossy_depth_[index],
        .specular = specular_depth_[index],
        .transmission = transmission_depth_[index],
        .volume = volume_depth_[index],
    };
    auto state =
        state_type::create(beta, radiance, counters, eta_scale_[index], wavelengths,
                           delta_flags_[index], MediumId{.value = current_medium_values_[index]});
    if (!state) {
        return std::unexpected(inconsistent_storage(
            "Path state SoA reconstruction failed validation: " + state.error().message));
    }
    if (state->depth() != depth_[index]) {
        return std::unexpected(inconsistent_storage(
            "Path state SoA depth column is inconsistent with its category counters."));
    }
    return state;
}

template <SpectrumScalar Scalar>
core::Result<std::vector<typename PathStateSoAT<Scalar>::state_type>>
PathStateSoAT<Scalar>::to_aos() const {
    auto result = std::vector<state_type>{};
    if (path_count_ > result.max_size()) {
        return std::unexpected(
            exhausted_storage("Path state AoS reconstruction exceeds host container limits."));
    }
    try {
        result.reserve(path_count_);
    } catch (const std::bad_alloc&) {
        return std::unexpected(
            exhausted_storage("Path state AoS reconstruction exhausted host memory."));
    } catch (const std::length_error&) {
        return std::unexpected(
            exhausted_storage("Path state AoS reconstruction exceeds host container limits."));
    }

    for (auto index = std::size_t{}; index < path_count_; ++index) {
        auto state = at(index);
        if (!state) {
            return std::unexpected(state.error());
        }
        result.push_back(std::move(*state));
    }
    return result;
}

template class PathStateSoAT<TransportScalar>;
template class PathStateSoAT<ReferenceScalar>;

} // namespace blackframe::renderer
