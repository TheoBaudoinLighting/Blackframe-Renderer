#include <Blackframe/Renderer/PathStateDiagnostics.hpp>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

namespace blackframe::renderer {
namespace {

static_assert(CurrentPathStateDiagnosticSchemaVersion == 2);

template <typename Unsigned> void append_hex(std::string& output, const Unsigned value) {
    static_assert(std::is_unsigned_v<Unsigned>);
    constexpr auto digits = "0123456789abcdef";
    output += "0x";
    for (auto index = sizeof(Unsigned) * 2; index > 0; --index) {
        const auto shift = (index - 1) * 4;
        const auto nibble = static_cast<std::size_t>((value >> shift) & Unsigned{0xFU});
        output += digits[nibble];
    }
}

template <SpectrumScalar Scalar> void append_scalar(std::string& output, const Scalar value) {
    if constexpr (std::same_as<Scalar, TransportScalar>) {
        append_hex(output, std::bit_cast<std::uint32_t>(value));
    } else {
        append_hex(output, std::bit_cast<std::uint64_t>(value));
    }
}

template <SpectrumScalar Scalar>
void append_spectrum(std::string& output,
                     const SampledSpectrum<TransportSpectrumSampleCount, Scalar>& spectrum) {
    output += "[";
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        if (lane != 0) {
            output += ",";
        }
        output += "\"";
        append_scalar(output, spectrum[lane]);
        output += "\"";
    }
    output += "]";
}

template <SpectrumScalar Scalar>
void append_wavelengths(std::string& output, const SampledWavelengthsT<Scalar>& wavelengths) {
    output += "[";
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        if (lane != 0) {
            output += ",";
        }
        output += R"({"nanometers_bits":")";
        append_scalar(output, wavelengths[lane].nanometers);
        output += R"(","pdf_bits":")";
        append_scalar(output, wavelengths[lane].probability.value);
        output += R"(","measure":"wavelength"})";
    }
    output += "]";
}

void append_depth_counters(std::string& output, const PathDepthCounters& counters) {
    output += R"({"diffuse":)";
    output += std::to_string(counters.diffuse);
    output += R"(,"glossy":)";
    output += std::to_string(counters.glossy);
    output += R"(,"specular":)";
    output += std::to_string(counters.specular);
    output += R"(,"transmission":)";
    output += std::to_string(counters.transmission);
    output += R"(,"volume":)";
    output += std::to_string(counters.volume);
    output += "}";
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<std::string> serialize(const PathStateT<Scalar>& state,
                                                  const std::uint32_t schema_version) {
    if (schema_version != CurrentPathStateDiagnosticSchemaVersion) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::incompatible,
            .message = "Unsupported path state diagnostic schema version " +
                       std::to_string(schema_version) + "; expected " +
                       std::to_string(CurrentPathStateDiagnosticSchemaVersion) + ".",
        });
    }

    auto output = std::string{};
    output.reserve(std::same_as<Scalar, TransportScalar> ? 800U : 1100U);
    output += R"({"schema_version":2,"precision":")";
    output += std::same_as<Scalar, TransportScalar> ? "float32" : "float64";
    output += R"(","throughput_bits":)";
    append_spectrum(output, state.beta());
    output += R"(,"accumulated_radiance_bits":)";
    append_spectrum(output, state.accumulated_radiance());
    output += R"(,"depth":)";
    output += std::to_string(state.depth());
    output += R"(,"depth_counters":)";
    append_depth_counters(output, state.depth_counters());
    output += R"(,"eta_scale_bits":")";
    append_scalar(output, state.eta_scale());
    output += R"(","wavelengths":)";
    append_wavelengths(output, state.wavelengths());
    output += R"(,"delta_flags":")";
    append_hex(output, static_cast<std::uint8_t>(state.delta_flags()));
    output += R"(","current_medium":")";
    append_hex(output, state.current_medium().value);
    output += R"("})";
    return output;
}

} // namespace

core::Result<std::string> serialize_path_state_diagnostic(const PathState& state,
                                                          const std::uint32_t schema_version) {
    return serialize(state, schema_version);
}

core::Result<std::string> serialize_path_state_diagnostic(const ReferencePathState& state,
                                                          const std::uint32_t schema_version) {
    return serialize(state, schema_version);
}

} // namespace blackframe::renderer
