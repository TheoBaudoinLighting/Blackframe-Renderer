#include <Blackframe/Renderer/SampleDimensionMap.hpp>
#include <cstddef>
#include <cstdint>
#include <string>

namespace blackframe::renderer {
namespace {

static_assert(CurrentSampleDimensionMapSchemaVersion == 1);
static_assert(FirstBounceSampleDimension == 4);
static_assert(SampleDimensionsPerBounce == 10);

constexpr auto bounce_dimension_offsets = BounceSampleDimensions{
    .light_selection = 0,
    .light_u = 1,
    .light_v = 2,
    .bsdf_component = 3,
    .bsdf_u = 4,
    .bsdf_v = 5,
    .medium_distance = 6,
    .medium_phase_u = 7,
    .medium_phase_v = 8,
    .russian_roulette = 9,
};

static_assert(bounce_dimension_offsets.light_selection == 0);
static_assert(bounce_dimension_offsets.russian_roulette + 1 == SampleDimensionsPerBounce);
constexpr auto last_mapped_bounce_dimension = FirstBounceSampleDimension +
                                              MaximumMappedBounceIndex * SampleDimensionsPerBounce +
                                              bounce_dimension_offsets.russian_roulette;
static_assert(last_mapped_bounce_dimension < PrimarySampleDimensionMap.camera_raster_x);
static_assert(last_mapped_bounce_dimension < PrimarySampleDimensionMap.camera_raster_y);

void append_hex(std::string& output, const std::uint64_t value) {
    constexpr auto digits = "0123456789abcdef";
    output += "0x";
    for (auto index = sizeof(value) * 2; index > 0; --index) {
        const auto shift = (index - 1) * 4;
        const auto nibble = static_cast<std::size_t>((value >> shift) & 0xFULL);
        output += digits[nibble];
    }
}

} // namespace

core::Result<BounceSampleDimensions>
sample_dimensions_for_bounce(const std::uint64_t bounce_index) {
    if (bounce_index > MaximumMappedBounceIndex) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::resource_exhausted,
            .message = "Sample dimension map supports bounce indices up to " +
                       std::to_string(MaximumMappedBounceIndex) + ".",
        });
    }

    const auto first = FirstBounceSampleDimension + bounce_index * SampleDimensionsPerBounce;
    return BounceSampleDimensions{
        .light_selection = first + bounce_dimension_offsets.light_selection,
        .light_u = first + bounce_dimension_offsets.light_u,
        .light_v = first + bounce_dimension_offsets.light_v,
        .bsdf_component = first + bounce_dimension_offsets.bsdf_component,
        .bsdf_u = first + bounce_dimension_offsets.bsdf_u,
        .bsdf_v = first + bounce_dimension_offsets.bsdf_v,
        .medium_distance = first + bounce_dimension_offsets.medium_distance,
        .medium_phase_u = first + bounce_dimension_offsets.medium_phase_u,
        .medium_phase_v = first + bounce_dimension_offsets.medium_phase_v,
        .russian_roulette = first + bounce_dimension_offsets.russian_roulette,
    };
}

core::Result<std::string> dump_sample_dimension_map(const std::uint32_t schema_version) {
    if (schema_version != CurrentSampleDimensionMapSchemaVersion) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::incompatible,
            .message = "Unsupported sample dimension map schema version " +
                       std::to_string(schema_version) + "; expected " +
                       std::to_string(CurrentSampleDimensionMapSchemaVersion) + ".",
        });
    }

    auto output = std::string{};
    output.reserve(640);
    output += R"({"schema_version":)";
    output += std::to_string(CurrentSampleDimensionMapSchemaVersion);
    output += R"(,"primary":{"camera":{"raster_x":")";
    append_hex(output, PrimarySampleDimensionMap.camera_raster_x);
    output += R"(","raster_y":")";
    append_hex(output, PrimarySampleDimensionMap.camera_raster_y);
    output += R"("},"lens":{"u":")";
    append_hex(output, PrimarySampleDimensionMap.lens_u);
    output += R"(","v":")";
    append_hex(output, PrimarySampleDimensionMap.lens_v);
    output += R"("},"time":")";
    append_hex(output, PrimarySampleDimensionMap.time);
    output += R"(","wavelength":")";
    append_hex(output, PrimarySampleDimensionMap.wavelength);
    output += R"("},"bounce":{"first":")";
    append_hex(output, FirstBounceSampleDimension);
    output += R"(","stride":)";
    output += std::to_string(SampleDimensionsPerBounce);
    output += R"(,"maximum_index":)";
    output += std::to_string(MaximumMappedBounceIndex);
    output += R"(,"light":{"selection":)";
    output += std::to_string(bounce_dimension_offsets.light_selection);
    output += R"(,"u":)";
    output += std::to_string(bounce_dimension_offsets.light_u);
    output += R"(,"v":)";
    output += std::to_string(bounce_dimension_offsets.light_v);
    output += R"(},"bsdf":{"component":)";
    output += std::to_string(bounce_dimension_offsets.bsdf_component);
    output += R"(,"u":)";
    output += std::to_string(bounce_dimension_offsets.bsdf_u);
    output += R"(,"v":)";
    output += std::to_string(bounce_dimension_offsets.bsdf_v);
    output += R"(},"medium":{"distance":)";
    output += std::to_string(bounce_dimension_offsets.medium_distance);
    output += R"(,"phase_u":)";
    output += std::to_string(bounce_dimension_offsets.medium_phase_u);
    output += R"(,"phase_v":)";
    output += std::to_string(bounce_dimension_offsets.medium_phase_v);
    output += R"(},"russian_roulette":)";
    output += std::to_string(bounce_dimension_offsets.russian_roulette);
    output += "}}";
    return output;
}

} // namespace blackframe::renderer
