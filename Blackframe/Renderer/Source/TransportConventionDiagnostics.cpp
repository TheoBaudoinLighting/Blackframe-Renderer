#include <Blackframe/Renderer/TransportConventionDiagnostics.hpp>
#include <Blackframe/Renderer/TransportConventions.hpp>
#include <string>
#include <string_view>

namespace blackframe::renderer {
namespace {

static_assert(CurrentBsdfConventionSchemaVersion == 1U);
static_assert(static_cast<std::uint8_t>(ProbabilityMeasure::discrete) == 0U);
static_assert(static_cast<std::uint8_t>(ProbabilityMeasure::solid_angle) == 1U);
static_assert(static_cast<std::uint8_t>(ProbabilityMeasure::area) == 2U);
static_assert(static_cast<std::uint8_t>(ProbabilityMeasure::distance) == 3U);
static_assert(static_cast<std::uint8_t>(ProbabilityMeasure::volume) == 4U);
static_assert(static_cast<std::uint8_t>(ProbabilityMeasure::wavelength) == 5U);
static_assert(scattering_lobe_bits(ScatteringLobe::none) == 0x00000000U);
static_assert(scattering_lobe_bits(ScatteringLobe::diffuse) == 0x00000001U);
static_assert(scattering_lobe_bits(ScatteringLobe::glossy) == 0x00000002U);
static_assert(scattering_lobe_bits(ScatteringLobe::specular) == 0x00000004U);
static_assert(scattering_lobe_bits(ScatteringLobe::reflection) == 0x00000008U);
static_assert(scattering_lobe_bits(ScatteringLobe::transmission) == 0x00000010U);
static_assert(scattering_lobe_bits(ScatteringLobe::volume) == 0x00000020U);
static_assert(scattering_lobe_bits(KnownScatteringLobeMask) == 0x0000003fU);
static_assert(static_cast<std::uint8_t>(TransportMode::radiance) == 0U);
static_assert(static_cast<std::uint8_t>(TransportMode::importance) == 1U);

constexpr auto BsdfConventionDocument = std::string_view{
    R"json({"schema_version":1,"pdf":{"measures":{"discrete":0,"solid_angle":1,"area":2,"distance":3,"volume":4,"wavelength":5},"bsdf":{"conditional":"p(wi|wo)","reverse":"swap_wo_wi","continuous":"solid_angle","delta":"discrete","directional_query_excludes_delta":true,"projected_solid_angle":false,"component_selection":"included_exactly_once","eval_contains_cosine":false,"throughput":"f*abs(wi.z)/p"}},"lobe_flags":{"bits":{"none":0,"diffuse":1,"glossy":2,"specular":4,"reflection":8,"transmission":16,"volume":32},"known_mask":63,"surface_event":"exactly_one_family_and_one_direction","volume_event":"volume_only","selection_mask":"any_combination_of_known_bits","specular_is_delta":true},"transport_modes":{"radiance":{"code":0,"transmission_adjoint_scale":"(eta_i/eta_t)^2"},"importance":{"code":1,"transmission_adjoint_scale":"1"},"eta_i":"wo_side","eta_t":"wi_side","reflection":"mode_invariant"},"directions":{"space":"local_closure_frame","frame_basis":"caller_supplied","normal_axis":"+z","orientation":"away_from_surface","wo":"surface_to_previous_vertex","wi":"surface_to_next_vertex","reflection":"same_nonzero_hemisphere","transmission":"opposite_nonzero_hemispheres","tangent":"zero_support"}})json"};

} // namespace

core::Result<std::string> dump_bsdf_conventions(const std::uint32_t schema_version) {
    if (schema_version != CurrentBsdfConventionSchemaVersion) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::incompatible,
            .message = "Unsupported BSDF convention schema version " +
                       std::to_string(schema_version) + "; expected " +
                       std::to_string(CurrentBsdfConventionSchemaVersion) + ".",
        });
    }
    return std::string{BsdfConventionDocument};
}

} // namespace blackframe::renderer
