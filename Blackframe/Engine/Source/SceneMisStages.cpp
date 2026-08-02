#include <Blackframe/Engine/SceneMisStages.hpp>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace blackframe::engine {
namespace {

static_assert(CurrentSceneMisStageTraceSchemaVersion == 1U);

[[nodiscard]] core::Error stage_error(const core::StatusCode code, std::string message) {
    return core::Error{
        .code = code,
        .message = std::move(message),
    };
}

[[nodiscard]] bool finite_non_negative(const renderer::TransportSpectrum& spectrum) noexcept {
    for (const auto value : spectrum.values) {
        if (!std::isfinite(value) || value < 0.0F) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] constexpr bool same_scalar_bits(const renderer::TransportScalar left,
                                              const renderer::TransportScalar right) noexcept {
    return std::bit_cast<std::uint32_t>(left) == std::bit_cast<std::uint32_t>(right);
}

[[nodiscard]] constexpr bool same_spectrum_bits(const renderer::TransportSpectrum& left,
                                                const renderer::TransportSpectrum& right) noexcept {
    for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        if (!same_scalar_bits(left[lane], right[lane])) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] constexpr bool same_point_bits(const renderer::Point3& left,
                                             const renderer::Point3& right) noexcept {
    return same_scalar_bits(left.x, right.x) && same_scalar_bits(left.y, right.y) &&
           same_scalar_bits(left.z, right.z);
}

[[nodiscard]] constexpr bool zero_spectrum(const renderer::TransportSpectrum& value) noexcept {
    for (const auto lane : value.values) {
        if (lane != 0.0F) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] constexpr bool
preserves_zero_lanes(const renderer::TransportSpectrum& throughput,
                     const renderer::TransportSpectrum& contribution_or_throughput) noexcept {
    for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        if (throughput[lane] == 0.0F && contribution_or_throughput[lane] != 0.0F) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] core::Result<renderer::TransportSpectrum>
checked_accumulation(const renderer::TransportSpectrum& radiance,
                     const renderer::TransportSpectrum& emitted_contribution,
                     const renderer::TransportSpectrum& direct_contribution) {
    auto result = renderer::TransportSpectrum{};
    for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        const auto after_emission = radiance[lane] + emitted_contribution[lane];
        result[lane] = after_emission + direct_contribution[lane];
        if (!std::isfinite(after_emission) || !std::isfinite(result[lane])) {
            return std::unexpected(
                stage_error(core::StatusCode::invalid_argument,
                            "The separated accumulation-stage contribution is not representable."));
        }
    }
    return result;
}

[[nodiscard]] constexpr bool same_ray(const renderer::Ray& left,
                                      const renderer::Ray& right) noexcept {
    return same_scalar_bits(left.origin().x, right.origin().x) &&
           same_scalar_bits(left.origin().y, right.origin().y) &&
           same_scalar_bits(left.origin().z, right.origin().z) &&
           same_scalar_bits(left.direction().x, right.direction().x) &&
           same_scalar_bits(left.direction().y, right.direction().y) &&
           same_scalar_bits(left.direction().z, right.direction().z) &&
           same_scalar_bits(left.t_min(), right.t_min()) &&
           same_scalar_bits(left.t_max(), right.t_max()) &&
           same_scalar_bits(left.time(), right.time()) && left.mask() == right.mask() &&
           left.current_medium() == right.current_medium();
}

[[nodiscard]] core::Status validate_vacuum_ray(const renderer::Ray& ray, const char* const stage) {
    if (ray.time() < 0.0F || ray.time() > 1.0F) {
        return std::unexpected(stage_error(core::StatusCode::invalid_argument,
                                           std::string{"The current scene MIS "} + stage +
                                               " stage requires normalized ray time in [0, 1]."));
    }
    if (ray.current_medium() != renderer::VacuumMedium) {
        return std::unexpected(stage_error(core::StatusCode::unavailable,
                                           std::string{"The current scene MIS "} + stage +
                                               " stage supports vacuum transmittance only."));
    }
    return {};
}

[[nodiscard]] bool known_termination(const renderer::BsdfOnlyPathTermination termination) noexcept {
    switch (termination) {
    case renderer::BsdfOnlyPathTermination::escaped_environment:
    case renderer::BsdfOnlyPathTermination::depth_limit:
    case renderer::BsdfOnlyPathTermination::outside_bsdf_support:
    case renderer::BsdfOnlyPathTermination::zero_throughput:
    case renderer::BsdfOnlyPathTermination::russian_roulette:
        return true;
    }
    return false;
}

[[nodiscard]] const char*
termination_name(const renderer::BsdfOnlyPathTermination termination) noexcept {
    switch (termination) {
    case renderer::BsdfOnlyPathTermination::escaped_environment:
        return "escaped_environment";
    case renderer::BsdfOnlyPathTermination::depth_limit:
        return "depth_limit";
    case renderer::BsdfOnlyPathTermination::outside_bsdf_support:
        return "outside_bsdf_support";
    case renderer::BsdfOnlyPathTermination::zero_throughput:
        return "zero_throughput";
    case renderer::BsdfOnlyPathTermination::russian_roulette:
        return "russian_roulette";
    }
    return "unknown";
}

template <typename Unsigned> void append_hex(std::string& output, const Unsigned value) {
    static_assert(std::is_unsigned_v<Unsigned>);
    constexpr auto digits = "0123456789abcdef";
    output += "0x";
    for (auto index = sizeof(Unsigned) * 2U; index > 0U; --index) {
        const auto shift = (index - 1U) * 4U;
        const auto nibble = static_cast<std::size_t>((value >> shift) & Unsigned{0xFU});
        output += digits[nibble];
    }
}

void append_scalar(std::string& output, const renderer::TransportScalar value) {
    append_hex(output, std::bit_cast<std::uint32_t>(value));
}

void append_spectrum(std::string& output, const renderer::TransportSpectrum& spectrum) {
    output += "[";
    for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        if (lane != 0U) {
            output += ",";
        }
        output += "\"";
        append_scalar(output, spectrum[lane]);
        output += "\"";
    }
    output += "]";
}

void append_ray(std::string& output, const renderer::Ray& ray) {
    output += R"({"origin_bits":[")";
    append_scalar(output, ray.origin().x);
    output += R"(",")";
    append_scalar(output, ray.origin().y);
    output += R"(",")";
    append_scalar(output, ray.origin().z);
    output += R"("],"direction_bits":[")";
    append_scalar(output, ray.direction().x);
    output += R"(",")";
    append_scalar(output, ray.direction().y);
    output += R"(",")";
    append_scalar(output, ray.direction().z);
    output += R"("],"t_min_bits":")";
    append_scalar(output, ray.t_min());
    output += R"(","t_max_bits":")";
    append_scalar(output, ray.t_max());
    output += R"(","time_bits":")";
    append_scalar(output, ray.time());
    output += R"(","mask":")";
    append_hex(output, ray.mask());
    output += R"(","current_medium":")";
    append_hex(output, ray.current_medium().value);
    output += R"("})";
}

void append_serialized_event(std::string& output, const std::size_t sequence,
                             const SceneMisCameraStageOutput& event) {
    output += R"({"sequence":)" + std::to_string(sequence) + R"(,"stage":"camera","ray":)";
    append_ray(output, event.primary_ray());
    output += R"(,"initial_state":{"throughput_bits":)";
    append_spectrum(output, event.initial_state().beta());
    output += R"(,"radiance_bits":)";
    append_spectrum(output, event.initial_state().accumulated_radiance());
    output += R"(,"depth":")";
    append_hex(output, event.initial_state().depth());
    output += R"(","eta_scale_bits":")";
    append_scalar(output, event.initial_state().eta_scale());
    output += R"(","delta_flags":")";
    append_hex(output, static_cast<std::uint8_t>(event.initial_state().delta_flags()));
    output += R"(","current_medium":")";
    append_hex(output, event.initial_state().current_medium().value);
    output += R"("}})";
}

void append_serialized_event(std::string& output, const std::size_t sequence,
                             const SceneMisIntersectionStageOutput& event) {
    output += R"({"sequence":)" + std::to_string(sequence) + R"(,"stage":"intersection","ray":)";
    append_ray(output, event.ray());
    output += R"(,"outcome":")";
    output += event.outcome() == SceneMisIntersectionOutcome::hit ? "hit" : "miss";
    output += "\"";
    if (event.surface_hit()) {
        const auto& hit = *event.surface_hit();
        output += R"(,"parameter_bits":")";
        append_scalar(output, hit.triangle.parameter);
        output += R"(","position_bits":[")";
        append_scalar(output, hit.triangle.position.x);
        output += R"(",")";
        append_scalar(output, hit.triangle.position.y);
        output += R"(",")";
        append_scalar(output, hit.triangle.position.z);
        output += R"("],"geometric_normal_bits":[")";
        append_scalar(output, hit.triangle.geometric_normal.x);
        output += R"(",")";
        append_scalar(output, hit.triangle.geometric_normal.y);
        output += R"(",")";
        append_scalar(output, hit.triangle.geometric_normal.z);
        output += R"("],"barycentrics_bits":[")";
        append_scalar(output, hit.triangle.barycentrics.vertex0);
        output += R"(",")";
        append_scalar(output, hit.triangle.barycentrics.vertex1);
        output += R"(",")";
        append_scalar(output, hit.triangle.barycentrics.vertex2);
        output += R"("],"object":")";
        append_hex(output, hit.object.value);
        output += R"(","instance":")";
        append_hex(output, hit.identifiers.instance.value);
        output += R"(","geometry":")";
        append_hex(output, hit.identifiers.geometry.value);
        output += R"(","primitive":")";
        append_hex(output, hit.identifiers.primitive.value);
        output += R"(","material":")";
        append_hex(output, hit.identifiers.material.value);
        output += "\"";
    }
    output += "}";
}

void append_serialized_event(std::string& output, const std::size_t sequence,
                             const SceneMisShadingStageOutput& event) {
    output +=
        R"({"sequence":)" + std::to_string(sequence) + R"(,"stage":"shading","throughput_bits":)";
    append_spectrum(output, event.throughput());
    output += R"(,"radiance_before_bits":)";
    append_spectrum(output, event.accumulated_radiance_before());
    output += R"(,"emitted_contribution_bits":)";
    append_spectrum(output, event.emitted_contribution());
    output += R"(,"requires_shadow":)";
    output += event.requires_shadow() ? "true" : "false";
    if (event.shadow_ray()) {
        output += R"(,"shadow_ray":)";
        append_ray(output, *event.shadow_ray());
    }
    output += R"(,"continuation":)";
    output += event.continuation() ? "true" : "false";
    if (event.continuation_ray()) {
        output += R"(,"continuation_ray":)";
        append_ray(output, *event.continuation_ray());
        output += R"(,"continuation_throughput_bits":)";
        append_spectrum(output, *event.continuation_throughput());
    }
    output += "}";
}

void append_serialized_event(std::string& output, const std::size_t sequence,
                             const SceneMisShadowStageOutput& event) {
    output += R"({"sequence":)" + std::to_string(sequence) + R"(,"stage":"shadow","ray":)";
    append_ray(output, event.shadow_ray());
    output += R"(,"outcome":")";
    output += event.outcome() == SceneMisShadowOutcome::visible ? "visible" : "occluded";
    output += R"(","direct_contribution_bits":)";
    append_spectrum(output, event.direct_contribution());
    output += "}";
}

void append_serialized_event(std::string& output, const std::size_t sequence,
                             const SceneMisAccumulationStageOutput& event) {
    output += R"({"sequence":)" + std::to_string(sequence) +
              R"(,"stage":"accumulation","radiance_bits":)";
    append_spectrum(output, event.accumulated_radiance());
    output += R"(,"route":")";
    output +=
        event.route() == SceneMisAccumulationRoute::continuation ? "continuation" : "terminated";
    output += R"(","termination":")";
    output += event.termination() ? termination_name(*event.termination()) : "none";
    if (event.continuation_ray()) {
        output += R"(","continuation_ray":)";
        append_ray(output, *event.continuation_ray());
        output += "}";
        return;
    }
    output += R"("})";
}

} // namespace

core::Result<SceneMisCameraStageOutput>
SceneMisCameraStageOutput::create(const SceneMisPixelAddress address,
                                  const renderer::Ray& primary_ray,
                                  const renderer::PathState& initial_state) {
    if (const auto ray_status = validate_vacuum_ray(primary_ray, "camera"); !ray_status) {
        return std::unexpected(ray_status.error());
    }
    if (initial_state.depth() != 0U) {
        return std::unexpected(
            stage_error(core::StatusCode::incompatible,
                        "The camera stage requires an initial path state at depth zero."));
    }
    if (initial_state.current_medium() != renderer::VacuumMedium) {
        return std::unexpected(
            stage_error(core::StatusCode::unavailable,
                        "The current scene MIS camera stage supports vacuum transmittance only."));
    }
    if (primary_ray.current_medium() != initial_state.current_medium()) {
        return std::unexpected(
            stage_error(core::StatusCode::invalid_argument,
                        "The camera-stage ray and path state must carry the same current medium."));
    }
    if (!finite_non_negative(initial_state.beta()) ||
        !finite_non_negative(initial_state.accumulated_radiance())) {
        return std::unexpected(stage_error(
            core::StatusCode::invalid_argument,
            "The camera stage requires finite non-negative initial transport spectra."));
    }
    return SceneMisCameraStageOutput{address, primary_ray, initial_state};
}

core::Result<SceneMisIntersectionStageOutput>
SceneMisIntersectionStageOutput::hit(const renderer::WavefrontPathSlot path_slot,
                                     const renderer::Ray& ray, const AccelHit& hit) {
    if (const auto ray_status = validate_vacuum_ray(ray, "intersection"); !ray_status) {
        return std::unexpected(ray_status.error());
    }
    const auto parameter = hit.triangle.parameter;
    const auto& barycentrics = hit.triangle.barycentrics;
    constexpr auto barycentric_tolerance =
        renderer::TransportScalar{32} * std::numeric_limits<renderer::TransportScalar>::epsilon();
    const auto barycentric_sum = barycentrics.vertex0 + barycentrics.vertex1 + barycentrics.vertex2;
    const auto ray_position = ray.at(parameter);
    const auto& normal = hit.triangle.geometric_normal;
    const auto normal_squared_length =
        std::fma(normal.x, normal.x, std::fma(normal.y, normal.y, normal.z * normal.z));
    constexpr auto normal_tolerance =
        renderer::TransportScalar{128} * std::numeric_limits<renderer::TransportScalar>::epsilon();
    if (!std::isfinite(parameter) || parameter < ray.t_min() || parameter > ray.t_max() ||
        !std::isfinite(barycentrics.vertex0) || barycentrics.vertex0 < -barycentric_tolerance ||
        barycentrics.vertex0 > 1.0F + barycentric_tolerance ||
        !std::isfinite(barycentrics.vertex1) || barycentrics.vertex1 < -barycentric_tolerance ||
        barycentrics.vertex1 > 1.0F + barycentric_tolerance ||
        !std::isfinite(barycentrics.vertex2) || barycentrics.vertex2 < -barycentric_tolerance ||
        barycentrics.vertex2 > 1.0F + barycentric_tolerance || !std::isfinite(barycentric_sum) ||
        std::abs(barycentric_sum - 1.0F) > barycentric_tolerance || !ray_position ||
        !same_point_bits(*ray_position, hit.triangle.position) || !std::isfinite(normal.x) ||
        !std::isfinite(normal.y) || !std::isfinite(normal.z) ||
        !std::isfinite(normal_squared_length) ||
        std::abs(normal_squared_length - 1.0F) > normal_tolerance) {
        return std::unexpected(
            stage_error(core::StatusCode::invalid_argument,
                        "The intersection-stage hit is inconsistent with its source ray."));
    }
    return SceneMisIntersectionStageOutput{path_slot, ray, hit};
}

core::Result<SceneMisIntersectionStageOutput>
SceneMisIntersectionStageOutput::miss(const renderer::WavefrontPathSlot path_slot,
                                      const renderer::Ray& ray) {
    if (const auto ray_status = validate_vacuum_ray(ray, "intersection"); !ray_status) {
        return std::unexpected(ray_status.error());
    }
    return SceneMisIntersectionStageOutput{path_slot, ray, std::nullopt};
}

core::Result<SceneMisShadingStageOutput> SceneMisShadingStageOutput::create(
    const renderer::WavefrontPathSlot path_slot, const renderer::PathState& state,
    const renderer::TransportSpectrum emitted_contribution, std::optional<renderer::Ray> shadow_ray,
    std::optional<renderer::Ray> continuation_ray,
    std::optional<renderer::TransportSpectrum> continuation_throughput) {
    if (state.current_medium() != renderer::VacuumMedium) {
        return std::unexpected(
            stage_error(core::StatusCode::unavailable,
                        "The current scene MIS shading stage supports vacuum transmittance only."));
    }
    if (!finite_non_negative(state.beta()) || !finite_non_negative(state.accumulated_radiance()) ||
        !finite_non_negative(emitted_contribution) ||
        (continuation_throughput && !finite_non_negative(*continuation_throughput))) {
        return std::unexpected(
            stage_error(core::StatusCode::invalid_argument,
                        "The shading stage requires finite non-negative transport spectra."));
    }
    if (continuation_ray.has_value() != continuation_throughput.has_value()) {
        return std::unexpected(
            stage_error(core::StatusCode::invalid_argument,
                        "A shading continuation requires both its ray and updated throughput."));
    }
    if (!preserves_zero_lanes(state.beta(), emitted_contribution) ||
        (continuation_throughput &&
         !preserves_zero_lanes(state.beta(), *continuation_throughput))) {
        return std::unexpected(stage_error(
            core::StatusCode::invalid_argument,
            "Shading contributions and continuation throughput cannot revive zero spectral "
            "lanes."));
    }
    if (zero_spectrum(state.beta()) && (shadow_ray || continuation_ray)) {
        return std::unexpected(
            stage_error(core::StatusCode::invalid_argument,
                        "Zero shading throughput cannot route shadow or continuation work."));
    }
    if (shadow_ray) {
        if (const auto ray_status = validate_vacuum_ray(*shadow_ray, "shading shadow request");
            !ray_status) {
            return std::unexpected(ray_status.error());
        }
    }
    if (continuation_ray) {
        if (const auto ray_status =
                validate_vacuum_ray(*continuation_ray, "shading continuation request");
            !ray_status) {
            return std::unexpected(ray_status.error());
        }
    }
    return SceneMisShadingStageOutput{path_slot,
                                      state.beta(),
                                      state.accumulated_radiance(),
                                      emitted_contribution,
                                      std::move(shadow_ray),
                                      std::move(continuation_ray),
                                      std::move(continuation_throughput)};
}

core::Result<SceneMisShadowStageOutput>
SceneMisShadowStageOutput::visible(const renderer::WavefrontPathSlot path_slot,
                                   const renderer::Ray& shadow_ray,
                                   const renderer::TransportSpectrum direct_contribution) {
    if (const auto ray_status = validate_vacuum_ray(shadow_ray, "shadow"); !ray_status) {
        return std::unexpected(ray_status.error());
    }
    if (!finite_non_negative(direct_contribution)) {
        return std::unexpected(
            stage_error(core::StatusCode::invalid_argument,
                        "The shadow stage requires a finite non-negative visible contribution."));
    }
    return SceneMisShadowStageOutput{path_slot, shadow_ray, SceneMisShadowOutcome::visible,
                                     direct_contribution};
}

core::Result<SceneMisShadowStageOutput>
SceneMisShadowStageOutput::occluded(const renderer::WavefrontPathSlot path_slot,
                                    const renderer::Ray& shadow_ray) {
    if (const auto ray_status = validate_vacuum_ray(shadow_ray, "shadow"); !ray_status) {
        return std::unexpected(ray_status.error());
    }
    return SceneMisShadowStageOutput{path_slot, shadow_ray, SceneMisShadowOutcome::occluded,
                                     renderer::TransportSpectrum{}};
}

core::Result<SceneMisAccumulationStageOutput> SceneMisAccumulationStageOutput::continuation(
    const renderer::WavefrontPathSlot path_slot,
    const renderer::TransportSpectrum accumulated_radiance, const renderer::Ray& continuation_ray) {
    if (!finite_non_negative(accumulated_radiance)) {
        return std::unexpected(
            stage_error(core::StatusCode::invalid_argument,
                        "The accumulation stage requires finite non-negative radiance."));
    }
    if (const auto ray_status = validate_vacuum_ray(continuation_ray, "accumulation continuation");
        !ray_status) {
        return std::unexpected(ray_status.error());
    }
    return SceneMisAccumulationStageOutput{path_slot, accumulated_radiance,
                                           SceneMisAccumulationRoute::continuation, std::nullopt,
                                           continuation_ray};
}

core::Result<SceneMisAccumulationStageOutput>
SceneMisAccumulationStageOutput::terminated(const renderer::WavefrontPathSlot path_slot,
                                            const renderer::TransportSpectrum accumulated_radiance,
                                            const renderer::BsdfOnlyPathTermination termination) {
    if (!finite_non_negative(accumulated_radiance)) {
        return std::unexpected(
            stage_error(core::StatusCode::invalid_argument,
                        "The accumulation stage requires finite non-negative radiance."));
    }
    if (!known_termination(termination)) {
        return std::unexpected(stage_error(core::StatusCode::invalid_argument,
                                           "The accumulation stage received an unknown path "
                                           "termination outcome."));
    }
    return SceneMisAccumulationStageOutput{path_slot, accumulated_radiance,
                                           SceneMisAccumulationRoute::terminated, termination,
                                           std::nullopt};
}

core::Result<SceneMisPixelTrace> SceneMisPixelTrace::create(const SceneMisPixelAddress address,
                                                            const std::uint32_t schema_version) {
    if (schema_version != CurrentSceneMisStageTraceSchemaVersion) {
        return std::unexpected(stage_error(
            core::StatusCode::incompatible,
            "Unsupported scene MIS stage trace schema version " + std::to_string(schema_version) +
                "; expected " + std::to_string(CurrentSceneMisStageTraceSchemaVersion) + "."));
    }
    return SceneMisPixelTrace{address, schema_version};
}

core::Status SceneMisPixelTrace::append_camera(const SceneMisCameraStageOutput& output) {
    const auto expected_status =
        validate_expected(ExpectedStage::camera, SceneMisStageKind::camera);
    if (!expected_status) {
        return expected_status;
    }
    if (output.address() != address_) {
        return std::unexpected(stage_error(
            core::StatusCode::invalid_argument,
            "The camera-stage output belongs to a different indexed pixel sample or path slot."));
    }
    const auto appended = append_event(output);
    if (!appended) {
        return appended;
    }
    expected_intersection_ray_ = output.primary_ray();
    expected_accumulated_radiance_ = output.initial_state().accumulated_radiance();
    expected_throughput_ = output.initial_state().beta();
    expected_ = ExpectedStage::intersection;
    return {};
}

core::Status
SceneMisPixelTrace::append_intersection(const SceneMisIntersectionStageOutput& output) {
    const auto expected_status =
        validate_expected(ExpectedStage::intersection, SceneMisStageKind::intersection);
    if (!expected_status) {
        return expected_status;
    }
    const auto slot_status = validate_slot(output.path_slot());
    if (!slot_status) {
        return slot_status;
    }
    if (!expected_intersection_ray_ || !same_ray(output.ray(), *expected_intersection_ray_)) {
        return std::unexpected(stage_error(
            core::StatusCode::invalid_argument,
            "The intersection-stage ray does not match the ray routed by the preceding stage."));
    }
    const auto appended = append_event(output);
    if (!appended) {
        return appended;
    }
    expected_intersection_ray_.reset();
    current_surface_ray_ = output.outcome() == SceneMisIntersectionOutcome::hit
                               ? std::optional{output.ray()}
                               : std::nullopt;
    expected_emitted_contribution_ = {};
    expected_direct_contribution_ = {};
    shadow_completed_ = false;
    shading_throughput_zero_ = false;
    accumulation_source_ = output.outcome() == SceneMisIntersectionOutcome::miss
                               ? AccumulationSource::miss
                               : AccumulationSource::none;
    expected_ = output.outcome() == SceneMisIntersectionOutcome::hit ? ExpectedStage::shading
                                                                     : ExpectedStage::accumulation;
    return {};
}

core::Status SceneMisPixelTrace::append_shading(const SceneMisShadingStageOutput& output) {
    const auto expected_status =
        validate_expected(ExpectedStage::shading, SceneMisStageKind::shading);
    if (!expected_status) {
        return expected_status;
    }
    const auto slot_status = validate_slot(output.path_slot());
    if (!slot_status) {
        return slot_status;
    }
    if (!same_spectrum_bits(output.accumulated_radiance_before(), expected_accumulated_radiance_)) {
        return std::unexpected(stage_error(
            core::StatusCode::invalid_argument,
            "The shading-stage radiance does not match the preceding accumulation state."));
    }
    if (!same_spectrum_bits(output.throughput(), expected_throughput_)) {
        return std::unexpected(
            stage_error(core::StatusCode::invalid_argument,
                        "The shading-stage throughput does not match the preceding path state."));
    }
    if (!current_surface_ray_) {
        return std::unexpected(stage_error(
            core::StatusCode::internal_error,
            "The shading stage has no preceding hit ray to preserve routing attributes."));
    }
    const auto preserves_ray_attributes = [this](const renderer::Ray& ray) {
        return same_scalar_bits(ray.time(), current_surface_ray_->time()) &&
               ray.mask() == current_surface_ray_->mask() &&
               ray.current_medium() == current_surface_ray_->current_medium();
    };
    if ((output.shadow_ray() && !preserves_ray_attributes(*output.shadow_ray())) ||
        (output.continuation_ray() && !preserves_ray_attributes(*output.continuation_ray()))) {
        return std::unexpected(stage_error(
            core::StatusCode::invalid_argument,
            "Shading-stage rays must preserve time, visibility mask, and current medium."));
    }
    const auto appended = append_event(output);
    if (!appended) {
        return appended;
    }
    accumulation_source_ = output.continuation() ? AccumulationSource::shading_with_continuation
                                                 : AccumulationSource::shading_without_continuation;
    expected_shadow_ray_ = output.shadow_ray();
    expected_continuation_ray_ = output.continuation_ray();
    expected_continuation_throughput_ = output.continuation_throughput();
    expected_emitted_contribution_ = output.emitted_contribution();
    expected_direct_contribution_ = {};
    shadow_completed_ = false;
    shading_throughput_zero_ = zero_spectrum(output.throughput());
    current_surface_ray_.reset();
    expected_ = output.requires_shadow() ? ExpectedStage::shadow : ExpectedStage::accumulation;
    return {};
}

core::Status SceneMisPixelTrace::append_shadow(const SceneMisShadowStageOutput& output) {
    const auto expected_status =
        validate_expected(ExpectedStage::shadow, SceneMisStageKind::shadow);
    if (!expected_status) {
        return expected_status;
    }
    const auto slot_status = validate_slot(output.path_slot());
    if (!slot_status) {
        return slot_status;
    }
    if (!expected_shadow_ray_ || !same_ray(output.shadow_ray(), *expected_shadow_ray_)) {
        return std::unexpected(stage_error(
            core::StatusCode::invalid_argument,
            "The shadow-stage ray does not match the request routed by the shading stage."));
    }
    if (!preserves_zero_lanes(expected_throughput_, output.direct_contribution())) {
        return std::unexpected(
            stage_error(core::StatusCode::invalid_argument,
                        "Direct lighting cannot revive a zero spectral throughput lane."));
    }
    const auto appended = append_event(output);
    if (!appended) {
        return appended;
    }
    expected_direct_contribution_ = output.direct_contribution();
    shadow_completed_ = true;
    expected_shadow_ray_.reset();
    expected_ = ExpectedStage::accumulation;
    return {};
}

core::Status
SceneMisPixelTrace::append_accumulation(const SceneMisAccumulationStageOutput& output) {
    const auto expected_status =
        validate_expected(ExpectedStage::accumulation, SceneMisStageKind::accumulation);
    if (!expected_status) {
        return expected_status;
    }
    const auto slot_status = validate_slot(output.path_slot());
    if (!slot_status) {
        return slot_status;
    }
    const auto termination = output.termination();
    if (accumulation_source_ == AccumulationSource::miss &&
        (output.route() != SceneMisAccumulationRoute::terminated || !termination ||
         *termination != renderer::BsdfOnlyPathTermination::escaped_environment)) {
        return std::unexpected(stage_error(
            core::StatusCode::invalid_argument,
            "A missed intersection must terminate through escaped environment accumulation."));
    }
    if (accumulation_source_ == AccumulationSource::shading_without_continuation) {
        const auto allowed_termination =
            termination &&
            ((!shadow_completed_ &&
              (*termination == renderer::BsdfOnlyPathTermination::depth_limit ||
               (shading_throughput_zero_ &&
                *termination == renderer::BsdfOnlyPathTermination::zero_throughput) ||
               (!shading_throughput_zero_ &&
                *termination == renderer::BsdfOnlyPathTermination::outside_bsdf_support))) ||
             (shadow_completed_ &&
              *termination == renderer::BsdfOnlyPathTermination::outside_bsdf_support));
        if (output.route() != SceneMisAccumulationRoute::terminated || !allowed_termination) {
            return std::unexpected(stage_error(
                core::StatusCode::invalid_argument,
                "A shaded hit without continuation has an impossible termination route."));
        }
        if (*termination == renderer::BsdfOnlyPathTermination::zero_throughput &&
            !shading_throughput_zero_) {
            return std::unexpected(stage_error(
                core::StatusCode::invalid_argument,
                "A pre-sampling zero-throughput termination requires zero shading throughput."));
        }
    }
    if (accumulation_source_ == AccumulationSource::shading_with_continuation) {
        if (!expected_continuation_throughput_) {
            return std::unexpected(
                stage_error(core::StatusCode::internal_error,
                            "A prepared continuation is missing its updated throughput."));
        }
        const auto continuation_is_zero = zero_spectrum(*expected_continuation_throughput_);
        if (output.route() == SceneMisAccumulationRoute::continuation) {
            if (!output.continuation_ray() || !expected_continuation_ray_ ||
                !same_ray(*output.continuation_ray(), *expected_continuation_ray_) ||
                continuation_is_zero) {
                return std::unexpected(stage_error(
                    core::StatusCode::invalid_argument,
                    "The accumulation-stage continuation is inconsistent with the shading "
                    "route."));
            }
        } else if (!termination ||
                   (continuation_is_zero &&
                    *termination != renderer::BsdfOnlyPathTermination::zero_throughput) ||
                   (!continuation_is_zero &&
                    *termination != renderer::BsdfOnlyPathTermination::russian_roulette)) {
            return std::unexpected(
                stage_error(core::StatusCode::invalid_argument,
                            "A prepared continuation has an impossible throughput termination."));
        }
    }
    if (accumulation_source_ == AccumulationSource::none) {
        return std::unexpected(
            stage_error(core::StatusCode::internal_error,
                        "The accumulation stage has no preceding miss or shading route."));
    }
    if (accumulation_source_ != AccumulationSource::miss) {
        const auto expected_radiance =
            checked_accumulation(expected_accumulated_radiance_, expected_emitted_contribution_,
                                 expected_direct_contribution_);
        if (!expected_radiance) {
            return std::unexpected(expected_radiance.error());
        }
        if (!same_spectrum_bits(output.accumulated_radiance(), *expected_radiance)) {
            return std::unexpected(stage_error(
                core::StatusCode::invalid_argument,
                "The accumulation-stage radiance does not match emission and direct lighting."));
        }
    } else {
        for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
            if ((expected_throughput_[lane] == 0.0F &&
                 !same_scalar_bits(output.accumulated_radiance()[lane],
                                   expected_accumulated_radiance_[lane])) ||
                output.accumulated_radiance()[lane] < expected_accumulated_radiance_[lane]) {
                return std::unexpected(stage_error(
                    core::StatusCode::invalid_argument,
                    "Environment accumulation cannot reduce radiance or revive a zero spectral "
                    "lane."));
            }
        }
    }
    const auto appended = append_event(output);
    if (!appended) {
        return appended;
    }
    if (output.route() == SceneMisAccumulationRoute::continuation) {
        expected_intersection_ray_ = output.continuation_ray();
        expected_accumulated_radiance_ = output.accumulated_radiance();
        expected_throughput_ = *expected_continuation_throughput_;
        expected_ = ExpectedStage::intersection;
    } else {
        expected_ = ExpectedStage::terminal;
    }
    accumulation_source_ = AccumulationSource::none;
    expected_continuation_ray_.reset();
    expected_continuation_throughput_.reset();
    expected_emitted_contribution_ = {};
    expected_direct_contribution_ = {};
    shadow_completed_ = false;
    shading_throughput_zero_ = false;
    return {};
}

bool SceneMisPixelTrace::complete() const noexcept {
    return expected_ == ExpectedStage::terminal;
}

core::Result<SceneMisStageKind> SceneMisPixelTrace::stage_at(const std::size_t index) const {
    if (index >= events_.size()) {
        return std::unexpected(stage_error(core::StatusCode::invalid_argument,
                                           "The stage trace index is outside the event domain."));
    }
    return std::visit(
        []<typename EventType>(const EventType&) {
            if constexpr (std::same_as<EventType, SceneMisCameraStageOutput>) {
                return SceneMisStageKind::camera;
            } else if constexpr (std::same_as<EventType, SceneMisIntersectionStageOutput>) {
                return SceneMisStageKind::intersection;
            } else if constexpr (std::same_as<EventType, SceneMisShadingStageOutput>) {
                return SceneMisStageKind::shading;
            } else if constexpr (std::same_as<EventType, SceneMisShadowStageOutput>) {
                return SceneMisStageKind::shadow;
            } else {
                static_assert(std::same_as<EventType, SceneMisAccumulationStageOutput>);
                return SceneMisStageKind::accumulation;
            }
        },
        events_[index]);
}

core::Result<std::string> SceneMisPixelTrace::serialize() const {
    if (!complete()) {
        return std::unexpected(stage_error(
            core::StatusCode::incompatible,
            "An incomplete scene MIS stage trace cannot be serialized as a finished pixel."));
    }

    try {
        auto output = std::string{};
        constexpr auto fixed_reserve = std::size_t{1024U};
        constexpr auto reserve_per_event = std::size_t{960U};
        if (events_.size() > (output.max_size() - fixed_reserve) / reserve_per_event) {
            return std::unexpected(
                stage_error(core::StatusCode::resource_exhausted,
                            "Scene MIS stage trace length exceeds host string limits."));
        }
        output.reserve(fixed_reserve + events_.size() * reserve_per_event);
        output += R"({"schema_version":1,"precision":"float32","pixel_x":")";
        append_hex(output, address_.sample.pixel_x);
        output += R"(","pixel_y":")";
        append_hex(output, address_.sample.pixel_y);
        output += R"(","sample_index":")";
        append_hex(output, address_.sample.sample_index);
        output += R"(","seed":")";
        append_hex(output, address_.sample.seed);
        output += R"(","path_slot":")";
        append_hex(output, address_.path_slot.value);
        output += R"(","events":[)";
        for (auto index = std::size_t{}; index < events_.size(); ++index) {
            if (index != 0U) {
                output += ",";
            }
            std::visit([&](const auto& event) { append_serialized_event(output, index, event); },
                       events_[index]);
        }
        output += "]}";
        return output;
    } catch (const std::bad_alloc&) {
        return std::unexpected(
            stage_error(core::StatusCode::resource_exhausted,
                        "Scene MIS stage trace serialization exhausted host memory."));
    } catch (const std::length_error&) {
        return std::unexpected(
            stage_error(core::StatusCode::resource_exhausted,
                        "Scene MIS stage trace serialization exceeds host string limits."));
    }
}

core::Status SceneMisPixelTrace::validate_slot(const renderer::WavefrontPathSlot path_slot) const {
    if (path_slot != address_.path_slot) {
        return std::unexpected(
            stage_error(core::StatusCode::invalid_argument,
                        "A stage output belongs to a different path slot than the pixel trace."));
    }
    return {};
}

core::Status SceneMisPixelTrace::validate_expected(const ExpectedStage expected,
                                                   const SceneMisStageKind received) const {
    if (expected_ == expected) {
        return {};
    }
    const auto expected_name = [this]() -> const char* {
        switch (expected_) {
        case ExpectedStage::camera:
            return "camera";
        case ExpectedStage::intersection:
            return "intersection";
        case ExpectedStage::shading:
            return "shading";
        case ExpectedStage::shadow:
            return "shadow";
        case ExpectedStage::accumulation:
            return "accumulation";
        case ExpectedStage::terminal:
            return "terminal";
        }
        return "unknown";
    }();
    return std::unexpected(
        stage_error(core::StatusCode::invalid_argument,
                    "Cannot append the " + std::string{scene_mis_stage_name(received)} +
                        " stage while the pixel trace expects " + expected_name + "."));
}

core::Status SceneMisPixelTrace::append_event(Event event) {
    try {
        events_.push_back(std::move(event));
        return {};
    } catch (const std::bad_alloc&) {
        return std::unexpected(stage_error(core::StatusCode::resource_exhausted,
                                           "Scene MIS stage tracing exhausted host memory."));
    } catch (const std::length_error&) {
        return std::unexpected(
            stage_error(core::StatusCode::resource_exhausted,
                        "Scene MIS stage trace length exceeds host container limits."));
    }
}

} // namespace blackframe::engine
