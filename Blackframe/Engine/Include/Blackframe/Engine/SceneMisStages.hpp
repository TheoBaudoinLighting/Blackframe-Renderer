#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Engine/AccelBackend.hpp>
#include <Blackframe/Renderer/BsdfOnlyPathLoop.hpp>
#include <Blackframe/Renderer/PathState.hpp>
#include <Blackframe/Renderer/Ray.hpp>
#include <Blackframe/Renderer/SampleStream.hpp>
#include <Blackframe/Renderer/Spectrum.hpp>
#include <Blackframe/Renderer/WavefrontQueues.hpp>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace blackframe::engine {

inline constexpr std::uint32_t CurrentSceneMisStageTraceSchemaVersion = 1U;

enum class SceneMisStageKind : std::uint8_t {
    camera = 0U,
    intersection = 1U,
    shading = 2U,
    shadow = 3U,
    accumulation = 4U,
};

enum class SceneMisIntersectionOutcome : std::uint8_t {
    hit = 0U,
    miss = 1U,
};

enum class SceneMisShadowOutcome : std::uint8_t {
    visible = 0U,
    occluded = 1U,
};

enum class SceneMisAccumulationRoute : std::uint8_t {
    continuation = 0U,
    terminated = 1U,
};

[[nodiscard]] constexpr const char* scene_mis_stage_name(const SceneMisStageKind stage) noexcept {
    switch (stage) {
    case SceneMisStageKind::camera:
        return "camera";
    case SceneMisStageKind::intersection:
        return "intersection";
    case SceneMisStageKind::shading:
        return "shading";
    case SceneMisStageKind::shadow:
        return "shadow";
    case SceneMisStageKind::accumulation:
        return "accumulation";
    }
    return "unknown";
}

// One trace is bound to one indexed sample and one path slot. Stage outputs repeat the slot so
// cross-path routing mistakes are rejected before the trace or a queue can accept them.
struct SceneMisPixelAddress final {
    renderer::SampleStreamIndex sample;
    renderer::WavefrontPathSlot path_slot;

    [[nodiscard]] constexpr bool operator==(const SceneMisPixelAddress&) const noexcept = default;
};

class SceneMisCameraStageOutput final {
  public:
    [[nodiscard]] static core::Result<SceneMisCameraStageOutput>
    create(SceneMisPixelAddress address, const renderer::Ray& primary_ray,
           const renderer::PathState& initial_state);

    [[nodiscard]] constexpr const SceneMisPixelAddress& address() const noexcept {
        return address_;
    }

    [[nodiscard]] constexpr const renderer::Ray& primary_ray() const noexcept {
        return primary_ray_;
    }

    [[nodiscard]] constexpr const renderer::PathState& initial_state() const noexcept {
        return initial_state_;
    }

  private:
    constexpr SceneMisCameraStageOutput(const SceneMisPixelAddress address,
                                        const renderer::Ray primary_ray,
                                        const renderer::PathState initial_state) noexcept
        : address_{address}, primary_ray_{primary_ray}, initial_state_{initial_state} {}

    SceneMisPixelAddress address_;
    renderer::Ray primary_ray_;
    renderer::PathState initial_state_;
};

class SceneMisIntersectionStageOutput final {
  public:
    [[nodiscard]] static core::Result<SceneMisIntersectionStageOutput>
    hit(renderer::WavefrontPathSlot path_slot, const renderer::Ray& ray, const AccelHit& hit);

    [[nodiscard]] static core::Result<SceneMisIntersectionStageOutput>
    miss(renderer::WavefrontPathSlot path_slot, const renderer::Ray& ray);

    [[nodiscard]] constexpr renderer::WavefrontPathSlot path_slot() const noexcept {
        return path_slot_;
    }

    [[nodiscard]] constexpr const renderer::Ray& ray() const noexcept {
        return ray_;
    }

    [[nodiscard]] constexpr SceneMisIntersectionOutcome outcome() const noexcept {
        return hit_ ? SceneMisIntersectionOutcome::hit : SceneMisIntersectionOutcome::miss;
    }

    [[nodiscard]] constexpr const std::optional<AccelHit>& surface_hit() const noexcept {
        return hit_;
    }

  private:
    constexpr SceneMisIntersectionStageOutput(const renderer::WavefrontPathSlot path_slot,
                                              const renderer::Ray ray,
                                              std::optional<AccelHit> hit) noexcept
        : path_slot_{path_slot}, ray_{ray}, hit_{std::move(hit)} {}

    renderer::WavefrontPathSlot path_slot_;
    renderer::Ray ray_;
    std::optional<AccelHit> hit_;
};

class SceneMisShadingStageOutput final {
  public:
    [[nodiscard]] static core::Result<SceneMisShadingStageOutput>
    create(renderer::WavefrontPathSlot path_slot, const renderer::PathState& state,
           renderer::TransportSpectrum emitted_contribution,
           std::optional<renderer::Ray> shadow_ray, std::optional<renderer::Ray> continuation_ray,
           std::optional<renderer::TransportSpectrum> continuation_throughput);

    [[nodiscard]] constexpr renderer::WavefrontPathSlot path_slot() const noexcept {
        return path_slot_;
    }

    [[nodiscard]] constexpr const renderer::TransportSpectrum& throughput() const noexcept {
        return throughput_;
    }

    [[nodiscard]] constexpr const renderer::TransportSpectrum&
    accumulated_radiance_before() const noexcept {
        return accumulated_radiance_before_;
    }

    [[nodiscard]] constexpr const renderer::TransportSpectrum&
    emitted_contribution() const noexcept {
        return emitted_contribution_;
    }

    [[nodiscard]] constexpr bool requires_shadow() const noexcept {
        return shadow_ray_.has_value();
    }

    [[nodiscard]] constexpr bool continuation() const noexcept {
        return continuation_ray_.has_value();
    }

    [[nodiscard]] constexpr const std::optional<renderer::Ray>& shadow_ray() const noexcept {
        return shadow_ray_;
    }

    [[nodiscard]] constexpr const std::optional<renderer::Ray>& continuation_ray() const noexcept {
        return continuation_ray_;
    }

    [[nodiscard]] constexpr const std::optional<renderer::TransportSpectrum>&
    continuation_throughput() const noexcept {
        return continuation_throughput_;
    }

  private:
    constexpr SceneMisShadingStageOutput(
        const renderer::WavefrontPathSlot path_slot, const renderer::TransportSpectrum throughput,
        const renderer::TransportSpectrum accumulated_radiance,
        const renderer::TransportSpectrum emitted_contribution,
        std::optional<renderer::Ray> shadow_ray, std::optional<renderer::Ray> continuation_ray,
        std::optional<renderer::TransportSpectrum> continuation_throughput) noexcept
        : path_slot_{path_slot}, throughput_{throughput},
          accumulated_radiance_before_{accumulated_radiance},
          emitted_contribution_{emitted_contribution}, shadow_ray_{std::move(shadow_ray)},
          continuation_ray_{std::move(continuation_ray)},
          continuation_throughput_{std::move(continuation_throughput)} {}

    renderer::WavefrontPathSlot path_slot_;
    renderer::TransportSpectrum throughput_;
    renderer::TransportSpectrum accumulated_radiance_before_;
    renderer::TransportSpectrum emitted_contribution_;
    std::optional<renderer::Ray> shadow_ray_;
    std::optional<renderer::Ray> continuation_ray_;
    std::optional<renderer::TransportSpectrum> continuation_throughput_;
};

class SceneMisShadowStageOutput final {
  public:
    [[nodiscard]] static core::Result<SceneMisShadowStageOutput>
    visible(renderer::WavefrontPathSlot path_slot, const renderer::Ray& shadow_ray,
            renderer::TransportSpectrum direct_contribution);

    [[nodiscard]] static core::Result<SceneMisShadowStageOutput>
    occluded(renderer::WavefrontPathSlot path_slot, const renderer::Ray& shadow_ray);

    [[nodiscard]] constexpr renderer::WavefrontPathSlot path_slot() const noexcept {
        return path_slot_;
    }

    [[nodiscard]] constexpr const renderer::Ray& shadow_ray() const noexcept {
        return shadow_ray_;
    }

    [[nodiscard]] constexpr SceneMisShadowOutcome outcome() const noexcept {
        return outcome_;
    }

    [[nodiscard]] constexpr const renderer::TransportSpectrum&
    direct_contribution() const noexcept {
        return direct_contribution_;
    }

  private:
    constexpr SceneMisShadowStageOutput(
        const renderer::WavefrontPathSlot path_slot, const renderer::Ray shadow_ray,
        const SceneMisShadowOutcome outcome,
        const renderer::TransportSpectrum direct_contribution) noexcept
        : path_slot_{path_slot}, shadow_ray_{shadow_ray}, outcome_{outcome},
          direct_contribution_{direct_contribution} {}

    renderer::WavefrontPathSlot path_slot_;
    renderer::Ray shadow_ray_;
    SceneMisShadowOutcome outcome_;
    renderer::TransportSpectrum direct_contribution_;
};

class SceneMisAccumulationStageOutput final {
  public:
    [[nodiscard]] static core::Result<SceneMisAccumulationStageOutput>
    continuation(renderer::WavefrontPathSlot path_slot,
                 renderer::TransportSpectrum accumulated_radiance,
                 const renderer::Ray& continuation_ray);

    [[nodiscard]] static core::Result<SceneMisAccumulationStageOutput>
    terminated(renderer::WavefrontPathSlot path_slot,
               renderer::TransportSpectrum accumulated_radiance,
               renderer::BsdfOnlyPathTermination termination);

    [[nodiscard]] constexpr renderer::WavefrontPathSlot path_slot() const noexcept {
        return path_slot_;
    }

    [[nodiscard]] constexpr const renderer::TransportSpectrum&
    accumulated_radiance() const noexcept {
        return accumulated_radiance_;
    }

    [[nodiscard]] constexpr SceneMisAccumulationRoute route() const noexcept {
        return route_;
    }

    [[nodiscard]] constexpr const std::optional<renderer::BsdfOnlyPathTermination>&
    termination() const noexcept {
        return termination_;
    }

    [[nodiscard]] constexpr const std::optional<renderer::Ray>& continuation_ray() const noexcept {
        return continuation_ray_;
    }

  private:
    constexpr SceneMisAccumulationStageOutput(
        const renderer::WavefrontPathSlot path_slot,
        const renderer::TransportSpectrum accumulated_radiance,
        const SceneMisAccumulationRoute route,
        const std::optional<renderer::BsdfOnlyPathTermination> termination,
        std::optional<renderer::Ray> continuation_ray) noexcept
        : path_slot_{path_slot}, accumulated_radiance_{accumulated_radiance}, route_{route},
          termination_{termination}, continuation_ray_{std::move(continuation_ray)} {}

    renderer::WavefrontPathSlot path_slot_;
    renderer::TransportSpectrum accumulated_radiance_;
    SceneMisAccumulationRoute route_;
    std::optional<renderer::BsdfOnlyPathTermination> termination_;
    std::optional<renderer::Ray> continuation_ray_;
};

// This local diagnostic object freezes the five stage boundaries without owning scheduling or
// mutating any queue. Invalid transitions and path-slot mismatches leave both the event sequence
// and the expected next stage unchanged. The scalar MIS loop remains the numerical transport
// oracle.
class SceneMisPixelTrace final {
  public:
    [[nodiscard]] static core::Result<SceneMisPixelTrace> create(SceneMisPixelAddress address,
                                                                 std::uint32_t schema_version);

    [[nodiscard]] core::Status append_camera(const SceneMisCameraStageOutput& output);
    [[nodiscard]] core::Status append_intersection(const SceneMisIntersectionStageOutput& output);
    [[nodiscard]] core::Status append_shading(const SceneMisShadingStageOutput& output);
    [[nodiscard]] core::Status append_shadow(const SceneMisShadowStageOutput& output);
    [[nodiscard]] core::Status append_accumulation(const SceneMisAccumulationStageOutput& output);

    [[nodiscard]] constexpr const SceneMisPixelAddress& address() const noexcept {
        return address_;
    }

    [[nodiscard]] constexpr std::uint32_t schema_version() const noexcept {
        return schema_version_;
    }

    [[nodiscard]] std::size_t event_count() const noexcept {
        return events_.size();
    }

    [[nodiscard]] bool complete() const noexcept;
    [[nodiscard]] core::Result<SceneMisStageKind> stage_at(std::size_t index) const;
    [[nodiscard]] core::Result<std::string> serialize() const;

  private:
    enum class ExpectedStage : std::uint8_t {
        camera,
        intersection,
        shading,
        shadow,
        accumulation,
        terminal,
    };

    enum class AccumulationSource : std::uint8_t {
        none,
        miss,
        shading_without_continuation,
        shading_with_continuation,
    };

    using Event = std::variant<SceneMisCameraStageOutput, SceneMisIntersectionStageOutput,
                               SceneMisShadingStageOutput, SceneMisShadowStageOutput,
                               SceneMisAccumulationStageOutput>;

    constexpr SceneMisPixelTrace(const SceneMisPixelAddress address,
                                 const std::uint32_t schema_version) noexcept
        : address_{address}, schema_version_{schema_version} {}

    [[nodiscard]] core::Status validate_slot(renderer::WavefrontPathSlot path_slot) const;
    [[nodiscard]] core::Status validate_expected(ExpectedStage expected,
                                                 SceneMisStageKind received) const;
    [[nodiscard]] core::Status append_event(Event event);

    SceneMisPixelAddress address_;
    std::uint32_t schema_version_;
    std::vector<Event> events_;
    ExpectedStage expected_{ExpectedStage::camera};
    AccumulationSource accumulation_source_{AccumulationSource::none};
    std::optional<renderer::Ray> expected_intersection_ray_;
    std::optional<renderer::Ray> current_surface_ray_;
    std::optional<renderer::Ray> expected_shadow_ray_;
    std::optional<renderer::Ray> expected_continuation_ray_;
    renderer::TransportSpectrum expected_accumulated_radiance_{};
    renderer::TransportSpectrum expected_throughput_{};
    std::optional<renderer::TransportSpectrum> expected_continuation_throughput_;
    renderer::TransportSpectrum expected_emitted_contribution_{};
    renderer::TransportSpectrum expected_direct_contribution_{};
    bool shadow_completed_{};
    bool shading_throughput_zero_{};
};

static_assert(sizeof(SceneMisStageKind) == sizeof(std::uint8_t));
static_assert(sizeof(SceneMisIntersectionOutcome) == sizeof(std::uint8_t));
static_assert(sizeof(SceneMisShadowOutcome) == sizeof(std::uint8_t));
static_assert(sizeof(SceneMisAccumulationRoute) == sizeof(std::uint8_t));
static_assert(std::is_standard_layout_v<SceneMisPixelAddress>);
static_assert(std::is_trivially_copyable_v<SceneMisPixelAddress>);

} // namespace blackframe::engine
