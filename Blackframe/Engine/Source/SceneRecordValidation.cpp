#include "SceneRecordValidation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>

namespace blackframe::engine::scene_record_validation {
namespace {

[[nodiscard]] core::Error scene_error(const core::StatusCode code, const std::string_view message) {
    return core::Error{
        .code = code,
        .message = std::string{message},
    };
}

template <typename Record, typename Identifier>
[[nodiscard]] std::optional<std::size_t> find_record_index(const std::vector<Record>& records,
                                                           const Identifier id) noexcept {
    const auto candidate = std::ranges::lower_bound(
        records, id.value, {}, [](const Record& record) { return record.id.value; });
    if (candidate == records.end() || candidate->id != id) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(candidate - records.begin());
}

[[nodiscard]] bool valid_ewa_limits(const renderer::HostImageEwaLimits limits) noexcept {
    return limits.maximum_anisotropy >= 1U &&
           limits.maximum_anisotropy <= renderer::HostImageEwaMaximumAnisotropy &&
           limits.maximum_texel_visits > 0U;
}

template <typename Binding>
[[nodiscard]] core::Result<std::reference_wrapper<const SceneHostImageTexture>>
validate_surface_map_binding(const std::vector<SceneHostImageTexture>& textures,
                             const Binding& binding) {
    if (!renderer::is_valid_texture_wrap_mode(binding.u_wrap) ||
        !renderer::is_valid_texture_wrap_mode(binding.v_wrap)) {
        return std::unexpected(scene_error(core::StatusCode::invalid_argument,
                                           "A frame scene surface map requires explicit valid wrap "
                                           "modes."));
    }
    if (!valid_ewa_limits(binding.ewa_limits)) {
        return std::unexpected(
            scene_error(core::StatusCode::invalid_argument,
                        "A frame scene surface map requires valid bounded EWA limits."));
    }
    const auto index = find_record_index(textures, binding.texture);
    if (!index) {
        return std::unexpected(scene_error(
            core::StatusCode::invalid_argument,
            "A frame scene surface map references an unknown host-image texture identifier."));
    }
    return std::cref(textures[*index]);
}

} // namespace

core::Status validate_constant_texture(const SceneConstantTexture& record) {
    if (record.texture.valueless_by_exception()) {
        return std::unexpected(scene_error(core::StatusCode::invalid_argument,
                                           "A frame scene constant-texture slot has no value."));
    }

    return std::visit(
        [](const auto& texture) -> core::Status {
            const auto canonical = std::remove_cvref_t<decltype(texture)>::create(texture.value());
            if (!canonical) {
                return std::unexpected(canonical.error());
            }
            return {};
        },
        record.texture);
}

core::Status validate_host_image_texture(const SceneHostImageTexture& record) {
    if (!record.image) {
        return std::unexpected(scene_error(core::StatusCode::invalid_argument,
                                           "A frame scene host-image texture requires an immutable "
                                           "mip chain."));
    }
    const auto source = record.image->source_image();
    if (!source || record.image->level_count() == 0U || source->channel_count() == 0U) {
        return std::unexpected(scene_error(
            core::StatusCode::invalid_argument,
            "A frame scene host-image texture requires a non-empty immutable mip chain."));
    }
    if (source->source_color_space() != renderer::TextureColorSpace::data ||
        source->storage_color_space() != renderer::TextureColorSpace::data) {
        return std::unexpected(scene_error(
            core::StatusCode::invalid_argument,
            "A frame scene surface-detail texture must use the explicit data source and storage "
            "color spaces."));
    }
    return {};
}

core::Status
reject_shared_texture_identifiers(const std::vector<SceneConstantTexture>& constants,
                                  const std::vector<SceneHostImageTexture>& host_images) {
    auto constant = constants.begin();
    auto host_image = host_images.begin();
    while (constant != constants.end() && host_image != host_images.end()) {
        if (constant->id == host_image->id) {
            return std::unexpected(
                scene_error(core::StatusCode::invalid_argument,
                            "A texture identifier cannot name both a constant and a host image."));
        }
        if (constant->id.value < host_image->id.value) {
            ++constant;
        } else {
            ++host_image;
        }
    }
    return {};
}

core::Status validate_normal_map_binding(const std::vector<SceneHostImageTexture>& textures,
                                         const SceneNormalMapBinding& binding) {
    switch (binding.y_convention) {
    case renderer::TangentSpaceNormalYConvention::positive_v:
    case renderer::TangentSpaceNormalYConvention::negative_v:
        break;
    default:
        return std::unexpected(
            scene_error(core::StatusCode::invalid_argument,
                        "A frame scene normal map requires an explicit supported Y convention."));
    }
    const auto texture = validate_surface_map_binding(textures, binding);
    if (!texture) {
        return std::unexpected(texture.error());
    }
    const auto source = texture->get().image->source_image();
    const auto channels = source->channel_count();
    if (binding.red_channel >= channels || binding.green_channel >= channels ||
        binding.blue_channel >= channels) {
        return std::unexpected(
            scene_error(core::StatusCode::invalid_argument,
                        "A frame scene normal map channel lies outside its host image."));
    }
    if (binding.red_channel == binding.green_channel ||
        binding.red_channel == binding.blue_channel ||
        binding.green_channel == binding.blue_channel) {
        return std::unexpected(
            scene_error(core::StatusCode::invalid_argument,
                        "A frame scene normal map requires three distinct source channels."));
    }
    return {};
}

core::Status validate_bump_map_binding(const std::vector<SceneHostImageTexture>& textures,
                                       const SceneBumpMapBinding& binding) {
    if (!std::isfinite(binding.scale)) {
        return std::unexpected(
            scene_error(core::StatusCode::invalid_argument,
                        "A frame scene bump map requires a finite signed scale."));
    }
    const auto texture = validate_surface_map_binding(textures, binding);
    if (!texture) {
        return std::unexpected(texture.error());
    }
    if (binding.channel >= texture->get().image->source_image()->channel_count()) {
        return std::unexpected(
            scene_error(core::StatusCode::invalid_argument,
                        "A frame scene bump map channel lies outside its host image."));
    }
    return {};
}

core::Status validate_closure_mixture(const SceneClosureMixture& closure_mixture) {
    const auto canonical = SceneClosureMixture::create(
        closure_mixture.closures, closure_mixture.active_component_probabilities(),
        closure_mixture.frame_mode, closure_mixture.tangent_rotation_radians);
    if (!canonical) {
        return std::unexpected(canonical.error());
    }

    const auto active_count = static_cast<std::size_t>(closure_mixture.closures.size());
    const auto inactive_probabilities = std::span<const renderer::TransportScalar>{
        closure_mixture.component_probabilities.data() + active_count,
        closure_mixture.component_probabilities.size() - active_count,
    };
    if (std::ranges::any_of(inactive_probabilities,
                            [](const auto probability) { return probability != 0.0F; })) {
        return std::unexpected(scene_error(
            core::StatusCode::invalid_argument,
            "A frame scene closure mixture requires an exactly zero inactive probability tail."));
    }
    if (!std::ranges::equal(canonical->active_component_probabilities(),
                            closure_mixture.active_component_probabilities())) {
        return std::unexpected(scene_error(
            core::StatusCode::invalid_argument,
            "A frame scene closure mixture requires canonical component probabilities."));
    }
    return {};
}

} // namespace blackframe::engine::scene_record_validation
