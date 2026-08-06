#include <Blackframe/Engine/ConstantTextureEvaluation.hpp>
#include <Blackframe/Renderer/ConstantTexture.hpp>
#include <functional>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>

namespace blackframe::engine {
namespace {

[[nodiscard]] core::Error texture_kind_error(const renderer::TextureId id,
                                             const std::string_view expected_kind) {
    return core::Error{
        .code = core::StatusCode::incompatible,
        .message = "Constant texture " + std::to_string(id.value) + " is not a " +
                   std::string{expected_kind} + ".",
    };
}

template <typename Texture>
[[nodiscard]] core::Result<std::reference_wrapper<const Texture>>
find_typed_texture(const FrameScene& scene, const renderer::TextureId id,
                   const std::string_view expected_kind) {
    const auto record = scene.constant_texture(id);
    if (!record) {
        return std::unexpected(record.error());
    }
    const auto* const texture = std::get_if<Texture>(&record->get().texture);
    if (texture == nullptr) {
        return std::unexpected(texture_kind_error(id, expected_kind));
    }
    return std::cref(*texture);
}

} // namespace

core::Result<renderer::ReferenceScalar>
evaluate_scalar_reference_constant_float_texture(const FrameScene& scene,
                                                 const renderer::TextureId texture) {
    const auto value =
        find_typed_texture<renderer::ConstantFloatTexture>(scene, texture, "float texture");
    if (!value) {
        return std::unexpected(value.error());
    }
    return renderer::widen_constant_texture_value(value->get());
}

core::Result<renderer::ReferenceLinearRGB>
evaluate_scalar_reference_constant_color_texture(const FrameScene& scene,
                                                 const renderer::TextureId texture) {
    const auto value =
        find_typed_texture<renderer::ConstantColorTexture>(scene, texture, "color texture");
    if (!value) {
        return std::unexpected(value.error());
    }
    return renderer::widen_constant_texture_value(value->get());
}

core::Result<renderer::ReferenceSpectrum>
evaluate_scalar_reference_constant_spectrum_texture(const FrameScene& scene,
                                                    const renderer::TextureId texture) {
    const auto value =
        find_typed_texture<renderer::ConstantSpectrumTexture>(scene, texture, "spectrum texture");
    if (!value) {
        return std::unexpected(value.error());
    }
    return renderer::widen_constant_texture_value(value->get());
}

core::Result<renderer::TransportScalar>
evaluate_cpu_transport_constant_float_texture(const FrameScene& scene,
                                              const renderer::TextureId texture) {
    const auto value =
        find_typed_texture<renderer::ConstantFloatTexture>(scene, texture, "float texture");
    if (!value) {
        return std::unexpected(value.error());
    }
    return value->get().value();
}

core::Result<renderer::LinearRGB>
evaluate_cpu_transport_constant_color_texture(const FrameScene& scene,
                                              const renderer::TextureId texture) {
    const auto value =
        find_typed_texture<renderer::ConstantColorTexture>(scene, texture, "color texture");
    if (!value) {
        return std::unexpected(value.error());
    }
    return value->get().value();
}

core::Result<renderer::TransportSpectrum>
evaluate_cpu_transport_constant_spectrum_texture(const FrameScene& scene,
                                                 const renderer::TextureId texture) {
    const auto value =
        find_typed_texture<renderer::ConstantSpectrumTexture>(scene, texture, "spectrum texture");
    if (!value) {
        return std::unexpected(value.error());
    }
    return value->get().value();
}

} // namespace blackframe::engine
