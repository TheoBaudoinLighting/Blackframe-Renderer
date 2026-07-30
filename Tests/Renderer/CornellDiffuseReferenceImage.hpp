#pragma once

#include "CornellDiffuseImageRenderer.hpp"

#include <Blackframe/Renderer/ExrWriter.hpp>
#include <IexBaseExc.h>
#include <ImathBox.h>
#include <ImfChannelList.h>
#include <ImfChromaticities.h>
#include <ImfChromaticitiesAttribute.h>
#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <ImfInputFile.h>
#include <ImfStringAttribute.h>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace blackframe::renderer::cornell_test {

struct LoadedCornellReference final {
    Film film;
    ExrRunMetadata metadata;
};

namespace reference_image_detail {

[[nodiscard]] inline std::string utf8_path(const std::filesystem::path& path) {
    const auto encoded = path.u8string();
    return std::string{reinterpret_cast<const char*>(encoded.data()), encoded.size()};
}

[[nodiscard]] inline bool lower_hexadecimal(const std::string_view value,
                                            const std::size_t digit_count) noexcept {
    if (value.size() != digit_count) {
        return false;
    }
    for (const auto character : value) {
        if (!((character >= '0' && character <= '9') || (character >= 'a' && character <= 'f'))) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] inline core::Error reference_error(const core::StatusCode code, std::string message) {
    return core::Error{
        .code = code,
        .message = std::move(message),
    };
}

[[nodiscard]] inline core::Result<std::string>
string_attribute(const OPENEXR_IMF_NAMESPACE::Header& header, const char* const name) {
    try {
        return header.typedAttribute<OPENEXR_IMF_NAMESPACE::StringAttribute>(name).value();
    } catch (const IEX_NAMESPACE::BaseExc&) {
        return std::unexpected(
            reference_error(core::StatusCode::incompatible,
                            "A Cornell reference EXR is missing required string attribute '" +
                                std::string{name} + "'."));
    }
}

[[nodiscard]] inline core::Result<std::uint64_t>
parse_seed(const std::string_view serialized_seed) {
    auto result = std::uint64_t{};
    const auto [end, error] = std::from_chars(
        serialized_seed.data(), serialized_seed.data() + serialized_seed.size(), result);
    if (error != std::errc{} || end != serialized_seed.data() + serialized_seed.size()) {
        return std::unexpected(
            reference_error(core::StatusCode::incompatible,
                            "A Cornell reference EXR contains an invalid decimal seed."));
    }
    return result;
}

[[nodiscard]] inline core::Result<ExrRunMetadata>
read_metadata(const OPENEXR_IMF_NAMESPACE::Header& header) {
    const auto metadata_version = string_attribute(header, "blackframe.run_metadata_version");
    const auto color_space = string_attribute(header, "blackframe.color_space");
    const auto scene = string_attribute(header, "blackframe.scene");
    const auto seed_text = string_attribute(header, "blackframe.seed");
    const auto commit = string_attribute(header, "blackframe.commit");
    const auto options = string_attribute(header, "blackframe.options");
    const auto backend = string_attribute(header, "blackframe.backend");
    const auto capabilities = string_attribute(header, "blackframe.capabilities");
    const auto asset_hashes = string_attribute(header, "blackframe.asset_hashes");
    for (const auto* const value : {&metadata_version, &color_space, &scene, &seed_text, &commit,
                                    &options, &backend, &capabilities, &asset_hashes}) {
        if (!value->has_value()) {
            return std::unexpected(value->error());
        }
    }
    if (*metadata_version != "1" || *color_space != "scene-linear-srgb") {
        return std::unexpected(reference_error(
            core::StatusCode::incompatible,
            "A Cornell reference EXR uses unsupported run metadata or color-space semantics."));
    }
    const auto seed = parse_seed(*seed_text);
    if (!seed.has_value()) {
        return std::unexpected(seed.error());
    }
    return ExrRunMetadata{
        .scene = *scene,
        .seed = *seed,
        .commit = *commit,
        .options = *options,
        .backend = *backend,
        .capabilities = *capabilities,
        .asset_hashes = *asset_hashes,
    };
}

[[nodiscard]] inline core::Status validate_header(const OPENEXR_IMF_NAMESPACE::Header& header,
                                                  const CornellImageSpecification& specification) {
    const auto expected_window = IMATH_NAMESPACE::Box2i{
        IMATH_NAMESPACE::V2i{0, 0},
        IMATH_NAMESPACE::V2i{static_cast<int>(specification.extent.width - 1U),
                             static_cast<int>(specification.extent.height - 1U)}};
    if (!(header.displayWindow() == expected_window) || !(header.dataWindow() == expected_window) ||
        header.compression() != OPENEXR_IMF_NAMESPACE::ZIP_COMPRESSION ||
        header.lineOrder() != OPENEXR_IMF_NAMESPACE::INCREASING_Y) {
        return std::unexpected(reference_error(
            core::StatusCode::incompatible,
            "A Cornell reference EXR has incompatible windows, compression, or line order."));
    }

    auto channel_count = std::size_t{};
    for (auto channel = header.channels().begin(); channel != header.channels().end(); ++channel) {
        ++channel_count;
    }
    if (channel_count != 3 || header.channels().findChannel("A") != nullptr) {
        return std::unexpected(reference_error(
            core::StatusCode::incompatible,
            "A Cornell reference EXR must contain exactly scene-linear RGB channels."));
    }
    for (const auto* const name : {"B", "G", "R"}) {
        const auto* const channel = header.channels().findChannel(name);
        if (channel == nullptr || channel->type != OPENEXR_IMF_NAMESPACE::FLOAT ||
            channel->xSampling != 1 || channel->ySampling != 1) {
            return std::unexpected(reference_error(
                core::StatusCode::incompatible,
                "A Cornell reference EXR requires full-resolution float RGB channels."));
        }
    }
    try {
        const auto& chromaticities =
            header.typedAttribute<OPENEXR_IMF_NAMESPACE::ChromaticitiesAttribute>("chromaticities")
                .value();
        if (!(chromaticities == OPENEXR_IMF_NAMESPACE::Chromaticities{})) {
            return std::unexpected(
                reference_error(core::StatusCode::incompatible,
                                "A Cornell reference EXR uses incompatible RGB chromaticities."));
        }
    } catch (const IEX_NAMESPACE::BaseExc&) {
        return std::unexpected(
            reference_error(core::StatusCode::incompatible,
                            "A Cornell reference EXR is missing its RGB chromaticities."));
    }
    return {};
}

[[nodiscard]] inline core::Status validate_metadata(const ExrRunMetadata& metadata,
                                                    const CornellImageSpecification& specification,
                                                    const std::string_view scene_sha256,
                                                    const std::string_view source_base_commit) {
    if (!lower_hexadecimal(scene_sha256, 64)) {
        return std::unexpected(reference_error(
            core::StatusCode::invalid_argument,
            "Cornell reference validation requires a complete lowercase scene SHA-256."));
    }
    if (!lower_hexadecimal(source_base_commit, 40)) {
        return std::unexpected(reference_error(
            core::StatusCode::invalid_argument,
            "Cornell reference validation requires a complete lowercase source base commit."));
    }
    if (metadata.scene != cornell_reference_scene_path(specification) ||
        metadata.seed != CornellReferenceSeed ||
        metadata.options != cornell_reference_options(specification) ||
        metadata.backend != CornellReferenceBackend ||
        metadata.capabilities != CornellReferenceCapabilities ||
        metadata.asset_hashes != "scene=sha256:" + std::string{scene_sha256} ||
        metadata.commit != source_base_commit) {
        return std::unexpected(reference_error(
            core::StatusCode::incompatible,
            "A Cornell reference EXR contains incompatible or incomplete run provenance."));
    }
    return {};
}

} // namespace reference_image_detail

[[nodiscard]] inline core::Result<LoadedCornellReference> load_cornell_reference(
    const std::filesystem::path& input_path, const CornellImageSpecification& specification,
    const std::string_view scene_sha256, const std::string_view source_base_commit) {
    if (!image_renderer_detail::known_specification(specification)) {
        return std::unexpected(reference_image_detail::reference_error(
            core::StatusCode::invalid_argument,
            "Cornell reference loading received an unsupported closed fixture specification."));
    }
    if (!input_path.is_absolute() || input_path.filename() != specification.reference_filename) {
        return std::unexpected(reference_image_detail::reference_error(
            core::StatusCode::invalid_argument,
            "A Cornell reference input must be absolute and use the canonical filename."));
    }
    std::error_code filesystem_error;
    if (!std::filesystem::is_regular_file(input_path, filesystem_error) || filesystem_error) {
        return std::unexpected(reference_image_detail::reference_error(
            core::StatusCode::not_found,
            "The required Cornell reference EXR is absent or inaccessible."));
    }

    try {
        auto input =
            OPENEXR_IMF_NAMESPACE::InputFile{reference_image_detail::utf8_path(input_path).c_str()};
        const auto& header = input.header();
        const auto header_status = reference_image_detail::validate_header(header, specification);
        if (!header_status.has_value()) {
            return std::unexpected(header_status.error());
        }
        const auto metadata = reference_image_detail::read_metadata(header);
        if (!metadata.has_value()) {
            return std::unexpected(metadata.error());
        }
        const auto metadata_status = reference_image_detail::validate_metadata(
            *metadata, specification, scene_sha256, source_base_commit);
        if (!metadata_status.has_value()) {
            return std::unexpected(metadata_status.error());
        }

        const auto pixel_count = static_cast<std::size_t>(specification.extent.width) *
                                 static_cast<std::size_t>(specification.extent.height);
        auto blue = std::vector<TransportScalar>(pixel_count);
        auto green = std::vector<TransportScalar>(pixel_count);
        auto red = std::vector<TransportScalar>(pixel_count);
        auto frame_buffer = OPENEXR_IMF_NAMESPACE::FrameBuffer{};
        frame_buffer.insert("B",
                            OPENEXR_IMF_NAMESPACE::Slice::Make(OPENEXR_IMF_NAMESPACE::FLOAT,
                                                               blue.data(), header.dataWindow()));
        frame_buffer.insert("G",
                            OPENEXR_IMF_NAMESPACE::Slice::Make(OPENEXR_IMF_NAMESPACE::FLOAT,
                                                               green.data(), header.dataWindow()));
        frame_buffer.insert("R",
                            OPENEXR_IMF_NAMESPACE::Slice::Make(OPENEXR_IMF_NAMESPACE::FLOAT,
                                                               red.data(), header.dataWindow()));
        input.setFrameBuffer(frame_buffer);
        input.readPixels(header.dataWindow().min.y, header.dataWindow().max.y);

        auto film = Film::create(specification.extent);
        if (!film.has_value()) {
            return std::unexpected(film.error());
        }
        auto has_non_zero_pixel = false;
        auto pixel_index = std::size_t{};
        for (auto pixel_y = std::uint32_t{0}; pixel_y < specification.extent.height; ++pixel_y) {
            for (auto pixel_x = std::uint32_t{0}; pixel_x < specification.extent.width; ++pixel_x) {
                const auto color = LinearRGB{
                    .red = red[pixel_index],
                    .green = green[pixel_index],
                    .blue = blue[pixel_index],
                };
                if (!std::isfinite(color.red) || !std::isfinite(color.green) ||
                    !std::isfinite(color.blue)) {
                    return std::unexpected(reference_image_detail::reference_error(
                        core::StatusCode::incompatible,
                        "A Cornell reference EXR contains a non-finite pixel."));
                }
                has_non_zero_pixel = has_non_zero_pixel || color.red != 0.0F ||
                                     color.green != 0.0F || color.blue != 0.0F;
                const auto status = film->add_sample(pixel_x, pixel_y, color, 1.0F);
                if (!status.has_value()) {
                    return std::unexpected(status.error());
                }
                ++pixel_index;
            }
        }
        if (!has_non_zero_pixel) {
            return std::unexpected(reference_image_detail::reference_error(
                core::StatusCode::incompatible,
                "A Cornell reference EXR cannot be an all-black image."));
        }
        return LoadedCornellReference{
            .film = std::move(*film),
            .metadata = *metadata,
        };
    } catch (const std::bad_alloc&) {
        return std::unexpected(reference_image_detail::reference_error(
            core::StatusCode::resource_exhausted,
            "Cornell reference loading exhausted host memory."));
    } catch (const std::length_error&) {
        return std::unexpected(reference_image_detail::reference_error(
            core::StatusCode::resource_exhausted,
            "Cornell reference loading exceeded a host container limit."));
    } catch (const IEX_NAMESPACE::BaseExc& error) {
        return std::unexpected(reference_image_detail::reference_error(
            core::StatusCode::platform_error,
            "OpenEXR could not read the Cornell reference: " + std::string{error.what()}));
    } catch (const std::filesystem::filesystem_error& error) {
        return std::unexpected(reference_image_detail::reference_error(
            core::StatusCode::platform_error,
            "The Cornell reference path failed: " + std::string{error.what()}));
    } catch (const std::exception& error) {
        return std::unexpected(reference_image_detail::reference_error(
            core::StatusCode::internal_error,
            "Cornell reference loading failed unexpectedly: " + std::string{error.what()}));
    }
}

} // namespace blackframe::renderer::cornell_test
