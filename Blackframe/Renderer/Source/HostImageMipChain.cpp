#include <Blackframe/Renderer/HostImageMipChain.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace blackframe::renderer {
namespace {

struct MipExtent final {
    std::uint32_t width{};
    std::uint32_t height{};
    std::size_t element_count{};
    std::uint64_t byte_count{};
};

[[nodiscard]] core::Error mip_error(const core::StatusCode code, std::string message) {
    return core::Error{
        .code = code,
        .message = std::move(message),
    };
}

[[nodiscard]] core::Result<std::uint64_t>
checked_product(const std::uint64_t left, const std::uint64_t right, const char* const message) {
    if (right != 0U && left > std::numeric_limits<std::uint64_t>::max() / right) {
        return std::unexpected(mip_error(core::StatusCode::resource_exhausted, message));
    }
    return left * right;
}

[[nodiscard]] core::Result<std::uint64_t>
checked_sum(const std::uint64_t left, const std::uint64_t right, const char* const message) {
    if (left > std::numeric_limits<std::uint64_t>::max() - right) {
        return std::unexpected(mip_error(core::StatusCode::resource_exhausted, message));
    }
    return left + right;
}

[[nodiscard]] core::Status validate_source_image(const HostImage& image) {
    if (image.width() == 0U || image.height() == 0U || image.channel_count() == 0U) {
        return std::unexpected(
            mip_error(core::StatusCode::incompatible,
                      "A mip chain source image must have non-zero dimensions and channels."));
    }
    if (image.channel_names().size() != image.channel_count()) {
        return std::unexpected(
            mip_error(core::StatusCode::incompatible,
                      "A mip chain source image must preserve one name for every stored channel."));
    }
    const auto pixel_count = checked_product(image.width(), image.height(),
                                             "The mip source pixel count is not representable.");
    if (!pixel_count) {
        return std::unexpected(pixel_count.error());
    }
    const auto element_count = checked_product(
        *pixel_count, image.channel_count(), "The mip source element count is not representable.");
    if (!element_count || *element_count != image.pixels().size()) {
        return std::unexpected(
            element_count ? mip_error(core::StatusCode::incompatible,
                                      "A mip chain source image must have exact pixel storage.")
                          : element_count.error());
    }
    if (!std::ranges::all_of(image.pixels(),
                             [](const auto value) { return std::isfinite(value); })) {
        return std::unexpected(mip_error(core::StatusCode::incompatible,
                                         "A mip chain source image must contain finite values."));
    }
    return {};
}

[[nodiscard]] core::Result<std::vector<MipExtent>>
generated_extents(const HostImage& source, const HostImageMipChainLimits limits) {
    if (limits.maximum_generated_pixel_bytes == 0U) {
        return std::unexpected(mip_error(core::StatusCode::invalid_argument,
                                         "A mip chain generated-pixel budget must be non-zero."));
    }

    auto extents = std::vector<MipExtent>{};
    extents.reserve(std::numeric_limits<std::uint32_t>::digits);
    auto width = source.width();
    auto height = source.height();
    auto generated_bytes = std::uint64_t{};
    while (width > 1U || height > 1U) {
        width = std::max(1U, width / 2U);
        height = std::max(1U, height / 2U);
        const auto pixel_count =
            checked_product(width, height, "A generated mip pixel count is not representable.");
        if (!pixel_count) {
            return std::unexpected(pixel_count.error());
        }
        const auto element_count =
            checked_product(*pixel_count, source.channel_count(),
                            "A generated mip element count is not representable.");
        if (!element_count || *element_count > std::numeric_limits<std::size_t>::max()) {
            return std::unexpected(
                element_count
                    ? mip_error(core::StatusCode::resource_exhausted,
                                "A generated mip element count exceeds host address space.")
                    : element_count.error());
        }
        const auto byte_count = checked_product(*element_count, sizeof(TransportScalar),
                                                "A generated mip byte count is not representable.");
        if (!byte_count) {
            return std::unexpected(byte_count.error());
        }
        const auto next_generated_bytes =
            checked_sum(generated_bytes, *byte_count,
                        "The complete generated mip byte count is not representable.");
        if (!next_generated_bytes) {
            return std::unexpected(next_generated_bytes.error());
        }
        if (*next_generated_bytes > limits.maximum_generated_pixel_bytes) {
            return std::unexpected(
                mip_error(core::StatusCode::resource_exhausted,
                          "The complete mip chain exceeds its generated-pixel budget."));
        }
        generated_bytes = *next_generated_bytes;
        extents.push_back(MipExtent{
            .width = width,
            .height = height,
            .element_count = static_cast<std::size_t>(*element_count),
            .byte_count = *byte_count,
        });
    }
    return extents;
}

[[nodiscard]] std::uint64_t interval_overlap(const std::uint64_t left_begin,
                                             const std::uint64_t left_end,
                                             const std::uint64_t right_begin,
                                             const std::uint64_t right_end) noexcept {
    const auto begin = std::max(left_begin, right_begin);
    const auto end = std::min(left_end, right_end);
    return end > begin ? end - begin : 0U;
}

[[nodiscard]] core::Result<std::shared_ptr<const std::vector<TransportScalar>>>
generate_level_pixels(const HostImage& source, const MipExtent target) {
    auto target_pixels = std::make_shared<std::vector<TransportScalar>>(target.element_count);
    const auto source_width = static_cast<std::uint64_t>(source.width());
    const auto source_height = static_cast<std::uint64_t>(source.height());
    const auto target_width = static_cast<std::uint64_t>(target.width);
    const auto target_height = static_cast<std::uint64_t>(target.height);
    const auto channel_count = static_cast<std::uint64_t>(source.channel_count());
    const auto expected_weight = checked_product(
        source_width, source_height, "A generated mip footprint weight is not representable.");
    if (!expected_weight) {
        return std::unexpected(expected_weight.error());
    }

    for (auto target_y = std::uint64_t{}; target_y < target_height; ++target_y) {
        const auto target_y_begin = target_y * source_height;
        const auto target_y_end = (target_y + 1U) * source_height;
        const auto source_y_begin = target_y_begin / target_height;
        const auto source_y_end = (target_y_end + target_height - 1U) / target_height;
        for (auto target_x = std::uint64_t{}; target_x < target_width; ++target_x) {
            const auto target_x_begin = target_x * source_width;
            const auto target_x_end = (target_x + 1U) * source_width;
            const auto source_x_begin = target_x_begin / target_width;
            const auto source_x_end = (target_x_end + target_width - 1U) / target_width;
            for (auto channel = std::uint64_t{}; channel < channel_count; ++channel) {
                auto mean = ReferenceScalar{};
                auto accumulated_weight = std::uint64_t{};
                for (auto source_y = source_y_begin; source_y < source_y_end; ++source_y) {
                    const auto source_y_interval_begin = source_y * target_height;
                    const auto source_y_interval_end = (source_y + 1U) * target_height;
                    const auto y_weight =
                        interval_overlap(target_y_begin, target_y_end, source_y_interval_begin,
                                         source_y_interval_end);
                    for (auto source_x = source_x_begin; source_x < source_x_end; ++source_x) {
                        const auto source_x_interval_begin = source_x * target_width;
                        const auto source_x_interval_end = (source_x + 1U) * target_width;
                        const auto x_weight =
                            interval_overlap(target_x_begin, target_x_end, source_x_interval_begin,
                                             source_x_interval_end);
                        const auto weight =
                            checked_product(x_weight, y_weight,
                                            "A generated mip sample weight is not representable.");
                        if (!weight) {
                            return std::unexpected(weight.error());
                        }
                        if (*weight == 0U) {
                            continue;
                        }
                        const auto next_weight =
                            checked_sum(accumulated_weight, *weight,
                                        "A generated mip accumulated weight is not representable.");
                        if (!next_weight) {
                            return std::unexpected(next_weight.error());
                        }
                        const auto source_element =
                            (source_y * source_width + source_x) * channel_count + channel;
                        const auto value =
                            static_cast<ReferenceScalar>(source.pixels()[source_element]);
                        if (accumulated_weight == 0U) {
                            mean = value;
                        } else {
                            const auto fraction = static_cast<ReferenceScalar>(*weight) /
                                                  static_cast<ReferenceScalar>(*next_weight);
                            mean = std::lerp(mean, value, fraction);
                        }
                        accumulated_weight = *next_weight;
                    }
                }
                if (accumulated_weight != *expected_weight) {
                    return std::unexpected(
                        mip_error(core::StatusCode::internal_error,
                                  "A generated mip footprint does not cover its source area."));
                }
                const auto narrowed = static_cast<TransportScalar>(mean);
                if (!std::isfinite(narrowed)) {
                    return std::unexpected(mip_error(
                        core::StatusCode::incompatible,
                        "A generated mip value is not representable in transport precision."));
                }
                const auto target_element =
                    (target_y * target_width + target_x) * channel_count + channel;
                (*target_pixels)[target_element] = narrowed;
            }
        }
    }
    return std::shared_ptr<const std::vector<TransportScalar>>{std::move(target_pixels)};
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Scalar>
filter_trilinear(const HostImageMipChain& mip_chain, const Point2T<Scalar> uv, const Scalar lod,
                 const std::uint32_t channel, const TextureWrapMode u_wrap,
                 const TextureWrapMode v_wrap) {
    if (!std::isfinite(lod)) {
        return std::unexpected(mip_error(core::StatusCode::invalid_argument,
                                         "A trilinear texture LOD must be finite."));
    }
    if (mip_chain.level_count() == 0U) {
        return std::unexpected(mip_error(core::StatusCode::incompatible,
                                         "A trilinear texture requires a complete mip chain."));
    }

    const auto last_level = mip_chain.level_count() - 1U;
    const auto clamped_lod = std::clamp(lod, Scalar{}, static_cast<Scalar>(last_level));
    const auto lower_level = static_cast<std::uint32_t>(std::floor(clamped_lod));
    const auto upper_level = std::min(lower_level + 1U, last_level);
    const auto fraction = clamped_lod - static_cast<Scalar>(lower_level);

    const auto lower_image = mip_chain.level(lower_level);
    if (!lower_image) {
        return std::unexpected(lower_image.error());
    }
    const auto lower = filter_host_image_channel(**lower_image, uv, channel,
                                                 TextureFilterMode::bilinear, u_wrap, v_wrap);
    if (!lower) {
        return std::unexpected(lower.error());
    }
    if (fraction == Scalar{} || upper_level == lower_level) {
        return *lower;
    }

    const auto upper_image = mip_chain.level(upper_level);
    if (!upper_image) {
        return std::unexpected(upper_image.error());
    }
    const auto upper = filter_host_image_channel(**upper_image, uv, channel,
                                                 TextureFilterMode::bilinear, u_wrap, v_wrap);
    if (!upper) {
        return std::unexpected(upper.error());
    }
    const auto filtered = std::lerp(*lower, *upper, fraction);
    if (!std::isfinite(filtered)) {
        return std::unexpected(
            mip_error(core::StatusCode::invalid_argument,
                      "A trilinear texture value must remain finite in the requested precision."));
    }
    return filtered;
}

} // namespace

HostImageMipChain::HostImageMipChain(std::vector<HostImageHandle> levels,
                                     const std::uint64_t generated_pixel_bytes,
                                     const std::uint64_t total_pixel_bytes) noexcept
    : levels_{std::move(levels)}, generated_pixel_bytes_{generated_pixel_bytes},
      total_pixel_bytes_{total_pixel_bytes} {}

core::Result<HostImageMipChainHandle>
HostImageMipChain::generate(HostImageHandle source_image, const HostImageMipChainLimits limits) {
    try {
        if (!source_image) {
            return std::unexpected(mip_error(core::StatusCode::invalid_argument,
                                             "A mip chain source image handle must not be null."));
        }
        if (const auto source_status = validate_source_image(*source_image); !source_status) {
            return std::unexpected(source_status.error());
        }
        const auto extents = generated_extents(*source_image, limits);
        if (!extents) {
            return std::unexpected(extents.error());
        }

        auto generated_bytes = std::uint64_t{};
        auto levels = std::vector<HostImageHandle>{};
        levels.reserve(extents->size() + 1U);
        levels.push_back(source_image);
        for (const auto extent : *extents) {
            const auto pixels = generate_level_pixels(*levels.back(), extent);
            if (!pixels) {
                return std::unexpected(pixels.error());
            }
            auto channel_names = std::vector<std::string>{source_image->channel_names().begin(),
                                                          source_image->channel_names().end()};
            levels.push_back(HostImageHandle{new HostImage{
                source_image->source_path(), std::string{source_image->format_name()},
                source_image->source_color_space(), source_image->storage_color_space(), 0, 0,
                extent.width, extent.height, std::move(channel_names), *pixels}});
            generated_bytes += extent.byte_count;
        }
        const auto total_bytes = checked_sum(source_image->pixel_byte_size(), generated_bytes,
                                             "The complete mip byte count is not representable.");
        if (!total_bytes) {
            return std::unexpected(total_bytes.error());
        }
        return HostImageMipChainHandle{
            new HostImageMipChain{std::move(levels), generated_bytes, *total_bytes}};
    } catch (const std::bad_alloc&) {
        return std::unexpected(mip_error(core::StatusCode::resource_exhausted,
                                         "The complete mip chain cannot be allocated."));
    } catch (const std::length_error&) {
        return std::unexpected(
            mip_error(core::StatusCode::resource_exhausted,
                      "The complete mip chain storage size is not representable."));
    } catch (const std::exception& error) {
        return std::unexpected(mip_error(core::StatusCode::internal_error,
                                         "Mip generation failed: " + std::string{error.what()}));
    } catch (...) {
        return std::unexpected(
            mip_error(core::StatusCode::internal_error, "Mip generation failed unexpectedly."));
    }
}

HostImageHandle HostImageMipChain::source_image() const noexcept {
    return levels_.front();
}

std::uint32_t HostImageMipChain::level_count() const noexcept {
    return static_cast<std::uint32_t>(levels_.size());
}

core::Result<HostImageHandle> HostImageMipChain::level(const std::uint32_t index) const {
    if (index >= levels_.size()) {
        return std::unexpected(mip_error(core::StatusCode::invalid_argument,
                                         "A mip level index must address the complete chain."));
    }
    return levels_[index];
}

std::uint64_t HostImageMipChain::generated_pixel_bytes() const noexcept {
    return generated_pixel_bytes_;
}

std::uint64_t HostImageMipChain::total_pixel_bytes() const noexcept {
    return total_pixel_bytes_;
}

core::Result<TransportScalar>
filter_host_image_trilinear_channel(const HostImageMipChain& mip_chain, const Point2 uv,
                                    const TransportScalar lod, const std::uint32_t channel,
                                    const TextureWrapMode u_wrap, const TextureWrapMode v_wrap) {
    return filter_trilinear(mip_chain, uv, lod, channel, u_wrap, v_wrap);
}

core::Result<ReferenceScalar>
filter_host_image_trilinear_channel(const HostImageMipChain& mip_chain, const ReferencePoint2 uv,
                                    const ReferenceScalar lod, const std::uint32_t channel,
                                    const TextureWrapMode u_wrap, const TextureWrapMode v_wrap) {
    return filter_trilinear(mip_chain, uv, lod, channel, u_wrap, v_wrap);
}

} // namespace blackframe::renderer
