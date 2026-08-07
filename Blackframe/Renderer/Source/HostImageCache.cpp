#include <Blackframe/Renderer/HostImageCache.hpp>
#include <Blackframe/Renderer/NumericConversion.hpp>
#include <OpenImageIO/imageio.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>

namespace blackframe::renderer {
namespace {

struct DecodedImage final {
    std::string format_name;
    std::int32_t origin_x{};
    std::int32_t origin_y{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::vector<std::string> channel_names;
    std::vector<TransportScalar> pixels;
};

using HostImageCacheKey = std::pair<std::filesystem::path, TextureColorSpace>;

struct ColorChannelIndices final {
    std::size_t red{};
    std::size_t green{};
    std::size_t blue{};
};

struct WorkingPixelStorage final {
    TextureColorSpace color_space{TextureColorSpace::data};
    std::shared_ptr<const std::vector<TransportScalar>> pixels;
    std::uint64_t additional_pixel_bytes{};
};

[[nodiscard]] core::Error cache_error(const core::StatusCode code, std::string message) {
    return core::Error{
        .code = code,
        .message = std::move(message),
    };
}

[[nodiscard]] core::Status validate_limits(const HostImageCacheLimits limits) {
    if (limits.maximum_width == 0U || limits.maximum_height == 0U ||
        limits.maximum_channels == 0U || limits.maximum_image_pixel_bytes == 0U ||
        limits.maximum_resident_pixel_bytes == 0U || limits.maximum_entries == 0U) {
        return std::unexpected(
            cache_error(core::StatusCode::invalid_argument,
                        "Host image cache limits must all be finite non-zero capacities."));
    }
    if (limits.maximum_image_pixel_bytes > limits.maximum_resident_pixel_bytes) {
        return std::unexpected(cache_error(
            core::StatusCode::invalid_argument,
            "The host image byte limit must not exceed the cache resident-byte limit."));
    }
    return {};
}

[[nodiscard]] core::Status validate_source_color_space(const TextureColorSpace color_space) {
    if (!is_valid_texture_color_space(color_space)) {
        return std::unexpected(
            cache_error(core::StatusCode::invalid_argument,
                        "A host image source color-space tag must be a supported explicit value."));
    }
    return {};
}

[[nodiscard]] core::Result<ColorChannelIndices>
color_channel_indices(const std::span<const std::string> channel_names) {
    constexpr auto required_names =
        std::array{std::string_view{"R"}, std::string_view{"G"}, std::string_view{"B"}};
    auto indices = std::array<std::size_t, required_names.size()>{};
    auto found = std::array<bool, required_names.size()>{};

    for (auto channel = std::size_t{}; channel < channel_names.size(); ++channel) {
        for (auto required = std::size_t{}; required < required_names.size(); ++required) {
            if (channel_names[channel] != required_names[required]) {
                continue;
            }
            if (found[required]) {
                return std::unexpected(cache_error(
                    core::StatusCode::incompatible,
                    "A color-tagged host image must contain exactly one R, G, and B channel."));
            }
            found[required] = true;
            indices[required] = channel;
        }
    }
    if (!std::ranges::all_of(found, [](const bool value) { return value; })) {
        return std::unexpected(
            cache_error(core::StatusCode::incompatible,
                        "A color-tagged host image must contain exactly one R, G, and B channel."));
    }
    return ColorChannelIndices{.red = indices[0], .green = indices[1], .blue = indices[2]};
}

[[nodiscard]] core::Result<TransportScalar> decode_srgb_component(const TransportScalar encoded) {
    const auto reference = to_reference_scalar(encoded);
    const auto linear =
        reference <= ReferenceScalar{0.04045}
            ? reference / ReferenceScalar{12.92}
            : std::pow((reference + ReferenceScalar{0.055}) / ReferenceScalar{1.055},
                       ReferenceScalar{2.4});
    const auto narrowed = to_transport_scalar(linear);
    if (!narrowed) {
        return std::unexpected(
            cache_error(core::StatusCode::incompatible,
                        "An sRGB texture channel cannot be represented in transport precision."));
    }
    return *narrowed;
}

[[nodiscard]] core::Result<WorkingPixelStorage>
convert_to_working_color_space(const std::shared_ptr<const DecodedImage>& source,
                               const TextureColorSpace source_color_space,
                               const std::uint64_t remaining_pixel_bytes) {
    const auto source_pixels =
        std::shared_ptr<const std::vector<TransportScalar>>{source, &source->pixels};
    if (source_color_space == TextureColorSpace::data) {
        return WorkingPixelStorage{
            .color_space = TextureColorSpace::data,
            .pixels = source_pixels,
        };
    }

    const auto channels = color_channel_indices(source->channel_names);
    if (!channels) {
        return std::unexpected(channels.error());
    }
    if (source_color_space == TextureColorSpace::scene_linear_srgb) {
        return WorkingPixelStorage{
            .color_space = TextureWorkingColorSpace,
            .pixels = source_pixels,
        };
    }

    const auto byte_size =
        static_cast<std::uint64_t>(source->pixels.size()) * sizeof(TransportScalar);
    if (byte_size > remaining_pixel_bytes) {
        return std::unexpected(cache_error(
            core::StatusCode::resource_exhausted,
            "The host image cache resident pixel budget cannot hold an sRGB conversion."));
    }

    auto converted_pixels = std::make_shared<std::vector<TransportScalar>>(source->pixels);
    const auto channel_count = source->channel_names.size();
    const auto pixel_count = converted_pixels->size() / channel_count;
    for (auto pixel = std::size_t{}; pixel < pixel_count; ++pixel) {
        const auto base = pixel * channel_count;
        for (const auto channel : {channels->red, channels->green, channels->blue}) {
            const auto converted = decode_srgb_component((*converted_pixels)[base + channel]);
            if (!converted) {
                return std::unexpected(converted.error());
            }
            (*converted_pixels)[base + channel] = *converted;
        }
    }
    return WorkingPixelStorage{
        .color_space = TextureWorkingColorSpace,
        .pixels = std::move(converted_pixels),
        .additional_pixel_bytes = byte_size,
    };
}

[[nodiscard]] core::Result<std::filesystem::path>
canonical_input_path(const std::filesystem::path& absolute_path) {
    if (absolute_path.empty() || !absolute_path.is_absolute()) {
        return std::unexpected(
            cache_error(core::StatusCode::invalid_argument,
                        "A host image source path must be explicit and absolute."));
    }

    auto error = std::error_code{};
    const auto exists = std::filesystem::exists(absolute_path, error);
    if (error) {
        return std::unexpected(
            cache_error(core::StatusCode::platform_error,
                        "The host image source path cannot be inspected: " + error.message()));
    }
    if (!exists) {
        return std::unexpected(
            cache_error(core::StatusCode::not_found, "The host image source file does not exist."));
    }
    if (!std::filesystem::is_regular_file(absolute_path, error) || error) {
        return std::unexpected(cache_error(
            error ? core::StatusCode::platform_error : core::StatusCode::invalid_argument,
            error ? "The host image source type cannot be inspected: " + error.message()
                  : "The host image source path must name a regular file."));
    }

    auto canonical = std::filesystem::canonical(absolute_path, error);
    if (error) {
        return std::unexpected(
            cache_error(core::StatusCode::platform_error,
                        "The host image source path cannot be canonicalized: " + error.message()));
    }
    return canonical;
}

[[nodiscard]] std::string utf8_path(const std::filesystem::path& path) {
    const auto encoded = path.u8string();
    return std::string{reinterpret_cast<const char*>(encoded.data()), encoded.size()};
}

[[nodiscard]] core::Result<std::string_view>
reader_name_for_path(const std::filesystem::path& canonical_path) {
    const auto encoded = canonical_path.extension().u8string();
    auto extension = std::string{reinterpret_cast<const char*>(encoded.data()), encoded.size()};
    std::ranges::transform(extension, extension.begin(), [](const char value) {
        if (value >= 'A' && value <= 'Z') {
            return static_cast<char>(value - 'A' + 'a');
        }
        return value;
    });

    if (extension == ".exr") {
        return std::string_view{"openexr"};
    }
    if (extension == ".png") {
        return std::string_view{"png"};
    }
    if (extension == ".pbm" || extension == ".pgm" || extension == ".ppm" || extension == ".pnm" ||
        extension == ".pfm") {
        return std::string_view{"pnm"};
    }
    return std::unexpected(
        cache_error(core::StatusCode::incompatible,
                    "The host image suffix does not select a supported embedded reader."));
}

[[nodiscard]] core::Result<std::size_t>
validated_element_count(const OIIO::ImageSpec& specification, const HostImageCacheLimits limits,
                        const std::uint64_t remaining_pixel_bytes) {
    if (specification.deep) {
        return std::unexpected(
            cache_error(core::StatusCode::incompatible,
                        "Deep images are not supported by the host image cache."));
    }
    if (specification.width <= 0 || specification.height <= 0 || specification.depth != 1 ||
        specification.nchannels <= 0) {
        return std::unexpected(
            cache_error(core::StatusCode::incompatible,
                        "The host image cache requires a non-empty two-dimensional image."));
    }

    const auto width = static_cast<std::uint64_t>(specification.width);
    const auto height = static_cast<std::uint64_t>(specification.height);
    const auto channels = static_cast<std::uint64_t>(specification.nchannels);
    if (width > limits.maximum_width || height > limits.maximum_height ||
        channels > limits.maximum_channels) {
        return std::unexpected(cache_error(
            core::StatusCode::resource_exhausted,
            "The host image dimensions or channel count exceed the configured limits."));
    }
    if (width > std::numeric_limits<std::uint64_t>::max() / height ||
        width * height > std::numeric_limits<std::uint64_t>::max() / channels) {
        return std::unexpected(cache_error(core::StatusCode::resource_exhausted,
                                           "The host image element count is not representable."));
    }
    const auto element_count = width * height * channels;
    if (element_count > std::numeric_limits<std::uint64_t>::max() / sizeof(TransportScalar)) {
        return std::unexpected(cache_error(core::StatusCode::resource_exhausted,
                                           "The host image byte count is not representable."));
    }
    const auto byte_count = element_count * sizeof(TransportScalar);
    if (byte_count > limits.maximum_image_pixel_bytes ||
        element_count > std::numeric_limits<std::size_t>::max()) {
        return std::unexpected(cache_error(core::StatusCode::resource_exhausted,
                                           "The host image exceeds the configured image budget."));
    }
    if (byte_count > remaining_pixel_bytes) {
        return std::unexpected(
            cache_error(core::StatusCode::resource_exhausted,
                        "The host image cache resident pixel budget cannot hold this image."));
    }
    return static_cast<std::size_t>(element_count);
}

[[nodiscard]] core::Result<DecodedImage> decode_image(const std::filesystem::path& canonical_path,
                                                      const HostImageCacheLimits limits,
                                                      const std::uint64_t remaining_pixel_bytes) {
    const auto reader_name = reader_name_for_path(canonical_path);
    if (!reader_name) {
        return std::unexpected(reader_name.error());
    }

    auto configuration = OIIO::ImageSpec{};
    configuration.attribute("oiio:UnassociatedAlpha", 1);
    auto input = OIIO::ImageInput::create(*reader_name, false, &configuration);
    if (!input) {
        auto detail = OIIO::geterror();
        if (detail.empty()) {
            detail = "OpenImageIO did not provide the required embedded image reader.";
        }
        return std::unexpected(
            cache_error(core::StatusCode::incompatible,
                        "The host image reader cannot be created: " + std::move(detail)));
    }

    auto specification = OIIO::ImageSpec{};
    if (!input->open(utf8_path(canonical_path), specification, configuration)) {
        auto detail = input->geterror();
        if (detail.empty()) {
            detail = "The selected reader rejected the file payload.";
        }
        return std::unexpected(cache_error(
            core::StatusCode::incompatible,
            "The host image source cannot be opened by its selected reader: " + std::move(detail)));
    }

    const auto element_count =
        validated_element_count(specification, limits, remaining_pixel_bytes);
    if (!element_count) {
        static_cast<void>(input->close());
        return std::unexpected(element_count.error());
    }
    if (specification.channelnames.size() != static_cast<std::size_t>(specification.nchannels)) {
        static_cast<void>(input->close());
        return std::unexpected(cache_error(core::StatusCode::incompatible,
                                           "The host image channel metadata is incomplete."));
    }
    constexpr auto maximum_channel_name_bytes = std::size_t{255U};
    if (!std::ranges::all_of(specification.channelnames,
                             [maximum_channel_name_bytes](const auto& name) {
                                 return !name.empty() && name.size() <= maximum_channel_name_bytes;
                             })) {
        static_cast<void>(input->close());
        return std::unexpected(
            cache_error(core::StatusCode::incompatible,
                        "The host image contains an empty or oversized channel name."));
    }

    auto decoded = DecodedImage{};
    decoded.format_name = input->format_name();
    decoded.origin_x = static_cast<std::int32_t>(specification.x);
    decoded.origin_y = static_cast<std::int32_t>(specification.y);
    decoded.width = static_cast<std::uint32_t>(specification.width);
    decoded.height = static_cast<std::uint32_t>(specification.height);
    decoded.channel_names.assign(specification.channelnames.begin(),
                                 specification.channelnames.end());
    decoded.pixels.resize(*element_count);

    const auto read = input->read_image(
        0, 0, 0, specification.nchannels,
        OIIO::span<TransportScalar>{decoded.pixels.data(), decoded.pixels.size()});
    auto detail = read ? std::string{} : input->geterror();
    const auto closed = input->close();
    if (!read) {
        if (detail.empty()) {
            detail = "OpenImageIO rejected the pixel payload.";
        }
        return std::unexpected(cache_error(core::StatusCode::platform_error,
                                           "The host image pixels cannot be read: " + detail));
    }
    if (!closed) {
        detail = input->geterror();
        return std::unexpected(
            cache_error(core::StatusCode::platform_error,
                        "The host image source cannot be closed cleanly: " + detail));
    }
    if (decoded.format_name.empty()) {
        return std::unexpected(cache_error(core::StatusCode::incompatible,
                                           "The host image format name is unavailable."));
    }
    if (decoded.format_name != *reader_name) {
        return std::unexpected(cache_error(
            core::StatusCode::incompatible,
            "The host image reader identity does not match the selected embedded format."));
    }
    if (!std::ranges::all_of(decoded.pixels,
                             [](const auto value) { return std::isfinite(value); })) {
        return std::unexpected(cache_error(core::StatusCode::incompatible,
                                           "The host image contains a non-finite channel value."));
    }
    return decoded;
}

} // namespace

struct HostImageCache::Impl final {
    explicit Impl(const HostImageCacheLimits configured_limits) noexcept
        : configured_limits{configured_limits} {}

    HostImageCacheLimits configured_limits;
    mutable std::mutex mutex;
    std::map<std::filesystem::path, std::shared_ptr<const DecodedImage>> source_images;
    std::map<HostImageCacheKey, HostImageHandle> images;
    std::uint64_t resident_pixel_byte_count{};
};

HostImage::HostImage(std::filesystem::path source_path, std::string format_name,
                     const TextureColorSpace source_color_space,
                     const TextureColorSpace storage_color_space, const std::int32_t origin_x,
                     const std::int32_t origin_y, const std::uint32_t width,
                     const std::uint32_t height, std::vector<std::string> channel_names,
                     std::shared_ptr<const std::vector<TransportScalar>> pixels) noexcept
    : source_path_{std::move(source_path)}, format_name_{std::move(format_name)},
      source_color_space_{source_color_space}, storage_color_space_{storage_color_space},
      origin_x_{origin_x}, origin_y_{origin_y}, width_{width}, height_{height},
      channel_names_{std::move(channel_names)}, pixels_{std::move(pixels)} {}

const std::filesystem::path& HostImage::source_path() const noexcept {
    return source_path_;
}

std::string_view HostImage::format_name() const noexcept {
    return format_name_;
}

TextureColorSpace HostImage::source_color_space() const noexcept {
    return source_color_space_;
}

TextureColorSpace HostImage::storage_color_space() const noexcept {
    return storage_color_space_;
}

std::int32_t HostImage::origin_x() const noexcept {
    return origin_x_;
}

std::int32_t HostImage::origin_y() const noexcept {
    return origin_y_;
}

std::uint32_t HostImage::width() const noexcept {
    return width_;
}

std::uint32_t HostImage::height() const noexcept {
    return height_;
}

std::uint32_t HostImage::channel_count() const noexcept {
    return static_cast<std::uint32_t>(channel_names_.size());
}

std::span<const std::string> HostImage::channel_names() const noexcept {
    return channel_names_;
}

std::span<const TransportScalar> HostImage::pixels() const noexcept {
    return *pixels_;
}

std::uint64_t HostImage::pixel_byte_size() const noexcept {
    return static_cast<std::uint64_t>(pixels_->size()) * sizeof(TransportScalar);
}

core::Result<HostImageCache> HostImageCache::create(const HostImageCacheLimits limits) {
    try {
        if (const auto status = validate_limits(limits); !status) {
            return std::unexpected(status.error());
        }
        return HostImageCache{std::make_unique<Impl>(limits)};
    } catch (const std::bad_alloc&) {
        return std::unexpected(cache_error(core::StatusCode::resource_exhausted,
                                           "The host image cache cannot allocate its state."));
    } catch (const std::exception& error) {
        return std::unexpected(
            cache_error(core::StatusCode::internal_error,
                        "The host image cache cannot initialize: " + std::string{error.what()}));
    } catch (...) {
        return std::unexpected(cache_error(core::StatusCode::internal_error,
                                           "The host image cache cannot initialize."));
    }
}

HostImageCache::HostImageCache(std::unique_ptr<Impl> implementation) noexcept
    : implementation_{std::move(implementation)} {}

HostImageCache::~HostImageCache() = default;
HostImageCache::HostImageCache(HostImageCache&&) noexcept = default;
HostImageCache& HostImageCache::operator=(HostImageCache&&) noexcept = default;

core::Result<HostImageHandle> HostImageCache::load(const std::filesystem::path& absolute_path,
                                                   const TextureColorSpace source_color_space) {
    try {
        if (!implementation_) {
            return std::unexpected(cache_error(core::StatusCode::incompatible,
                                               "The host image cache was moved from."));
        }
        if (const auto status = validate_source_color_space(source_color_space); !status) {
            return std::unexpected(status.error());
        }
        if (absolute_path.empty() || !absolute_path.is_absolute()) {
            return std::unexpected(
                cache_error(core::StatusCode::invalid_argument,
                            "A host image source path must be explicit and absolute."));
        }
        const auto normalized_path = absolute_path.lexically_normal();

        auto lock = std::scoped_lock{implementation_->mutex};
        if (const auto cached = implementation_->images.find(
                HostImageCacheKey{normalized_path, source_color_space});
            cached != implementation_->images.end()) {
            return cached->second;
        }

        auto canonical_path = normalized_path;
        auto source_image = std::shared_ptr<const DecodedImage>{};
        if (const auto source = implementation_->source_images.find(normalized_path);
            source != implementation_->source_images.end()) {
            canonical_path = source->first;
            source_image = source->second;
        } else {
            const auto canonical = canonical_input_path(normalized_path);
            if (!canonical) {
                return std::unexpected(canonical.error());
            }
            canonical_path = *canonical;
        }

        const auto cache_key = HostImageCacheKey{canonical_path, source_color_space};
        if (const auto found = implementation_->images.find(cache_key);
            found != implementation_->images.end()) {
            return found->second;
        }
        if (implementation_->images.size() == implementation_->configured_limits.maximum_entries) {
            return std::unexpected(cache_error(core::StatusCode::resource_exhausted,
                                               "The host image cache entry limit is exhausted."));
        }

        if (!source_image) {
            if (const auto source = implementation_->source_images.find(canonical_path);
                source != implementation_->source_images.end()) {
                source_image = source->second;
            }
        }

        auto new_source_byte_size = std::uint64_t{};
        if (!source_image) {
            const auto remaining_pixel_bytes =
                implementation_->configured_limits.maximum_resident_pixel_bytes -
                implementation_->resident_pixel_byte_count;
            auto decoded = decode_image(canonical_path, implementation_->configured_limits,
                                        remaining_pixel_bytes);
            if (!decoded) {
                return std::unexpected(decoded.error());
            }
            new_source_byte_size =
                static_cast<std::uint64_t>(decoded->pixels.size()) * sizeof(TransportScalar);
            source_image = std::make_shared<DecodedImage>(std::move(*decoded));
        }

        const auto remaining_pixel_bytes =
            implementation_->configured_limits.maximum_resident_pixel_bytes -
            implementation_->resident_pixel_byte_count - new_source_byte_size;
        const auto working_pixels =
            convert_to_working_color_space(source_image, source_color_space, remaining_pixel_bytes);
        if (!working_pixels) {
            return std::unexpected(working_pixels.error());
        }

        auto image = HostImageHandle{
            new HostImage{canonical_path, source_image->format_name, source_color_space,
                          working_pixels->color_space, source_image->origin_x,
                          source_image->origin_y, source_image->width, source_image->height,
                          source_image->channel_names, working_pixels->pixels}};
        if (new_source_byte_size != 0U) {
            const auto source_insertion =
                implementation_->source_images.emplace(canonical_path, source_image);
            if (!source_insertion.second) {
                return std::unexpected(
                    cache_error(core::StatusCode::internal_error,
                                "A host image source snapshot was inserted concurrently."));
            }
            implementation_->resident_pixel_byte_count += new_source_byte_size;
        }
        const auto insertion = implementation_->images.emplace(cache_key, image);
        if (!insertion.second) {
            return std::unexpected(
                cache_error(core::StatusCode::internal_error,
                            "A host image cache key was inserted concurrently."));
        }
        implementation_->resident_pixel_byte_count += working_pixels->additional_pixel_bytes;
        return image;
    } catch (const std::filesystem::filesystem_error& error) {
        return std::unexpected(
            cache_error(core::StatusCode::platform_error,
                        "The host image path failed: " + std::string{error.what()}));
    } catch (const std::bad_alloc&) {
        return std::unexpected(cache_error(core::StatusCode::resource_exhausted,
                                           "The host image cache exhausted memory."));
    } catch (const std::length_error&) {
        return std::unexpected(cache_error(core::StatusCode::resource_exhausted,
                                           "The host image storage size is not representable."));
    } catch (const std::exception& error) {
        return std::unexpected(
            cache_error(core::StatusCode::internal_error,
                        "The host image load failed: " + std::string{error.what()}));
    } catch (...) {
        return std::unexpected(cache_error(core::StatusCode::internal_error,
                                           "The host image load failed unexpectedly."));
    }
}

core::Result<std::size_t> HostImageCache::entry_count() const {
    if (!implementation_) {
        return std::unexpected(
            cache_error(core::StatusCode::incompatible, "The host image cache was moved from."));
    }
    try {
        auto lock = std::scoped_lock{implementation_->mutex};
        return implementation_->images.size();
    } catch (const std::system_error& error) {
        return std::unexpected(
            cache_error(core::StatusCode::platform_error,
                        "The host image cache cannot lock its state: " + error.code().message()));
    }
}

core::Result<std::uint64_t> HostImageCache::resident_pixel_bytes() const {
    if (!implementation_) {
        return std::unexpected(
            cache_error(core::StatusCode::incompatible, "The host image cache was moved from."));
    }
    try {
        auto lock = std::scoped_lock{implementation_->mutex};
        return implementation_->resident_pixel_byte_count;
    } catch (const std::system_error& error) {
        return std::unexpected(
            cache_error(core::StatusCode::platform_error,
                        "The host image cache cannot lock its state: " + error.code().message()));
    }
}

core::Result<HostImageCacheLimits> HostImageCache::limits() const {
    if (!implementation_) {
        return std::unexpected(
            cache_error(core::StatusCode::incompatible, "The host image cache was moved from."));
    }
    try {
        auto lock = std::scoped_lock{implementation_->mutex};
        return implementation_->configured_limits;
    } catch (const std::system_error& error) {
        return std::unexpected(
            cache_error(core::StatusCode::platform_error,
                        "The host image cache cannot lock its state: " + error.code().message()));
    }
}

} // namespace blackframe::renderer
