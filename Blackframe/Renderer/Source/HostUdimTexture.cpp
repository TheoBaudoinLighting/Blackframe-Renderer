#include <Blackframe/Renderer/HostUdimTexture.hpp>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace blackframe::renderer {
namespace {

inline constexpr auto UdimToken = std::u8string_view{u8"<UDIM>"};

[[nodiscard]] core::Error udim_error(const core::StatusCode code, std::string message) {
    return core::Error{
        .code = code,
        .message = std::move(message),
    };
}

[[nodiscard]] std::string utf8_path(const std::filesystem::path& path) {
    const auto encoded = path.u8string();
    return std::string{reinterpret_cast<const char*>(encoded.data()), encoded.size()};
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<UdimAddressT<Scalar>> resolve_address(const Point2T<Scalar> uv) try {
    if (!std::isfinite(uv.x) || !std::isfinite(uv.y) || uv.x < Scalar{} || uv.y < Scalar{}) {
        return std::unexpected(udim_error(core::StatusCode::invalid_argument,
                                          "UDIM coordinates must be finite and non-negative."));
    }

    // Widening float to double is exact, and every uint32 value is exactly representable in
    // double. This avoids accepting a rounded float version of the maximum legal row.
    const auto column_value = std::floor(static_cast<ReferenceScalar>(uv.x));
    const auto row_value = std::floor(static_cast<ReferenceScalar>(uv.y));
    if (column_value > static_cast<ReferenceScalar>(UdimMaximumColumn)) {
        return std::unexpected(
            udim_error(core::StatusCode::invalid_argument,
                       "A UDIM column must be in [0, 9] to keep tile numbering unambiguous."));
    }

    const auto column = static_cast<std::uint32_t>(column_value);
    constexpr auto maximum_number = std::numeric_limits<std::uint32_t>::max();
    const auto maximum_row = (maximum_number - UdimFirstTileNumber - column) / UdimColumnsPerRow;
    if (row_value > static_cast<ReferenceScalar>(maximum_row)) {
        return std::unexpected(
            udim_error(core::StatusCode::invalid_argument,
                       "The UDIM row would produce an unrepresentable tile number."));
    }

    const auto row = static_cast<std::uint32_t>(row_value);
    const auto number_wide = static_cast<std::uint64_t>(UdimFirstTileNumber) +
                             static_cast<std::uint64_t>(column) +
                             static_cast<std::uint64_t>(UdimColumnsPerRow) * row;
    if (number_wide > maximum_number) {
        return std::unexpected(udim_error(core::StatusCode::invalid_argument,
                                          "The UDIM tile number is not representable."));
    }

    const auto local_uv = Point2T<Scalar>{
        .x = uv.x - static_cast<Scalar>(column),
        .y = uv.y - static_cast<Scalar>(row),
    };
    if (!std::isfinite(local_uv.x) || !std::isfinite(local_uv.y) || local_uv.x < Scalar{} ||
        local_uv.x >= Scalar{1} || local_uv.y < Scalar{} || local_uv.y >= Scalar{1}) {
        return std::unexpected(
            udim_error(core::StatusCode::invalid_argument,
                       "The UDIM local coordinates are not representable in their precision."));
    }

    return UdimAddressT<Scalar>{
        .tile =
            UdimTile{
                .number = static_cast<std::uint32_t>(number_wide),
                .column = column,
                .row = row,
            },
        .local_uv = local_uv,
    };
} catch (const std::bad_alloc&) {
    return std::unexpected(udim_error(core::StatusCode::resource_exhausted,
                                      "UDIM address resolution exhausted host memory."));
} catch (const std::length_error&) {
    return std::unexpected(udim_error(core::StatusCode::resource_exhausted,
                                      "A UDIM address diagnostic is too large to represent."));
} catch (const std::exception& error) {
    return std::unexpected(
        udim_error(core::StatusCode::internal_error,
                   "UDIM address resolution failed: " + std::string{error.what()}));
} catch (...) {
    return std::unexpected(udim_error(core::StatusCode::internal_error,
                                      "UDIM address resolution failed unexpectedly."));
}

[[nodiscard]] core::Result<std::filesystem::path>
substitute_tile_number(const std::u8string& encoded_pattern, const std::size_t token_offset,
                       const UdimTile tile) try {
    auto encoded_path = encoded_pattern;
    const auto decimal_number = std::to_string(tile.number);
    auto encoded_number = std::u8string{};
    encoded_number.reserve(decimal_number.size());
    for (const auto digit : decimal_number) {
        encoded_number.push_back(static_cast<char8_t>(digit));
    }
    encoded_path.replace(token_offset, UdimToken.size(), encoded_number);
    return std::filesystem::path{encoded_path};
} catch (const std::bad_alloc&) {
    return std::unexpected(udim_error(core::StatusCode::resource_exhausted,
                                      "The concrete UDIM tile path cannot be allocated."));
} catch (const std::length_error&) {
    return std::unexpected(udim_error(core::StatusCode::resource_exhausted,
                                      "The concrete UDIM tile path length is not representable."));
} catch (const std::filesystem::filesystem_error& error) {
    return std::unexpected(
        udim_error(core::StatusCode::platform_error,
                   "The concrete UDIM tile path cannot be formed: " + std::string{error.what()}));
} catch (const std::exception& error) {
    return std::unexpected(
        udim_error(core::StatusCode::internal_error,
                   "The concrete UDIM tile path cannot be formed: " + std::string{error.what()}));
} catch (...) {
    return std::unexpected(udim_error(core::StatusCode::internal_error,
                                      "The concrete UDIM tile path cannot be formed."));
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<HostUdimResolvedTileT<Scalar>>
resolve_tile(const std::u8string& encoded_pattern, const std::size_t token_offset,
             const TextureColorSpace source_color_space, HostImageCache& cache,
             const UdimAddressT<Scalar> address) try {
    const auto concrete_path = substitute_tile_number(encoded_pattern, token_offset, address.tile);
    if (!concrete_path) {
        return std::unexpected(concrete_path.error());
    }

    auto image = cache.load(*concrete_path, source_color_space);
    if (!image) {
        auto message = "UDIM tile " + std::to_string(address.tile.number) + " at '" +
                       utf8_path(*concrete_path) +
                       "' could not be loaded: " + image.error().message;
        return std::unexpected(udim_error(image.error().code, std::move(message)));
    }
    return HostUdimResolvedTileT<Scalar>{
        .address = address,
        .image = std::move(*image),
    };
} catch (const std::bad_alloc&) {
    return std::unexpected(udim_error(core::StatusCode::resource_exhausted,
                                      "UDIM tile resolution exhausted host memory."));
} catch (const std::length_error&) {
    return std::unexpected(udim_error(core::StatusCode::resource_exhausted,
                                      "A UDIM tile diagnostic is too large to represent."));
} catch (const std::filesystem::filesystem_error& error) {
    return std::unexpected(
        udim_error(core::StatusCode::platform_error,
                   "UDIM tile resolution failed for its path: " + std::string{error.what()}));
} catch (const std::exception& error) {
    return std::unexpected(udim_error(core::StatusCode::internal_error,
                                      "UDIM tile resolution failed: " + std::string{error.what()}));
} catch (...) {
    return std::unexpected(
        udim_error(core::StatusCode::internal_error, "UDIM tile resolution failed unexpectedly."));
}

} // namespace

core::Result<UdimAddress> resolve_udim_address(const Point2 uv) {
    return resolve_address(uv);
}

core::Result<ReferenceUdimAddress> resolve_udim_address(const ReferencePoint2 uv) {
    return resolve_address(uv);
}

core::Result<HostUdimTexture>
HostUdimTexture::create(const std::filesystem::path& absolute_pattern,
                        const TextureColorSpace source_color_space) try {
    if (!is_valid_texture_color_space(source_color_space)) {
        return std::unexpected(
            udim_error(core::StatusCode::invalid_argument,
                       "A host UDIM texture requires a supported explicit color-space tag."));
    }
    if (absolute_pattern.empty() || !absolute_pattern.is_absolute()) {
        return std::unexpected(
            udim_error(core::StatusCode::invalid_argument,
                       "A host UDIM texture pattern must be explicit and absolute."));
    }

    auto normalized_pattern = absolute_pattern.lexically_normal();
    auto encoded_pattern = normalized_pattern.u8string();
    const auto token_offset = encoded_pattern.find(UdimToken);
    if (token_offset == std::u8string::npos ||
        encoded_pattern.find(UdimToken, token_offset + UdimToken.size()) != std::u8string::npos) {
        return std::unexpected(
            udim_error(core::StatusCode::invalid_argument,
                       "A host UDIM texture pattern must contain exactly one <UDIM> token."));
    }

    return HostUdimTexture{std::move(normalized_pattern), std::move(encoded_pattern), token_offset,
                           source_color_space};
} catch (const std::bad_alloc&) {
    return std::unexpected(udim_error(core::StatusCode::resource_exhausted,
                                      "The host UDIM texture pattern cannot be allocated."));
} catch (const std::length_error&) {
    return std::unexpected(
        udim_error(core::StatusCode::resource_exhausted,
                   "The host UDIM texture pattern length is not representable."));
} catch (const std::filesystem::filesystem_error& error) {
    return std::unexpected(udim_error(core::StatusCode::platform_error,
                                      "The host UDIM texture pattern cannot be normalized: " +
                                          std::string{error.what()}));
} catch (const std::exception& error) {
    return std::unexpected(
        udim_error(core::StatusCode::internal_error,
                   "The host UDIM texture cannot be created: " + std::string{error.what()}));
} catch (...) {
    return std::unexpected(
        udim_error(core::StatusCode::internal_error, "The host UDIM texture cannot be created."));
}

HostUdimTexture::HostUdimTexture(std::filesystem::path pattern, std::u8string encoded_pattern,
                                 const std::size_t token_offset,
                                 const TextureColorSpace source_color_space) noexcept
    : pattern_{std::move(pattern)}, encoded_pattern_{std::move(encoded_pattern)},
      token_offset_{token_offset}, source_color_space_{source_color_space} {}

const std::filesystem::path& HostUdimTexture::pattern() const noexcept {
    return pattern_;
}

TextureColorSpace HostUdimTexture::source_color_space() const noexcept {
    return source_color_space_;
}

core::Result<HostUdimResolvedTile> HostUdimTexture::resolve(HostImageCache& cache,
                                                            const Point2 uv) const {
    const auto address = resolve_udim_address(uv);
    if (!address) {
        return std::unexpected(address.error());
    }
    return resolve_tile(encoded_pattern_, token_offset_, source_color_space_, cache, *address);
}

core::Result<ReferenceHostUdimResolvedTile>
HostUdimTexture::resolve(HostImageCache& cache, const ReferencePoint2 uv) const {
    const auto address = resolve_udim_address(uv);
    if (!address) {
        return std::unexpected(address.error());
    }
    return resolve_tile(encoded_pattern_, token_offset_, source_color_space_, cache, *address);
}

} // namespace blackframe::renderer
