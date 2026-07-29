#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <Blackframe/Renderer/SampleStream.hpp>
#include <cstdint>

namespace blackframe::renderer {

// Center mode explicitly disables jitter and always selects the exact pixel center.
enum class PixelJitterMode : std::uint8_t {
    center = 0,
    uniform = 1,
};

using PixelSampleIndex = SampleStreamIndex;

// Offsets are positions inside [0, 1) in the addressed pixel. Integer pixel
// coordinates remain separate so float transport cannot round a valid sample
// in the last pixel up to the image extent.
template <GeometryScalar Scalar> struct PixelSampleT final {
    std::uint32_t pixel_x{};
    std::uint32_t pixel_y{};
    Scalar offset_x{};
    Scalar offset_y{};

    [[nodiscard]] constexpr bool operator==(const PixelSampleT&) const noexcept = default;
};

using PixelSample = PixelSampleT<TransportScalar>;
using ReferencePixelSample = PixelSampleT<ReferenceScalar>;

// The uniform mode is a stateless, indexed mapping. Replaying a seed, pixel,
// and sample index produces the same two values independently of call order.
template <GeometryScalar Scalar>
[[nodiscard]] core::Result<PixelSampleT<Scalar>> generate_pixel_sample(const PixelSampleIndex index,
                                                                       const PixelJitterMode mode) {
    switch (mode) {
    case PixelJitterMode::center:
        return PixelSampleT<Scalar>{
            .pixel_x = index.pixel_x,
            .pixel_y = index.pixel_y,
            .offset_x = Scalar{0.5},
            .offset_y = Scalar{0.5},
        };
    case PixelJitterMode::uniform:
        const auto stream = SampleStreamT<Scalar>{index};
        return PixelSampleT<Scalar>{
            .pixel_x = index.pixel_x,
            .pixel_y = index.pixel_y,
            .offset_x = stream.sample_1d(0xA24BAED4963EE407ULL),
            .offset_y = stream.sample_1d(0x9FB21C651E98DF25ULL),
        };
    }

    return std::unexpected(core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = "Unsupported pixel jitter mode.",
    });
}

} // namespace blackframe::renderer
