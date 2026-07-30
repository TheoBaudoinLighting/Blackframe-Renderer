#include <Blackframe/Renderer/SobolSampling.hpp>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <string>

namespace blackframe::renderer {
namespace {

using SobolPackedParameters = std::array<std::uint64_t, 3>;

// Packed from dimensions 2 through 21201 of Joe and Kuo's new-joe-kuo-6.21201 data.
// Source: https://web.maths.unsw.edu.au/~fkuo/sobol/new-joe-kuo-6.21201
// Source SHA-256: 68eedd2a4e3b659b9695e7aff0f8ac68718bcf620730fc3d3a8c65df2a067441.
// The applicable BSD licence is retained in Licenses/JoeKuoSobol.txt.
constexpr auto packed_parameters = std::array<SobolPackedParameters, SobolDimensionCount - 1>{{
#include "SobolDirectionNumbers.inc"
}};

// Each entry is the exclusive row end for one primitive-polynomial degree, starting at degree 1.
constexpr auto degree_row_ends = std::array<std::uint32_t, 18>{
    1, 2, 4, 6, 12, 18, 36, 52, 100, 160, 336, 480, 1'110, 1'866, 3'666, 5'714, 13'424, 21'200,
};

[[nodiscard]] constexpr std::uint32_t degree_for_row(const std::uint64_t row) noexcept {
    auto degree = std::uint32_t{1};
    while (row >= degree_row_ends[degree - 1]) {
        ++degree;
    }
    return degree;
}

[[nodiscard]] constexpr std::uint64_t unpack_bits(const SobolPackedParameters& parameters,
                                                  const std::uint32_t offset,
                                                  const std::uint32_t width) noexcept {
    if (width == 0) {
        return 0;
    }

    const auto word = offset / 64U;
    const auto shift = offset % 64U;
    auto value = parameters[word] >> shift;
    if (shift + width > 64U) {
        value |= parameters[word + 1] << (64U - shift);
    }
    return value & ((std::uint64_t{1} << width) - 1U);
}

[[nodiscard]] constexpr std::array<std::uint64_t, SobolDirectionBitCount>
direction_numbers(const std::uint64_t sobol_dimension) noexcept {
    auto directions = std::array<std::uint64_t, SobolDirectionBitCount>{};
    if (sobol_dimension == 0) {
        for (auto column = std::uint32_t{0}; column < SobolDirectionBitCount; ++column) {
            directions[column] = std::uint64_t{1} << (63U - column);
        }
        return directions;
    }

    const auto row = sobol_dimension - 1;
    const auto degree = degree_for_row(row);
    const auto& parameters = packed_parameters[static_cast<std::size_t>(row)];
    const auto coefficient = unpack_bits(parameters, 0, degree - 1);
    auto offset = degree - 1;

    for (auto column = std::uint32_t{0}; column < degree; ++column) {
        const auto width = column + 1;
        const auto initial_direction = unpack_bits(parameters, offset, width);
        offset += width;
        directions[column] = initial_direction << (63U - column);
    }

    for (auto column = degree; column < SobolDirectionBitCount; ++column) {
        auto direction = directions[column - degree] ^ (directions[column - degree] >> degree);
        for (auto coefficient_index = std::uint32_t{1}; coefficient_index < degree;
             ++coefficient_index) {
            const auto coefficient_shift = degree - 1U - coefficient_index;
            if (((coefficient >> coefficient_shift) & 1U) != 0) {
                direction ^= directions[column - coefficient_index];
            }
        }
        directions[column] = direction;
    }
    return directions;
}

static_assert(packed_parameters.size() + 1 == SobolDimensionCount);
static_assert(degree_row_ends.back() == packed_parameters.size());
static_assert(degree_for_row(0) == 1);
static_assert(degree_for_row(packed_parameters.size() - 1) == 18);

} // namespace

core::Result<std::uint64_t> sobol_sample_bits(const std::uint64_t sample_index,
                                              const std::uint64_t sobol_dimension) {
    if (sobol_dimension >= SobolDimensionCount) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::resource_exhausted,
            .message = "Sobol sampling supports dimensions [0, " +
                       std::to_string(SobolDimensionCount - 1) + "].",
        });
    }

    auto gray_code = sample_index ^ (sample_index >> 1U);
    if (gray_code == 0) {
        return std::uint64_t{0};
    }

    const auto directions = direction_numbers(sobol_dimension);
    auto sample = std::uint64_t{0};
    while (gray_code != 0) {
        const auto column = static_cast<std::uint32_t>(std::countr_zero(gray_code));
        sample ^= directions[column];
        gray_code &= gray_code - 1U;
    }
    return sample;
}

} // namespace blackframe::renderer
