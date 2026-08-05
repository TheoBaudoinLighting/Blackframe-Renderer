#include <Blackframe/Renderer/SampleDimensionMap.hpp>
#include <Blackframe/Renderer/SampleStream.hpp>
#include <Blackframe/XPU/CUDA/DeviceMemory.hpp>
#include <Blackframe/XPU/CUDA/SampleStreamKernel.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace cuda = blackframe::xpu::cuda;
namespace renderer = blackframe::renderer;
namespace shared = blackframe::xpu::shared;

using cuda::SampleStreamDimensionManifest;
using cuda::SampleStreamDumpRequest;
using cuda::SampleStreamDumpResult;
using cuda::SampleStreamDumpStatus;

constexpr auto PrimaryDimensionCount = std::size_t{6U};
constexpr auto BounceDimensionCount = std::size_t{10U};
constexpr auto FrozenCanonicalDumpFnv1a = std::uint64_t{0x59DC894B1E63155DULL};

static_assert(std::is_standard_layout_v<SampleStreamDumpRequest>);
static_assert(std::is_trivially_copyable_v<SampleStreamDumpRequest>);
static_assert(std::is_standard_layout_v<SampleStreamDumpResult>);
static_assert(std::is_trivially_copyable_v<SampleStreamDumpResult>);
static_assert(std::is_standard_layout_v<SampleStreamDimensionManifest>);
static_assert(std::is_trivially_copyable_v<SampleStreamDimensionManifest>);

struct DownloadedDump final {
    SampleStreamDimensionManifest manifest{};
    std::vector<SampleStreamDumpResult> results{};
};

[[nodiscard]] testing::AssertionResult select_test_device() {
    auto device_count = int{};
    const auto count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess) {
        return testing::AssertionFailure()
               << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_status);
    }
    if (device_count <= 0) {
        return testing::AssertionFailure() << "No CUDA device is available.";
    }
    const auto select_status = cudaSetDevice(0);
    if (select_status != cudaSuccess) {
        return testing::AssertionFailure()
               << "cudaSetDevice failed: " << cudaGetErrorString(select_status);
    }
    return testing::AssertionSuccess();
}

void require_cuda_success(const cudaError_t status, const char* const operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error{std::string{operation} + ": " + cudaGetErrorString(status)};
    }
}

template <typename Value>
[[nodiscard]] cuda::DeviceBuffer<Value> allocate_device(const std::size_t count) {
    auto result = cuda::DeviceBuffer<Value>::allocate(count);
    if (!result) {
        throw std::runtime_error{result.error().message};
    }
    return std::move(*result);
}

[[nodiscard]] DownloadedDump launch_dump(const std::span<const SampleStreamDumpRequest> requests) {
    auto device_requests = allocate_device<SampleStreamDumpRequest>(requests.size());
    auto device_results = allocate_device<SampleStreamDumpResult>(requests.size());
    auto device_manifest = allocate_device<SampleStreamDimensionManifest>(1U);

    require_cuda_success(cudaMemcpy(device_requests.data(), requests.data(), requests.size_bytes(),
                                    cudaMemcpyHostToDevice),
                         "sample-stream request upload");
    const auto launch_status = blackframe_cuda_launch_sample_stream_dump(
        renderer::CurrentSampleDimensionMapSchemaVersion, device_requests.data(),
        static_cast<std::uint32_t>(requests.size()), device_manifest.data(), device_results.data());
    require_cuda_success(static_cast<cudaError_t>(launch_status), "sample-stream dump launch");
    require_cuda_success(cudaDeviceSynchronize(), "sample-stream dump synchronization");

    auto dump = DownloadedDump{};
    dump.results.resize(requests.size());
    require_cuda_success(cudaMemcpy(&dump.manifest, device_manifest.data(), sizeof(dump.manifest),
                                    cudaMemcpyDeviceToHost),
                         "sample-stream manifest download");
    require_cuda_success(cudaMemcpy(dump.results.data(), device_results.data(),
                                    dump.results.size() * sizeof(SampleStreamDumpResult),
                                    cudaMemcpyDeviceToHost),
                         "sample-stream result download");
    return dump;
}

[[nodiscard]] constexpr std::array<renderer::SampleDimension, PrimaryDimensionCount>
cpu_primary_dimensions() noexcept {
    return {
        renderer::PrimarySampleDimensionMap.camera_raster_x,
        renderer::PrimarySampleDimensionMap.camera_raster_y,
        renderer::PrimarySampleDimensionMap.lens_u,
        renderer::PrimarySampleDimensionMap.lens_v,
        renderer::PrimarySampleDimensionMap.time,
        renderer::PrimarySampleDimensionMap.wavelength,
    };
}

[[nodiscard]] constexpr std::array<renderer::SampleDimension, BounceDimensionCount>
bounce_dimensions_as_array(const renderer::BounceSampleDimensions dimensions) noexcept {
    return {
        dimensions.light_selection,  dimensions.light_u,        dimensions.light_v,
        dimensions.bsdf_component,   dimensions.bsdf_u,         dimensions.bsdf_v,
        dimensions.medium_distance,  dimensions.medium_phase_u, dimensions.medium_phase_v,
        dimensions.russian_roulette,
    };
}

[[nodiscard]] constexpr std::array<std::uint64_t, PrimaryDimensionCount>
device_primary_dimensions_as_array(const cuda::SampleStreamPrimaryDimensions dimensions) noexcept {
    return {
        dimensions.camera_raster_x,
        dimensions.camera_raster_y,
        dimensions.lens_u,
        dimensions.lens_v,
        dimensions.time,
        dimensions.wavelength,
    };
}

[[nodiscard]] constexpr std::array<std::uint64_t, BounceDimensionCount>
device_bounce_dimensions_as_array(const cuda::SampleStreamBounceDimensions dimensions) noexcept {
    return {
        dimensions.light_selection,  dimensions.light_u,        dimensions.light_v,
        dimensions.bsdf_component,   dimensions.bsdf_u,         dimensions.bsdf_v,
        dimensions.medium_distance,  dimensions.medium_phase_u, dimensions.medium_phase_v,
        dimensions.russian_roulette,
    };
}

[[nodiscard]] constexpr cuda::SampleStreamPrimaryDimensions device_primary_dimensions(
    const std::array<renderer::SampleDimension, PrimaryDimensionCount> values) noexcept {
    return {
        .camera_raster_x = values[0U],
        .camera_raster_y = values[1U],
        .lens_u = values[2U],
        .lens_v = values[3U],
        .time = values[4U],
        .wavelength = values[5U],
    };
}

[[nodiscard]] constexpr cuda::SampleStreamBounceDimensions device_bounce_dimensions(
    const std::array<renderer::SampleDimension, BounceDimensionCount> values) noexcept {
    return {
        .light_selection = values[0U],
        .light_u = values[1U],
        .light_v = values[2U],
        .bsdf_component = values[3U],
        .bsdf_u = values[4U],
        .bsdf_v = values[5U],
        .medium_distance = values[6U],
        .medium_phase_u = values[7U],
        .medium_phase_v = values[8U],
        .russian_roulette = values[9U],
    };
}

[[nodiscard]] std::array<renderer::SampleDimension, BounceDimensionCount> cpu_bounce_offsets() {
    const auto first = renderer::sample_dimensions_for_bounce(0U);
    if (!first) {
        throw std::runtime_error{first.error().message};
    }
    auto offsets = bounce_dimensions_as_array(*first);
    for (auto& dimension : offsets) {
        dimension -= renderer::FirstBounceSampleDimension;
    }
    return offsets;
}

[[nodiscard]] renderer::SampleStreamIndex
renderer_index(const shared::SampleStreamIndex index) noexcept {
    return renderer::SampleStreamIndex{
        .pixel_x = index.pixel_x,
        .pixel_y = index.pixel_y,
        .sample_index = index.sample_index,
        .seed = index.seed,
    };
}

[[nodiscard]] std::vector<SampleStreamDumpRequest> sample_requests() {
    constexpr auto maximum_u32 = std::numeric_limits<std::uint32_t>::max();
    constexpr auto maximum_u64 = std::numeric_limits<std::uint64_t>::max();
    constexpr auto high_bit = std::uint64_t{1U} << 63U;
    constexpr auto base_index = shared::SampleStreamIndex{
        .pixel_x = 0x01234567U,
        .pixel_y = 0x89ABCDEFU,
        .sample_index = 0x0123456789ABCDEFULL,
        .seed = 0xFEDCBA9876543210ULL,
    };

    auto requests = std::vector<SampleStreamDumpRequest>{
        SampleStreamDumpRequest{
            .sample_stream = {},
            .dimension = 0U,
            .bounce_index = 0U,
        },
        SampleStreamDumpRequest{
            .sample_stream =
                {.pixel_x = 17U, .pixel_y = 29U, .sample_index = 0U, .seed = 0x0123456789ABCDEFULL},
            .dimension = 0U,
            .bounce_index = 1U,
        },
        SampleStreamDumpRequest{
            .sample_stream = {.pixel_x = 17U,
                              .pixel_y = 29U,
                              .sample_index = 4095U,
                              .seed = 0x0123456789ABCDEFULL},
            .dimension = 1U,
            .bounce_index = renderer::MaximumMappedBounceIndex,
        },
        SampleStreamDumpRequest{
            .sample_stream = {.pixel_x = maximum_u32,
                              .pixel_y = maximum_u32,
                              .sample_index = maximum_u64,
                              .seed = maximum_u64},
            .dimension = maximum_u64,
            .bounce_index = renderer::MaximumMappedBounceIndex,
        },
        SampleStreamDumpRequest{
            .sample_stream = {},
            .dimension = renderer::PrimarySampleDimensionMap.camera_raster_x,
            .bounce_index = renderer::MaximumMappedBounceIndex + 1U,
        },
        SampleStreamDumpRequest{
            .sample_stream = base_index,
            .dimension = high_bit | 0x25U,
            .bounce_index = 0U,
        },
    };

    auto changed = base_index;
    changed.pixel_x ^= 0x80000000U;
    requests.push_back(
        {.sample_stream = changed, .dimension = high_bit | 0x25U, .bounce_index = 0U});
    changed = base_index;
    changed.pixel_y ^= 0x80000000U;
    requests.push_back(
        {.sample_stream = changed, .dimension = high_bit | 0x25U, .bounce_index = 0U});
    changed = base_index;
    changed.sample_index ^= high_bit;
    requests.push_back(
        {.sample_stream = changed, .dimension = high_bit | 0x25U, .bounce_index = 0U});
    changed = base_index;
    changed.seed ^= high_bit;
    requests.push_back(
        {.sample_stream = changed, .dimension = high_bit | 0x25U, .bounce_index = 0U});
    requests.push_back({.sample_stream = base_index, .dimension = 0x25U, .bounce_index = 0U});
    return requests;
}

void expect_manifest_matches_cpu(const SampleStreamDimensionManifest& manifest) {
    EXPECT_EQ(manifest.schema_version, renderer::CurrentSampleDimensionMapSchemaVersion);
    EXPECT_EQ(manifest.reserved, 0U);
    EXPECT_EQ(manifest.first_bounce_dimension, renderer::FirstBounceSampleDimension);
    EXPECT_EQ(manifest.dimensions_per_bounce, renderer::SampleDimensionsPerBounce);
    EXPECT_EQ(manifest.maximum_bounce_index, renderer::MaximumMappedBounceIndex);

    const auto primary = cpu_primary_dimensions();
    const auto offsets = cpu_bounce_offsets();
    const auto device_primary = device_primary_dimensions_as_array(manifest.primary_dimensions);
    const auto device_offsets = device_bounce_dimensions_as_array(manifest.bounce_offsets);
    for (auto index = std::size_t{}; index < primary.size(); ++index) {
        EXPECT_EQ(device_primary[index], primary[index]) << "primary " << index;
    }
    for (auto index = std::size_t{}; index < offsets.size(); ++index) {
        EXPECT_EQ(device_offsets[index], offsets[index]) << "bounce offset " << index;
    }
}

void expect_result_matches_cpu(const SampleStreamDumpRequest& request,
                               const SampleStreamDumpResult& result) {
    EXPECT_EQ(result.reserved, 0U);
    EXPECT_EQ(result.reserved_tail, 0U);
    const auto index = renderer_index(request.sample_stream);
    const auto expected_bits =
        renderer::sample_stream_detail::indexed_bits(index, request.dimension);
    const auto expected_sample = renderer::SampleStream{index}.sample_1d(request.dimension);
    EXPECT_EQ(result.indexed_bits, expected_bits);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(result.sample),
              std::bit_cast<std::uint32_t>(expected_sample));

    const auto expected_dimensions = renderer::sample_dimensions_for_bounce(request.bounce_index);
    if (!expected_dimensions) {
        EXPECT_EQ(result.status, SampleStreamDumpStatus::bounce_out_of_range);
        for (const auto dimension : device_bounce_dimensions_as_array(result.bounce_dimensions)) {
            EXPECT_EQ(dimension, 0U);
        }
        return;
    }

    EXPECT_EQ(result.status, SampleStreamDumpStatus::success);
    const auto dimensions = bounce_dimensions_as_array(*expected_dimensions);
    const auto device_dimensions = device_bounce_dimensions_as_array(result.bounce_dimensions);
    for (auto dimension_index = std::size_t{}; dimension_index < dimensions.size();
         ++dimension_index) {
        EXPECT_EQ(device_dimensions[dimension_index], dimensions[dimension_index])
            << "bounce field " << dimension_index;
    }
}

void append_hex(std::string& output, const std::uint64_t value, const std::size_t digits) {
    constexpr auto alphabet = "0123456789abcdef";
    output += "0x";
    for (auto index = digits; index > 0U; --index) {
        const auto shift = (index - 1U) * 4U;
        output += alphabet[static_cast<std::size_t>((value >> shift) & 0xFULL)];
    }
}

template <typename Integer> void append_decimal(std::string& output, const Integer value) {
    auto buffer = std::array<char, 32U>{};
    const auto converted = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    if (converted.ec != std::errc{}) {
        throw std::runtime_error{"Canonical sample dump integer formatting failed."};
    }
    output.append(buffer.data(), converted.ptr);
}

[[nodiscard]] std::string canonical_dump(const SampleStreamDimensionManifest& manifest,
                                         const std::span<const SampleStreamDumpRequest> requests,
                                         const std::span<const SampleStreamDumpResult> results) {
    if (requests.size() != results.size()) {
        throw std::runtime_error{"Canonical sample dump inputs have different lengths."};
    }

    auto output = std::string{};
    output.reserve(4096U);
    output += "schema=";
    append_decimal(output, manifest.schema_version);
    output += "\nprimary=";
    const auto primary = device_primary_dimensions_as_array(manifest.primary_dimensions);
    for (auto index = std::size_t{}; index < PrimaryDimensionCount; ++index) {
        if (index != 0U) {
            output += ',';
        }
        append_hex(output, primary[index], 16U);
    }
    output += "\nbounce=";
    append_hex(output, manifest.first_bounce_dimension, 16U);
    output += ',';
    append_decimal(output, manifest.dimensions_per_bounce);
    output += ',';
    append_decimal(output, manifest.maximum_bounce_index);
    for (const auto offset : device_bounce_dimensions_as_array(manifest.bounce_offsets)) {
        output += ',';
        append_decimal(output, offset);
    }
    output += '\n';

    for (auto index = std::size_t{}; index < requests.size(); ++index) {
        const auto& request = requests[index];
        const auto& result = results[index];
        output += "request=";
        append_hex(output, request.sample_stream.pixel_x, 8U);
        output += ',';
        append_hex(output, request.sample_stream.pixel_y, 8U);
        output += ',';
        append_hex(output, request.sample_stream.sample_index, 16U);
        output += ',';
        append_hex(output, request.sample_stream.seed, 16U);
        output += ',';
        append_hex(output, request.dimension, 16U);
        output += ',';
        append_decimal(output, request.bounce_index);
        output += ";result=";
        append_decimal(output, static_cast<std::uint32_t>(result.status));
        output += ',';
        append_hex(output, result.indexed_bits, 16U);
        output += ',';
        append_hex(output, std::bit_cast<std::uint32_t>(result.sample), 8U);
        for (const auto dimension : device_bounce_dimensions_as_array(result.bounce_dimensions)) {
            output += ',';
            append_hex(output, dimension, 16U);
        }
        output += '\n';
    }
    return output;
}

[[nodiscard]] SampleStreamDimensionManifest cpu_manifest() {
    auto manifest = SampleStreamDimensionManifest{};
    manifest.schema_version = renderer::CurrentSampleDimensionMapSchemaVersion;
    const auto primary = cpu_primary_dimensions();
    manifest.primary_dimensions = device_primary_dimensions(primary);
    manifest.first_bounce_dimension = renderer::FirstBounceSampleDimension;
    manifest.dimensions_per_bounce = renderer::SampleDimensionsPerBounce;
    manifest.maximum_bounce_index = renderer::MaximumMappedBounceIndex;
    const auto offsets = cpu_bounce_offsets();
    manifest.bounce_offsets = device_bounce_dimensions(offsets);
    return manifest;
}

[[nodiscard]] std::vector<SampleStreamDumpResult>
cpu_results(const std::span<const SampleStreamDumpRequest> requests) {
    auto results = std::vector<SampleStreamDumpResult>{};
    results.reserve(requests.size());
    for (const auto& request : requests) {
        auto result = SampleStreamDumpResult{};
        const auto index = renderer_index(request.sample_stream);
        result.indexed_bits =
            renderer::sample_stream_detail::indexed_bits(index, request.dimension);
        result.sample = renderer::SampleStream{index}.sample_1d(request.dimension);
        const auto dimensions = renderer::sample_dimensions_for_bounce(request.bounce_index);
        if (dimensions) {
            result.status = SampleStreamDumpStatus::success;
            const auto values = bounce_dimensions_as_array(*dimensions);
            result.bounce_dimensions = device_bounce_dimensions(values);
        } else {
            result.status = SampleStreamDumpStatus::bounce_out_of_range;
        }
        results.push_back(result);
    }
    return results;
}

[[nodiscard]] constexpr std::uint64_t fnv1a64(const std::string_view text) noexcept {
    auto hash = std::uint64_t{0xCBF29CE484222325ULL};
    for (const auto character : text) {
        hash ^= static_cast<std::uint8_t>(character);
        hash *= 0x100000001B3ULL;
    }
    return hash;
}

TEST(CudaSampleStream, ManifestAndSamplesMatchTheCompleteCpuDimensionMapBitwise) {
    ASSERT_TRUE(select_test_device());
    const auto requests = sample_requests();
    const auto dump = launch_dump(requests);

    expect_manifest_matches_cpu(dump.manifest);
    ASSERT_EQ(dump.results.size(), requests.size());
    for (auto index = std::size_t{}; index < requests.size(); ++index) {
        SCOPED_TRACE(testing::Message{} << "request " << index);
        expect_result_matches_cpu(requests[index], dump.results[index]);
    }

    constexpr auto known_bits = std::array{
        std::uint64_t{0x2F37B19CDAB08AA0ULL},
        std::uint64_t{0xDE27E521323C8AFFULL},
        std::uint64_t{0xDC4661423B6A00FAULL},
        std::uint64_t{0x8F4FD2AABC4AC602ULL},
    };
    constexpr auto known_sample_bits = std::array{
        std::uint32_t{0x3E3CDEC4U},
        std::uint32_t{0x3F5E27E5U},
        std::uint32_t{0x3F5C4661U},
        std::uint32_t{0x3F0F4FD2U},
    };
    for (auto index = std::size_t{}; index < known_bits.size(); ++index) {
        EXPECT_EQ(dump.results[index].indexed_bits, known_bits[index]);
        EXPECT_EQ(std::bit_cast<std::uint32_t>(dump.results[index].sample),
                  known_sample_bits[index]);
    }

    constexpr auto high_bit_baseline = std::size_t{5U};
    for (auto index = high_bit_baseline + 1U; index < requests.size(); ++index) {
        EXPECT_NE(dump.results[index].indexed_bits, dump.results[high_bit_baseline].indexed_bits)
            << "high-bit variant " << index;
    }
}

TEST(CudaSampleStream, ReplayAndRequestOrderAreDeterministic) {
    ASSERT_TRUE(select_test_device());
    const auto requests = sample_requests();
    const auto first = launch_dump(requests);
    const auto replay = launch_dump(requests);

    EXPECT_EQ(canonical_dump(first.manifest, requests, first.results),
              canonical_dump(replay.manifest, requests, replay.results));

    auto reversed_requests = requests;
    std::ranges::reverse(reversed_requests);
    const auto reversed = launch_dump(reversed_requests);
    auto restored_results = reversed.results;
    std::ranges::reverse(restored_results);
    EXPECT_EQ(canonical_dump(first.manifest, requests, first.results),
              canonical_dump(reversed.manifest, requests, restored_results));
}

TEST(CudaSampleStream, CanonicalDeviceDumpMatchesCpuAndItsFrozenFingerprint) {
    ASSERT_TRUE(select_test_device());
    const auto requests = sample_requests();
    const auto device = launch_dump(requests);
    const auto host_manifest = cpu_manifest();
    const auto host_results = cpu_results(requests);

    const auto device_text = canonical_dump(device.manifest, requests, device.results);
    const auto host_text = canonical_dump(host_manifest, requests, host_results);
    EXPECT_EQ(device_text, host_text);
    EXPECT_EQ(fnv1a64(device_text), FrozenCanonicalDumpFnv1a) << device_text;
}

TEST(CudaSampleStream, UnsupportedSchemasAndInvalidPointersAreRejectedBeforeDispatch) {
    ASSERT_TRUE(select_test_device());
    const auto requests = sample_requests();
    auto device_requests = allocate_device<SampleStreamDumpRequest>(requests.size());
    auto device_results = allocate_device<SampleStreamDumpResult>(requests.size());
    auto device_manifest = allocate_device<SampleStreamDimensionManifest>(1U);
    require_cuda_success(cudaMemcpy(device_requests.data(), requests.data(),
                                    requests.size() * sizeof(SampleStreamDumpRequest),
                                    cudaMemcpyHostToDevice),
                         "sample-stream invalid-contract request upload");

    for (const auto version : std::array{std::uint32_t{0U}, std::uint32_t{2U}}) {
        EXPECT_EQ(blackframe_cuda_launch_sample_stream_dump(
                      version, device_requests.data(), static_cast<std::uint32_t>(requests.size()),
                      device_manifest.data(), device_results.data()),
                  static_cast<int>(cudaErrorInvalidValue));
    }
    EXPECT_EQ(blackframe_cuda_launch_sample_stream_dump(
                  renderer::CurrentSampleDimensionMapSchemaVersion, nullptr,
                  static_cast<std::uint32_t>(requests.size()), device_manifest.data(),
                  device_results.data()),
              static_cast<int>(cudaErrorInvalidValue));
    EXPECT_EQ(blackframe_cuda_launch_sample_stream_dump(
                  renderer::CurrentSampleDimensionMapSchemaVersion, device_requests.data(),
                  static_cast<std::uint32_t>(requests.size()), nullptr, device_results.data()),
              static_cast<int>(cudaErrorInvalidValue));
    EXPECT_EQ(blackframe_cuda_launch_sample_stream_dump(
                  renderer::CurrentSampleDimensionMapSchemaVersion, device_requests.data(),
                  static_cast<std::uint32_t>(requests.size()), device_manifest.data(), nullptr),
              static_cast<int>(cudaErrorInvalidValue));
    EXPECT_EQ(blackframe_cuda_launch_sample_stream_dump(
                  renderer::CurrentSampleDimensionMapSchemaVersion, device_requests.data(), 0U,
                  device_manifest.data(), device_results.data()),
              static_cast<int>(cudaErrorInvalidValue));
}

} // namespace
