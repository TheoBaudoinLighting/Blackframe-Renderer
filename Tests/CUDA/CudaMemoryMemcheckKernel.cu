#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace {

__global__ void fill_bytes(std::uint8_t* const destination, const std::size_t byte_count,
                           const std::uint8_t value) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < byte_count) {
        destination[index] = value;
    }
}

} // namespace

extern "C" int blackframe_cuda_memory_memcheck_fill(void* const destination,
                                                    const std::size_t byte_count,
                                                    const std::uint8_t value) noexcept {
    if (destination == nullptr || byte_count == 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    constexpr auto thread_count = std::uint32_t{128};
    const auto block_count = static_cast<std::uint32_t>(
        (byte_count + static_cast<std::size_t>(thread_count) - 1) / thread_count);
    fill_bytes<<<block_count, thread_count>>>(static_cast<std::uint8_t*>(destination), byte_count,
                                              value);

    auto status = cudaGetLastError();
    if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
    }
    return static_cast<int>(status);
}
