#pragma once

#include <cstdint>

extern "C" int blackframe_cuda_run_smoke(std::uint32_t input, std::uint32_t xor_mask,
                                         std::uint32_t* output, int* device_count) noexcept;

extern "C" std::uint64_t blackframe_cuda_shared_header_language_level() noexcept;
