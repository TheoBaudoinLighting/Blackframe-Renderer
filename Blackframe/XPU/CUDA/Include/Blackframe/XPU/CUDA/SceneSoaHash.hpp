#pragma once

#include <cstddef>
#include <cstdint>

// Launches one deterministic device hash over the serialized scene bytes. The
// output pointer is device storage owned by the caller; synchronization and
// readback are explicit caller responsibilities.
extern "C" int blackframe_cuda_launch_scene_soa_hash(const std::uint8_t* scene_bytes,
                                                     std::size_t scene_byte_count,
                                                     std::uint64_t* output_hash) noexcept;
