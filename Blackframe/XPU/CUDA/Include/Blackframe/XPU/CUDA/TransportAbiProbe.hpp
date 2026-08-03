#pragma once

#include <Blackframe/XPU/Shared/TransportAbi.hpp>

extern "C" int
blackframe_cuda_query_transport_abi_layout(blackframe::xpu::shared::LayoutManifest* output,
                                           int* device_count) noexcept;
