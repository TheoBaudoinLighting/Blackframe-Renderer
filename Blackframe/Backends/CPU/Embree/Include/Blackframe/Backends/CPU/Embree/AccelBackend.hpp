#pragma once

#include <Blackframe/Engine/AccelBackend.hpp>

namespace blackframe::engine {

[[nodiscard]] core::Result<std::unique_ptr<AccelBackend>>
create_embree_accel_backend(std::span<const AccelGeometry> geometries);

} // namespace blackframe::engine
