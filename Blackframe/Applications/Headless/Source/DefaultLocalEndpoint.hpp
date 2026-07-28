#pragma once

#include <Blackframe/IPC/LocalTransport.hpp>

namespace blackframe::application {

[[nodiscard]] ipc::LocalEndpoint default_local_endpoint();

} // namespace blackframe::application
