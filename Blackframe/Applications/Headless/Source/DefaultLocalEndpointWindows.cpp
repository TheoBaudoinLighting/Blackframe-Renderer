#include "DefaultLocalEndpoint.hpp"

namespace blackframe::application {

ipc::LocalEndpoint default_local_endpoint() {
    return ipc::LocalEndpoint{.address = "Blackframe"};
}

} // namespace blackframe::application
