#include <Blackframe/Renderer/CapabilityRegistry.hpp>
#include <array>

#if !defined(BLACKFRAME_CAPABILITY_REFERENCE_CPU) || !defined(BLACKFRAME_CAPABILITY_EMBREE) ||     \
    !defined(BLACKFRAME_CAPABILITY_CUDA)
#error "Blackframe backend capability definitions are missing."
#endif

namespace blackframe::renderer {
namespace {

constexpr auto reference_cpu_status = [] {
    if constexpr (BLACKFRAME_CAPABILITY_REFERENCE_CPU != 0) {
        return BackendCapabilityStatus::experimental;
    }
    return BackendCapabilityStatus::unavailable;
}();

constexpr auto embree_status = [] {
    if constexpr (BLACKFRAME_CAPABILITY_EMBREE != 0) {
        return BackendCapabilityStatus::supported;
    }
    return BackendCapabilityStatus::unavailable;
}();

constexpr auto cuda_status = [] {
    if constexpr (BLACKFRAME_CAPABILITY_CUDA != 0) {
        return BackendCapabilityStatus::supported;
    }
    return BackendCapabilityStatus::unavailable;
}();

constexpr auto registered_backends = std::array{
    BackendCapability{
        .identifier = "reference_cpu",
        .status = reference_cpu_status,
        .required_dependency = "BlackframeXpuReferenceCpu",
    },
    BackendCapability{
        .identifier = "cpu_embree",
        .status = embree_status,
        .required_dependency = "Embree",
    },
    BackendCapability{
        .identifier = "gpu_cuda",
        .status = cuda_status,
        .required_dependency = "CUDA Toolkit",
    },
};

} // namespace

std::string_view backend_capability_status_name(const BackendCapabilityStatus status) noexcept {
    switch (status) {
    case BackendCapabilityStatus::supported:
        return "supported";
    case BackendCapabilityStatus::experimental:
        return "experimental";
    case BackendCapabilityStatus::unavailable:
        return "unavailable";
    }
    return "unknown";
}

std::span<const BackendCapability> backend_capabilities() noexcept {
    return registered_backends;
}

std::string backend_capability_manifest() {
    auto manifest = std::string{"{\n"
                                "  \"schema_version\": " +
                                std::to_string(CapabilityManifestSchemaVersion) +
                                ",\n"
                                "  \"backends\": [\n"};

    for (std::size_t index = 0; index < registered_backends.size(); ++index) {
        const auto& backend = registered_backends[index];
        manifest += "    {\"id\": \"" + std::string{backend.identifier} + "\", \"status\": \"" +
                    std::string{backend_capability_status_name(backend.status)} + "\"}";
        manifest += index + 1 == registered_backends.size() ? "\n" : ",\n";
    }
    manifest += "  ]\n}\n";
    return manifest;
}

core::Status require_backend_capability(const std::string_view identifier) {
    for (const auto& backend : registered_backends) {
        if (backend.identifier != identifier) {
            continue;
        }
        if (backend.status != BackendCapabilityStatus::unavailable) {
            return {};
        }
        return std::unexpected(core::Error{
            .code = core::StatusCode::unavailable,
            .message = "Backend capability '" + std::string{identifier} +
                       "' is unavailable; missing dependency '" +
                       std::string{backend.required_dependency} + "'.",
        });
    }

    return std::unexpected(core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = "Unknown backend capability '" + std::string{identifier} + "'.",
    });
}

} // namespace blackframe::renderer
