#include <Blackframe/XPU/XpuBackend.hpp>
#include <utility>

namespace blackframe::xpu {
namespace {

constexpr std::uint32_t maximum_device_count = 1024;
constexpr std::uint32_t maximum_label_size = 4096;

[[nodiscard]] auto make_error(const XpuHostErrorCode code,
                              const std::filesystem::path& extension_path, std::string message,
                              const blackframe_status status = BLACKFRAME_STATUS_SUCCESS)
    -> XpuHostError {
    return XpuHostError{
        .code = code,
        .extension_status = status,
        .extension_path = extension_path,
        .message = std::move(message),
    };
}

[[nodiscard]] auto copy_view(const blackframe_utf8_view view)
    -> std::expected<std::string, std::string> {
    if (view.reserved != 0 || view.size > maximum_label_size ||
        (view.data == nullptr && view.size != 0)) {
        return std::unexpected("The extension returned an invalid string view.");
    }

    if (view.size == 0) {
        return std::string{};
    }
    return std::string{view.data, view.size};
}

[[nodiscard]] constexpr auto to_device_kind(const blackframe_xpu_device_kind kind) noexcept
    -> DeviceKind {
    switch (kind) {
    case BLACKFRAME_XPU_DEVICE_KIND_CPU:
        return DeviceKind::Cpu;
    case BLACKFRAME_XPU_DEVICE_KIND_GPU:
        return DeviceKind::Gpu;
    case BLACKFRAME_XPU_DEVICE_KIND_ACCELERATOR:
        return DeviceKind::Accelerator;
    default:
        return DeviceKind::Unknown;
    }
}

} // namespace

XpuBackend::XpuBackend(extensions::ExtensionModule extension,
                       const blackframe_xpu_backend_api_v1 backend_api,
                       std::string backend_name) noexcept
    : extension_(std::move(extension)), backend_api_(backend_api),
      backend_name_(std::move(backend_name)) {}

auto XpuBackend::load(const std::filesystem::path& absolute_extension_path)
    -> std::expected<XpuBackend, XpuHostError> {
    auto extension = extensions::ExtensionModule::load(absolute_extension_path);
    if (!extension) {
        return std::unexpected(make_error(XpuHostErrorCode::ExtensionLoadFailed,
                                          extension.error().path, extension.error().message,
                                          extension.error().extension_status));
    }

    blackframe_xpu_backend_api_v1 backend_api{};
    backend_api.struct_size = sizeof(blackframe_xpu_backend_api_v1);

    constexpr blackframe_interface_id interface_id{
        .high = BLACKFRAME_XPU_BACKEND_INTERFACE_ID_HIGH,
        .low = BLACKFRAME_XPU_BACKEND_INTERFACE_ID_LOW,
    };
    const blackframe_status interface_status = extension->get_interface(
        interface_id, BLACKFRAME_XPU_BACKEND_ABI_MAJOR, BLACKFRAME_XPU_BACKEND_ABI_MINOR,
        &backend_api, sizeof(backend_api));
    if (interface_status != BLACKFRAME_STATUS_SUCCESS) {
        return std::unexpected(
            make_error(XpuHostErrorCode::InterfaceUnavailable, extension->path(),
                       "The extension does not provide the Blackframe XPU backend interface.",
                       interface_status));
    }

    const bool malformed = backend_api.struct_size < sizeof(blackframe_xpu_backend_api_v1) ||
                           backend_api.abi_major != BLACKFRAME_XPU_BACKEND_ABI_MAJOR ||
                           backend_api.abi_minor > BLACKFRAME_XPU_BACKEND_ABI_MINOR ||
                           backend_api.reserved != 0 || backend_api.get_device_count == nullptr ||
                           backend_api.get_device_descriptor == nullptr;
    if (malformed) {
        return std::unexpected(
            make_error(XpuHostErrorCode::MalformedBackendApi, extension->path(),
                       "The extension returned an incomplete or incompatible XPU API table."));
    }

    auto backend_name = copy_view(backend_api.backend_name);
    if (!backend_name || backend_name->empty()) {
        return std::unexpected(
            make_error(XpuHostErrorCode::MalformedBackendApi, extension->path(),
                       backend_name ? "The XPU backend name is empty." : backend_name.error()));
    }

    return XpuBackend{
        std::move(*extension),
        backend_api,
        std::move(*backend_name),
    };
}

auto XpuBackend::enumerate_devices() const
    -> std::expected<std::vector<DeviceDescriptor>, XpuHostError> {
    std::uint32_t device_count{};
    const blackframe_status count_status =
        backend_api_.get_device_count(backend_api_.backend_context, &device_count);
    if (count_status != BLACKFRAME_STATUS_SUCCESS) {
        return std::unexpected(make_error(XpuHostErrorCode::EnumerationFailed, extension_.path(),
                                          "The XPU backend failed to return its device count.",
                                          count_status));
    }
    if (device_count > maximum_device_count) {
        return std::unexpected(
            make_error(XpuHostErrorCode::InvalidDeviceCount, extension_.path(),
                       "The XPU backend returned an unreasonable device count."));
    }

    std::vector<DeviceDescriptor> devices;
    devices.reserve(device_count);

    for (std::uint32_t index = 0; index < device_count; ++index) {
        blackframe_xpu_device_descriptor_v1 descriptor{};
        descriptor.struct_size = sizeof(blackframe_xpu_device_descriptor_v1);

        const blackframe_status descriptor_status =
            backend_api_.get_device_descriptor(backend_api_.backend_context, index, &descriptor);
        if (descriptor_status != BLACKFRAME_STATUS_SUCCESS) {
            return std::unexpected(
                make_error(XpuHostErrorCode::EnumerationFailed, extension_.path(),
                           "The XPU backend failed to describe a device.", descriptor_status));
        }
        if (descriptor.struct_size < sizeof(blackframe_xpu_device_descriptor_v1)) {
            return std::unexpected(
                make_error(XpuHostErrorCode::MalformedDeviceDescriptor, extension_.path(),
                           "The XPU backend returned an incomplete device descriptor."));
        }

        auto name = copy_view(descriptor.name);
        auto vendor = copy_view(descriptor.vendor);
        if (!name || !vendor || name->empty()) {
            return std::unexpected(make_error(XpuHostErrorCode::MalformedDeviceDescriptor,
                                              extension_.path(),
                                              "The XPU backend returned invalid device text."));
        }

        devices.push_back(DeviceDescriptor{
            .identifier =
                DeviceIdentifier{
                    .high = descriptor.identifier_high,
                    .low = descriptor.identifier_low,
                },
            .kind = to_device_kind(descriptor.kind),
            .backend_name = backend_name_,
            .name = std::move(*name),
            .vendor = std::move(*vendor),
        });
    }

    return devices;
}

auto XpuBackend::name() const noexcept -> std::string_view {
    return backend_name_;
}

auto XpuBackend::extension_path() const noexcept -> const std::filesystem::path& {
    return extension_.path();
}

} // namespace blackframe::xpu
