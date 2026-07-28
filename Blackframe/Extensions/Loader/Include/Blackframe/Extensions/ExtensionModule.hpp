#pragma once

#include <Blackframe/Extensions/DynamicLibrary.hpp>
#include <Blackframe/Extensions/ExtensionAbi.h>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>

namespace blackframe::extensions {

enum class ExtensionLoadErrorCode : std::uint8_t {
    DynamicLibraryFailure,
    QuerySymbolMissing,
    QueryRejected,
    MalformedExtensionApi,
};

struct ExtensionLoadError final {
    ExtensionLoadErrorCode code{};
    blackframe_status extension_status{BLACKFRAME_STATUS_SUCCESS};
    std::uint32_t native_error{};
    std::filesystem::path path;
    std::string message;
};

class ExtensionModule final {
  public:
    ~ExtensionModule();

    ExtensionModule(const ExtensionModule&) = delete;
    auto operator=(const ExtensionModule&) -> ExtensionModule& = delete;

    ExtensionModule(ExtensionModule&& other) noexcept;
    auto operator=(ExtensionModule&& other) noexcept -> ExtensionModule&;

    [[nodiscard]] static auto load(const std::filesystem::path& absolute_path)
        -> std::expected<ExtensionModule, ExtensionLoadError>;

    [[nodiscard]] auto get_interface(blackframe_interface_id interface_id,
                                     std::uint32_t requested_major, std::uint32_t requested_minor,
                                     void* out_interface,
                                     std::uint32_t out_interface_size) const noexcept
        -> blackframe_status;

    void shutdown() noexcept;

    [[nodiscard]] auto is_active() const noexcept -> bool;
    [[nodiscard]] auto path() const noexcept -> const std::filesystem::path&;

  private:
    ExtensionModule(DynamicLibrary library, blackframe_extension_api_v1 extension_api) noexcept;

    DynamicLibrary library_;
    blackframe_extension_api_v1 extension_api_{};
};

} // namespace blackframe::extensions
