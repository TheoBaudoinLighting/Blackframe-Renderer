#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <string_view>

namespace blackframe::extensions {

enum class DynamicLibraryErrorCode : std::uint8_t {
    PathMustBeAbsolute,
    PathResolutionFailed,
    PathIsNotAFile,
    OpenFailed,
    InvalidSymbolName,
    SymbolNotFound,
};

struct DynamicLibraryError final {
    DynamicLibraryErrorCode code{};
    std::uint32_t native_error{};
    std::filesystem::path path;
    std::string message;
};

class DynamicLibrary final {
  public:
    DynamicLibrary() noexcept = default;
    ~DynamicLibrary();

    DynamicLibrary(const DynamicLibrary&) = delete;
    auto operator=(const DynamicLibrary&) -> DynamicLibrary& = delete;

    DynamicLibrary(DynamicLibrary&& other) noexcept;
    auto operator=(DynamicLibrary&& other) noexcept -> DynamicLibrary&;

    [[nodiscard]] static auto open(const std::filesystem::path& absolute_path)
        -> std::expected<DynamicLibrary, DynamicLibraryError>;

    [[nodiscard]] auto find_symbol(std::string_view symbol_name) const
        -> std::expected<void*, DynamicLibraryError>;

    [[nodiscard]] auto is_open() const noexcept -> bool;
    [[nodiscard]] auto path() const noexcept -> const std::filesystem::path&;

  private:
    DynamicLibrary(void* native_handle, std::filesystem::path canonical_path) noexcept;
    void close() noexcept;

    void* native_handle_{};
    std::filesystem::path path_;
};

} // namespace blackframe::extensions
