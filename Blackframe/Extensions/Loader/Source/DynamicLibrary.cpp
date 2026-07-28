#include <Blackframe/Extensions/DynamicLibrary.hpp>
#include <algorithm>
#include <system_error>
#include <utility>

#if defined(_WIN32)
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace blackframe::extensions {
namespace {

[[nodiscard]] auto make_error(const DynamicLibraryErrorCode code, const std::filesystem::path& path,
                              std::string message, const std::uint32_t native_error = 0)
    -> DynamicLibraryError {
    return DynamicLibraryError{
        .code = code,
        .native_error = native_error,
        .path = path,
        .message = std::move(message),
    };
}

#if defined(_WIN32)
[[nodiscard]] auto windows_error_message(const DWORD error) -> std::string {
    return std::system_category().message(static_cast<int>(error));
}
#endif

} // namespace

DynamicLibrary::DynamicLibrary(void* const native_handle,
                               std::filesystem::path canonical_path) noexcept
    : native_handle_(native_handle), path_(std::move(canonical_path)) {}

DynamicLibrary::~DynamicLibrary() {
    close();
}

DynamicLibrary::DynamicLibrary(DynamicLibrary&& other) noexcept
    : native_handle_(std::exchange(other.native_handle_, nullptr)), path_(std::move(other.path_)) {}

auto DynamicLibrary::operator=(DynamicLibrary&& other) noexcept -> DynamicLibrary& {
    if (this != &other) {
        close();
        native_handle_ = std::exchange(other.native_handle_, nullptr);
        path_ = std::move(other.path_);
    }
    return *this;
}

auto DynamicLibrary::open(const std::filesystem::path& absolute_path)
    -> std::expected<DynamicLibrary, DynamicLibraryError> {
    if (!absolute_path.is_absolute()) {
        return std::unexpected(
            make_error(DynamicLibraryErrorCode::PathMustBeAbsolute, absolute_path,
                       "Extension paths must be absolute; library search paths are not used."));
    }

    std::error_code path_error;
    auto canonical_path = std::filesystem::canonical(absolute_path, path_error);
    if (path_error) {
        return std::unexpected(make_error(DynamicLibraryErrorCode::PathResolutionFailed,
                                          absolute_path, path_error.message(),
                                          static_cast<std::uint32_t>(path_error.value())));
    }

    const bool is_regular_file = std::filesystem::is_regular_file(canonical_path, path_error);
    if (path_error || !is_regular_file) {
        return std::unexpected(make_error(DynamicLibraryErrorCode::PathIsNotAFile, canonical_path,
                                          path_error ? path_error.message()
                                                     : "The extension path is not a regular file.",
                                          static_cast<std::uint32_t>(path_error.value())));
    }

#if defined(_WIN32)
    constexpr DWORD safe_search_flags =
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32;
    HMODULE const module = LoadLibraryExW(canonical_path.c_str(), nullptr, safe_search_flags);
    if (module == nullptr) {
        const DWORD error = GetLastError();
        return std::unexpected(make_error(DynamicLibraryErrorCode::OpenFailed, canonical_path,
                                          windows_error_message(error), error));
    }
    return DynamicLibrary{static_cast<void*>(module), std::move(canonical_path)};
#else
    dlerror();
    void* const module = dlopen(canonical_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (module == nullptr) {
        const char* const error = dlerror();
        return std::unexpected(
            make_error(DynamicLibraryErrorCode::OpenFailed, canonical_path,
                       error != nullptr ? error : "dlopen failed without an error message."));
    }
    return DynamicLibrary{module, std::move(canonical_path)};
#endif
}

auto DynamicLibrary::find_symbol(const std::string_view symbol_name) const
    -> std::expected<void*, DynamicLibraryError> {
    if (native_handle_ == nullptr || symbol_name.empty() ||
        std::ranges::find(symbol_name, '\0') != symbol_name.end()) {
        return std::unexpected(make_error(
            DynamicLibraryErrorCode::InvalidSymbolName, path_,
            "The symbol name is empty, contains a null byte, or the library is closed."));
    }

    const std::string terminated_name{symbol_name};

#if defined(_WIN32)
    FARPROC const symbol =
        GetProcAddress(static_cast<HMODULE>(native_handle_), terminated_name.c_str());
    if (symbol == nullptr) {
        const DWORD error = GetLastError();
        return std::unexpected(make_error(DynamicLibraryErrorCode::SymbolNotFound, path_,
                                          windows_error_message(error), error));
    }
    return reinterpret_cast<void*>(symbol);
#else
    dlerror();
    void* const symbol = dlsym(native_handle_, terminated_name.c_str());
    const char* const error = dlerror();
    if (error != nullptr) {
        return std::unexpected(make_error(DynamicLibraryErrorCode::SymbolNotFound, path_, error));
    }
    return symbol;
#endif
}

auto DynamicLibrary::is_open() const noexcept -> bool {
    return native_handle_ != nullptr;
}

auto DynamicLibrary::path() const noexcept -> const std::filesystem::path& {
    return path_;
}

void DynamicLibrary::close() noexcept {
    if (native_handle_ == nullptr) {
        return;
    }

#if defined(_WIN32)
    static_cast<void>(FreeLibrary(static_cast<HMODULE>(native_handle_)));
#else
    static_cast<void>(dlclose(native_handle_));
#endif

    native_handle_ = nullptr;
}

} // namespace blackframe::extensions
