#include <Blackframe/Extensions/ExtensionModule.hpp>
#include <cstring>
#include <utility>

namespace blackframe::extensions {
namespace {

[[nodiscard]] auto from_dynamic_library_error(const DynamicLibraryError& error,
                                              const ExtensionLoadErrorCode code)
    -> ExtensionLoadError {
    return ExtensionLoadError{
        .code = code,
        .extension_status = BLACKFRAME_STATUS_SUCCESS,
        .native_error = error.native_error,
        .path = error.path,
        .message = error.message,
    };
}

[[nodiscard]] auto make_extension_error(const ExtensionLoadErrorCode code,
                                        const std::filesystem::path& path, std::string message,
                                        const blackframe_status status = BLACKFRAME_STATUS_SUCCESS)
    -> ExtensionLoadError {
    return ExtensionLoadError{
        .code = code,
        .extension_status = status,
        .native_error = 0,
        .path = path,
        .message = std::move(message),
    };
}

} // namespace

ExtensionModule::ExtensionModule(DynamicLibrary library,
                                 const blackframe_extension_api_v1 extension_api) noexcept
    : library_(std::move(library)), extension_api_(extension_api) {}

ExtensionModule::~ExtensionModule() {
    shutdown();
}

ExtensionModule::ExtensionModule(ExtensionModule&& other) noexcept
    : library_(std::move(other.library_)), extension_api_(std::exchange(other.extension_api_, {})) {
}

auto ExtensionModule::operator=(ExtensionModule&& other) noexcept -> ExtensionModule& {
    if (this != &other) {
        shutdown();
        library_ = std::move(other.library_);
        extension_api_ = std::exchange(other.extension_api_, {});
    }
    return *this;
}

auto ExtensionModule::load(const std::filesystem::path& absolute_path)
    -> std::expected<ExtensionModule, ExtensionLoadError> {
    auto library = DynamicLibrary::open(absolute_path);
    if (!library) {
        return std::unexpected(from_dynamic_library_error(
            library.error(), ExtensionLoadErrorCode::DynamicLibraryFailure));
    }

    auto query_symbol = library->find_symbol(BLACKFRAME_EXTENSION_QUERY_SYMBOL);
    if (!query_symbol) {
        return std::unexpected(from_dynamic_library_error(
            query_symbol.error(), ExtensionLoadErrorCode::QuerySymbolMissing));
    }

    static_assert(sizeof(blackframe_query_extension_fn) == sizeof(void*));
    blackframe_query_extension_fn query_extension{};
    void* const raw_query_symbol = *query_symbol;
    std::memcpy(&query_extension, &raw_query_symbol, sizeof(query_extension));

    const blackframe_extension_query_v1 query{
        .struct_size = sizeof(blackframe_extension_query_v1),
        .abi_major = BLACKFRAME_EXTENSION_ABI_MAJOR,
        .abi_minor = BLACKFRAME_EXTENSION_ABI_MINOR,
        .reserved = 0,
    };

    blackframe_extension_api_v1 extension_api{};
    extension_api.struct_size = sizeof(blackframe_extension_api_v1);

    const blackframe_status query_status = query_extension(&query, &extension_api);
    if (query_status != BLACKFRAME_STATUS_SUCCESS) {
        return std::unexpected(make_extension_error(
            ExtensionLoadErrorCode::QueryRejected, library->path(),
            "The extension rejected the Blackframe extension ABI.", query_status));
    }

    const bool malformed = extension_api.struct_size < sizeof(blackframe_extension_api_v1) ||
                           extension_api.abi_major != BLACKFRAME_EXTENSION_ABI_MAJOR ||
                           extension_api.abi_minor > BLACKFRAME_EXTENSION_ABI_MINOR ||
                           extension_api.reserved != 0 || extension_api.get_interface == nullptr ||
                           extension_api.shutdown == nullptr;
    if (malformed) {
        if (extension_api.shutdown != nullptr) {
            extension_api.shutdown(extension_api.extension_context);
        }
        return std::unexpected(make_extension_error(
            ExtensionLoadErrorCode::MalformedExtensionApi, library->path(),
            "The extension returned an incomplete or incompatible API table."));
    }

    return ExtensionModule{std::move(*library), extension_api};
}

auto ExtensionModule::get_interface(const blackframe_interface_id interface_id,
                                    const std::uint32_t requested_major,
                                    const std::uint32_t requested_minor, void* const out_interface,
                                    const std::uint32_t out_interface_size) const noexcept
    -> blackframe_status {
    if (!is_active()) {
        return BLACKFRAME_STATUS_INVALID_STATE;
    }
    if (out_interface == nullptr || out_interface_size == 0) {
        return BLACKFRAME_STATUS_INVALID_ARGUMENT;
    }

    return extension_api_.get_interface(extension_api_.extension_context, interface_id,
                                        requested_major, requested_minor, out_interface,
                                        out_interface_size);
}

void ExtensionModule::shutdown() noexcept {
    if (!is_active()) {
        return;
    }

    extension_api_.shutdown(extension_api_.extension_context);
    extension_api_ = {};
}

auto ExtensionModule::is_active() const noexcept -> bool {
    return library_.is_open() && extension_api_.shutdown != nullptr;
}

auto ExtensionModule::path() const noexcept -> const std::filesystem::path& {
    return library_.path();
}

} // namespace blackframe::extensions
