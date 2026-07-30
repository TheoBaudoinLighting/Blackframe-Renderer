#include "CornellDiffuseImageRenderer.hpp"

#include <Blackframe/Renderer/ExrWriter.hpp>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#if !defined(BLACKFRAME_CORNELL_SCENE_64_SHA256) || !defined(BLACKFRAME_CORNELL_SCENE_256_SHA256)
#error "Cornell reference generator scene provenance is incomplete."
#endif

namespace blackframe::renderer::cornell_test {
namespace {

struct GeneratorArguments final {
    const CornellImageSpecification* specification{};
    std::filesystem::path output_path;
    std::string source_base_commit;
    std::string scene_sha256;
    std::uint32_t worker_count{};
};

class TemporaryReferenceFile final {
  public:
    TemporaryReferenceFile(const TemporaryReferenceFile&) = delete;
    TemporaryReferenceFile& operator=(const TemporaryReferenceFile&) = delete;

    TemporaryReferenceFile(TemporaryReferenceFile&& other) noexcept
        : path_{std::exchange(other.path_, {})} {}

    TemporaryReferenceFile& operator=(TemporaryReferenceFile&&) = delete;

    ~TemporaryReferenceFile() {
        if (!path_.empty()) {
            auto ignored_error = std::error_code{};
            std::filesystem::remove(path_, ignored_error);
        }
    }

    [[nodiscard]] static core::Result<TemporaryReferenceFile>
    reserve_next_to(const std::filesystem::path& output_path) {
        for (auto suffix = std::uint32_t{0}; suffix < 1024; ++suffix) {
            auto candidate =
                output_path.parent_path() /
                (output_path.stem().string() + ".partial." + std::to_string(suffix) + ".exr");
            auto reservation = std::ofstream{
                candidate,
                std::ios::binary | std::ios::out | std::ios::noreplace,
            };
            if (reservation.is_open()) {
                reservation.close();
                if (!reservation) {
                    auto ignored_error = std::error_code{};
                    std::filesystem::remove(candidate, ignored_error);
                    return std::unexpected(core::Error{
                        .code = core::StatusCode::platform_error,
                        .message = "The reference generator could not close its temporary output.",
                    });
                }
                return TemporaryReferenceFile{std::move(candidate)};
            }

            auto inspection_error = std::error_code{};
            const auto candidate_exists = std::filesystem::exists(candidate, inspection_error);
            if (inspection_error || !candidate_exists) {
                return std::unexpected(core::Error{
                    .code = core::StatusCode::platform_error,
                    .message = "The reference generator could not reserve a temporary output.",
                });
            }
        }
        return std::unexpected(core::Error{
            .code = core::StatusCode::resource_exhausted,
            .message = "The reference generator exhausted its temporary output namespace.",
        });
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

    [[nodiscard]] core::Status install_at(const std::filesystem::path& output_path) {
        auto link_error = std::error_code{};
        std::filesystem::create_hard_link(path_, output_path, link_error);
        if (link_error) {
            auto inspection_error = std::error_code{};
            const auto output_exists = std::filesystem::exists(output_path, inspection_error);
            if (!inspection_error && output_exists) {
                return std::unexpected(core::Error{
                    .code = core::StatusCode::invalid_argument,
                    .message = "The reference generator refuses to overwrite an existing output.",
                });
            }
            return std::unexpected(core::Error{
                .code = core::StatusCode::platform_error,
                .message =
                    "The reference generator could not atomically install its completed output.",
            });
        }

        auto removal_error = std::error_code{};
        std::filesystem::remove(path_, removal_error);
        if (removal_error) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::platform_error,
                .message = "The reference generator installed its output but could not remove "
                           "the temporary hard link.",
            });
        }
        path_.clear();
        return {};
    }

  private:
    explicit TemporaryReferenceFile(std::filesystem::path path) noexcept : path_{std::move(path)} {}

    std::filesystem::path path_;
};

[[nodiscard]] bool is_lower_hexadecimal(const std::string_view value,
                                        const std::size_t digit_count) noexcept {
    if (value.size() != digit_count) {
        return false;
    }
    for (const auto character : value) {
        if (!((character >= '0' && character <= '9') || (character >= 'a' && character <= 'f'))) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] core::Result<std::uint32_t> parse_worker_count(const std::string_view text) {
    auto value = std::uint32_t{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value == 0 ||
        value > CornellMaximumWorkerCount) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "The reference generator worker count must be an integer in [1, 64].",
        });
    }
    return value;
}

[[nodiscard]] core::Result<GeneratorArguments> parse_arguments(const int argc, char** const argv) {
    if (argc != 11) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "Usage: BlackframeCornellReferenceGenerator --variant <64x64|256x256> "
                       "--output <absolute.exr> --source-base-commit <40-lowercase-hex> "
                       "--scene-sha256 <64-lowercase-hex> --workers <1..64>",
        });
    }

    auto arguments = GeneratorArguments{};
    for (auto index = 1; index < argc; index += 2) {
        const auto option = std::string_view{argv[index]};
        const auto value = std::string_view{argv[index + 1]};
        if (option == "--variant") {
            if (arguments.specification != nullptr) {
                return std::unexpected(core::Error{
                    .code = core::StatusCode::invalid_argument,
                    .message = "The reference generator variant was provided more than once.",
                });
            }
            if (value == Cornell64Specification.variant) {
                arguments.specification = &Cornell64Specification;
            } else if (value == Cornell256Specification.variant) {
                arguments.specification = &Cornell256Specification;
            } else {
                return std::unexpected(core::Error{
                    .code = core::StatusCode::invalid_argument,
                    .message = "The reference generator received an unsupported scene variant.",
                });
            }
        } else if (option == "--output") {
            if (!arguments.output_path.empty()) {
                return std::unexpected(core::Error{
                    .code = core::StatusCode::invalid_argument,
                    .message = "The reference generator output was provided more than once.",
                });
            }
            arguments.output_path = std::filesystem::path{value};
        } else if (option == "--source-base-commit") {
            if (!arguments.source_base_commit.empty()) {
                return std::unexpected(core::Error{
                    .code = core::StatusCode::invalid_argument,
                    .message = "The reference source base commit was provided more than once.",
                });
            }
            arguments.source_base_commit = value;
        } else if (option == "--scene-sha256") {
            if (!arguments.scene_sha256.empty()) {
                return std::unexpected(core::Error{
                    .code = core::StatusCode::invalid_argument,
                    .message = "The reference scene hash was provided more than once.",
                });
            }
            arguments.scene_sha256 = value;
        } else if (option == "--workers") {
            if (arguments.worker_count != 0) {
                return std::unexpected(core::Error{
                    .code = core::StatusCode::invalid_argument,
                    .message = "The reference generator worker count was provided more than once.",
                });
            }
            const auto workers = parse_worker_count(value);
            if (!workers.has_value()) {
                return std::unexpected(workers.error());
            }
            arguments.worker_count = *workers;
        } else {
            return std::unexpected(core::Error{
                .code = core::StatusCode::invalid_argument,
                .message = "The reference generator received an unknown command-line option.",
            });
        }
    }

    if (arguments.specification == nullptr || arguments.output_path.empty() ||
        !is_lower_hexadecimal(arguments.source_base_commit, 40) ||
        !is_lower_hexadecimal(arguments.scene_sha256, 64) || arguments.worker_count == 0) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "The reference generator requires every explicit argument.",
        });
    }
    if (!arguments.output_path.is_absolute() ||
        arguments.output_path.filename() != arguments.specification->reference_filename) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "The reference output must be absolute and use the canonical filename.",
        });
    }
    const auto expected_scene_sha256 = arguments.specification == &Cornell64Specification
                                           ? std::string_view{BLACKFRAME_CORNELL_SCENE_64_SHA256}
                                           : std::string_view{BLACKFRAME_CORNELL_SCENE_256_SHA256};
    if (arguments.scene_sha256 != expected_scene_sha256) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::incompatible,
            .message =
                "The requested scene SHA-256 does not match the fixture compiled into this tool.",
        });
    }
    std::error_code filesystem_error;
    if (!std::filesystem::is_directory(arguments.output_path.parent_path(), filesystem_error) ||
        filesystem_error) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::not_found,
            .message = "The reference output directory does not exist.",
        });
    }
    filesystem_error.clear();
    if (std::filesystem::exists(arguments.output_path, filesystem_error) || filesystem_error) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "The reference generator refuses to overwrite an existing output.",
        });
    }
    return arguments;
}

[[nodiscard]] core::Status generate_reference(const GeneratorArguments& arguments) {
    auto temporary_output = TemporaryReferenceFile::reserve_next_to(arguments.output_path);
    if (!temporary_output.has_value()) {
        return std::unexpected(temporary_output.error());
    }
    const auto reference = render_cornell_image<ReferenceScalar>(
        *arguments.specification, arguments.specification->reference_samples_per_pixel,
        CornellReferenceSeed, arguments.worker_count);
    if (!reference.has_value()) {
        return std::unexpected(reference.error());
    }
    const auto quantized = quantize_cornell_reference(*reference);
    if (!quantized.has_value()) {
        return std::unexpected(quantized.error());
    }

    const auto write_status =
        write_scene_linear_exr(*quantized, temporary_output->path(),
                               ExrRunMetadata{
                                   .scene = cornell_reference_scene_path(*arguments.specification),
                                   .seed = CornellReferenceSeed,
                                   .commit = arguments.source_base_commit,
                                   .options = cornell_reference_options(*arguments.specification),
                                   .backend = CornellReferenceBackend,
                                   .capabilities = CornellReferenceCapabilities,
                                   .asset_hashes = "scene=sha256:" + arguments.scene_sha256,
                               });
    if (!write_status.has_value()) {
        return std::unexpected(write_status.error());
    }
    return temporary_output->install_at(arguments.output_path);
}

} // namespace
} // namespace blackframe::renderer::cornell_test

int main(const int argc, char** const argv) {
    const auto arguments = blackframe::renderer::cornell_test::parse_arguments(argc, argv);
    if (!arguments.has_value()) {
        std::cerr << arguments.error().message << '\n';
        return 2;
    }
    const auto status = blackframe::renderer::cornell_test::generate_reference(*arguments);
    if (!status.has_value()) {
        std::cerr << status.error().message << '\n';
        return 1;
    }
    std::cout << arguments->output_path.string() << '\n';
    return 0;
}
