#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/Film.hpp>
#include <cstdint>
#include <filesystem>
#include <string>

namespace blackframe::renderer {

struct ExrRunMetadata final {
    std::string scene;
    std::uint64_t seed{};
    std::string commit;
    std::string options;
    std::string backend;
    std::string capabilities;
    std::string asset_hashes;

    [[nodiscard]] bool operator==(const ExrRunMetadata&) const noexcept = default;
};

// Writes the active film crop as losslessly compressed 32-bit scene-linear RGB.
// The display window retains the full film extent and every run field is mandatory.
[[nodiscard]] core::Status write_scene_linear_exr(const Film& film,
                                                  const std::filesystem::path& output_path,
                                                  const ExrRunMetadata& metadata);

} // namespace blackframe::renderer
