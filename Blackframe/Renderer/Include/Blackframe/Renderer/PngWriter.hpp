#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/Film.hpp>
#include <filesystem>

namespace blackframe::renderer {

// Writes the active film crop as an 8-bit RGB PNG preview. The display pipeline
// is fixed to zero-stop exposure, clipping to the sRGB display range, the
// IEC 61966-2-1 transfer function, and round-to-nearest quantization.
[[nodiscard]] core::Status write_png_preview(const Film& film,
                                             const std::filesystem::path& output_path);

} // namespace blackframe::renderer
