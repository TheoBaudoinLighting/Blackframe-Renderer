#include <Blackframe/Renderer/AreaLights.hpp>
#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/PngWriter.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <gtest/gtest.h>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

#if BLACKFRAME_HAS_PNG_PREVIEW
#include <stb_image.h>
#endif

namespace blackframe::renderer {
namespace {

#if BLACKFRAME_HAS_PNG_PREVIEW

inline constexpr auto AreaLightAtlasExtent = RenderExtent{.width = 256U, .height = 128U};
inline constexpr auto AreaLightPanelWidth = std::uint32_t{64U};
inline constexpr auto AreaLightHalfPanelWidth = std::uint32_t{32U};
inline constexpr auto AreaLightPanelHeight = std::uint32_t{64U};

[[nodiscard]] std::optional<std::filesystem::path> area_checksum_output_path() {
#if defined(_WIN32)
    auto* value = static_cast<char*>(nullptr);
    auto value_size = std::size_t{};
    if (_dupenv_s(&value, &value_size, "BLACKFRAME_PNG_CHECKSUM_OUTPUT") != 0 || value == nullptr) {
        return std::nullopt;
    }
    const auto path = value_size > 1 ? std::optional{std::filesystem::path{value}} : std::nullopt;
    std::free(value);
    return path;
#else
    const auto* const value = std::getenv("BLACKFRAME_PNG_CHECKSUM_OUTPUT");
    if (value == nullptr || *value == '\0') {
        return std::nullopt;
    }
    return std::filesystem::path{value};
#endif
}

[[nodiscard]] std::filesystem::path area_atlas_output_path() {
    if (const auto checksum_output = area_checksum_output_path(); checksum_output.has_value()) {
        return *checksum_output;
    }
    return std::filesystem::path{BLACKFRAME_LIGHT_TEST_OUTPUT_DIR} / "area-light-pdf-sides.png";
}

[[nodiscard]] TransportSpectrum area_constant_spectrum(const TransportScalar value) {
    auto result = TransportSpectrum{};
    result.values.fill(value);
    return result;
}

struct AtlasQuery final {
    Point3 context_position;
    Vector3 direction_to_light;
};

[[nodiscard]] Point2 atlas_local_position(const std::uint32_t x, const std::uint32_t y) noexcept {
    const auto half_x = x % AreaLightHalfPanelWidth;
    const auto panel_y = y % AreaLightPanelHeight;
    return Point2{
        .x = 2.4F * (static_cast<TransportScalar>(half_x) + 0.5F) /
                 static_cast<TransportScalar>(AreaLightHalfPanelWidth) -
             1.2F,
        .y = 1.2F - 2.4F * (static_cast<TransportScalar>(panel_y) + 0.5F) /
                        static_cast<TransportScalar>(AreaLightPanelHeight),
    };
}

[[nodiscard]] AtlasQuery planar_query(const Point2 local, const bool back) noexcept {
    return back ? AtlasQuery{
                      .context_position = {.x = local.x, .y = local.y, .z = -2.0F},
                      .direction_to_light = {.z = 1.0F},
                  }
                : AtlasQuery{
                      .context_position = {.x = local.x, .y = local.y, .z = 2.0F},
                      .direction_to_light = {.z = -1.0F},
                  };
}

[[nodiscard]] AtlasQuery sphere_query(const Point2 local, const bool interior) noexcept {
    return interior ? AtlasQuery{
                          .context_position = {.x = local.x, .y = local.y, .z = 0.0F},
                          .direction_to_light = {.z = 1.0F},
                      }
                    : AtlasQuery{
                          .context_position = {.x = local.x, .y = local.y, .z = 2.5F},
                          .direction_to_light = {.z = -1.0F},
                      };
}

template <typename Light>
[[nodiscard]] core::Result<TransportScalar> atlas_pdf(const Light& light, const AtlasQuery query,
                                                      const SampledWavelengths& wavelengths) {
    const auto context = LightSampleContext::create(query.context_position, 0.0F);
    if (!context) {
        return std::unexpected(context.error());
    }
    const auto pdf = light.pdf_li(*context, query.direction_to_light, wavelengths);
    if (!pdf) {
        return std::unexpected(pdf.error());
    }
    return pdf->value();
}

[[nodiscard]] std::vector<Point3> atlas_mesh_positions() {
    return {
        {.x = 0.0F, .y = -0.9F, .z = 0.0F},
        {.x = 0.95F, .y = 0.0F, .z = 0.0F},
        {.x = 0.0F, .y = 0.9F, .z = 0.0F},
        {.x = -0.65F, .y = 0.0F, .z = 0.0F},
    };
}

[[nodiscard]] std::vector<AreaLightTriangleIndices> atlas_mesh_triangles() {
    return {
        {.vertex0 = 0U, .vertex1 = 1U, .vertex2 = 2U},
        {.vertex0 = 0U, .vertex1 = 2U, .vertex2 = 3U},
    };
}

TEST(AreaLightImageTest, WritesStablePdfAndSidednessAtlas) {
    const auto wavelengths = sample_uniform_visible_wavelengths(0.25F);
    ASSERT_TRUE(wavelengths.has_value()) << wavelengths.error().message;
    const auto radiance = area_constant_spectrum(1.0F);

    const auto rectangle_one = RectangleAreaLight::create(
        Point3{}, Normal3{.z = 1.0F}, Vector3{.x = 1.0F}, 0.85F, 0.65F, Vector3{},
        AreaLightSidedness::one_sided, *wavelengths, radiance);
    const auto rectangle_two = RectangleAreaLight::create(
        Point3{}, Normal3{.z = 1.0F}, Vector3{.x = 1.0F}, 0.85F, 0.65F, Vector3{},
        AreaLightSidedness::two_sided, *wavelengths, radiance);
    const auto disk_one =
        DiskAreaLight::create(Point3{}, Normal3{.z = 1.0F}, 0.85F, Vector3{},
                              AreaLightSidedness::one_sided, *wavelengths, radiance);
    const auto disk_two =
        DiskAreaLight::create(Point3{}, Normal3{.z = 1.0F}, 0.85F, Vector3{},
                              AreaLightSidedness::two_sided, *wavelengths, radiance);
    const auto sphere_one = SphereAreaLight::create(
        Point3{}, 0.85F, Vector3{}, AreaLightSidedness::one_sided, *wavelengths, radiance);
    const auto sphere_two = SphereAreaLight::create(
        Point3{}, 0.85F, Vector3{}, AreaLightSidedness::two_sided, *wavelengths, radiance);
    const auto mesh_one =
        MeshAreaLight::create(atlas_mesh_positions(), atlas_mesh_triangles(), Vector3{},
                              AreaLightSidedness::one_sided, *wavelengths, radiance);
    const auto mesh_two =
        MeshAreaLight::create(atlas_mesh_positions(), atlas_mesh_triangles(), Vector3{},
                              AreaLightSidedness::two_sided, *wavelengths, radiance);
    ASSERT_TRUE(rectangle_one.has_value()) << rectangle_one.error().message;
    ASSERT_TRUE(rectangle_two.has_value()) << rectangle_two.error().message;
    ASSERT_TRUE(disk_one.has_value()) << disk_one.error().message;
    ASSERT_TRUE(disk_two.has_value()) << disk_two.error().message;
    ASSERT_TRUE(sphere_one.has_value()) << sphere_one.error().message;
    ASSERT_TRUE(sphere_two.has_value()) << sphere_two.error().message;
    ASSERT_TRUE(mesh_one.has_value()) << mesh_one.error().message;
    ASSERT_TRUE(mesh_two.has_value()) << mesh_two.error().message;

    auto film = Film::create(AreaLightAtlasExtent);
    ASSERT_TRUE(film.has_value()) << film.error().message;
    for (auto y = std::uint32_t{}; y < AreaLightAtlasExtent.height; ++y) {
        for (auto x = std::uint32_t{}; x < AreaLightAtlasExtent.width; ++x) {
            const auto panel = x / AreaLightPanelWidth;
            const auto back_or_interior = (x % AreaLightPanelWidth) >= AreaLightHalfPanelWidth;
            const auto two_sided = y >= AreaLightPanelHeight;
            const auto local = atlas_local_position(x, y);
            const auto query = panel == 2U ? sphere_query(local, back_or_interior)
                                           : planar_query(local, back_or_interior);

            auto response = core::Result<TransportScalar>{};
            switch (panel) {
            case 0U:
                response =
                    atlas_pdf(two_sided ? *rectangle_two : *rectangle_one, query, *wavelengths);
                break;
            case 1U:
                response = atlas_pdf(two_sided ? *disk_two : *disk_one, query, *wavelengths);
                break;
            case 2U:
                response = atlas_pdf(two_sided ? *sphere_two : *sphere_one, query, *wavelengths);
                break;
            case 3U:
                response = atlas_pdf(two_sided ? *mesh_two : *mesh_one, query, *wavelengths);
                break;
            default:
                FAIL() << "Area-light atlas panel index escaped its fixed layout.";
            }
            ASSERT_TRUE(response.has_value())
                << "pixel (" << x << ", " << y << "): " << response.error().message;
            const auto gray = std::min(*response * 0.35F, 1.0F);
            const auto accumulated =
                film->add_sample(x, y, LinearRGB{.red = gray, .green = gray, .blue = gray}, 1.0F);
            ASSERT_TRUE(accumulated.has_value())
                << "pixel (" << x << ", " << y << "): " << accumulated.error().message;
        }
    }

    const auto output_path = area_atlas_output_path();
    std::error_code cleanup_error;
    std::filesystem::remove(output_path, cleanup_error);
    ASSERT_FALSE(cleanup_error) << "Cannot replace '" << output_path.string()
                                << "': " << cleanup_error.message();
    const auto write_status = write_png_preview(*film, output_path);
    ASSERT_TRUE(write_status.has_value()) << write_status.error().message;
    ASSERT_TRUE(std::filesystem::is_regular_file(output_path));

    auto width = int{};
    auto height = int{};
    auto components = int{};
    auto* const decoded = stbi_load(output_path.string().c_str(), &width, &height, &components, 3);
    ASSERT_NE(decoded, nullptr) << "Cannot decode '" << output_path.string()
                                << "': " << stbi_failure_reason();
    EXPECT_EQ(width, static_cast<int>(AreaLightAtlasExtent.width));
    EXPECT_EQ(height, static_cast<int>(AreaLightAtlasExtent.height));
    EXPECT_EQ(components, 3);
    stbi_image_free(decoded);
}

#else

TEST(AreaLightImageTest, WritesStablePdfAndSidednessAtlas) {
    GTEST_SKIP() << "PNG preview support is disabled explicitly.";
}

#endif

} // namespace
} // namespace blackframe::renderer
