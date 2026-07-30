#include <Blackframe/Engine/TriangleMeshImport.hpp>
#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/GeometryOperations.hpp>
#include <Blackframe/Renderer/LocalFrame.hpp>
#include <Blackframe/Renderer/PinholeCamera.hpp>
#include <Blackframe/Renderer/SurfaceInteraction.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

constexpr auto ReferenceExtent = renderer::RenderExtent{.width = 8, .height = 8};
constexpr auto ReferenceTime = renderer::TransportScalar{0.25F};
constexpr auto ReferenceNormalChecksum = std::uint64_t{4624172394029642977ULL};
constexpr auto ReferenceUvChecksum = std::uint64_t{2983303244280021857ULL};

[[nodiscard]] std::filesystem::path fixture_path(const char* const filename) {
    return std::filesystem::path{BLACKFRAME_ENGINE_MESH_FIXTURE_DIR} / filename;
}

[[nodiscard]] std::filesystem::path artifact_path(const char* const filename) {
    return std::filesystem::path{BLACKFRAME_ENGINE_TEST_OUTPUT_DIR} / filename;
}

template <typename Result>
void expect_error(const Result& result, const core::StatusCode code,
                  const bool require_line = false) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, code);
    EXPECT_FALSE(result.error().message.empty());
    if (require_line) {
        EXPECT_NE(result.error().message.find("line"), std::string::npos);
    }
}

struct ClosestMeshHit final {
    std::size_t triangle_index;
    renderer::TriangleHit hit;
};

struct MeshRender final {
    renderer::Film normal;
    renderer::Film uv;
    std::uint32_t hit_count;
};

[[nodiscard]] core::Result<std::vector<renderer::Triangle>>
make_geometric_triangles(const TriangleMesh& mesh) {
    auto result = std::vector<renderer::Triangle>{};
    result.reserve(mesh.triangles().size());
    for (auto triangle_index = std::size_t{}; triangle_index < mesh.triangles().size();
         ++triangle_index) {
        auto triangle = mesh.geometric_triangle(triangle_index);
        if (!triangle) {
            return std::unexpected(triangle.error());
        }
        result.push_back(std::move(*triangle));
    }
    return result;
}

[[nodiscard]] core::Result<std::optional<ClosestMeshHit>>
trace_mesh(const renderer::Ray& ray, const std::vector<renderer::Triangle>& triangles) {
    auto closest = std::optional<ClosestMeshHit>{};
    for (auto triangle_index = std::size_t{}; triangle_index < triangles.size(); ++triangle_index) {
        const auto intersection = triangles[triangle_index].intersect(ray);
        if (!intersection) {
            return std::unexpected(intersection.error());
        }
        if (!intersection->has_value()) {
            continue;
        }
        if (closest && (**intersection).parameter >= closest->hit.parameter) {
            continue;
        }
        closest = ClosestMeshHit{
            .triangle_index = triangle_index,
            .hit = **intersection,
        };
    }
    return closest;
}

[[nodiscard]] core::Result<renderer::SurfaceInteraction>
make_surface_interaction(const TriangleMesh& mesh, const ClosestMeshHit& closest,
                         const renderer::Ray& ray) {
    const auto& vertex_indices = mesh.triangles()[closest.triangle_index].vertices;
    const auto& barycentrics = closest.hit.barycentrics;
    const auto weights = std::array{
        barycentrics.vertex0,
        barycentrics.vertex1,
        barycentrics.vertex2,
    };

    const auto interpolate = [&weights](const auto& values) {
        using Value = std::remove_cvref_t<decltype(values[0])>;
        return Value{
            .x = std::fma(values[0].x, weights[0],
                          std::fma(values[1].x, weights[1], values[2].x * weights[2])),
            .y = std::fma(values[0].y, weights[0],
                          std::fma(values[1].y, weights[1], values[2].y * weights[2])),
        };
    };

    const auto mesh_normals = mesh.normals();
    const auto corner_normals = std::array{
        mesh_normals[vertex_indices[0]],
        mesh_normals[vertex_indices[1]],
        mesh_normals[vertex_indices[2]],
    };
    const auto interpolated_normal = renderer::Normal3{
        .x = std::fma(corner_normals[0].x, weights[0],
                      std::fma(corner_normals[1].x, weights[1], corner_normals[2].x * weights[2])),
        .y = std::fma(corner_normals[0].y, weights[0],
                      std::fma(corner_normals[1].y, weights[1], corner_normals[2].y * weights[2])),
        .z = std::fma(corner_normals[0].z, weights[0],
                      std::fma(corner_normals[1].z, weights[1], corner_normals[2].z * weights[2])),
    };
    const auto shading_normal = renderer::normalized(interpolated_normal);
    if (!shading_normal) {
        return std::unexpected(shading_normal.error());
    }

    const auto mesh_uvs = mesh.texture_coordinates();
    const auto corner_uvs = std::array{
        mesh_uvs[vertex_indices[0]],
        mesh_uvs[vertex_indices[1]],
        mesh_uvs[vertex_indices[2]],
    };
    const auto uv = interpolate(corner_uvs);

    const auto mesh_positions = mesh.positions();
    const auto edge1 = mesh_positions[vertex_indices[1]] - mesh_positions[vertex_indices[0]];
    const auto edge2 = mesh_positions[vertex_indices[2]] - mesh_positions[vertex_indices[0]];
    const auto duv1 = renderer::Point2{
        .x = corner_uvs[1].x - corner_uvs[0].x,
        .y = corner_uvs[1].y - corner_uvs[0].y,
    };
    const auto duv2 = renderer::Point2{
        .x = corner_uvs[2].x - corner_uvs[0].x,
        .y = corner_uvs[2].y - corner_uvs[0].y,
    };
    const auto determinant = duv1.x * duv2.y - duv1.y * duv2.x;
    if (!std::isfinite(determinant) || determinant == renderer::TransportScalar{0}) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "Reference mesh UV derivatives must remain non-singular.",
        });
    }
    const auto dpdu = (edge1 * duv2.y - edge2 * duv1.y) / determinant;
    const auto dpdv = (edge2 * duv1.x - edge1 * duv2.x) / determinant;

    return renderer::SurfaceInteraction::create(
        closest.hit.position, closest.hit.geometric_normal, *shading_normal, uv, dpdu, dpdv,
        renderer::SurfaceIdentifiers{
            .instance = {.value = 1},
            .geometry = {.value = 2},
            .primitive = {.value = static_cast<std::uint32_t>(closest.triangle_index)},
            .material = {.value = 3},
        },
        ray.time());
}

[[nodiscard]] core::Result<MeshRender> render_reference_mesh(const TriangleMesh& mesh) {
    const auto frame = renderer::OrthonormalFrame::from_normal_and_tangent(
        renderer::Normal3{.z = 1.0F}, renderer::Vector3{.x = 1.0F});
    if (!frame) {
        return std::unexpected(frame.error());
    }
    const auto camera =
        renderer::PinholeCamera::create(renderer::Point3{}, *frame, ReferenceExtent,
                                        std::numbers::pi_v<renderer::TransportScalar> / 2.0F, 0.0F,
                                        4.0F, renderer::AllRayVisibility, renderer::VacuumMedium);
    if (!camera) {
        return std::unexpected(camera.error());
    }
    const auto triangles = make_geometric_triangles(mesh);
    if (!triangles) {
        return std::unexpected(triangles.error());
    }
    auto normal_film = renderer::Film::create(ReferenceExtent);
    auto uv_film = renderer::Film::create(ReferenceExtent);
    if (!normal_film) {
        return std::unexpected(normal_film.error());
    }
    if (!uv_film) {
        return std::unexpected(uv_film.error());
    }

    auto hit_count = std::uint32_t{};
    for (auto y = std::uint32_t{}; y < ReferenceExtent.height; ++y) {
        for (auto x = std::uint32_t{}; x < ReferenceExtent.width; ++x) {
            const auto ray = camera->generate_primary_ray(
                renderer::PixelSampleIndex{
                    .pixel_x = x,
                    .pixel_y = y,
                    .sample_index = 0,
                    .seed = 0,
                },
                renderer::PixelJitterMode::center, ReferenceTime);
            if (!ray) {
                return std::unexpected(ray.error());
            }
            const auto closest = trace_mesh(*ray, *triangles);
            if (!closest) {
                return std::unexpected(closest.error());
            }

            auto normal_color = renderer::LinearRGB{};
            auto uv_color = renderer::LinearRGB{};
            if (closest->has_value()) {
                const auto interaction = make_surface_interaction(mesh, **closest, *ray);
                if (!interaction) {
                    return std::unexpected(interaction.error());
                }
                const auto& normal = interaction->shading_normal();
                normal_color = renderer::LinearRGB{
                    .red = (normal.x + 1.0F) / 2.0F,
                    .green = (normal.y + 1.0F) / 2.0F,
                    .blue = (normal.z + 1.0F) / 2.0F,
                };
                uv_color = renderer::LinearRGB{
                    .red = interaction->uv().x,
                    .green = interaction->uv().y,
                };
                ++hit_count;
            }

            const auto normal_status = normal_film->add_sample(x, y, normal_color, 1.0F);
            const auto uv_status = uv_film->add_sample(x, y, uv_color, 1.0F);
            if (!normal_status) {
                return std::unexpected(normal_status.error());
            }
            if (!uv_status) {
                return std::unexpected(uv_status.error());
            }
        }
    }

    return MeshRender{
        .normal = std::move(*normal_film),
        .uv = std::move(*uv_film),
        .hit_count = hit_count,
    };
}

[[nodiscard]] renderer::LinearRGB resolved(const renderer::Film& film, const std::uint32_t x,
                                           const std::uint32_t y) {
    const auto pixel = film.resolved_pixel(x, y);
    EXPECT_TRUE(pixel.has_value());
    return pixel.value_or(renderer::LinearRGB{});
}

void expect_same_film(const renderer::Film& left, const renderer::Film& right) {
    ASSERT_EQ(left.extent().width, right.extent().width);
    ASSERT_EQ(left.extent().height, right.extent().height);
    for (auto y = std::uint32_t{}; y < left.extent().height; ++y) {
        for (auto x = std::uint32_t{}; x < left.extent().width; ++x) {
            EXPECT_EQ(resolved(left, x, y), resolved(right, x, y)) << x << ", " << y;
        }
    }
}

[[nodiscard]] std::uint64_t quantized_film_checksum(const renderer::Film& film) {
    auto checksum = std::uint64_t{14695981039346656037ULL};
    const auto append = [&checksum](const std::uint16_t value) {
        for (auto shift = 0U; shift < 16U; shift += 8U) {
            checksum ^= static_cast<std::uint8_t>(value >> shift);
            checksum *= std::uint64_t{1099511628211ULL};
        }
    };

    for (auto y = std::uint32_t{}; y < film.extent().height; ++y) {
        for (auto x = std::uint32_t{}; x < film.extent().width; ++x) {
            const auto color = resolved(film, x, y);
            const auto channels = std::array{color.red, color.green, color.blue};
            for (const auto channel : channels) {
                EXPECT_TRUE(std::isfinite(channel));
                EXPECT_GE(channel, 0.0F);
                EXPECT_LE(channel, 1.0F);
                const auto bounded = std::clamp(channel, 0.0F, 1.0F);
                append(static_cast<std::uint16_t>(std::lround(bounded * 4095.0F)));
            }
        }
    }
    return checksum;
}

void write_artifact(const std::filesystem::path& path, const std::string_view contents) {
    auto output = std::ofstream{path, std::ios::binary | std::ios::trunc};
    ASSERT_TRUE(output.is_open());
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    ASSERT_TRUE(output.good());
}

void remove_artifact(const std::filesystem::path& path) {
    auto error = std::error_code{};
    static_cast<void>(std::filesystem::remove(path, error));
    EXPECT_FALSE(error);
}

TEST(TriangleMeshImportTest, LoadsIndependentObjIndicesAndPlyPropertyOrderIntoTheSameMesh) {
    const auto obj = load_obj_triangle_mesh(fixture_path("reference-mesh.obj"));
    const auto ply = load_ply_triangle_mesh(fixture_path("reference-mesh.ply"));
    ASSERT_TRUE(obj.has_value()) << obj.error().message;
    ASSERT_TRUE(ply.has_value()) << ply.error().message;

    ASSERT_EQ(obj->positions().size(), std::size_t{6});
    ASSERT_EQ(obj->normals().size(), std::size_t{6});
    ASSERT_EQ(obj->texture_coordinates().size(), std::size_t{6});
    ASSERT_EQ(obj->triangles().size(), std::size_t{2});
    ASSERT_EQ(ply->positions().size(), std::size_t{6});
    ASSERT_EQ(ply->normals().size(), std::size_t{6});
    ASSERT_EQ(ply->texture_coordinates().size(), std::size_t{6});
    ASSERT_EQ(ply->triangles().size(), std::size_t{2});
    EXPECT_EQ(obj->triangles()[0], (TriangleVertexIndices{.vertices = {0, 1, 2}}));
    EXPECT_EQ(obj->triangles()[1], (TriangleVertexIndices{.vertices = {3, 4, 5}}));
    EXPECT_EQ(ply->triangles()[0], (TriangleVertexIndices{.vertices = {1, 3, 0}}));
    EXPECT_EQ(ply->triangles()[1], (TriangleVertexIndices{.vertices = {5, 4, 2}}));

    for (auto triangle_index = std::size_t{}; triangle_index < obj->triangles().size();
         ++triangle_index) {
        for (auto corner = std::size_t{}; corner < 3; ++corner) {
            const auto obj_vertex = obj->triangles()[triangle_index].vertices[corner];
            const auto ply_vertex = ply->triangles()[triangle_index].vertices[corner];
            EXPECT_EQ(obj->positions()[obj_vertex], ply->positions()[ply_vertex]);
            EXPECT_EQ(obj->normals()[obj_vertex], ply->normals()[ply_vertex]);
            EXPECT_EQ(obj->texture_coordinates()[obj_vertex],
                      ply->texture_coordinates()[ply_vertex]);
        }
    }

    EXPECT_EQ(obj->positions()[0], obj->positions()[3]);
    EXPECT_EQ(obj->texture_coordinates()[0], obj->texture_coordinates()[3]);
    EXPECT_EQ(obj->positions()[2], obj->positions()[4]);
    EXPECT_NE(obj->texture_coordinates()[2], obj->texture_coordinates()[4]);
    EXPECT_NE(obj->normals()[2], obj->normals()[4]);
}

TEST(TriangleMeshImportTest, RendersReferenceObjAndPlyWithIdenticalExpectedAttributes) {
    const auto obj = load_obj_triangle_mesh(fixture_path("reference-mesh.obj"));
    const auto ply = load_ply_triangle_mesh(fixture_path("reference-mesh.ply"));
    ASSERT_TRUE(obj.has_value()) << obj.error().message;
    ASSERT_TRUE(ply.has_value()) << ply.error().message;

    const auto obj_render = render_reference_mesh(*obj);
    const auto ply_render = render_reference_mesh(*ply);
    ASSERT_TRUE(obj_render.has_value()) << obj_render.error().message;
    ASSERT_TRUE(ply_render.has_value()) << ply_render.error().message;
    EXPECT_EQ(obj_render->hit_count, std::uint32_t{16});
    EXPECT_EQ(ply_render->hit_count, std::uint32_t{16});
    expect_same_film(obj_render->normal, ply_render->normal);
    expect_same_film(obj_render->uv, ply_render->uv);
    EXPECT_EQ(quantized_film_checksum(obj_render->normal), ReferenceNormalChecksum);
    EXPECT_EQ(quantized_film_checksum(ply_render->normal), ReferenceNormalChecksum);
    EXPECT_EQ(quantized_film_checksum(obj_render->uv), ReferenceUvChecksum);
    EXPECT_EQ(quantized_film_checksum(ply_render->uv), ReferenceUvChecksum);

    EXPECT_EQ(resolved(obj_render->normal, 0, 0), renderer::LinearRGB{});
    EXPECT_EQ(resolved(obj_render->uv, 0, 0), renderer::LinearRGB{});
    const auto lower_right_uv = resolved(obj_render->uv, 5, 5);
    EXPECT_NEAR(lower_right_uv.red, 0.875F, 1.0e-6F);
    EXPECT_NEAR(lower_right_uv.green, 0.125F, 1.0e-6F);
    const auto upper_left_uv = resolved(obj_render->uv, 2, 2);
    EXPECT_NEAR(upper_left_uv.red, 0.75F, 1.0e-6F);
    EXPECT_NEAR(upper_left_uv.green, 0.875F, 1.0e-6F);

    const auto lower_right_normal = resolved(obj_render->normal, 5, 5);
    EXPECT_GT(lower_right_normal.red, 0.5F);
    EXPECT_GT(lower_right_normal.green, 0.5F);
    EXPECT_GT(lower_right_normal.blue, 0.9F);
    const auto upper_left_normal = resolved(obj_render->normal, 2, 2);
    EXPECT_LT(upper_left_normal.red, 0.5F);
    EXPECT_LT(upper_left_normal.green, 0.5F);
    EXPECT_GT(upper_left_normal.blue, 0.9F);
}

TEST(TriangleMeshImportTest, RejectsImplicitPathsMissingFilesAndUnknownTriangles) {
    expect_error(load_obj_triangle_mesh("reference-mesh.obj"), core::StatusCode::invalid_argument);
    expect_error(load_obj_triangle_mesh(fixture_path("reference-mesh.ply")),
                 core::StatusCode::invalid_argument);
    expect_error(load_obj_triangle_mesh(fixture_path("missing.obj")), core::StatusCode::not_found);

    const auto mesh = load_obj_triangle_mesh(fixture_path("reference-mesh.obj"));
    ASSERT_TRUE(mesh.has_value()) << mesh.error().message;
    expect_error(mesh->geometric_triangle(2), core::StatusCode::not_found);
}

TEST(TriangleMeshImportTest, RejectsMalformedObjDataWithoutGeneratingAttributesOrTriangles) {
    const auto path = artifact_path("malformed.obj");
    constexpr auto incomplete_attributes = R"OBJ(v 0 0 -2
v 1 0 -2
v 0 1 -2
vt 0 0
vt 1 0
vt 0 1
vn 0 0 1
f 1/1 2/2 3/3
)OBJ";
    write_artifact(path, incomplete_attributes);
    expect_error(load_obj_triangle_mesh(path), core::StatusCode::invalid_argument, true);

    constexpr auto polygon = R"OBJ(v -1 -1 -2
v 1 -1 -2
v 1 1 -2
v -1 1 -2
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 1
f 1/1/1 2/2/1 3/3/1 4/4/1
)OBJ";
    write_artifact(path, polygon);
    expect_error(load_obj_triangle_mesh(path), core::StatusCode::invalid_argument, true);

    constexpr auto non_unit_normal = R"OBJ(v 0 0 -2
v 1 0 -2
v 0 1 -2
vt 0 0
vt 1 0
vt 0 1
vn 0 0 2
f 1/1/1 2/2/1 3/3/1
)OBJ";
    write_artifact(path, non_unit_normal);
    expect_error(load_obj_triangle_mesh(path), core::StatusCode::invalid_argument, true);

    constexpr auto non_finite_position = R"OBJ(v nan 0 -2
v 1 0 -2
v 0 1 -2
vt 0 0
vt 1 0
vt 0 1
vn 0 0 1
f 1/1/1 2/2/1 3/3/1
)OBJ";
    write_artifact(path, non_finite_position);
    expect_error(load_obj_triangle_mesh(path), core::StatusCode::invalid_argument, true);

    constexpr auto non_finite_uv = R"OBJ(v 0 0 -2
v 1 0 -2
v 0 1 -2
vt nan 0
vt 1 0
vt 0 1
vn 0 0 1
f 1/1/1 2/2/1 3/3/1
)OBJ";
    write_artifact(path, non_finite_uv);
    expect_error(load_obj_triangle_mesh(path), core::StatusCode::invalid_argument, true);

    constexpr auto degenerate_triangle = R"OBJ(v 0 0 -2
v 1 0 -2
v 2 0 -2
vt 0 0
vt 1 0
vt 0 1
vn 0 0 1
f 1/1/1 2/2/1 3/3/1
)OBJ";
    write_artifact(path, degenerate_triangle);
    expect_error(load_obj_triangle_mesh(path), core::StatusCode::invalid_argument);

    constexpr auto invalid_index = R"OBJ(v 0 0 -2
v 1 0 -2
v 0 1 -2
vt 0 0
vt 1 0
vt 0 1
vn 0 0 1
f 0/1/1 2/2/1 3/3/1
)OBJ";
    write_artifact(path, invalid_index);
    expect_error(load_obj_triangle_mesh(path), core::StatusCode::invalid_argument, true);

    constexpr auto face_record = std::string_view{"f 1/1/1 1/1/1 1/1/1\n"};
    constexpr auto excessive_face_count = std::size_t{600'000};
    auto excessive_decoded_storage = std::string{R"OBJ(v 0 0 -2
vt 0 0
vn 0 0 1
)OBJ"};
    excessive_decoded_storage.reserve(excessive_decoded_storage.size() +
                                      excessive_face_count * face_record.size());
    for (auto face = std::size_t{}; face < excessive_face_count; ++face) {
        excessive_decoded_storage.append(face_record);
    }
    write_artifact(path, excessive_decoded_storage);
    expect_error(load_obj_triangle_mesh(path), core::StatusCode::resource_exhausted);
    remove_artifact(path);
}

TEST(TriangleMeshImportTest, RejectsUnsupportedOrMalformedPlyDataWithoutFallback) {
    const auto path = artifact_path("malformed.ply");
    auto binary = std::string{R"PLY(ply
format binary_little_endian 1.0
end_header
)PLY"};
    binary.push_back('\0');
    binary.append("binary payload");
    write_artifact(path, binary);
    expect_error(load_ply_triangle_mesh(path), core::StatusCode::incompatible, true);

    constexpr auto missing_uv = R"PLY(ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
property float nx
property float ny
property float nz
element face 1
property list uchar uint vertex_indices
end_header
)PLY";
    write_artifact(path, missing_uv);
    expect_error(load_ply_triangle_mesh(path), core::StatusCode::invalid_argument, true);

    constexpr auto invalid_index = R"PLY(ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float u
property float v
element face 1
property list uchar uint vertex_indices
end_header
0 0 -2 0 0 1 0 0
1 0 -2 0 0 1 1 0
0 1 -2 0 0 1 0 1
3 0 1 9
)PLY";
    write_artifact(path, invalid_index);
    expect_error(load_ply_triangle_mesh(path), core::StatusCode::invalid_argument, true);

    constexpr auto polygon = R"PLY(ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float u
property float v
element face 1
property list uchar uint vertex_indices
end_header
-1 -1 -2 0 0 1 0 0
1 -1 -2 0 0 1 1 0
1 1 -2 0 0 1 1 1
-1 1 -2 0 0 1 0 1
4 0 1 2 3
)PLY";
    write_artifact(path, polygon);
    expect_error(load_ply_triangle_mesh(path), core::StatusCode::invalid_argument, true);

    constexpr auto non_finite_position = R"PLY(ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float u
property float v
element face 1
property list uchar uint vertex_indices
end_header
nan 0 -2 0 0 1 0 0
1 0 -2 0 0 1 1 0
0 1 -2 0 0 1 0 1
3 0 1 2
)PLY";
    write_artifact(path, non_finite_position);
    expect_error(load_ply_triangle_mesh(path), core::StatusCode::invalid_argument, true);

    constexpr auto non_finite_uv = R"PLY(ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float u
property float v
element face 1
property list uchar uint vertex_indices
end_header
0 0 -2 0 0 1 nan 0
1 0 -2 0 0 1 1 0
0 1 -2 0 0 1 0 1
3 0 1 2
)PLY";
    write_artifact(path, non_finite_uv);
    expect_error(load_ply_triangle_mesh(path), core::StatusCode::invalid_argument, true);

    constexpr auto degenerate_triangle = R"PLY(ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float u
property float v
element face 1
property list uchar uint vertex_indices
end_header
0 0 -2 0 0 1 0 0
1 0 -2 0 0 1 1 0
2 0 -2 0 0 1 0 1
3 0 1 2
)PLY";
    write_artifact(path, degenerate_triangle);
    expect_error(load_ply_triangle_mesh(path), core::StatusCode::invalid_argument);

    constexpr auto excessive_declaration = R"PLY(ply
format ascii 1.0
element vertex 4294967295
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float u
property float v
element face 1
property list uchar uint vertex_indices
end_header
)PLY";
    write_artifact(path, excessive_declaration);
    expect_error(load_ply_triangle_mesh(path), core::StatusCode::resource_exhausted);

    constexpr auto truncated = R"PLY(ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float u
property float v
element face 1
property list uchar uint vertex_indices
end_header
0 0 -2 0 0 1 0 0
)PLY";
    write_artifact(path, truncated);
    expect_error(load_ply_triangle_mesh(path), core::StatusCode::invalid_argument, true);
    remove_artifact(path);
}

} // namespace
} // namespace blackframe::engine
