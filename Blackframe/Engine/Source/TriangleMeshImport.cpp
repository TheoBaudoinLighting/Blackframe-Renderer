#include <Blackframe/Engine/TriangleMeshImport.hpp>
#include <Blackframe/Renderer/NumericConversion.hpp>
#include <array>
#include <charconv>
#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

constexpr auto MaximumMeshFileBytes = std::uintmax_t{256} * 1024U * 1024U;
constexpr auto MaximumMeshStorageBytes = std::uintmax_t{256} * 1024U * 1024U;
constexpr auto MaximumMeshEntries =
    static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());

[[nodiscard]] core::Error import_error(const core::StatusCode code, std::string message) {
    return core::Error{
        .code = code,
        .message = std::move(message),
    };
}

[[nodiscard]] core::Error
line_error(const char* const format, const std::size_t line, std::string_view detail,
           const core::StatusCode code = core::StatusCode::invalid_argument) {
    return import_error(code, std::string{format} + " line " + std::to_string(line) + ": " +
                                  std::string{detail});
}

struct SourceLine final {
    std::string_view text;
    std::size_t number;
};

class LineCursor final {
  public:
    explicit LineCursor(const std::string_view source) noexcept : source_{source} {}

    [[nodiscard]] std::optional<SourceLine> next() noexcept {
        if (offset_ == source_.size()) {
            return std::nullopt;
        }

        const auto line_end = source_.find('\n', offset_);
        const auto text_end = line_end == std::string_view::npos ? source_.size() : line_end;
        auto line = source_.substr(offset_, text_end - offset_);
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }
        offset_ = line_end == std::string_view::npos ? source_.size() : line_end + 1;
        ++line_number_;
        return SourceLine{.text = line, .number = line_number_};
    }

    [[nodiscard]] std::size_t next_line_number() const noexcept {
        return line_number_ + 1;
    }

  private:
    std::string_view source_;
    std::size_t offset_{};
    std::size_t line_number_{};
};

[[nodiscard]] constexpr bool ascii_space(const char value) noexcept {
    return value == ' ' || value == '\t' || value == '\r' || value == '\v' || value == '\f';
}

[[nodiscard]] std::vector<std::string_view> tokens(std::string_view line, const bool obj_comments,
                                                   const std::size_t maximum_tokens) {
    if (obj_comments) {
        if (const auto comment = line.find('#'); comment != std::string_view::npos) {
            line = line.substr(0, comment);
        }
    }

    auto result = std::vector<std::string_view>{};
    result.reserve(maximum_tokens);
    auto offset = std::size_t{};
    while (offset < line.size()) {
        while (offset < line.size() && ascii_space(line[offset])) {
            ++offset;
        }
        if (offset == line.size()) {
            break;
        }
        const auto beginning = offset;
        while (offset < line.size() && !ascii_space(line[offset])) {
            ++offset;
        }
        if (result.size() == maximum_tokens) {
            break;
        }
        result.push_back(line.substr(beginning, offset - beginning));
    }
    return result;
}

[[nodiscard]] std::optional<std::string_view> first_obj_token(std::string_view line) noexcept {
    if (const auto comment = line.find('#'); comment != std::string_view::npos) {
        line = line.substr(0, comment);
    }
    auto offset = std::size_t{};
    while (offset < line.size() && ascii_space(line[offset])) {
        ++offset;
    }
    if (offset == line.size()) {
        return std::nullopt;
    }
    const auto beginning = offset;
    while (offset < line.size() && !ascii_space(line[offset])) {
        ++offset;
    }
    return line.substr(beginning, offset - beginning);
}

[[nodiscard]] std::string_view without_leading_plus(std::string_view token) noexcept {
    if (!token.empty() && token.front() == '+') {
        token.remove_prefix(1);
    }
    return token;
}

[[nodiscard]] core::Result<std::int64_t>
parse_signed_decimal(std::string_view token, const char* const format, const std::size_t line) {
    token = without_leading_plus(token);
    auto value = std::int64_t{};
    const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value, 10);
    if (token.empty() || parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size()) {
        return std::unexpected(
            line_error(format, line, "expected a complete signed decimal integer."));
    }
    return value;
}

[[nodiscard]] core::Result<std::uint64_t>
parse_unsigned_decimal(std::string_view token, const char* const format, const std::size_t line) {
    token = without_leading_plus(token);
    auto value = std::uint64_t{};
    const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value, 10);
    if (token.empty() || parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size()) {
        return std::unexpected(
            line_error(format, line, "expected a complete unsigned decimal integer."));
    }
    return value;
}

[[nodiscard]] core::Result<renderer::TransportScalar>
parse_transport_scalar(std::string_view token, const char* const format, const std::size_t line) {
    token = without_leading_plus(token);
    auto reference_value = renderer::ReferenceScalar{};
    const auto parsed = std::from_chars(token.data(), token.data() + token.size(), reference_value,
                                        std::chars_format::general);
    if (token.empty() || parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size()) {
        return std::unexpected(
            line_error(format, line, "expected a complete locale-independent scalar."));
    }
    const auto converted = renderer::to_transport_scalar(reference_value);
    if (!converted) {
        return std::unexpected(line_error(format, line, converted.error().message));
    }
    return *converted;
}

[[nodiscard]] bool unit_normal(const renderer::Normal3 normal) noexcept {
    const auto squared_length =
        std::fma(normal.x, normal.x, std::fma(normal.y, normal.y, normal.z * normal.z));
    constexpr auto tolerance =
        std::numeric_limits<renderer::TransportScalar>::epsilon() * renderer::TransportScalar{256};
    return std::isfinite(squared_length) &&
           std::abs(squared_length - renderer::TransportScalar{1}) <= tolerance;
}

[[nodiscard]] core::Result<std::uint32_t>
resolve_obj_index(const std::string_view token, const std::size_t count, const std::size_t line) {
    const auto source_index = parse_signed_decimal(token, "OBJ", line);
    if (!source_index) {
        return std::unexpected(source_index.error());
    }
    if (*source_index == 0) {
        return std::unexpected(line_error("OBJ", line, "index zero is not valid."));
    }
    if (count > MaximumMeshEntries) {
        return std::unexpected(
            import_error(core::StatusCode::resource_exhausted,
                         "OBJ attribute count exceeds the supported 32-bit index domain."));
    }

    const auto signed_count = static_cast<std::int64_t>(count);
    const auto resolved = *source_index > 0 ? *source_index - 1 : signed_count + *source_index;
    if (resolved < 0 || resolved >= signed_count) {
        return std::unexpected(
            line_error("OBJ", line, "index references an attribute outside its current domain."));
    }
    return static_cast<std::uint32_t>(resolved);
}

struct ObjVertexKey final {
    std::uint32_t position;
    std::uint32_t texture_coordinate;
    std::uint32_t normal;

    [[nodiscard]] constexpr auto operator<=>(const ObjVertexKey&) const noexcept = default;
};

struct ObjRecordCounts final {
    std::size_t positions{};
    std::size_t normals{};
    std::size_t texture_coordinates{};
    std::size_t triangles{};
};

struct ObjStoragePlan final {
    ObjRecordCounts records;
    std::size_t maximum_aligned_vertices;
};

[[nodiscard]] bool consume_storage_budget(std::uintmax_t& remaining, const std::uintmax_t count,
                                          const std::uintmax_t bytes_per_entry) noexcept {
    if (bytes_per_entry != 0 && count > remaining / bytes_per_entry) {
        return false;
    }
    remaining -= count * bytes_per_entry;
    return true;
}

[[nodiscard]] core::Result<ObjStoragePlan> make_obj_storage_plan(const std::string_view source) {
    auto records = ObjRecordCounts{};
    auto cursor = LineCursor{source};
    while (const auto line = cursor.next()) {
        const auto record = first_obj_token(line->text);
        if (!record) {
            continue;
        }

        auto* count = static_cast<std::size_t*>(nullptr);
        if (*record == "v") {
            count = &records.positions;
        } else if (*record == "vn") {
            count = &records.normals;
        } else if (*record == "vt") {
            count = &records.texture_coordinates;
        } else if (*record == "f") {
            count = &records.triangles;
        }
        if (count == nullptr) {
            continue;
        }
        if (*count == MaximumMeshEntries) {
            return std::unexpected(
                import_error(core::StatusCode::resource_exhausted,
                             "OBJ record count exceeds the supported 32-bit index domain."));
        }
        ++*count;
    }

    constexpr auto corners_per_triangle = std::uintmax_t{3};
    const auto triangle_count = static_cast<std::uintmax_t>(records.triangles);
    if (triangle_count > MaximumMeshEntries / corners_per_triangle) {
        return std::unexpected(
            import_error(core::StatusCode::resource_exhausted,
                         "OBJ faces can exceed the supported 32-bit aligned-vertex domain."));
    }
    const auto maximum_aligned_vertices = triangle_count * corners_per_triangle;

    // std::map node layout is implementation-defined. This deliberately conservative allowance
    // bounds decoded importer storage; it is not presented as an exact resident-memory measure.
    constexpr auto map_entry_storage_allowance = std::uintmax_t{128};
    auto remaining = MaximumMeshStorageBytes;
    const auto fits =
        consume_storage_budget(remaining, records.positions, sizeof(renderer::Point3)) &&
        consume_storage_budget(remaining, records.normals, sizeof(renderer::Normal3)) &&
        consume_storage_budget(remaining, records.texture_coordinates, sizeof(renderer::Point2)) &&
        consume_storage_budget(remaining, maximum_aligned_vertices, sizeof(renderer::Point3)) &&
        consume_storage_budget(remaining, maximum_aligned_vertices, sizeof(renderer::Normal3)) &&
        consume_storage_budget(remaining, maximum_aligned_vertices, sizeof(renderer::Point2)) &&
        consume_storage_budget(remaining, records.triangles, sizeof(TriangleVertexIndices)) &&
        consume_storage_budget(remaining, maximum_aligned_vertices, map_entry_storage_allowance);
    if (!fits) {
        return std::unexpected(import_error(
            core::StatusCode::resource_exhausted,
            "OBJ records exceed the conservative 256 MiB decoded mesh-storage limit."));
    }

    return ObjStoragePlan{
        .records = records,
        .maximum_aligned_vertices = static_cast<std::size_t>(maximum_aligned_vertices),
    };
}

[[nodiscard]] core::Result<ObjVertexKey>
parse_obj_corner(const std::string_view token, const std::size_t position_count,
                 const std::size_t texture_coordinate_count, const std::size_t normal_count,
                 const std::size_t line) {
    const auto first_separator = token.find('/');
    const auto second_separator = first_separator == std::string_view::npos
                                      ? std::string_view::npos
                                      : token.find('/', first_separator + 1);
    if (first_separator == std::string_view::npos || second_separator == std::string_view::npos ||
        token.find('/', second_separator + 1) != std::string_view::npos || first_separator == 0 ||
        second_separator == first_separator + 1 || second_separator + 1 == token.size()) {
        return std::unexpected(
            line_error("OBJ", line, "triangle corners must use complete v/vt/vn indices."));
    }

    const auto position = resolve_obj_index(token.substr(0, first_separator), position_count, line);
    const auto texture_coordinate =
        resolve_obj_index(token.substr(first_separator + 1, second_separator - first_separator - 1),
                          texture_coordinate_count, line);
    const auto normal = resolve_obj_index(token.substr(second_separator + 1), normal_count, line);
    if (!position) {
        return std::unexpected(position.error());
    }
    if (!texture_coordinate) {
        return std::unexpected(texture_coordinate.error());
    }
    if (!normal) {
        return std::unexpected(normal.error());
    }
    return ObjVertexKey{
        .position = *position,
        .texture_coordinate = *texture_coordinate,
        .normal = *normal,
    };
}

[[nodiscard]] core::Result<TriangleMesh> parse_obj_triangle_mesh(const std::string_view source) {
    if (source.find('\0') != std::string_view::npos) {
        return std::unexpected(import_error(core::StatusCode::invalid_argument,
                                            "OBJ text contains an embedded null byte."));
    }
    const auto storage_plan = make_obj_storage_plan(source);
    if (!storage_plan) {
        return std::unexpected(storage_plan.error());
    }

    auto source_positions = std::vector<renderer::Point3>{};
    auto source_normals = std::vector<renderer::Normal3>{};
    auto source_texture_coordinates = std::vector<renderer::Point2>{};
    auto positions = std::vector<renderer::Point3>{};
    auto normals = std::vector<renderer::Normal3>{};
    auto texture_coordinates = std::vector<renderer::Point2>{};
    auto triangles = std::vector<TriangleVertexIndices>{};
    auto aligned_vertices = std::map<ObjVertexKey, std::uint32_t>{};
    if (storage_plan->records.positions > source_positions.max_size() ||
        storage_plan->records.normals > source_normals.max_size() ||
        storage_plan->records.texture_coordinates > source_texture_coordinates.max_size() ||
        storage_plan->records.triangles > triangles.max_size() ||
        storage_plan->maximum_aligned_vertices > positions.max_size() ||
        storage_plan->maximum_aligned_vertices > normals.max_size() ||
        storage_plan->maximum_aligned_vertices > texture_coordinates.max_size() ||
        storage_plan->maximum_aligned_vertices > aligned_vertices.max_size()) {
        return std::unexpected(import_error(core::StatusCode::resource_exhausted,
                                            "OBJ records exceed host container limits."));
    }
    source_positions.reserve(storage_plan->records.positions);
    source_normals.reserve(storage_plan->records.normals);
    source_texture_coordinates.reserve(storage_plan->records.texture_coordinates);
    positions.reserve(storage_plan->maximum_aligned_vertices);
    normals.reserve(storage_plan->maximum_aligned_vertices);
    texture_coordinates.reserve(storage_plan->maximum_aligned_vertices);
    triangles.reserve(storage_plan->records.triangles);

    auto cursor = LineCursor{source};
    while (const auto line = cursor.next()) {
        const auto fields = tokens(line->text, true, 5);
        if (fields.empty()) {
            continue;
        }

        const auto record = fields.front();
        if (record == "v") {
            if (fields.size() != 4) {
                return std::unexpected(
                    line_error("OBJ", line->number, "positions require exactly x, y, and z."));
            }
            if (source_positions.size() == MaximumMeshEntries) {
                return std::unexpected(
                    import_error(core::StatusCode::resource_exhausted,
                                 "OBJ position count exceeds the supported 32-bit index domain."));
            }
            const auto x = parse_transport_scalar(fields[1], "OBJ", line->number);
            const auto y = parse_transport_scalar(fields[2], "OBJ", line->number);
            const auto z = parse_transport_scalar(fields[3], "OBJ", line->number);
            if (!x) {
                return std::unexpected(x.error());
            }
            if (!y) {
                return std::unexpected(y.error());
            }
            if (!z) {
                return std::unexpected(z.error());
            }
            source_positions.push_back(renderer::Point3{.x = *x, .y = *y, .z = *z});
            continue;
        }
        if (record == "vn") {
            if (fields.size() != 4) {
                return std::unexpected(
                    line_error("OBJ", line->number, "normals require exactly x, y, and z."));
            }
            if (source_normals.size() == MaximumMeshEntries) {
                return std::unexpected(
                    import_error(core::StatusCode::resource_exhausted,
                                 "OBJ normal count exceeds the supported 32-bit index domain."));
            }
            const auto x = parse_transport_scalar(fields[1], "OBJ", line->number);
            const auto y = parse_transport_scalar(fields[2], "OBJ", line->number);
            const auto z = parse_transport_scalar(fields[3], "OBJ", line->number);
            if (!x) {
                return std::unexpected(x.error());
            }
            if (!y) {
                return std::unexpected(y.error());
            }
            if (!z) {
                return std::unexpected(z.error());
            }
            const auto normal = renderer::Normal3{.x = *x, .y = *y, .z = *z};
            if (!unit_normal(normal)) {
                return std::unexpected(
                    line_error("OBJ", line->number, "normals must be finite unit-length values."));
            }
            source_normals.push_back(normal);
            continue;
        }
        if (record == "vt") {
            if (fields.size() != 3) {
                return std::unexpected(line_error("OBJ", line->number,
                                                  "texture coordinates require exactly u and v."));
            }
            if (source_texture_coordinates.size() == MaximumMeshEntries) {
                return std::unexpected(import_error(
                    core::StatusCode::resource_exhausted,
                    "OBJ texture-coordinate count exceeds the supported 32-bit index domain."));
            }
            const auto u = parse_transport_scalar(fields[1], "OBJ", line->number);
            const auto v = parse_transport_scalar(fields[2], "OBJ", line->number);
            if (!u) {
                return std::unexpected(u.error());
            }
            if (!v) {
                return std::unexpected(v.error());
            }
            source_texture_coordinates.push_back(renderer::Point2{.x = *u, .y = *v});
            continue;
        }
        if (record == "f") {
            if (fields.size() != 4) {
                return std::unexpected(line_error(
                    "OBJ", line->number,
                    "only explicitly triangulated faces with three corners are supported."));
            }
            if (triangles.size() == MaximumMeshEntries) {
                return std::unexpected(
                    import_error(core::StatusCode::resource_exhausted,
                                 "OBJ triangle count exceeds the supported 32-bit index domain."));
            }

            auto triangle = TriangleVertexIndices{};
            for (auto corner = std::size_t{}; corner < triangle.vertices.size(); ++corner) {
                const auto key = parse_obj_corner(fields[corner + 1], source_positions.size(),
                                                  source_texture_coordinates.size(),
                                                  source_normals.size(), line->number);
                if (!key) {
                    return std::unexpected(key.error());
                }

                const auto existing = aligned_vertices.find(*key);
                if (existing != aligned_vertices.end()) {
                    triangle.vertices[corner] = existing->second;
                    continue;
                }
                if (positions.size() == MaximumMeshEntries) {
                    return std::unexpected(import_error(
                        core::StatusCode::resource_exhausted,
                        "OBJ aligned vertex count exceeds the supported 32-bit index domain."));
                }

                const auto aligned_index = static_cast<std::uint32_t>(positions.size());
                positions.push_back(source_positions[key->position]);
                normals.push_back(source_normals[key->normal]);
                texture_coordinates.push_back(source_texture_coordinates[key->texture_coordinate]);
                aligned_vertices.emplace(*key, aligned_index);
                triangle.vertices[corner] = aligned_index;
            }
            triangles.push_back(triangle);
            continue;
        }
        if (record == "g") {
            if (fields.size() > 2) {
                return std::unexpected(
                    line_error("OBJ", line->number,
                               "at most one opaque group name is supported by this importer."));
            }
            continue;
        }
        if (record == "o" || record == "s" || record == "usemtl" || record == "mtllib") {
            if (fields.size() != 2) {
                return std::unexpected(line_error(
                    "OBJ", line->number,
                    "opaque object, smoothing, and material metadata requires one value."));
            }
            continue;
        }
        return std::unexpected(
            line_error("OBJ", line->number, "record is not supported by the triangle importer."));
    }

    return TriangleMesh::create(std::move(positions), std::move(normals),
                                std::move(texture_coordinates), std::move(triangles));
}

enum class PlyScalarType {
    int8,
    uint8,
    int16,
    uint16,
    int32,
    uint32,
    float32,
    float64,
};

[[nodiscard]] std::optional<PlyScalarType> ply_scalar_type(const std::string_view name) noexcept {
    if (name == "char" || name == "int8") {
        return PlyScalarType::int8;
    }
    if (name == "uchar" || name == "uint8") {
        return PlyScalarType::uint8;
    }
    if (name == "short" || name == "int16") {
        return PlyScalarType::int16;
    }
    if (name == "ushort" || name == "uint16") {
        return PlyScalarType::uint16;
    }
    if (name == "int" || name == "int32") {
        return PlyScalarType::int32;
    }
    if (name == "uint" || name == "uint32") {
        return PlyScalarType::uint32;
    }
    if (name == "float" || name == "float32") {
        return PlyScalarType::float32;
    }
    if (name == "double" || name == "float64") {
        return PlyScalarType::float64;
    }
    return std::nullopt;
}

[[nodiscard]] constexpr bool ply_integer_type(const PlyScalarType type) noexcept {
    return type != PlyScalarType::float32 && type != PlyScalarType::float64;
}

[[nodiscard]] core::Result<std::int64_t>
parse_bounded_signed_ply_integer(const std::string_view token, const PlyScalarType type,
                                 const std::size_t line) {
    const auto value = parse_signed_decimal(token, "PLY", line);
    if (!value) {
        return std::unexpected(value.error());
    }
    auto minimum = std::int64_t{};
    auto maximum = std::int64_t{};
    switch (type) {
    case PlyScalarType::int8:
        minimum = std::numeric_limits<std::int8_t>::min();
        maximum = std::numeric_limits<std::int8_t>::max();
        break;
    case PlyScalarType::int16:
        minimum = std::numeric_limits<std::int16_t>::min();
        maximum = std::numeric_limits<std::int16_t>::max();
        break;
    case PlyScalarType::int32:
        minimum = std::numeric_limits<std::int32_t>::min();
        maximum = std::numeric_limits<std::int32_t>::max();
        break;
    default:
        return std::unexpected(
            line_error("PLY", line, "declared list integer type is inconsistent."));
    }
    if (*value < minimum || *value > maximum) {
        return std::unexpected(
            line_error("PLY", line, "integer value exceeds its declared scalar type."));
    }
    return *value;
}

[[nodiscard]] core::Result<std::uint64_t>
parse_bounded_unsigned_ply_integer(const std::string_view token, const PlyScalarType type,
                                   const std::size_t line) {
    const auto value = parse_unsigned_decimal(token, "PLY", line);
    if (!value) {
        return std::unexpected(value.error());
    }
    auto maximum = std::uint64_t{};
    switch (type) {
    case PlyScalarType::uint8:
        maximum = std::numeric_limits<std::uint8_t>::max();
        break;
    case PlyScalarType::uint16:
        maximum = std::numeric_limits<std::uint16_t>::max();
        break;
    case PlyScalarType::uint32:
        maximum = std::numeric_limits<std::uint32_t>::max();
        break;
    default:
        return std::unexpected(
            line_error("PLY", line, "declared list integer type is inconsistent."));
    }
    if (*value > maximum) {
        return std::unexpected(
            line_error("PLY", line, "integer value exceeds its declared scalar type."));
    }
    return *value;
}

[[nodiscard]] constexpr bool ply_signed_integer_type(const PlyScalarType type) noexcept {
    return type == PlyScalarType::int8 || type == PlyScalarType::int16 ||
           type == PlyScalarType::int32;
}

[[nodiscard]] core::Result<std::uint64_t>
parse_non_negative_ply_integer(const std::string_view token, const PlyScalarType type,
                               const std::size_t line) {
    if (!ply_integer_type(type)) {
        return std::unexpected(
            line_error("PLY", line, "list counts and indices require an integer scalar type."));
    }
    if (ply_signed_integer_type(type)) {
        const auto value = parse_bounded_signed_ply_integer(token, type, line);
        if (!value) {
            return std::unexpected(value.error());
        }
        if (*value < 0) {
            return std::unexpected(
                line_error("PLY", line, "list counts and indices cannot be negative."));
        }
        return static_cast<std::uint64_t>(*value);
    }
    return parse_bounded_unsigned_ply_integer(token, type, line);
}

[[nodiscard]] core::Result<renderer::TransportScalar>
parse_ply_transport_scalar(const std::string_view token, const PlyScalarType type,
                           const std::size_t line) {
    if (type == PlyScalarType::float32 || type == PlyScalarType::float64) {
        return parse_transport_scalar(token, "PLY", line);
    }

    auto reference_value = renderer::ReferenceScalar{};
    if (ply_signed_integer_type(type)) {
        const auto value = parse_bounded_signed_ply_integer(token, type, line);
        if (!value) {
            return std::unexpected(value.error());
        }
        reference_value = static_cast<renderer::ReferenceScalar>(*value);
    } else {
        const auto value = parse_bounded_unsigned_ply_integer(token, type, line);
        if (!value) {
            return std::unexpected(value.error());
        }
        reference_value = static_cast<renderer::ReferenceScalar>(*value);
    }
    const auto converted = renderer::to_transport_scalar(reference_value);
    if (!converted) {
        return std::unexpected(line_error("PLY", line, converted.error().message));
    }
    return *converted;
}

enum class PlyVertexSemantic : std::size_t {
    x,
    y,
    z,
    nx,
    ny,
    nz,
    u,
    v,
    count,
};

[[nodiscard]] std::optional<PlyVertexSemantic>
ply_vertex_semantic(const std::string_view name) noexcept {
    if (name == "x") {
        return PlyVertexSemantic::x;
    }
    if (name == "y") {
        return PlyVertexSemantic::y;
    }
    if (name == "z") {
        return PlyVertexSemantic::z;
    }
    if (name == "nx") {
        return PlyVertexSemantic::nx;
    }
    if (name == "ny") {
        return PlyVertexSemantic::ny;
    }
    if (name == "nz") {
        return PlyVertexSemantic::nz;
    }
    if (name == "u" || name == "s" || name == "texture_u") {
        return PlyVertexSemantic::u;
    }
    if (name == "v" || name == "t" || name == "texture_v") {
        return PlyVertexSemantic::v;
    }
    return std::nullopt;
}

struct PlyVertexProperty final {
    PlyVertexSemantic semantic;
    PlyScalarType type;
};

enum class PlyElement {
    none,
    vertex,
    face,
};

struct PlyHeader final {
    std::size_t vertex_count;
    std::size_t face_count;
    std::vector<PlyVertexProperty> vertex_properties;
    PlyScalarType face_count_type;
    PlyScalarType face_index_type;
};

[[nodiscard]] core::Result<PlyHeader> parse_ply_header(LineCursor& cursor) {
    const auto magic_line = cursor.next();
    if (!magic_line) {
        return std::unexpected(line_error("PLY", 1, "file is empty."));
    }
    const auto magic = tokens(magic_line->text, false, 2);
    if (magic.size() != 1 || magic.front() != "ply") {
        return std::unexpected(line_error("PLY", magic_line->number, "missing ply magic."));
    }

    auto format_seen = false;
    auto vertex_seen = false;
    auto face_seen = false;
    auto end_seen = false;
    auto current_element = PlyElement::none;
    auto vertex_count = std::size_t{};
    auto face_count = std::size_t{};
    auto vertex_properties = std::vector<PlyVertexProperty>{};
    auto semantics_seen = std::array<bool, static_cast<std::size_t>(PlyVertexSemantic::count)>{};
    auto face_count_type = std::optional<PlyScalarType>{};
    auto face_index_type = std::optional<PlyScalarType>{};

    while (const auto line = cursor.next()) {
        const auto fields = tokens(line->text, false, 6);
        if (fields.empty()) {
            continue;
        }
        if (fields.front() == "comment" || fields.front() == "obj_info") {
            continue;
        }
        if (fields.front() == "format") {
            if (format_seen || fields.size() != 3) {
                return std::unexpected(line_error("PLY", line->number,
                                                  "format declaration is malformed or repeated."));
            }
            format_seen = true;
            if (fields[1] != "ascii" || fields[2] != "1.0") {
                return std::unexpected(line_error(
                    "PLY", line->number,
                    "only the ASCII 1.0 encoding is supported; no encoding fallback is available.",
                    core::StatusCode::incompatible));
            }
            continue;
        }
        if (fields.front() == "element") {
            if (!format_seen || fields.size() != 3) {
                return std::unexpected(
                    line_error("PLY", line->number, "element declaration is malformed."));
            }
            const auto count = parse_unsigned_decimal(fields[2], "PLY", line->number);
            if (!count) {
                return std::unexpected(count.error());
            }
            if (*count > MaximumMeshEntries) {
                return std::unexpected(line_error("PLY", line->number,
                                                  "element count exceeds the 32-bit mesh domain.",
                                                  core::StatusCode::resource_exhausted));
            }
            if (fields[1] == "vertex") {
                if (vertex_seen || face_seen) {
                    return std::unexpected(line_error(
                        "PLY", line->number, "vertex must be the first and only vertex element."));
                }
                vertex_seen = true;
                current_element = PlyElement::vertex;
                vertex_count = static_cast<std::size_t>(*count);
                continue;
            }
            if (fields[1] == "face") {
                if (!vertex_seen || face_seen) {
                    return std::unexpected(line_error(
                        "PLY", line->number, "face must follow exactly one vertex element."));
                }
                face_seen = true;
                current_element = PlyElement::face;
                face_count = static_cast<std::size_t>(*count);
                continue;
            }
            return std::unexpected(line_error("PLY", line->number,
                                              "element is not supported by the triangle importer.",
                                              core::StatusCode::incompatible));
        }
        if (fields.front() == "property") {
            if (current_element == PlyElement::vertex) {
                if (fields.size() != 3 || fields[1] == "list") {
                    return std::unexpected(line_error(
                        "PLY", line->number, "vertex properties must be supported scalar values."));
                }
                const auto type = ply_scalar_type(fields[1]);
                const auto semantic = ply_vertex_semantic(fields[2]);
                if (!type || !semantic) {
                    return std::unexpected(line_error(
                        "PLY", line->number, "vertex property type or semantic is not supported.",
                        core::StatusCode::incompatible));
                }
                const auto semantic_index = static_cast<std::size_t>(*semantic);
                if (semantics_seen[semantic_index]) {
                    return std::unexpected(
                        line_error("PLY", line->number,
                                   "vertex property semantic is duplicated or aliased twice."));
                }
                semantics_seen[semantic_index] = true;
                vertex_properties.push_back(
                    PlyVertexProperty{.semantic = *semantic, .type = *type});
                continue;
            }
            if (current_element == PlyElement::face) {
                if (fields.size() != 5 || fields[1] != "list" ||
                    (fields[4] != "vertex_indices" && fields[4] != "vertex_index")) {
                    return std::unexpected(
                        line_error("PLY", line->number,
                                   "faces require exactly one vertex_indices list property.",
                                   core::StatusCode::incompatible));
                }
                if (face_count_type || face_index_type) {
                    return std::unexpected(
                        line_error("PLY", line->number, "face index property is duplicated."));
                }
                const auto count_type = ply_scalar_type(fields[2]);
                const auto index_type = ply_scalar_type(fields[3]);
                if (!count_type || !index_type || !ply_integer_type(*count_type) ||
                    !ply_integer_type(*index_type)) {
                    return std::unexpected(
                        line_error("PLY", line->number,
                                   "face list counts and indices require supported integer types.",
                                   core::StatusCode::incompatible));
                }
                face_count_type = *count_type;
                face_index_type = *index_type;
                continue;
            }
            return std::unexpected(
                line_error("PLY", line->number, "property appears before an element."));
        }
        if (fields.front() == "end_header") {
            if (fields.size() != 1) {
                return std::unexpected(
                    line_error("PLY", line->number, "end_header cannot carry extra fields."));
            }
            end_seen = true;
            break;
        }
        return std::unexpected(line_error("PLY", line->number, "header directive is not supported.",
                                          core::StatusCode::incompatible));
    }

    const auto every_semantic = [&semantics_seen] {
        for (const auto seen : semantics_seen) {
            if (!seen) {
                return false;
            }
        }
        return true;
    }();
    if (!end_seen) {
        return std::unexpected(
            line_error("PLY", cursor.next_line_number(), "header is truncated."));
    }
    if (!format_seen || !vertex_seen || !face_seen || !every_semantic || !face_count_type ||
        !face_index_type) {
        return std::unexpected(line_error(
            "PLY", cursor.next_line_number(),
            "header must define vertices with x/y/z, nx/ny/nz, u/v and triangular faces."));
    }
    if (vertex_count == 0 || face_count == 0) {
        return std::unexpected(line_error("PLY", cursor.next_line_number(),
                                          "vertex and face elements must be non-empty."));
    }

    return PlyHeader{
        .vertex_count = vertex_count,
        .face_count = face_count,
        .vertex_properties = std::move(vertex_properties),
        .face_count_type = *face_count_type,
        .face_index_type = *face_index_type,
    };
}

struct PlyVertexComponents final {
    renderer::TransportScalar x;
    renderer::TransportScalar y;
    renderer::TransportScalar z;
    renderer::TransportScalar nx;
    renderer::TransportScalar ny;
    renderer::TransportScalar nz;
    renderer::TransportScalar u;
    renderer::TransportScalar v;
};

void assign_ply_component(PlyVertexComponents& components, const PlyVertexSemantic semantic,
                          const renderer::TransportScalar value) noexcept {
    switch (semantic) {
    case PlyVertexSemantic::x:
        components.x = value;
        break;
    case PlyVertexSemantic::y:
        components.y = value;
        break;
    case PlyVertexSemantic::z:
        components.z = value;
        break;
    case PlyVertexSemantic::nx:
        components.nx = value;
        break;
    case PlyVertexSemantic::ny:
        components.ny = value;
        break;
    case PlyVertexSemantic::nz:
        components.nz = value;
        break;
    case PlyVertexSemantic::u:
        components.u = value;
        break;
    case PlyVertexSemantic::v:
        components.v = value;
        break;
    case PlyVertexSemantic::count:
        break;
    }
}

[[nodiscard]] core::Result<TriangleMesh> parse_ply_triangle_mesh(const std::string_view source) {
    auto cursor = LineCursor{source};
    auto header = parse_ply_header(cursor);
    if (!header) {
        return std::unexpected(header.error());
    }
    if (source.find('\0') != std::string_view::npos) {
        return std::unexpected(import_error(core::StatusCode::invalid_argument,
                                            "PLY ASCII data contains an embedded null byte."));
    }

    auto positions = std::vector<renderer::Point3>{};
    auto normals = std::vector<renderer::Normal3>{};
    auto texture_coordinates = std::vector<renderer::Point2>{};
    auto triangles = std::vector<TriangleVertexIndices>{};
    constexpr auto vertex_storage_bytes =
        sizeof(renderer::Point3) + sizeof(renderer::Normal3) + sizeof(renderer::Point2);
    const auto required_vertex_bytes =
        static_cast<std::uintmax_t>(header->vertex_count) * vertex_storage_bytes;
    const auto required_triangle_bytes =
        static_cast<std::uintmax_t>(header->face_count) * sizeof(TriangleVertexIndices);
    if (required_vertex_bytes > MaximumMeshStorageBytes ||
        required_triangle_bytes > MaximumMeshStorageBytes - required_vertex_bytes) {
        return std::unexpected(
            import_error(core::StatusCode::resource_exhausted,
                         "PLY declarations exceed the 256 MiB decoded mesh-storage limit."));
    }

    constexpr auto minimum_vertex_record_bytes = std::uintmax_t{15};
    constexpr auto minimum_triangle_record_bytes = std::uintmax_t{7};
    const auto minimum_vertex_bytes =
        static_cast<std::uintmax_t>(header->vertex_count) * minimum_vertex_record_bytes;
    const auto minimum_triangle_bytes =
        static_cast<std::uintmax_t>(header->face_count) * minimum_triangle_record_bytes;
    if (minimum_vertex_bytes > source.size() ||
        minimum_triangle_bytes >
            static_cast<std::uintmax_t>(source.size()) - minimum_vertex_bytes) {
        return std::unexpected(
            import_error(core::StatusCode::invalid_argument,
                         "PLY declared element counts cannot fit in the available source data."));
    }

    if (header->vertex_count > positions.max_size() || header->vertex_count > normals.max_size() ||
        header->vertex_count > texture_coordinates.max_size() ||
        header->face_count > triangles.max_size()) {
        return std::unexpected(import_error(core::StatusCode::resource_exhausted,
                                            "PLY element counts exceed host container limits."));
    }
    positions.reserve(header->vertex_count);
    normals.reserve(header->vertex_count);
    texture_coordinates.reserve(header->vertex_count);
    triangles.reserve(header->face_count);

    for (auto vertex_index = std::size_t{}; vertex_index < header->vertex_count; ++vertex_index) {
        const auto line = cursor.next();
        if (!line) {
            return std::unexpected(
                line_error("PLY", cursor.next_line_number(), "vertex records are truncated."));
        }
        const auto fields = tokens(line->text, false, header->vertex_properties.size() + 1);
        if (fields.size() != header->vertex_properties.size()) {
            return std::unexpected(
                line_error("PLY", line->number,
                           "vertex record does not match the declared property count and order."));
        }

        auto components = PlyVertexComponents{};
        for (auto property_index = std::size_t{}; property_index < header->vertex_properties.size();
             ++property_index) {
            const auto& property = header->vertex_properties[property_index];
            const auto value =
                parse_ply_transport_scalar(fields[property_index], property.type, line->number);
            if (!value) {
                return std::unexpected(value.error());
            }
            assign_ply_component(components, property.semantic, *value);
        }
        positions.push_back(
            renderer::Point3{.x = components.x, .y = components.y, .z = components.z});
        normals.push_back(
            renderer::Normal3{.x = components.nx, .y = components.ny, .z = components.nz});
        texture_coordinates.push_back(renderer::Point2{.x = components.u, .y = components.v});
    }

    for (auto face_index = std::size_t{}; face_index < header->face_count; ++face_index) {
        const auto line = cursor.next();
        if (!line) {
            return std::unexpected(
                line_error("PLY", cursor.next_line_number(), "face records are truncated."));
        }
        const auto fields = tokens(line->text, false, 5);
        if (fields.empty()) {
            return std::unexpected(line_error("PLY", line->number, "face record cannot be empty."));
        }
        const auto list_size =
            parse_non_negative_ply_integer(fields.front(), header->face_count_type, line->number);
        if (!list_size) {
            return std::unexpected(list_size.error());
        }
        if (*list_size > std::numeric_limits<std::size_t>::max() - 1 ||
            fields.size() != static_cast<std::size_t>(*list_size) + 1) {
            return std::unexpected(line_error(
                "PLY", line->number, "face record does not match its declared list count."));
        }
        if (*list_size != 3) {
            return std::unexpected(
                line_error("PLY", line->number,
                           "only explicitly triangulated faces with three indices are supported."));
        }

        auto triangle = TriangleVertexIndices{};
        for (auto corner = std::size_t{}; corner < triangle.vertices.size(); ++corner) {
            const auto index = parse_non_negative_ply_integer(
                fields[corner + 1], header->face_index_type, line->number);
            if (!index) {
                return std::unexpected(index.error());
            }
            if (*index >= header->vertex_count) {
                return std::unexpected(
                    line_error("PLY", line->number, "face index references an unknown vertex."));
            }
            triangle.vertices[corner] = static_cast<std::uint32_t>(*index);
        }
        triangles.push_back(triangle);
    }

    while (const auto line = cursor.next()) {
        if (!tokens(line->text, false, 1).empty()) {
            return std::unexpected(
                line_error("PLY", line->number, "unexpected data follows the declared elements."));
        }
    }
    return TriangleMesh::create(std::move(positions), std::move(normals),
                                std::move(texture_coordinates), std::move(triangles));
}

[[nodiscard]] core::Result<std::string>
read_mesh_text_file(const std::filesystem::path& absolute_path,
                    const std::filesystem::path& required_extension, const char* const format) {
    if (absolute_path.empty() || !absolute_path.is_absolute()) {
        return std::unexpected(
            import_error(core::StatusCode::invalid_argument,
                         std::string{format} + " mesh paths must be explicit absolute paths."));
    }
    if (absolute_path.extension() != required_extension) {
        return std::unexpected(
            import_error(core::StatusCode::invalid_argument,
                         std::string{format} + " mesh path has the wrong lower-case extension."));
    }

    auto status_error = std::error_code{};
    const auto status = std::filesystem::status(absolute_path, status_error);
    if (status_error) {
        if (status_error == std::errc::no_such_file_or_directory) {
            return std::unexpected(import_error(core::StatusCode::not_found,
                                                std::string{format} + " mesh file was not found."));
        }
        return std::unexpected(
            import_error(core::StatusCode::platform_error,
                         std::string{format} + " mesh file status could not be inspected."));
    }
    if (status.type() == std::filesystem::file_type::not_found) {
        return std::unexpected(import_error(core::StatusCode::not_found,
                                            std::string{format} + " mesh file was not found."));
    }
    if (!std::filesystem::is_regular_file(status)) {
        return std::unexpected(
            import_error(core::StatusCode::invalid_argument,
                         std::string{format} + " mesh path must identify a regular file."));
    }

    auto size_error = std::error_code{};
    const auto file_size = std::filesystem::file_size(absolute_path, size_error);
    if (size_error) {
        return std::unexpected(
            import_error(core::StatusCode::platform_error,
                         std::string{format} + " mesh file size could not be inspected."));
    }
    if (file_size > MaximumMeshFileBytes) {
        return std::unexpected(
            import_error(core::StatusCode::resource_exhausted,
                         std::string{format} + " mesh file exceeds the 256 MiB import limit."));
    }

    auto input = std::ifstream{absolute_path, std::ios::binary};
    if (!input.is_open()) {
        return std::unexpected(
            import_error(core::StatusCode::platform_error,
                         std::string{format} + " mesh file could not be opened for reading."));
    }

    auto contents = std::string{};
    if (file_size > contents.max_size()) {
        return std::unexpected(
            import_error(core::StatusCode::resource_exhausted,
                         std::string{format} + " mesh file exceeds host string limits."));
    }
    contents.resize(static_cast<std::size_t>(file_size));
    if (!contents.empty()) {
        input.read(contents.data(), static_cast<std::streamsize>(contents.size()));
        if (input.gcount() != static_cast<std::streamsize>(contents.size()) || input.bad()) {
            return std::unexpected(
                import_error(core::StatusCode::platform_error,
                             std::string{format} + " mesh file could not be read completely."));
        }
    }
    if (input.peek() != std::char_traits<char>::eof()) {
        return std::unexpected(
            import_error(core::StatusCode::platform_error,
                         std::string{format} + " mesh file changed while it was being read."));
    }
    return contents;
}

using MeshParser = core::Result<TriangleMesh> (*)(std::string_view);

[[nodiscard]] core::Result<TriangleMesh>
load_triangle_mesh(const std::filesystem::path& absolute_path,
                   const std::filesystem::path& extension, const char* const format,
                   const MeshParser parser) {
    try {
        auto contents = read_mesh_text_file(absolute_path, extension, format);
        if (!contents) {
            return std::unexpected(std::move(contents.error()));
        }
        return parser(*contents);
    } catch (const std::bad_alloc&) {
        return std::unexpected(
            import_error(core::StatusCode::resource_exhausted,
                         std::string{format} + " mesh import exhausted host memory."));
    } catch (const std::length_error&) {
        return std::unexpected(
            import_error(core::StatusCode::resource_exhausted,
                         std::string{format} + " mesh import exceeded host container limits."));
    }
}

} // namespace

core::Result<TriangleMesh> load_obj_triangle_mesh(const std::filesystem::path& absolute_path) {
    return load_triangle_mesh(absolute_path, ".obj", "OBJ", parse_obj_triangle_mesh);
}

core::Result<TriangleMesh> load_ply_triangle_mesh(const std::filesystem::path& absolute_path) {
    return load_triangle_mesh(absolute_path, ".ply", "PLY", parse_ply_triangle_mesh);
}

} // namespace blackframe::engine
