#pragma once

#include <cstdint>
#include <type_traits>

namespace blackframe::renderer {

struct FilmId final {
    std::uint32_t value{};

    [[nodiscard]] constexpr bool operator==(const FilmId&) const noexcept = default;
};

struct CameraId final {
    std::uint32_t value{};

    [[nodiscard]] constexpr bool operator==(const CameraId&) const noexcept = default;
};

struct RenderOptionsId final {
    std::uint32_t value{};

    [[nodiscard]] constexpr bool operator==(const RenderOptionsId&) const noexcept = default;
};

struct ObjectId final {
    std::uint32_t value{};

    [[nodiscard]] constexpr bool operator==(const ObjectId&) const noexcept = default;
};

struct InstanceId final {
    std::uint32_t value{};

    [[nodiscard]] constexpr bool operator==(const InstanceId&) const noexcept = default;
};

struct GeometryId final {
    std::uint32_t value{};

    [[nodiscard]] constexpr bool operator==(const GeometryId&) const noexcept = default;
};

struct PrimitiveId final {
    std::uint32_t value{};

    [[nodiscard]] constexpr bool operator==(const PrimitiveId&) const noexcept = default;
};

struct MaterialId final {
    std::uint32_t value{};

    [[nodiscard]] constexpr bool operator==(const MaterialId&) const noexcept = default;
};

struct TextureId final {
    std::uint32_t value{};

    [[nodiscard]] constexpr bool operator==(const TextureId&) const noexcept = default;
};

struct LightId final {
    std::uint32_t value{};

    [[nodiscard]] constexpr bool operator==(const LightId&) const noexcept = default;
};

struct SurfaceIdentifiers final {
    InstanceId instance{};
    GeometryId geometry{};
    PrimitiveId primitive{};
    MaterialId material{};

    [[nodiscard]] constexpr bool operator==(const SurfaceIdentifiers&) const noexcept = default;
};

static_assert(!std::is_same_v<FilmId, CameraId>);
static_assert(!std::is_same_v<FilmId, RenderOptionsId>);
static_assert(!std::is_same_v<FilmId, ObjectId>);
static_assert(!std::is_same_v<FilmId, InstanceId>);
static_assert(!std::is_same_v<FilmId, GeometryId>);
static_assert(!std::is_same_v<FilmId, PrimitiveId>);
static_assert(!std::is_same_v<FilmId, MaterialId>);
static_assert(!std::is_same_v<FilmId, TextureId>);
static_assert(!std::is_same_v<FilmId, LightId>);
static_assert(!std::is_same_v<CameraId, RenderOptionsId>);
static_assert(!std::is_same_v<CameraId, ObjectId>);
static_assert(!std::is_same_v<CameraId, InstanceId>);
static_assert(!std::is_same_v<CameraId, GeometryId>);
static_assert(!std::is_same_v<CameraId, PrimitiveId>);
static_assert(!std::is_same_v<CameraId, MaterialId>);
static_assert(!std::is_same_v<CameraId, TextureId>);
static_assert(!std::is_same_v<CameraId, LightId>);
static_assert(!std::is_same_v<RenderOptionsId, ObjectId>);
static_assert(!std::is_same_v<RenderOptionsId, InstanceId>);
static_assert(!std::is_same_v<RenderOptionsId, GeometryId>);
static_assert(!std::is_same_v<RenderOptionsId, PrimitiveId>);
static_assert(!std::is_same_v<RenderOptionsId, MaterialId>);
static_assert(!std::is_same_v<RenderOptionsId, TextureId>);
static_assert(!std::is_same_v<RenderOptionsId, LightId>);
static_assert(!std::is_same_v<ObjectId, InstanceId>);
static_assert(!std::is_same_v<ObjectId, GeometryId>);
static_assert(!std::is_same_v<ObjectId, PrimitiveId>);
static_assert(!std::is_same_v<ObjectId, MaterialId>);
static_assert(!std::is_same_v<ObjectId, TextureId>);
static_assert(!std::is_same_v<InstanceId, GeometryId>);
static_assert(!std::is_same_v<InstanceId, PrimitiveId>);
static_assert(!std::is_same_v<InstanceId, MaterialId>);
static_assert(!std::is_same_v<InstanceId, TextureId>);
static_assert(!std::is_same_v<GeometryId, PrimitiveId>);
static_assert(!std::is_same_v<GeometryId, MaterialId>);
static_assert(!std::is_same_v<GeometryId, TextureId>);
static_assert(!std::is_same_v<PrimitiveId, MaterialId>);
static_assert(!std::is_same_v<PrimitiveId, TextureId>);
static_assert(!std::is_same_v<MaterialId, TextureId>);
static_assert(!std::is_same_v<ObjectId, LightId>);
static_assert(!std::is_same_v<InstanceId, LightId>);
static_assert(!std::is_same_v<GeometryId, LightId>);
static_assert(!std::is_same_v<PrimitiveId, LightId>);
static_assert(!std::is_same_v<MaterialId, LightId>);
static_assert(!std::is_same_v<TextureId, LightId>);
static_assert(sizeof(FilmId) == sizeof(std::uint32_t));
static_assert(sizeof(CameraId) == sizeof(std::uint32_t));
static_assert(sizeof(RenderOptionsId) == sizeof(std::uint32_t));
static_assert(sizeof(ObjectId) == sizeof(std::uint32_t));
static_assert(sizeof(InstanceId) == sizeof(std::uint32_t));
static_assert(sizeof(GeometryId) == sizeof(std::uint32_t));
static_assert(sizeof(PrimitiveId) == sizeof(std::uint32_t));
static_assert(sizeof(MaterialId) == sizeof(std::uint32_t));
static_assert(sizeof(TextureId) == sizeof(std::uint32_t));
static_assert(sizeof(LightId) == sizeof(std::uint32_t));
static_assert(std::is_standard_layout_v<FilmId>);
static_assert(std::is_trivially_copyable_v<FilmId>);
static_assert(std::is_standard_layout_v<CameraId>);
static_assert(std::is_trivially_copyable_v<CameraId>);
static_assert(std::is_standard_layout_v<RenderOptionsId>);
static_assert(std::is_trivially_copyable_v<RenderOptionsId>);
static_assert(std::is_standard_layout_v<LightId>);
static_assert(std::is_trivially_copyable_v<LightId>);
static_assert(std::is_standard_layout_v<SurfaceIdentifiers>);
static_assert(std::is_trivially_copyable_v<SurfaceIdentifiers>);

} // namespace blackframe::renderer
