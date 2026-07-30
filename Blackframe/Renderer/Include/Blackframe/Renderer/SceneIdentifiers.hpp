#pragma once

#include <cstdint>
#include <type_traits>

namespace blackframe::renderer {

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

struct SurfaceIdentifiers final {
    InstanceId instance{};
    GeometryId geometry{};
    PrimitiveId primitive{};
    MaterialId material{};

    [[nodiscard]] constexpr bool operator==(const SurfaceIdentifiers&) const noexcept = default;
};

static_assert(!std::is_same_v<ObjectId, InstanceId>);
static_assert(!std::is_same_v<ObjectId, GeometryId>);
static_assert(!std::is_same_v<ObjectId, PrimitiveId>);
static_assert(!std::is_same_v<ObjectId, MaterialId>);
static_assert(!std::is_same_v<InstanceId, GeometryId>);
static_assert(!std::is_same_v<InstanceId, PrimitiveId>);
static_assert(!std::is_same_v<InstanceId, MaterialId>);
static_assert(!std::is_same_v<GeometryId, PrimitiveId>);
static_assert(!std::is_same_v<GeometryId, MaterialId>);
static_assert(!std::is_same_v<PrimitiveId, MaterialId>);
static_assert(sizeof(ObjectId) == sizeof(std::uint32_t));
static_assert(sizeof(InstanceId) == sizeof(std::uint32_t));
static_assert(sizeof(GeometryId) == sizeof(std::uint32_t));
static_assert(sizeof(PrimitiveId) == sizeof(std::uint32_t));
static_assert(sizeof(MaterialId) == sizeof(std::uint32_t));
static_assert(std::is_standard_layout_v<SurfaceIdentifiers>);
static_assert(std::is_trivially_copyable_v<SurfaceIdentifiers>);

} // namespace blackframe::renderer
