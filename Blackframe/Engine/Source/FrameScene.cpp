#include <Blackframe/Engine/FrameScene.hpp>
#include <algorithm>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <utility>

namespace blackframe::engine {
namespace {

[[nodiscard]] core::Error scene_error(const core::StatusCode code, const std::string_view message) {
    return core::Error{
        .code = code,
        .message = std::string{message},
    };
}

template <typename Record> void sort_by_identifier(std::vector<Record>& records) {
    std::ranges::sort(records, {}, [](const Record& record) { return record.id.value; });
}

template <typename Record>
[[nodiscard]] core::Status reject_duplicate_identifiers(const std::vector<Record>& records,
                                                        const std::string_view message) {
    const auto duplicate = std::ranges::adjacent_find(
        records, [](const Record& left, const Record& right) { return left.id == right.id; });
    if (duplicate != records.end()) {
        return std::unexpected(scene_error(core::StatusCode::invalid_argument, message));
    }
    return {};
}

template <typename Record, typename Identifier>
[[nodiscard]] bool contains_identifier(const std::vector<Record>& records,
                                       const Identifier id) noexcept {
    const auto candidate = std::ranges::lower_bound(
        records, id.value, {}, [](const Record& record) { return record.id.value; });
    return candidate != records.end() && candidate->id == id;
}

template <typename Record, typename Identifier>
[[nodiscard]] core::Result<std::reference_wrapper<const Record>>
find_record(const std::vector<Record>& records, const Identifier id,
            const std::string_view missing_message) {
    const auto candidate = std::ranges::lower_bound(
        records, id.value, {}, [](const Record& record) { return record.id.value; });
    if (candidate == records.end() || candidate->id != id) {
        return std::unexpected(scene_error(core::StatusCode::not_found, missing_message));
    }
    return std::cref(*candidate);
}

} // namespace

core::Result<FrameSceneHandle> FrameScene::create(const FrameSceneDescription& description) {
    try {
        auto owned_description = description;
        return create(std::move(owned_description));
    } catch (const std::bad_alloc&) {
        return std::unexpected(scene_error(core::StatusCode::resource_exhausted,
                                           "Frame scene storage exhausted host memory."));
    }
}

core::Result<FrameSceneHandle> FrameScene::create(FrameSceneDescription&& description) {
    sort_by_identifier(description.objects);
    sort_by_identifier(description.geometries);
    sort_by_identifier(description.materials);
    sort_by_identifier(description.instances);

    if (auto status = reject_duplicate_identifiers(
            description.objects, "A frame scene contains duplicate object identifiers.");
        !status) {
        return std::unexpected(std::move(status.error()));
    }
    if (auto status = reject_duplicate_identifiers(
            description.geometries, "A frame scene contains duplicate geometry identifiers.");
        !status) {
        return std::unexpected(std::move(status.error()));
    }
    if (auto status = reject_duplicate_identifiers(
            description.materials, "A frame scene contains duplicate material identifiers.");
        !status) {
        return std::unexpected(std::move(status.error()));
    }
    if (auto status = reject_duplicate_identifiers(
            description.instances, "A frame scene contains duplicate instance identifiers.");
        !status) {
        return std::unexpected(std::move(status.error()));
    }

    for (const auto& instance : description.instances) {
        if (!contains_identifier(description.objects, instance.object)) {
            return std::unexpected(
                scene_error(core::StatusCode::invalid_argument,
                            "A frame scene instance references an unknown object identifier."));
        }
        if (!contains_identifier(description.geometries, instance.geometry)) {
            return std::unexpected(
                scene_error(core::StatusCode::invalid_argument,
                            "A frame scene instance references an unknown geometry identifier."));
        }
        if (!contains_identifier(description.materials, instance.material)) {
            return std::unexpected(
                scene_error(core::StatusCode::invalid_argument,
                            "A frame scene instance references an unknown material identifier."));
        }
    }

    try {
        return FrameSceneHandle{new FrameScene{std::move(description)}};
    } catch (const std::bad_alloc&) {
        return std::unexpected(scene_error(core::StatusCode::resource_exhausted,
                                           "Frame scene storage exhausted host memory."));
    }
}

FrameScene::FrameScene(FrameSceneDescription&& description) noexcept
    : objects_{std::move(description.objects)}, geometries_{std::move(description.geometries)},
      materials_{std::move(description.materials)}, instances_{std::move(description.instances)} {}

std::span<const SceneObject> FrameScene::objects() const noexcept {
    return objects_;
}

std::span<const SceneGeometry> FrameScene::geometries() const noexcept {
    return geometries_;
}

std::span<const SceneMaterial> FrameScene::materials() const noexcept {
    return materials_;
}

std::span<const SceneInstance> FrameScene::instances() const noexcept {
    return instances_;
}

core::Result<std::reference_wrapper<const SceneObject>>
FrameScene::object(const renderer::ObjectId id) const {
    return find_record(objects_, id, "The frame scene does not contain the requested object.");
}

core::Result<std::reference_wrapper<const SceneGeometry>>
FrameScene::geometry(const renderer::GeometryId id) const {
    return find_record(geometries_, id, "The frame scene does not contain the requested geometry.");
}

core::Result<std::reference_wrapper<const SceneMaterial>>
FrameScene::material(const renderer::MaterialId id) const {
    return find_record(materials_, id, "The frame scene does not contain the requested material.");
}

core::Result<std::reference_wrapper<const SceneInstance>>
FrameScene::instance(const renderer::InstanceId id) const {
    return find_record(instances_, id, "The frame scene does not contain the requested instance.");
}

} // namespace blackframe::engine
