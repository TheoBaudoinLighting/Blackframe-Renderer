#include <Blackframe/Engine/FrameScene.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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
[[nodiscard]] std::optional<std::size_t> find_record_index(const std::vector<Record>& records,
                                                           const Identifier id) noexcept {
    const auto candidate = std::ranges::lower_bound(
        records, id.value, {}, [](const Record& record) { return record.id.value; });
    if (candidate == records.end() || candidate->id != id) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(candidate - records.begin());
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

enum class HierarchyState : std::uint8_t {
    unresolved,
    resolving,
    resolved,
};

struct ResolvedInstanceTransforms final {
    std::vector<renderer::AffineTransform> local;
    std::vector<renderer::AffineTransform> world;
};

[[nodiscard]] core::Result<ResolvedInstanceTransforms>
resolve_instance_transforms(const std::vector<SceneInstance>& instances) {
    auto resolved = ResolvedInstanceTransforms{};
    auto states = std::vector<HierarchyState>{};
    auto pending_world = std::vector<std::optional<renderer::AffineTransform>>{};
    auto chain = std::vector<std::size_t>{};
    const auto instance_count = instances.size();
    if (instance_count > resolved.local.max_size() || instance_count > resolved.world.max_size() ||
        instance_count > states.max_size() || instance_count > pending_world.max_size() ||
        instance_count > chain.max_size()) {
        return std::unexpected(scene_error(core::StatusCode::resource_exhausted,
                                           "Frame scene hierarchy exceeds host container limits."));
    }

    resolved.local.reserve(instances.size());
    for (const auto& instance : instances) {
        auto local = renderer::AffineTransform::from_matrix(instance.local_to_parent);
        if (!local) {
            return std::unexpected(
                scene_error(core::StatusCode::invalid_argument,
                            "A frame scene instance local transform must be finite, affine, "
                            "and invertible."));
        }
        resolved.local.push_back(std::move(*local));
    }

    states.assign(instances.size(), HierarchyState::unresolved);
    pending_world.resize(instances.size());
    chain.reserve(instances.size());

    for (std::size_t start = 0; start < instances.size(); ++start) {
        if (states[start] == HierarchyState::resolved) {
            continue;
        }

        chain.clear();
        auto current = start;
        while (states[current] != HierarchyState::resolved) {
            if (states[current] == HierarchyState::resolving) {
                return std::unexpected(
                    scene_error(core::StatusCode::invalid_argument,
                                "A frame scene instance hierarchy contains a cycle."));
            }

            states[current] = HierarchyState::resolving;
            chain.push_back(current);
            if (!instances[current].parent) {
                break;
            }

            const auto parent_index = find_record_index(instances, *instances[current].parent);
            if (!parent_index) {
                return std::unexpected(
                    scene_error(core::StatusCode::invalid_argument,
                                "A frame scene instance references an unknown parent instance."));
            }
            current = *parent_index;
        }

        while (!chain.empty()) {
            const auto instance_index = chain.back();
            chain.pop_back();
            const auto& instance = instances[instance_index];

            if (instance.parent) {
                const auto parent_index = find_record_index(instances, *instance.parent);
                if (!parent_index || states[*parent_index] != HierarchyState::resolved ||
                    !pending_world[*parent_index]) {
                    return std::unexpected(
                        scene_error(core::StatusCode::internal_error,
                                    "Frame scene hierarchy resolution lost a validated parent."));
                }

                auto world =
                    renderer::AffineTransform::from_matrix(pending_world[*parent_index]->matrix() *
                                                           resolved.local[instance_index].matrix());
                if (!world) {
                    return std::unexpected(scene_error(
                        core::StatusCode::invalid_argument,
                        "A frame scene hierarchy produced an invalid world transform."));
                }
                pending_world[instance_index] = std::move(*world);
            } else {
                pending_world[instance_index] = resolved.local[instance_index];
            }
            states[instance_index] = HierarchyState::resolved;
        }
    }

    resolved.world.reserve(instances.size());
    for (auto& world : pending_world) {
        if (!world) {
            return std::unexpected(
                scene_error(core::StatusCode::internal_error,
                            "Frame scene hierarchy resolution left an instance unresolved."));
        }
        resolved.world.push_back(std::move(*world));
    }
    return resolved;
}

} // namespace

core::Result<FrameSceneHandle> FrameScene::create(const FrameSceneDescription& description) {
    try {
        auto owned_description = description;
        return create(std::move(owned_description));
    } catch (const std::bad_alloc&) {
        return std::unexpected(scene_error(core::StatusCode::resource_exhausted,
                                           "Frame scene storage exhausted host memory."));
    } catch (const std::length_error&) {
        return std::unexpected(scene_error(core::StatusCode::resource_exhausted,
                                           "Frame scene storage exceeds host container limits."));
    }
}

core::Result<FrameSceneHandle> FrameScene::create(FrameSceneDescription&& description) {
    try {
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
                return std::unexpected(scene_error(
                    core::StatusCode::invalid_argument,
                    "A frame scene instance references an unknown geometry identifier."));
            }
            if (!contains_identifier(description.materials, instance.material)) {
                return std::unexpected(scene_error(
                    core::StatusCode::invalid_argument,
                    "A frame scene instance references an unknown material identifier."));
            }
            if (instance.parent && !contains_identifier(description.instances, *instance.parent)) {
                return std::unexpected(
                    scene_error(core::StatusCode::invalid_argument,
                                "A frame scene instance references an unknown parent instance."));
            }
        }

        auto transforms = resolve_instance_transforms(description.instances);
        if (!transforms) {
            return std::unexpected(std::move(transforms.error()));
        }

        return FrameSceneHandle{new FrameScene{std::move(description), std::move(transforms->local),
                                               std::move(transforms->world)}};
    } catch (const std::bad_alloc&) {
        return std::unexpected(scene_error(core::StatusCode::resource_exhausted,
                                           "Frame scene storage exhausted host memory."));
    } catch (const std::length_error&) {
        return std::unexpected(scene_error(core::StatusCode::resource_exhausted,
                                           "Frame scene storage exceeds host container limits."));
    }
}

FrameScene::FrameScene(FrameSceneDescription&& description,
                       std::vector<renderer::AffineTransform>&& local_transforms,
                       std::vector<renderer::AffineTransform>&& world_transforms) noexcept
    : objects_{std::move(description.objects)}, geometries_{std::move(description.geometries)},
      materials_{std::move(description.materials)}, instances_{std::move(description.instances)},
      local_transforms_{std::move(local_transforms)},
      world_transforms_{std::move(world_transforms)} {}

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

core::Result<std::reference_wrapper<const renderer::AffineTransform>>
FrameScene::local_transform(const renderer::InstanceId id) const {
    const auto index = find_record_index(instances_, id);
    if (!index) {
        return std::unexpected(scene_error(
            core::StatusCode::not_found,
            "The frame scene does not contain the requested instance local transform."));
    }
    return std::cref(local_transforms_[*index]);
}

core::Result<std::reference_wrapper<const renderer::AffineTransform>>
FrameScene::world_transform(const renderer::InstanceId id) const {
    const auto index = find_record_index(instances_, id);
    if (!index) {
        return std::unexpected(scene_error(
            core::StatusCode::not_found,
            "The frame scene does not contain the requested instance world transform."));
    }
    return std::cref(world_transforms_[*index]);
}

} // namespace blackframe::engine
