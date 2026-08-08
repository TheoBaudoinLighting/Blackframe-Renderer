#include "SceneHierarchyValidation.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace blackframe::engine::scene_hierarchy_validation {
namespace {

[[nodiscard]] core::Error scene_error(const core::StatusCode code, const std::string_view message) {
    return core::Error{
        .code = code,
        .message = std::string{message},
    };
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

enum class HierarchyState : std::uint8_t {
    unresolved,
    resolving,
    resolved,
};

} // namespace

core::Result<ResolvedTransforms>
resolve_instance_transforms(const std::vector<SceneInstance>& instances) {
    auto resolved = ResolvedTransforms{};
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

} // namespace blackframe::engine::scene_hierarchy_validation
