#include <Blackframe/Renderer/RenderSessionState.hpp>

namespace blackframe::renderer {

std::string_view render_session_state_name(const RenderSessionState state) noexcept {
    switch (state) {
    case RenderSessionState::created:
        return "created";
    case RenderSessionState::ready:
        return "ready";
    case RenderSessionState::rendering:
        return "rendering";
    case RenderSessionState::cancelling:
        return "cancelling";
    case RenderSessionState::completed:
        return "completed";
    case RenderSessionState::failed:
        return "failed";
    }

    return "unknown";
}

} // namespace blackframe::renderer
