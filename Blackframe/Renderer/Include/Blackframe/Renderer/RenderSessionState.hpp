#pragma once

#include <string_view>

namespace blackframe::renderer {

enum class RenderSessionState {
    created,
    ready,
    rendering,
    cancelling,
    completed,
    failed,
};

[[nodiscard]] std::string_view render_session_state_name(RenderSessionState state) noexcept;

} // namespace blackframe::renderer
