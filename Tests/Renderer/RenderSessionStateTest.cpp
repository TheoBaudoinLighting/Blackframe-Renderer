#include <Blackframe/Renderer/RenderSessionState.hpp>
#include <gtest/gtest.h>

namespace blackframe::renderer {
namespace {

TEST(RenderSessionStateTest, ExposesStableDiagnosticNames) {
    EXPECT_EQ(render_session_state_name(RenderSessionState::created), "created");
    EXPECT_EQ(render_session_state_name(RenderSessionState::ready), "ready");
    EXPECT_EQ(render_session_state_name(RenderSessionState::rendering), "rendering");
    EXPECT_EQ(render_session_state_name(RenderSessionState::cancelling), "cancelling");
    EXPECT_EQ(render_session_state_name(RenderSessionState::completed), "completed");
    EXPECT_EQ(render_session_state_name(RenderSessionState::failed), "failed");
}

} // namespace
} // namespace blackframe::renderer
