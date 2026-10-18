//! CPU tests of contract inputs, independent references and offline launch metadata; no CUDA driver is opened.
test {
    _ = @import("contract_glm_attention_data.zig");
    _ = @import("contract_glm_experts_data.zig");
    _ = @import("contract_glm_experts.zig");
    _ = @import("contract_glm_dense.zig");
    _ = @import("contract_glm_launches_test.zig");
    _ = @import("contract_limits.zig");
}
