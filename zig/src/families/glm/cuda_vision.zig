//! Verified native host units for GLM5-Next CUDA vision.
//! No CUDA encoder or image codec is registered here; the native engine must continue refusing vision.
//! Oracle: Mia b5808ac6 vision/glm.py == archived da6282f9, SHA256
//! 07da2a80bd37a95c5728026024ccdddbc36725d1d84066aee116860987b93e01.
pub const config = @import("cuda_vision_config.zig");
pub const preprocess = @import("cuda_vision_preprocess.zig");
pub const weights = @import("cuda_vision_weights.zig");
pub const rope = @import("cuda_vision_rope.zig");
pub const resize = @import("cuda_vision_resize.zig");
pub const image = @import("cuda_vision_image.zig");
pub const Config = config.Config;
pub const Weights = weights.Weights;
pub const prepareRgb = preprocess.prepareRgb;
pub const decodeInline = preprocess.decodeInline;
pub const imagePositions = preprocess.imagePositions;

// F.layer_norm in the oracle's merger uses its default epsilon, independently of vision RMSNorm.
pub const projector_layernorm_eps: f32 = 1e-5;

test {
    _ = config;
    _ = preprocess;
    _ = weights;
    _ = rope;
    _ = @import("cuda_vision_test.zig");
}
