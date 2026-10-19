// Device metadata for the native GLM target-forward graph path.
// Mirrors cuda_target_meta.zig. Sequence addresses and geometry are refreshed
// before replay; captured kernels receive only the stable table addresses.
#pragma once
#include <stdint.h>
#include <stddef.h>

struct TfGlmTargetRow {
    uint32_t segment;
    uint32_t local;
};

struct TfGlmTargetKdaSegment {
    uint32_t row0;
    uint32_t rows;
    uint64_t conv;
    uint64_t state_in;
    uint64_t state_out;
    uint64_t proj;
    uint64_t k;
    uint64_t v;
    uint64_t g;
    uint64_t b;
};

struct TfGlmTargetDsaRow {
    uint64_t latent;
    uint64_t index_keys;
    uint64_t index_gates;
    uint64_t pooled;
    uint32_t position;
    uint32_t ring_rows;
    uint32_t pooled_rows;
    uint32_t owner;
    uint32_t flags;
    uint32_t pad; // Segment-end merge chunk count (1..5), not unused padding.
};

static_assert(sizeof(TfGlmTargetRow) == 8, "GLM target row ABI");
static_assert(sizeof(TfGlmTargetKdaSegment) == 72, "GLM target KDA ABI");
static_assert(sizeof(TfGlmTargetDsaRow) == 56, "GLM target DSA ABI");
static_assert(offsetof(TfGlmTargetKdaSegment, conv) == 8, "GLM target KDA pointer alignment");
static_assert(offsetof(TfGlmTargetDsaRow, position) == 32, "GLM target DSA position ABI");
static constexpr uint32_t TF_GLM_TARGET_MAX_ROWS = 32;
static constexpr uint32_t TF_GLM_TARGET_MAX_SEGMENTS = 8;
static constexpr uint32_t TF_GLM_TARGET_UNOWNED = UINT32_MAX;
static constexpr uint32_t TF_GLM_TARGET_RESIDENT = 1;
static constexpr uint32_t TF_GLM_TARGET_SPARSE = 2;
