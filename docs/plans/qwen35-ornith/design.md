# Native Qwen3.5 / Qwen3.6 ModelOpt text inference

The delivery target is now a runnable production deployment. Follow the
[production plan](production.md) for the shared-engine integration and
qualification work after this initial correctness baseline.

Use candidate A's CUDA-local checkpoint and execution boundary. The independent
cross-judge preferred it on completeness, quantization semantics, state
ownership, existing-backend isolation and verification. Candidate B's shared
format would require migrating the working HIP decoder contract. Retain B's
per-tensor validation, incremental expert uploads and CPU inspection report.

The first runnable path uses serial token prefill and decode, BF16 KV, FP32
GDN state and a contiguous layer split. It is a correctness baseline. It makes
no throughput or production-readiness claim. Vision, MTP, tensor parallelism,
prefix caching and spill are subsequent work.

## Ownership and implementation contract

All files below are in `zig/src/families/qwen3_5/` unless stated otherwise.
The implementation workers own disjoint files in this task worktree. The
coordinator owns build integration, the engine, native adapter and docs.

### Checkpoint and device weights

`cuda_checkpoint.zig` owns `Config` and a pure host checkpoint view over
`core.Checkpoint`. Reuse `config.zig`'s general `declared`, `textTower` and
`geometry` validation. Pin admission to Qwen3.5/3.6 35B geometry initially.
Expose `Config.read(gpa, io, dir)`, `.spec` of the existing `Spec` type, and
`.context`. Official raw HF tensors use `model.language_model.*` and
`lm_head.*`. Norms use explicit raw HF zero-centered semantics. Conv weights
are `[8192,1,4]`. No global quantization mode controls individual projections.

`cuda_weights.zig` exposes:

```zig
pub const Projection = cuda.qlinear.Weight;
pub const Dense = struct { gate: Projection, up: Projection, down: Projection };
pub const Moe = struct {
    router: u64, shared_gate: u64,
    experts: cuda.experts.Layer, shared: Dense,
};
pub const Full = struct {
    q: Projection, k: Projection, v: Projection, out: Projection,
    q_norm: u64, k_norm: u64,
};
pub const Gdn = struct {
    qkv: Projection, z: Projection, a: Projection, b: Projection, out: Projection,
    conv: u64, a_log: u64, dt_bias: u64, norm: u64,
};
pub const Layer = struct {
    input_norm: u64, post_norm: u64, moe: Moe,
    mixer: union(enum) { full: Full, gdn: Gdn },
};
pub const Weights = struct {
    layers: []Layer,
    embed: ?u64, head: ?Projection, norm: ?u64,
    // Own device buffers. Layer indices in layers are local to first..end.
    pub fn load(gpa: Allocator, io: Io, ctx: *const cuda.Context,
        dir: []const u8, cfg: Config, first: usize, end: usize,
        with_embed: bool, with_head: bool) !Weights;
    pub fn deinit(self: *Weights) void;
};
```

FP8 `Weight.fp8` carries static input scale `.act` and uses
`qlinear.Linear.ck` checkpoint quantization. W4A16 uses `Weight.nvfp4` and
never requantizes activations to FP4. Gate/up have independent global scales.
The shared expert is separate from the top-8 route. Norms/conv/state parameters
are uploaded as FP32; router and shared gate are BF16. The embedding is BF16.
Upload and pack one expert at a time, release temporary device staging before
the next one. Expose buffer byte totals for admission.

### Kernel bindings

`cuda_kernels.zig` owns Qwen-specific `Ops` and the existing `qlinear`,
ModelOpt/checkpoint, grouped-expert and norm kernels it requires. Add
`zig/kernels/cuda/qwen35_ops.cu` for missing glue, and any necessary attention
kernel in the same image initially. Existing family code remains untouched.
`Ops.load(ctx)` and `.unload()` manage modules. Expose bindings over addresses
and the fixed admitted geometry, with explicit `cuda.Stream` arguments:

```zig
embed(stream, table, token: u32, out);
rms(stream, x, weight_f32, out, width: usize, eps: f32);
add(stream, residual, update, width: usize); // BF16 residual += update
gdnPre(stream, qkv, conv_weights, history, a, b, a_log, dt_bias,
    q, k, v, decay, beta);
gdnStep(stream, q, k, v, decay, beta, recurrent_state, out);
gdnOut(stream, y_f32, z_bf16, norm_f32, out_bf16, eps: f32);
attention(stream, q_gate, k, v, q_norm_f32, k_norm_f32,
    keys, values, position: usize, capacity: usize, out_bf16, eps: f32);
route(stream, x, router_bf16, shared_gate_bf16, picks_i32, weights_f32);
combine(stream, pairs_f32, weights_f32, shared_bf16, out_bf16);
swiglu(stream, gate, up, out, width: usize);
```

Routing writes eight ids and nine weights. Weights 0..7 are softmax normalized
over the selected logits, with lower ids winning ties; weight 8 is the sigmoid
of the shared gate. The grouped plan for eight pairs has 256 experts and tile
16. Expert up output is BF16 `[8,512]`, down FP32 `[8,2048]`, combined in pick
order plus shared. Norm conversions happen once at load. GDN caches store the
last three BF16 qkv rows, and FP32 recurrent matrices `[32,128,128]`.

### Engine and proof

`cuda_engine.zig` owns device contexts, per-partition streams, Ops, immutable
weights, scratch and sessions. `Session.append` uses one forward path for
prefill and decode. Synchronize each partition before host-staged hidden
transfer. Reset clears both devices. A failed partial token poisons the session
until reset. Validate token ids and context before state writes.

The native API and a teacher-forcing probe use the same engine. Capability
refusals happen before allocation. Check free GPU memory before loading and
after each load/session allocation. Never cross the 1536 MiB per-GPU floor.
The initial default is a small context; the model's 262144 geometry limit is
not an admitted-capacity promise.

CPU tests and native/fatbin builds precede any approved device window. Run the
same pinned checkpoint through a trusted reference to compare teacher-forced
logits and greedy text, then exercise cancellation and repeated requests.
Build success alone leaves the model unqualified.
