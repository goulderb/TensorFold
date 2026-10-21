//! cuBLASLt opened at run time: descriptors, layouts, heuristics and matmul, with fp32 compute only (no TF32 or fast modes).

const std = @import("std");
const platform = @import("sys");
const abi = @import("abi.zig");

pub const Status = c_int;
pub const Handle = ?*opaque {};
pub const MatmulDesc = ?*opaque {};
pub const Layout = ?*opaque {};
pub const Preference = ?*opaque {};

pub const Algo = extern struct { data: [8]u64 };

pub const Heuristic = extern struct {
    algo: Algo,
    workspace_size: usize,
    state: Status,
    waves_count: f32,
    reserved: [4]c_int,
};

pub const DataType = enum(c_int) { f32 = 0, f16 = 2, bf16 = 14 };
pub const compute_32f: c_int = 68;
pub const Op = enum(c_int) { n = 0, t = 1 };
pub const Order = enum(c_int) { col = 0, row = 1 };

pub const desc_transa: c_int = 3;
pub const desc_transb: c_int = 4;
// CUDA 13.3.73 cublasLt.h:1420,1439,1646,2065; library_types.h:61.
pub const desc_epilogue: c_int = 7;
pub const desc_bias_pointer: c_int = 8;
pub const desc_bias_data_type: c_int = 26;
pub const layout_order: c_int = 1;
pub const pref_max_workspace_bytes: c_int = 1;
pub const pref_min_alignment_a_bytes: c_int = 5;
pub const pref_min_alignment_b_bytes: c_int = 6;
pub const pref_min_alignment_c_bytes: c_int = 7;
pub const pref_min_alignment_d_bytes: c_int = 8;

/// What D gets after the product: nothing, or a bias over its columns added in f32 before D's one rounding.
pub const Epilogue = enum(u32) { none = 1, bias = 4 };

pub const Error = platform.optional_libraries.Failure || error{ LibraryUnavailable, MissingSymbol, CublasFailed, NoAlgorithm, Invalid };

const S = Status;
const CPtr = ?*const anyopaque;

/// Device pointers are declared as u64: the same register class as `void*` on every LP64 target.
pub const Api = struct {
    cublasLtCreate: *const fn (*Handle) callconv(.c) S,
    cublasLtDestroy: *const fn (Handle) callconv(.c) S,
    cublasLtGetVersion: *const fn () callconv(.c) usize,
    cublasLtGetStatusName: *const fn (S) callconv(.c) ?[*:0]const u8,
    cublasLtMatmulDescCreate: *const fn (*MatmulDesc, c_int, DataType) callconv(.c) S,
    cublasLtMatmulDescDestroy: *const fn (MatmulDesc) callconv(.c) S,
    cublasLtMatmulDescSetAttribute: *const fn (MatmulDesc, c_int, CPtr, usize) callconv(.c) S,
    cublasLtMatrixLayoutCreate: *const fn (*Layout, DataType, u64, u64, i64) callconv(.c) S,
    cublasLtMatrixLayoutDestroy: *const fn (Layout) callconv(.c) S,
    cublasLtMatrixLayoutSetAttribute: *const fn (Layout, c_int, CPtr, usize) callconv(.c) S,
    cublasLtMatmulPreferenceCreate: *const fn (*Preference) callconv(.c) S,
    cublasLtMatmulPreferenceDestroy: *const fn (Preference) callconv(.c) S,
    cublasLtMatmulPreferenceSetAttribute: *const fn (Preference, c_int, CPtr, usize) callconv(.c) S,
    cublasLtMatmulAlgoGetHeuristic: *const fn (Handle, MatmulDesc, Layout, Layout, Layout, Layout, Preference, c_int, [*]Heuristic, *c_int) callconv(.c) S,
    cublasLtMatmulAlgoCheck: *const fn (Handle, MatmulDesc, Layout, Layout, Layout, Layout, *const Algo, *Heuristic) callconv(.c) S,
    cublasLtMatmul: *const fn (Handle, MatmulDesc, CPtr, abi.DevicePtr, Layout, abi.DevicePtr, Layout, CPtr, abi.DevicePtr, Layout, abi.DevicePtr, Layout, ?*const Algo, abi.DevicePtr, usize, abi.Stream) callconv(.c) S,
};

pub const Library = struct {
    lib: platform.DynLib,
    api: Api,

    /// The CUDA 13 soname first, then the unversioned link name; on Windows the toolkit's DLL.
    pub fn open() Error!Library {
        const names: []const []const u8 = if (@import("builtin").os.tag == .windows) &.{"cublasLt64_13.dll"} else &.{ "libcublasLt.so.13", "libcublasLt.so" };
        for (names) |path| {
            return openCandidate(path) catch |e| switch (e) {
                error.LibraryUnavailable => continue,
                error.MissingSymbol => return platform.optional_libraries.unavailable(.cublaslt),
                else => return e,
            };
        }
        return platform.optional_libraries.unavailable(.cublaslt);
    }

    pub fn openPath(path: []const u8) Error!Library {
        return openCandidate(path) catch |e| switch (e) {
            error.LibraryUnavailable, error.MissingSymbol => platform.optional_libraries.unavailable(.cublaslt),
            else => e,
        };
    }

    fn openCandidate(path: []const u8) Error!Library {
        var lib = platform.DynLib.open(path) catch return error.LibraryUnavailable;
        errdefer lib.close();
        var api: Api = undefined;
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse {
                std.log.err("{s} has no {s}", .{ path, name });
                return error.MissingSymbol;
            };
        }
        return .{ .lib = lib, .api = api };
    }

    pub fn close(self: *Library) void {
        self.lib.close();
    }

    pub fn check(self: *const Library, s: Status, what: []const u8) Error!void {
        if (s == 0) return;
        const n = self.api.cublasLtGetStatusName(s);
        // Fault-injection tests assert the returned error; Zig's runner treats error logs as test failures.
        if (!@import("builtin").is_test) std.log.err("{s}: {s} ({d})", .{ what, if (n) |p| std.mem.span(p) else "?", s });
        return error.CublasFailed;
    }
};

/// Borrowed device storage, with capacity in bytes (not elements). No host dereference occurs.
pub const DeviceSpan = struct {
    ptr: abi.DevicePtr,
    len: usize,

    fn validate(self: DeviceSpan, need: usize, alignment: u64) Error!void {
        if (self.len < need) return error.Invalid;
        if (self.len == 0 and need == 0) return;
        if (self.ptr == 0 or self.ptr % alignment != 0) return error.Invalid;
        const len = std.math.cast(u64, self.len) orelse return error.Invalid;
        _ = std.math.add(u64, self.ptr, len) catch return error.Invalid;
    }
};

const Sizes = struct {
    x: usize,
    w: usize,
    d: usize,
    bias: usize,

    fn init(m: u64, n: u64, k: u64, out: DataType) Error!Sizes {
        for ([_]u64{ m, n, k }) |dim| {
            if (dim == 0 or dim > std.math.maxInt(i64)) return error.Invalid;
        }
        const out_bytes: u64 = if (out == .f32) 4 else 2;
        return .{
            .x = try bytes(m, k, 2),
            .w = try bytes(n, k, 2),
            .d = try bytes(m, n, out_bytes),
            .bias = try bytes(n, 1, 2),
        };
    }

    fn bytes(rows: u64, cols: u64, element_bytes: u64) Error!usize {
        const elements = std.math.mul(u64, rows, cols) catch return error.Invalid;
        const size = std.math.mul(u64, elements, element_bytes) catch return error.Invalid;
        return std.math.cast(usize, size) orelse error.Invalid;
    }
};

/// Row-major D[m,n] = X[m,k] . W[n,k]^T, with fp32 accumulation.
/// Owns cuBLASLt resources, not the library or any device storage. Do not copy a live owner.
pub const Linear = struct {
    lt: *const Library,
    handle: Handle,
    desc: MatmulDesc,
    a: Layout,
    b: Layout,
    c: Layout,
    pref: Preference,
    algo: Algo,
    workspace_need: usize,
    sizes: Sizes,
    bias: ?DeviceSpan,
    epilogue: Epilogue,

    pub fn init(lt: *const Library, m: u64, n: u64, k: u64, out: DataType, workspace_limit: usize) Error!Linear {
        return initEpilogue(lt, m, n, k, out, workspace_limit, .none);
    }

    /// Requests optional fusion; epilogue reports whether the selected algorithm supports it.
    pub fn initEpilogue(lt: *const Library, m: u64, n: u64, k: u64, out: DataType, workspace_limit: usize, epilogue: Epilogue) Error!Linear {
        return initEpilogueChecked(lt, m, n, k, out, workspace_limit, epilogue, null);
    }

    /// A bound BF16 bias requires fusion and remains borrowed through completion of outstanding work.
    /// Bound plans must use runChecked, which validates all caller-reported capacities and aliases.
    pub fn initEpilogueChecked(lt: *const Library, m: u64, n: u64, k: u64, out: DataType, workspace_limit: usize, epilogue: Epilogue, bias: ?DeviceSpan) Error!Linear {
        if (bias != null and (epilogue != .bias or out != .bf16)) return error.Invalid;
        const sizes = try Sizes.init(m, n, k, out);
        if (bias) |span| try span.validate(sizes.bias, 2);
        var self: Linear = .{
            .lt = lt,
            .handle = null,
            .desc = null,
            .a = null,
            .b = null,
            .c = null,
            .pref = null,
            .algo = undefined,
            .workspace_need = 0,
            .sizes = sizes,
            .bias = bias,
            .epilogue = .none,
        };
        // Arm cleanup before each create: even a failed call that sets a handle must unwind.
        errdefer self.release();
        try lt.check(lt.api.cublasLtCreate(&self.handle), "cublasLtCreate");
        if (self.handle == null) return error.CublasFailed;
        try lt.check(lt.api.cublasLtMatmulDescCreate(&self.desc, compute_32f, .f32), "cublasLtMatmulDescCreate");
        if (self.desc == null) return error.CublasFailed;
        const ta: c_int = @backingInt(Op.t);
        const tb: c_int = @backingInt(Op.n);
        try lt.check(lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_transa, &ta, @sizeOf(c_int)), "set transa");
        try lt.check(lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_transb, &tb, @sizeOf(c_int)), "set transb");
        if (bias) |span| {
            const dtype: DataType = .bf16;
            try lt.check(lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_epilogue, &@as(u32, @backingInt(Epilogue.bias)), @sizeOf(u32)), "set bias epilogue");
            try lt.check(lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_bias_data_type, &dtype, @sizeOf(DataType)), "set bias dtype");
            try lt.check(lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_bias_pointer, &span.ptr, @sizeOf(abi.DevicePtr)), "set bias pointer");
        }
        // Column-major D^T[n,m] = W^T[k,n]^T . X^T[k,m]. Bias[n] broadcasts over m columns.
        try lt.check(lt.api.cublasLtMatrixLayoutCreate(&self.a, .bf16, k, n, @intCast(k)), "layout W");
        if (self.a == null) return error.CublasFailed;
        try lt.check(lt.api.cublasLtMatrixLayoutCreate(&self.b, .bf16, k, m, @intCast(k)), "layout X");
        if (self.b == null) return error.CublasFailed;
        try lt.check(lt.api.cublasLtMatrixLayoutCreate(&self.c, out, n, m, @intCast(n)), "layout D");
        if (self.c == null) return error.CublasFailed;
        if (bias != null) {
            const order: c_int = @backingInt(Order.col);
            for ([_]Layout{ self.a, self.b, self.c }) |layout| {
                try lt.check(lt.api.cublasLtMatrixLayoutSetAttribute(layout, layout_order, &order, @sizeOf(c_int)), "set column order");
            }
        }
        try lt.check(lt.api.cublasLtMatmulPreferenceCreate(&self.pref), "cublasLtMatmulPreferenceCreate");
        if (self.pref == null) return error.CublasFailed;
        const limit: u64 = workspace_limit;
        try lt.check(lt.api.cublasLtMatmulPreferenceSetAttribute(self.pref, pref_max_workspace_bytes, &limit, @sizeOf(u64)), "set workspace");
        if (bias != null) {
            // Default matrix alignment is 256. Select only algorithms safe for BF16 subviews.
            const alignment: u32 = 2;
            for ([_]c_int{ pref_min_alignment_a_bytes, pref_min_alignment_b_bytes, pref_min_alignment_c_bytes, pref_min_alignment_d_bytes }) |attr| {
                try lt.check(lt.api.cublasLtMatmulPreferenceSetAttribute(self.pref, attr, &alignment, @sizeOf(u32)), "set BF16 alignment");
            }
        }
        var found: [1]Heuristic = undefined;
        var count: c_int = 0;
        try lt.check(lt.api.cublasLtMatmulAlgoGetHeuristic(self.handle, self.desc, self.a, self.b, self.c, self.c, self.pref, 1, &found, &count), "cublasLtMatmulAlgoGetHeuristic");
        if (count != 1 or found[0].state != 0) return error.NoAlgorithm;
        if (found[0].workspace_size > workspace_limit) return error.NoAlgorithm;
        self.algo = found[0].algo;
        self.workspace_need = found[0].workspace_size;
        if (bias != null) {
            self.epilogue = .bias;
        } else if (epilogue != .none) try self.fuse(epilogue, workspace_limit);
        return self;
    }

    fn release(self: *Linear) void {
        const api = self.lt.api;
        if (self.pref != null) _ = api.cublasLtMatmulPreferenceDestroy(self.pref);
        if (self.c != null) _ = api.cublasLtMatrixLayoutDestroy(self.c);
        if (self.b != null) _ = api.cublasLtMatrixLayoutDestroy(self.b);
        if (self.a != null) _ = api.cublasLtMatrixLayoutDestroy(self.a);
        if (self.desc != null) _ = api.cublasLtMatmulDescDestroy(self.desc);
        if (self.handle != null) _ = api.cublasLtDestroy(self.handle);
    }

    pub fn deinit(self: *Linear) void {
        self.release();
        self.* = undefined;
    }

    /// Legacy unbiased dispatch. The caller still guarantees actual device allocation capacities.
    pub fn run(self: *const Linear, x: abi.DevicePtr, w: abi.DevicePtr, d: abi.DevicePtr, workspace: abi.DevicePtr, workspace_len: usize, stream: abi.Stream) Error!void {
        if (self.bias != null) return error.Invalid;
        // Unbiased heuristics retain cuBLASLt's default 256-byte matrix alignment preference.
        try (DeviceSpan{ .ptr = x, .len = self.sizes.x }).validate(self.sizes.x, 256);
        try (DeviceSpan{ .ptr = w, .len = self.sizes.w }).validate(self.sizes.w, 256);
        try (DeviceSpan{ .ptr = d, .len = self.sizes.d }).validate(self.sizes.d, 256);
        try (DeviceSpan{ .ptr = workspace, .len = workspace_len }).validate(self.workspace_need, 256);
        try self.dispatch(x, w, d, workspace, workspace_len, stream);
    }

    /// Checked dispatch with caller-reported capacities; output/workspace may not alias any input.
    /// Biased matrix pointers need 2-byte alignment; unbiased owners retain the 256-byte heuristic default.
    pub fn runChecked(self: *const Linear, x: DeviceSpan, w: DeviceSpan, d: DeviceSpan, workspace: DeviceSpan, stream: abi.Stream) Error!void {
        const alignment: u64 = if (self.bias != null) 2 else 256;
        try x.validate(self.sizes.x, alignment);
        try w.validate(self.sizes.w, alignment);
        try d.validate(self.sizes.d, alignment);
        try workspace.validate(self.workspace_need, 256);
        if (self.bias) |span| try span.validate(self.sizes.bias, 2);
        const read_ptrs = [_]abi.DevicePtr{ x.ptr, w.ptr, if (self.bias) |span| span.ptr else 0 };
        const read_lens = [_]usize{ self.sizes.x, self.sizes.w, if (self.bias != null) self.sizes.bias else 0 };
        for (read_ptrs, read_lens) |ptr, len| {
            if (overlap(d.ptr, self.sizes.d, ptr, len) or overlap(workspace.ptr, self.workspace_need, ptr, len)) return error.Invalid;
        }
        if (overlap(d.ptr, self.sizes.d, workspace.ptr, self.workspace_need)) return error.Invalid;
        try self.dispatch(x.ptr, w.ptr, d.ptr, if (self.workspace_need == 0) 0 else workspace.ptr, self.workspace_need, stream);
    }

    fn overlap(a: abi.DevicePtr, a_len: usize, b: abi.DevicePtr, b_len: usize) bool {
        if (a_len == 0 or b_len == 0) return false;
        // All spans were validated, so their exclusive ends cannot wrap.
        return a < b + @as(u64, @intCast(b_len)) and b < a + @as(u64, @intCast(a_len));
    }

    fn dispatch(self: *const Linear, x: abi.DevicePtr, w: abi.DevicePtr, d: abi.DevicePtr, workspace: abi.DevicePtr, workspace_len: usize, stream: abi.Stream) Error!void {
        const one: f32 = 1;
        const zero: f32 = 0;
        try self.lt.check(self.lt.api.cublasLtMatmul(self.handle, self.desc, &one, w, self.a, x, self.b, &zero, d, self.c, d, self.c, &self.algo, workspace, workspace_len, stream), "cublasLtMatmul");
    }

    /// D = X . W^T + bias (bias [n], D's type) on a plan that fused `.bias`, one rounding to D.
    pub fn runBias(self: *const Linear, x: abi.DevicePtr, w: abi.DevicePtr, bias: abi.DevicePtr, d: abi.DevicePtr, workspace: abi.DevicePtr, workspace_len: usize, stream: abi.Stream) Error!void {
        if (self.epilogue != .bias or self.bias != null) return error.Invalid;
        try self.lt.check(self.lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_bias_pointer, &bias, @sizeOf(abi.DevicePtr)), "set bias");
        try self.run(x, w, d, workspace, workspace_len, stream);
    }

    /// Fuses `epilogue` into the chosen algorithm when it supports it within `workspace_limit`; else leaves it unfused.
    fn fuse(self: *Linear, epilogue: Epilogue, workspace_limit: usize) Error!void {
        const api = self.lt.api;
        const e: u32 = @backingInt(epilogue);
        try self.lt.check(api.cublasLtMatmulDescSetAttribute(self.desc, desc_epilogue, &e, @sizeOf(u32)), "set epilogue");
        var checked: Heuristic = undefined;
        if (api.cublasLtMatmulAlgoCheck(self.handle, self.desc, self.a, self.b, self.c, self.c, &self.algo, &checked) == 0 and checked.state == 0 and checked.workspace_size <= workspace_limit) {
            self.epilogue = epilogue;
            self.workspace_need = @max(self.workspace_need, checked.workspace_size);
            return;
        }
        const none: u32 = @backingInt(Epilogue.none);
        try self.lt.check(api.cublasLtMatmulDescSetAttribute(self.desc, desc_epilogue, &none, @sizeOf(u32)), "unset epilogue");
    }
};

comptime {
    std.debug.assert(@sizeOf(Algo) == 64 and @sizeOf(Heuristic) == 96);
    std.debug.assert(@sizeOf(abi.DevicePtr) == @sizeOf(?*const anyopaque));
    std.debug.assert(@sizeOf(DataType) == 4 and @sizeOf(c_int) == 4);
}
