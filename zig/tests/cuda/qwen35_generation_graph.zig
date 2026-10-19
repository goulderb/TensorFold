const std = @import("std");
const cuda = @import("cuda");
const qwen = @import("qwen35_cuda");

const Kind = enum { embed, kv, attention };
const Patch = struct {
    node: cuda.graph.Node,
    kind: Kind,
    params: cuda.abi.KernelNodeParams,
    args: cuda.Args,

    fn read(comptime T: type, params: cuda.abi.KernelNodeParams, index: usize) !T {
        const ptr = (params.params orelse return error.MissingGraphArguments)[index] orelse return error.MissingGraphArgument;
        const bytes: [*]const u8 = @ptrCast(ptr);
        var value: T = undefined;
        @memcpy(std.mem.asBytes(&value), bytes[0..@sizeOf(T)]);
        return value;
    }

    fn init(node: cuda.graph.Node, kind: Kind, params: cuda.abi.KernelNodeParams) !Patch {
        if (params.extra != null) return error.UnsupportedGraphArguments;
        var args: cuda.Args = .{};
        switch (kind) {
            .embed => {
                args.add(try read(u64, params, 0));
                args.add(try read(u32, params, 1));
                args.add(try read(u64, params, 2));
            },
            .kv => {
                for (0..5) |i| args.add(try read(u64, params, i));
                args.add(try read(i32, params, 5));
                args.add(try read(f32, params, 6));
            },
            .attention => {
                for (0..4) |i| args.add(try read(u64, params, i));
                args.add(try read(i32, params, 4));
                args.add(try read(u64, params, 5));
                args.add(try read(f32, params, 6));
            },
        }
        return .{ .node = node, .kind = kind, .params = params, .args = args };
    }

    fn update(self: *Patch, exec: cuda.graph.Exec, token: u32, position: usize) !void {
        const index: usize = switch (self.kind) {
            .embed => 1,
            .kv => 5,
            .attention => 4,
        };
        const at = self.args.offsets[index];
        std.mem.writeInt(u32, self.args.storage[at..][0..4], if (self.kind == .embed) token else @intCast(position), .little);
        var params = self.params;
        params.params = self.args.pointers();
        try exec.d.check(exec.d.api.cuGraphExecKernelNodeSetParams_v2(exec.handle, self.node, &params), "cuGraphExecKernelNodeSetParams");
    }
};

const Partition = struct {
    graph: cuda.graph.Graph,
    exec: cuda.graph.Exec,
    patches: []Patch,
    nodes: usize,

    fn init(a: std.mem.Allocator, session: *qwen.Session, rank: usize) !Partition {
        const p = &session.engine.partitions[rank];
        try p.context.makeCurrent();
        try cuda.graph.beginCapture(p.stream, .global);
        session.diagnosticPartition(rank, 0, 0, qwen.DiagnosticNoTap{}) catch |err| {
            if (cuda.graph.endCapture(p.stream)) |g| {
                var dead = g;
                dead.deinit();
            } else |_| {}
            return err;
        };
        var graph = try cuda.graph.endCapture(p.stream);
        errdefer graph.deinit();
        var exec = try graph.instantiate();
        errdefer exec.deinit();
        try exec.upload(p.stream);
        try p.stream.synchronize();
        const count = try graph.nodeCount();
        const nodes = try a.alloc(cuda.graph.Node, count);
        var patches: std.ArrayList(Patch) = .empty;
        var counts: [3]usize = @splat(0);
        for (try graph.nodes(nodes)) |node| {
            var params: cuda.abi.KernelNodeParams = undefined;
            try exec.d.check(exec.d.api.cuGraphKernelNodeGetParams_v2(node, &params), "cuGraphKernelNodeGetParams");
            const kind: Kind = if (params.func == p.ops.embed_fn.handle) .embed else if (params.func == p.ops.attention_kv_fn.handle) .kv else if (params.func == p.ops.attention_fn.handle) .attention else continue;
            counts[@backingInt(kind)] += 1;
            try patches.append(a, try Patch.init(node, kind, params));
        }
        var full: usize = 0;
        for (session.layers[p.first..p.end]) |layer| if (layer == .full) {
            full += 1;
        };
        if (counts[0] != @as(usize, if (rank == 0) 1 else 0) or counts[1] != full or counts[2] != full)
            return error.DynamicNodeCountMismatch;
        if ((try p.context.memInfo()).free < qwen.floor_bytes) return error.GpuMemoryFloor;
        return .{ .graph = graph, .exec = exec, .patches = try patches.toOwnedSlice(a), .nodes = count };
    }

    fn deinit(self: *Partition) void {
        self.exec.deinit();
        self.graph.deinit();
    }
};

pub const Replay = struct {
    session: *qwen.Session,
    partitions: [2]Partition,
    appends: usize = 0,
    launches: usize = 0,
    updates: usize = 0,

    pub fn init(a: std.mem.Allocator, session: *qwen.Session) !Replay {
        if (session.engine.partitions.len != 2) return error.UnsupportedTopology;
        var self: Replay = .{ .session = session, .partitions = undefined };
        var made: usize = 0;
        errdefer for (self.partitions[0..made], 0..) |*p, rank| {
            session.engine.partitions[rank].context.makeCurrent() catch {};
            p.deinit();
        };
        for (&self.partitions, 0..) |*p, rank| {
            p.* = try Partition.init(a, session, rank);
            made += 1;
        }
        return self;
    }

    pub fn deinit(self: *Replay) void {
        for (&self.partitions, 0..) |*p, rank| {
            self.session.engine.partitions[rank].context.makeCurrent() catch {};
            p.deinit();
        }
    }

    pub fn clearCounts(self: *Replay) void {
        self.appends = 0;
        self.launches = 0;
        self.updates = 0;
    }

    pub fn append(self: *Replay, tokens: []const u32) !void {
        return self.appendImpl(tokens, false);
    }

    pub fn appendFrozenPosition(self: *Replay, tokens: []const u32) !void {
        return self.appendImpl(tokens, true);
    }

    fn appendImpl(self: *Replay, tokens: []const u32, comptime frozen: bool) !void {
        const session = self.session;
        const e = session.engine;
        if (session.poisoned) return error.PoisonedSession;
        if (tokens.len > e.capacity - session.position) return error.ContextFull;
        for (tokens) |id| if (id >= 248320) return error.InvalidToken;
        if (tokens.len == 0) return;
        errdefer session.poisoned = true;
        for (tokens) |token| {
            for (e.partitions, &self.partitions, 0..) |*p, *replay, rank| {
                try p.context.makeCurrent();
                for (replay.patches) |*patch| {
                    try patch.update(replay.exec, token, if (frozen) 0 else session.position);
                    self.updates += 1;
                }
                if (rank != 0) try p.scratch.upload(@intCast(p.b.residual - p.scratch.ptr), e.transfer.slice(u8));
                try replay.exec.launchOn(p.stream);
                self.launches += 1;
                try p.stream.synchronize();
                if (rank + 1 < e.partitions.len) try p.scratch.download(@intCast(p.b.residual - p.scratch.ptr), e.transfer.slice(u8));
                if ((try p.context.memInfo()).free < qwen.floor_bytes) return error.GpuMemoryFloor;
            }
            session.position += 1;
            self.appends += 1;
        }
        const last = &e.partitions[e.partitions.len - 1];
        try last.scratch.download(@intCast(last.b.logits - last.scratch.ptr), std.mem.sliceAsBytes(session.logits));
    }
};
