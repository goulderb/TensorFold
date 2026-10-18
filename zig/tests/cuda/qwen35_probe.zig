const std = @import("std");
const qwen = @import("qwen35_cuda");
const core = @import("core");

const Input = struct {
    prompt: []const u32,
    forced: []const u32 = &.{},
    max_tokens: usize = 8,
    context: usize = 4096,
    devices: []const c_int = &.{ 0, 1 },
    cut: usize = 20,
};

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len == 3 and std.mem.eql(u8, args[1], "--inspect")) {
        const cfg = try qwen.Config.read(init.gpa, init.io, args[2]);
        var ck = try core.Checkpoint.openModel(init.gpa, init.io, args[2]);
        defer ck.close();
        try qwen.checkpoint.validateText(&ck, cfg, 0, 40, true, true);
        try qwen.checkpoint.refuseUnusedText(&ck, 0, 40, true, true);
        try std.Io.File.stdout().writeStreamingAll(init.io, "Qwen35 ModelOpt text checkpoint validated on CPU; no CUDA context created.\n");
        return 0;
    }
    if (args.len != 4) {
        try std.Io.File.stderr().writeStreamingAll(init.io, "usage: tf-qwen35-probe --inspect MODEL\n       tf-qwen35-probe MODEL INPUT.json OUTPUT_DIR\n");
        return 2;
    }
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], a, .limited(4 << 20));
    const input = try std.json.parseFromSliceLeaky(Input, a, bytes, .{});
    if (input.prompt.len == 0 or input.context == 0 or input.max_tokens > input.context or
        input.forced.len > input.max_tokens or input.prompt.len > input.context - input.max_tokens)
        return error.InvalidProbeInput;
    const model = try qwen.Engine.open(init.gpa, init.io, args[1], .{ .devices = input.devices, .context = input.context, .cut = input.cut });
    defer model.deinit();
    var main_session = try model.createSession();
    defer main_session.deinit();
    var peer = try model.createSession();
    defer peer.deinit();
    try std.Io.Dir.cwd().createDirPath(init.io, args[3]);
    const logits = try a.alloc(f32, 248320);
    const peer_logits = try a.alloc(f32, 248320);
    try main_session.append(input.prompt);
    try main_session.copyLogits(logits);
    try peer.append(input.prompt);
    try peer.copyLogits(peer_logits);
    if (!std.mem.eql(u8, std.mem.sliceAsBytes(logits), std.mem.sliceAsBytes(peer_logits))) return error.SessionIsolationMismatch;
    try main_session.copyLogits(peer_logits);
    if (!std.mem.eql(u8, std.mem.sliceAsBytes(logits), std.mem.sliceAsBytes(peer_logits))) return error.SessionLogitsAliased;
    const tokens = try a.alloc(u32, input.max_tokens);
    for (tokens, 0..) |*token, row| {
        try main_session.copyLogits(logits);
        try finite(logits);
        const filename = try std.fmt.allocPrint(a, "{s}/row-{d:0>4}.f32", .{ args[3], row });
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = filename, .data = std.mem.sliceAsBytes(logits) });
        token.* = if (row < input.forced.len) input.forced[row] else argmax(logits);
        try main_session.append(&.{token.*});
        if (row == 0) {
            try peer.append(&.{token.*});
            try main_session.copyLogits(logits);
            try peer.copyLogits(peer_logits);
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(logits), std.mem.sliceAsBytes(peer_logits))) return error.InterleavedStateMismatch;
        }
    }
    try main_session.reset();
    try main_session.append(input.prompt);
    try main_session.copyLogits(logits);
    try peer.reset();
    try peer.append(input.prompt);
    try peer.copyLogits(peer_logits);
    if (!std.mem.eql(u8, std.mem.sliceAsBytes(logits), std.mem.sliceAsBytes(peer_logits))) return error.ResetMismatch;
    var free_mib: [2]usize = @splat(0);
    for (model.partitions, 0..) |*part, i| free_mib[i] = (try part.context.memInfo()).free / (1 << 20);
    const config_sha = try fileHash(a, init.io, args[1], "config.json");
    const index_sha = try fileHash(a, init.io, args[1], "model.safetensors.index.json");
    const report = .{ .prompt = input.prompt, .forced = input.forced, .tokens = tokens, .vocab = 248320, .context = input.context, .devices = input.devices, .cut = input.cut, .free_mib = free_mib[0..model.partitions.len], .config_sha256 = config_sha[0..], .index_sha256 = index_sha[0..], .session_isolation = true, .reset = true, .teacher_parity = "not checked by this executable" };
    var out: std.Io.Writer.Allocating = .init(a);
    try std.json.Stringify.value(report, .{ .whitespace = .indent_2 }, &out.writer);
    try out.writer.writeByte('\n');
    const manifest = try std.fs.path.join(a, &.{ args[3], "result.json" });
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = manifest, .data = out.written() });
    try std.Io.File.stdout().writeStreamingAll(init.io, out.written());
    return 0;
}

fn fileHash(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![64]u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20));
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

fn finite(logits: []const f32) !void {
    for (logits) |value| if (!std.math.isFinite(value)) return error.NonfiniteLogits;
}

fn argmax(logits: []const f32) u32 {
    var best: usize = 0;
    for (logits, 0..) |value, i| if (value > logits[best]) {
        best = i;
    };
    return @intCast(best);
}
