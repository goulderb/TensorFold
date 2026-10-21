//! Pinned runtime arithmetic identity from kernels/dflash2-aot/provenance.json.
const std = @import("std");
const File = struct { name: []const u8, sha256: []const u8 };
const pinned = [_]File{
    .{ .name = "aot.json", .sha256 = "191a4e3145e5da0a9609a66397b7c036f301a8a7e32baca6822740e317ef5e51" },
    .{ .name = "cubins/076465dc4c83edb8834accb20cdd5576ba9f7fbdcfa2862bce8cb8965c570bb6.cubin", .sha256 = "64a1a37997cf6fd46ed5f59d04ee2bec4b6ac7a1379026ca9338ff3cfb2507d7" },
    .{ .name = "cubins/08b04587a4e37c8ec012ab98a2074e564d5cdc918ac2928bbbd78501761575d0.cubin", .sha256 = "e682f5d22853a7a5d18c474a4ab76df6e67b8e9b8c5a847240969de1de0a5a04" },
    .{ .name = "cubins/0a3625fcc3d25304e782284d25e311e2d2f187a87be1678770fb8c4d4b1b0715.cubin", .sha256 = "80477e078d06974f2222e0080dba9b5d49171d433c5089b0043ab74485857ff6" },
    .{ .name = "cubins/12e1f02393b3d517b72a1aedbdcae62e09921e71b9a2b66c285c98a081513e53.cubin", .sha256 = "505cd8b19263ac9805a4f4b10dfcc59c154f164640f9a22498ff6b49c3abd60f" },
    .{ .name = "cubins/1354789dda340695148448edfea9d31a88fca85beace31de0228d62f3826056e.cubin", .sha256 = "79231e44b121bfbcb942f2c01fa3edf0005b8a3e6b4d5ba7f71f8a31ddaaf56a" },
    .{ .name = "cubins/14e582a8230e78e0bc11992687f511e77c13782014c6f7eb4b54142065002438.cubin", .sha256 = "ff39928af9a20d5c59a7da2667f20541471fdb56d207303128e478c13a47f1f0" },
    .{ .name = "cubins/392d3c3fb4b084240d51121d729b0d6ffab7b784edca6899a6a8c565e6e1a383.cubin", .sha256 = "4506feb100984e62431de6d94340a946f83562e7918380f50632f8ab18fe4e31" },
    .{ .name = "cubins/43d676f97a1432cdbf2693b84dd4ddf10f865bb3ea6c8bd0831359187f1c072e.cubin", .sha256 = "3ebe8d2898628767bbc9838bc60186a3e9e61b51ece69f99b05ef201ba9222ec" },
    .{ .name = "cubins/4501bc97eab5fe280ad0a95e66861d41da1e1444c0d85eab9f7ea0bd0373d412.cubin", .sha256 = "b09832a05030f726c2f702fb106007129b54cb9aa28d7104c0e26bf6349f32d1" },
    .{ .name = "cubins/5cd90b80b6723a2eeff7747d27d9969a1e96ac40cc5323dd4c18c31cdc989199.cubin", .sha256 = "5c3cd8d341e7f8b43dffade9db3830f4939caffbfbebab7a0cef02e4be24de25" },
    .{ .name = "cubins/6276feeb834335b74942f61bad16c895c881e7036dbe6c3a9c3f1076d30cbd84.cubin", .sha256 = "22288e2b611d15bfce92c801bc2b83fdc5634b4d44d92102085852f38ba6c38f" },
    .{ .name = "cubins/62dd24bcf35428e828d38eac270b11ac74a2980962088e52dd2d61d090c724c9.cubin", .sha256 = "2258c07a3e4438f852021ca7b70233d5e672e16f81fbaf5300eaa87215e13bef" },
    .{ .name = "cubins/6c10dc68271c091c29c5544c28c2432e708552662261defeff8797ba4538da91.cubin", .sha256 = "3f2fac0094caed58e3b979565d68fa39c9e92dcd74a23380c4cfef450ff3def1" },
    .{ .name = "cubins/77fc73d57fbbf33695e979ad31d7c414f3b237472ceb3705ece64fa9a0b02ec4.cubin", .sha256 = "89f3973a13312ad7fb055bbe1a9c38ac8388c7c1e5ed2875bae0c18e5a4200a0" },
    .{ .name = "cubins/7d3cdc8e1cb17a6824a84a230c0e12ad89984af8910f238b75a7a613a3ee83d6.cubin", .sha256 = "505cd8b19263ac9805a4f4b10dfcc59c154f164640f9a22498ff6b49c3abd60f" },
    .{ .name = "cubins/8dcb5f08f1162759dee17398f6f93a6729c450d17b0a0ee1099e40b2a3b9eadc.cubin", .sha256 = "7b432a0fadfd9b93588ed2defc1fd7bd68c2fb6ede6e26e84e7ce750bb0d63bb" },
    .{ .name = "cubins/917b8fd568079cc8ab3300feea1bd3a33fa56d9212598505fd69c1841956a3c9.cubin", .sha256 = "2f8db85afe09f06425a56432638ba76e48371828a3a42dd54511f599952d1209" },
    .{ .name = "cubins/960246046efe54dce67bca329f7d8ecab96c719b61b7e6478d9d415463d61d4d.cubin", .sha256 = "62680823b164cb21b316d2d6128df5ea1ad00940325d6b01ab40ba65db3ef476" },
    .{ .name = "cubins/9836036c40e1e88bc10eda36a75df740ff636410921ee809edfa798aed3288a5.cubin", .sha256 = "2e777cefa8ce92265b3904f8883d0c08c8fb2e57172a53e7534625ee982b1767" },
    .{ .name = "cubins/a1e02f512f8ba725366024ba1c2e61f32b3a612a5371dff7df8af47b2872b08f.cubin", .sha256 = "b83083767f87e908d4626645b8c3302c338a15971bf7f61b49e77660023ba347" },
    .{ .name = "cubins/b8997b95bd2893357669a89d641936b7a19b09b42a85ca927c68a7ff918c3e04.cubin", .sha256 = "d642e0e1f435d493a9c9b4b18a85e98f1e26657697ba3ff02b8682c2ff6b6788" },
    .{ .name = "cubins/c46158e9ca06ae269d40516d6a6110b322a1384175e7d25adbe956dc06067359.cubin", .sha256 = "80477e078d06974f2222e0080dba9b5d49171d433c5089b0043ab74485857ff6" },
    .{ .name = "cubins/c8dc76d671c0a16e4b07c2700a831160d6ba2ddd44a78e055a451b1806758be3.cubin", .sha256 = "043756dab360f3322c71d90026c8e4dd52f6976d30d386034f07e2b5aa88e109" },
    .{ .name = "cubins/f6a439d0582a7e6dd073c60802c64d6ecf0e4c4daf390d01e7629f92a9567104.cubin", .sha256 = "7b432a0fadfd9b93588ed2defc1fd7bd68c2fb6ede6e26e84e7ce750bb0d63bb" },
};
pub fn directory(gpa: std.mem.Allocator, io: std.Io, explicit: ?[]const u8, draft_dir: []const u8) ![]u8 {
    if (explicit) |dir| return gpa.dupe(u8, dir);
    const exe = try std.process.executableDirPathAlloc(io, gpa);
    defer gpa.free(exe);
    const packaged = try std.fs.path.join(gpa, &.{ exe, "..", "share", "tensorfold", "dflash2", "sm121" });
    defer gpa.free(packaged);
    const manifest = try std.fs.path.join(gpa, &.{ packaged, "aot.json" });
    defer gpa.free(manifest);
    std.Io.Dir.cwd().access(io, manifest, .{}) catch |err| {
        if (err != error.FileNotFound) return err;
        return std.fs.path.join(gpa, &.{ draft_dir, "cuda-aot" });
    };
    return gpa.dupe(u8, packaged);
}
pub fn validate(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    for (pinned) |file| {
        const path = try std.fs.path.join(gpa, &.{ dir, file.name });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
        defer gpa.free(bytes);
        try verify(bytes, file.sha256);
    }
}
fn verify(bytes: []const u8, hex: []const u8) !void {
    var expected: [32]u8 = undefined;
    _ = try std.fmt.hexToBytes(&expected, hex);
    var actual: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &actual, .{});
    if (!std.mem.eql(u8, &expected, &actual)) return error.DraftKernelIdentityMismatch;
}
test "runtime kernel bytes are checked independently of a Triton cache key" {
    const abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    try verify("abc", abc);
    try std.testing.expectError(error.DraftKernelIdentityMismatch, verify("abd", abc));
    try std.testing.expectError(error.DraftKernelIdentityMismatch, verify("abc ", abc));
    try std.testing.expectError(error.DraftKernelIdentityMismatch, verify("{}", pinned[0].sha256));
    try std.testing.expectEqual(@as(usize, 25), pinned.len);
    try std.testing.expectEqualStrings("aot.json", pinned[0].name);
}
const shipped = "zig/kernels/dflash2-aot";
fn stage(gpa: std.mem.Allocator, io: std.Io, tmp: std.testing.TmpDir) !void {
    try tmp.dir.createDirPath(io, "cubins");
    for (pinned) |file| {
        const path = try std.fs.path.join(gpa, &.{ shipped, file.name });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
        defer gpa.free(bytes);
        try tmp.dir.writeFile(io, .{ .sub_path = file.name, .data = bytes });
    }
}
test "the dflash2-aot directory shipped from source satisfies every pin" {
    // A text-converting checkout or archive (core.autocrlf) rewrites aot.json and is refused at engine start.
    try validate(std.testing.allocator, std.testing.io, shipped);
}
test "a changed aot.json or cubin in a staged copy is refused" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try stage(gpa, io, tmp);
    var path_buf: [64]u8 = undefined;
    const dir = try std.fmt.bufPrint(&path_buf, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    try validate(gpa, io, dir);

    const manifest = try tmp.dir.readFileAlloc(io, "aot.json", gpa, .limited(1 << 26));
    defer gpa.free(manifest);
    const crlf = try std.mem.replaceOwned(u8, gpa, manifest, "\n", "\r\n");
    defer gpa.free(crlf);
    try std.testing.expect(crlf.len > manifest.len);
    try tmp.dir.writeFile(io, .{ .sub_path = "aot.json", .data = crlf });
    try std.testing.expectError(error.DraftKernelIdentityMismatch, validate(gpa, io, dir));
    manifest[manifest.len - 2] ^= 1;
    try tmp.dir.writeFile(io, .{ .sub_path = "aot.json", .data = manifest });
    try std.testing.expectError(error.DraftKernelIdentityMismatch, validate(gpa, io, dir));
    manifest[manifest.len - 2] ^= 1;
    try tmp.dir.writeFile(io, .{ .sub_path = "aot.json", .data = manifest });
    try validate(gpa, io, dir);

    const cubin = try tmp.dir.readFileAlloc(io, pinned[1].name, gpa, .limited(1 << 26));
    defer gpa.free(cubin);
    cubin[cubin.len / 2] ^= 1;
    try tmp.dir.writeFile(io, .{ .sub_path = pinned[1].name, .data = cubin });
    try std.testing.expectError(error.DraftKernelIdentityMismatch, validate(gpa, io, dir));
}
