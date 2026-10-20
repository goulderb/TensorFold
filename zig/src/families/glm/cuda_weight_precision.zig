const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const Dense = enum { bf16, q4, fp8, nvfp4, fp8_native, nvfp4_native, source };
pub const Head = enum { bf16, fp8 };
pub const Group = enum { kda, dsa, mlp, shared, mtp_eh };
pub const groups = std.meta.tags(Group);

pub const DensePolicy = struct {
    values: [groups.len]Dense = @splat(.bf16),

    pub fn uniform(value: Dense) DensePolicy {
        return .{ .values = @splat(value) };
    }
    pub fn get(p: DensePolicy, group: Group) Dense {
        return p.values[@backingInt(group)];
    }

    pub fn native(p: DensePolicy) bool {
        for (p.values) |value| if (value == .fp8_native or value == .nvfp4_native) return true;
        return false;
    }

    pub fn source(p: DensePolicy) bool {
        for (p.values) |value| if (value == .source) return true;
        return false;
    }

    pub fn read(getEnv: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !DensePolicy {
        const default = parseDense(getEnv("TF_GLM_DENSE")) catch {
            problem.* = "TF_GLM_DENSE: bf16, q4, fp8, nvfp4, fp8_native, nvfp4_native or source";
            return error.BadSetting;
        };
        var result = uniform(default);
        inline for (groups) |group| {
            const name = switch (group) {
                .kda => "TF_GLM_DENSE_KDA",
                .dsa => "TF_GLM_DENSE_DSA",
                .mlp => "TF_GLM_DENSE_MLP",
                .shared => "TF_GLM_DENSE_SHARED",
                .mtp_eh => "TF_GLM_DENSE_MTP_EH",
            };
            if (getEnv(name)) |value| {
                result.values[@backingInt(group)] = std.meta.stringToEnum(Dense, value) orelse {
                    problem.* = name ++ ": bf16, q4, fp8, nvfp4, fp8_native, nvfp4_native or source";
                    return error.BadSetting;
                };
            }
        }
        return result;
    }

    pub fn hash(p: DensePolicy, h: *Sha256) void {
        h.update("glm-dense-conversion-v1\x00");
        inline for (groups) |group| {
            h.update(@tagName(group));
            h.update("=");
            h.update(@tagName(p.get(group)));
            h.update("\x00");
        }
        if (p.native()) h.update(@import("cuda").ck.numerical_identity);
        if (p.get(.dsa) == .q4) h.update("glm-dsa-combined-q4-v1\x00");
        if (p.get(.mlp) == .q4) h.update("glm-mlp-combined-q4-v1\x00");
    }

    pub fn digest(p: DensePolicy) [32]u8 {
        var h = Sha256.init(.{});
        p.hash(&h);
        return h.finalResult();
    }

    pub fn format(p: DensePolicy, out: []u8) ![]const u8 {
        var at: usize = 0;
        inline for (groups, 0..) |group, i| {
            const field = try std.fmt.bufPrint(out[at..], "{s}{s}={s}", .{ if (i == 0) "" else ",", @tagName(group), @tagName(p.get(group)) });
            at += field.len;
        }
        return out[0..at];
    }
};

pub const Settings = struct { dense: DensePolicy = .{}, head: Head = .bf16 };

pub fn parseDense(value: ?[]const u8) !Dense {
    const v = value orelse return .bf16;
    if (v.len == 0) return .bf16;
    return std.meta.stringToEnum(Dense, v) orelse error.InvalidDensePrecision;
}

pub fn parseHead(value: ?[]const u8) !Head {
    const v = value orelse return .bf16;
    if (v.len == 0) return .bf16;
    return std.meta.stringToEnum(Head, v) orelse error.InvalidHeadPrecision;
}

test "precision is explicit and defaults to checkpoint BF16" {
    try std.testing.expectEqual(Dense.bf16, try parseDense(null));
    try std.testing.expectEqual(Head.bf16, try parseHead(""));
    try std.testing.expectEqual(Dense.q4, try parseDense("q4"));
    try std.testing.expectEqual(Head.fp8, try parseHead("fp8"));
    try std.testing.expectEqual(Dense.fp8, try parseDense("fp8"));
    try std.testing.expectEqual(Dense.nvfp4, try parseDense("nvfp4"));
    try std.testing.expectError(error.InvalidDensePrecision, parseDense("int8"));
    try std.testing.expectError(error.InvalidHeadPrecision, parseHead("q4"));
    try std.testing.expectEqual(Dense.fp8_native, try parseDense("fp8_native"));
    try std.testing.expectEqual(Dense.nvfp4_native, try parseDense("nvfp4_native"));
    try std.testing.expectEqual(Dense.source, try parseDense("source"));
    try std.testing.expect(!DensePolicy.uniform(.fp8).native());
    try std.testing.expect(DensePolicy.uniform(.source).source());
    try std.testing.expect(!DensePolicy.uniform(.source).native());
    try std.testing.expect(DensePolicy.uniform(.fp8_native).native());
    try std.testing.expect(!std.mem.eql(u8, &DensePolicy.uniform(.fp8).digest(), &DensePolicy.uniform(.fp8_native).digest()));
}

test "global and family settings resolve to one effective policy" {
    const E = struct {
        fn empty(_: [:0]const u8) ?[]const u8 {
            return null;
        }
        fn mixed(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_DENSE")) return "fp8";
            if (std.mem.eql(u8, name, "TF_GLM_DENSE_DSA")) return "bf16";
            return null;
        }
        fn bad(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_DENSE_DSA")) "" else null;
        }
    };
    var problem: []const u8 = "";
    const base = try DensePolicy.read(E.empty, &problem);
    for (groups) |group| try std.testing.expectEqual(Dense.bf16, base.get(group));
    const mixed = try DensePolicy.read(E.mixed, &problem);
    try std.testing.expectEqual(Dense.bf16, mixed.get(.dsa));
    try std.testing.expectEqual(Dense.fp8, mixed.get(.kda));
    try std.testing.expectError(error.BadSetting, DensePolicy.read(E.bad, &problem));
    try std.testing.expectEqualStrings("TF_GLM_DENSE_DSA: bf16, q4, fp8, nvfp4, fp8_native, nvfp4_native or source", problem);
    const Redundant = struct {
        fn get(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_DENSE")) return "fp8";
            inline for (.{ "TF_GLM_DENSE_KDA", "TF_GLM_DENSE_DSA", "TF_GLM_DENSE_MLP", "TF_GLM_DENSE_SHARED", "TF_GLM_DENSE_MTP_EH" }) |key| {
                if (std.mem.eql(u8, name, key)) return "bf16";
            }
            return null;
        }
    };
    const equal = try DensePolicy.read(Redundant.get, &problem);
    try std.testing.expectEqualSlices(u8, &base.digest(), &equal.digest());
    var formatted: [128]u8 = undefined;
    try std.testing.expectEqualStrings("kda=bf16,dsa=bf16,mlp=bf16,shared=bf16,mtp_eh=bf16", try equal.format(&formatted));
    try std.testing.expect(!std.mem.eql(u8, &base.digest(), &mixed.digest()));
}
