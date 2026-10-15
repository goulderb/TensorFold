const std = @import("std");
const pre = @import("cuda_vision_preprocess.zig");
const resize = @import("cuda_vision_resize.zig");
const testing = std.testing;
const Allocator = std.mem.Allocator;

const Kind = enum { ramp, checker, impulse, constant };
fn pattern(a: Allocator, shape: pre.Shape, kind: Kind) ![]u8 {
    const bytes = try a.alloc(u8, @as(usize, shape.height) * shape.width * 3);
    for (0..shape.height) |y| for (0..shape.width) |x| {
        const i = (y * shape.width + x) * 3;
        switch (kind) {
            .ramp => {
                bytes[i] = @intCast((y * 7 + x * 3) % 256);
                bytes[i + 1] = @intCast((y * 2 + x * 11) % 256);
                bytes[i + 2] = @intCast((y + x) % 256);
            },
            .checker => {
                bytes[i] = @intCast(((y + x) % 2) * 255);
                bytes[i + 1] = @intCast(((y / 2 + x / 3) % 2) * 255);
                bytes[i + 2] = @intCast(((y * 13 + x * 17) % 2) * 255);
            },
            .constant => @memcpy(bytes[i..][0..3], &[_]u8{ 0, 127, 255 }),
            .impulse => @memset(bytes[i..][0..3], 0),
        }
    };
    if (kind == .impulse) {
        @memcpy(bytes[0..3], &[_]u8{ 255, 17, 199 });
        @memcpy(bytes[bytes.len - 3 ..], &[_]u8{ 73, 255, 5 });
        const mid = (@as(usize, shape.height / 2) * shape.width + shape.width / 2) * 3;
        @memset(bytes[mid..][0..3], 255);
    }
    return bytes;
}
fn hashEqual(expected: []const u8, bytes: []const u8) !void {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    try testing.expectEqualStrings(expected, &std.fmt.bytesToHex(digest, .lower));
}

// Complete HWC uint8 outputs from Torch 2.9.0 git 0fabc3ba, CPU-only, one thread.
// torchvision v0.24.0 resize_image's direct uint8 interpolate branch was inspected.
test "bicubic antialias complete-byte oracle covers noninteger scales, edges, clipping and degenerate axes" {
    const cases = [_]struct { input: [2]u32, output: [2]u32, kind: Kind, hash: []const u8 }{
        .{ .input = .{ 5, 7 }, .output = .{ 9, 11 }, .kind = .ramp, .hash = "5bdfbcb68ada75172798b0ebeea9d81052898ba662fa0018daef32f34990d8d8" },
        .{ .input = .{ 13, 17 }, .output = .{ 107, 140 }, .kind = .ramp, .hash = "d400a419f49fff2fb27fb358c03c8366ce2d8c987d23756c409b613571ba5dae" },
        .{ .input = .{ 31, 47 }, .output = .{ 19, 23 }, .kind = .ramp, .hash = "f79ad109559f5b193e4753a58c35b7c361cf202ef2d98a1c17c5e15f4933fea2" },
        .{ .input = .{ 31, 47 }, .output = .{ 19, 73 }, .kind = .ramp, .hash = "e9efe7ef7cda8238e4d4e20c3cf3dfa173f694aff5f0362ef9fc0d2fdf35e7a9" },
        .{ .input = .{ 31, 47 }, .output = .{ 53, 23 }, .kind = .ramp, .hash = "811983b695b1b8e4c86dc7fa7fbffce005ba830cd43ae3ff21cc0b845142a7ec" },
        .{ .input = .{ 28, 56 }, .output = .{ 17, 31 }, .kind = .ramp, .hash = "ac88849fd39ba2a4138b6597a43ca6992c3c809999752149dfe6f604ffefe216" },
        .{ .input = .{ 1, 1 }, .output = .{ 13, 17 }, .kind = .constant, .hash = "df0984521461e92467cb9613893543b2960210c5205e4312fbed806ec1623432" },
        .{ .input = .{ 1, 97 }, .output = .{ 9, 13 }, .kind = .ramp, .hash = "612ccd4159afb1093488ec78fff5d9b2dbb0d5c6870381720de13f1ebb9f009e" },
        .{ .input = .{ 97, 1 }, .output = .{ 13, 9 }, .kind = .ramp, .hash = "3a35b70fef5bbb3b26e2446361f883d248c0a671d9de0d485e12c3db6a663b43" },
        .{ .input = .{ 19, 23 }, .output = .{ 19, 23 }, .kind = .ramp, .hash = "51b77ef1d8caafd2acd39d18de0b7b0d73f64dd93416f660d3fbd3e95704e42e" },
        .{ .input = .{ 19, 23 }, .output = .{ 19, 7 }, .kind = .ramp, .hash = "d4c130417b4a3dda4d16f969f8e869f681b1aa826cde704d61e24048a1cb7b5b" },
        .{ .input = .{ 19, 23 }, .output = .{ 7, 23 }, .kind = .ramp, .hash = "ef5c87e37c680bb56099974f8c35f323c1b2b81f93b35df105e1a5f2f74d11a2" },
        .{ .input = .{ 3, 5 }, .output = .{ 1, 1 }, .kind = .impulse, .hash = "5b7409d8a9bcd9963a75c17de2e7238cc8fa9ad4fc86d645a038f62339a4ed33" },
        .{ .input = .{ 5, 3 }, .output = .{ 11, 7 }, .kind = .impulse, .hash = "00d467b19ec5244175a1bc241adaf796b0e6d3f7d91301d1b620316583d0d07c" },
        .{ .input = .{ 7, 9 }, .output = .{ 4, 3 }, .kind = .checker, .hash = "f8a2fecc6ef895573425bff703fc003059d4362cbc8a667019391b7c3ecc5dcb" },
        .{ .input = .{ 7, 9 }, .output = .{ 19, 23 }, .kind = .checker, .hash = "6a75a1d740fe0c85f06ecd9ffa0de763e6b05cf4947682e1ee613b3a388b642d" },
        .{ .input = .{ 33, 35 }, .output = .{ 11, 13 }, .kind = .constant, .hash = "b00df498f98d6461d9fd86cfb8c8d778712eaa6cae16f1e6e03caaf388e7853e" },
        .{ .input = .{ 2, 2 }, .output = .{ 29, 31 }, .kind = .checker, .hash = "0d2ecc7d88261754a153e43f4e80f680e0ed4cced67adfa5c9d9117873e4346f" },
        .{ .input = .{ 17, 19 }, .output = .{ 1, 11 }, .kind = .ramp, .hash = "43c19ee0a3a404a4145699581282f60bd2c2e39b659c818cc26c4846339bad7b" },
        .{ .input = .{ 17, 19 }, .output = .{ 13, 1 }, .kind = .ramp, .hash = "adb67161f5aaf768956b0051b0c3ca2cb392cd8419f8c623918e3e17838c37d7" },
        .{ .input = .{ 1, 8192 }, .output = .{ 1, 7 }, .kind = .ramp, .hash = "87229e9e289f757fa9c6ab64c3b3552657db845bcbca2b8994cf64e87241c1d4" },
        .{ .input = .{ 8192, 1 }, .output = .{ 7, 1 }, .kind = .ramp, .hash = "c32c25ad0c8e092a6da0b81833acdadf1626bfd73300eae43e4d5743fcc5714b" },
        .{ .input = .{ 1080, 1920 }, .output = .{ 945, 1680 }, .kind = .ramp, .hash = "6cb3ead6ef4d8e6470c567f6847790dcad123f7e683206f01ebefbf6a2e13d32" },
        .{ .input = .{ 1080, 1920 }, .output = .{ 251, 447 }, .kind = .checker, .hash = "4fb290420419a24c7eed450cbe701ad0e68beaa8757b094076bee66f1dbaaf27" },
        .{ .input = .{ 4, 512 }, .output = .{ 4, 1 }, .kind = .impulse, .hash = "3d9039569b37dd0d6d0c65a54690da213a7a373d7239300b4100a60e80676a12" },
    };
    for (cases) |cs| {
        const a = testing.allocator;
        const shape: pre.Shape = .{ .height = cs.input[0], .width = cs.input[1] };
        const bytes = try pattern(a, shape, cs.kind);
        defer a.free(bytes);
        var out = try resize.resize(a, .{ .shape = shape, .bytes = bytes }, .{ .height = cs.output[0], .width = cs.output[1] });
        defer out.deinit(a);
        try testing.expectEqual(@as(usize, cs.output[0]) * cs.output[1] * 3, out.bytes.len);
        try hashEqual(cs.hash, out.bytes);
    }
}

test "384 complete resize outputs match the CPU oracle aggregate" {
    const a = testing.allocator;
    var digest = std.crypto.hash.sha2.Sha256.init(.{});
    for (0..384) |n| {
        const shape: pre.Shape = .{ .height = @intCast(1 + (n * 37) % 67), .width = @intCast(1 + (n * 53) % 71) };
        const target: pre.Shape = .{ .height = @intCast(1 + (n * 31) % 79), .width = @intCast(1 + (n * 47) % 83) };
        const kinds = [_]Kind{ .ramp, .checker, .impulse, .constant };
        const bytes = try pattern(a, shape, kinds[n % 4]);
        defer a.free(bytes);
        var out = try resize.resize(a, .{ .shape = shape, .bytes = bytes }, target);
        defer out.deinit(a);
        digest.update(out.bytes);
    }
    const final = digest.finalResult();
    try testing.expectEqualStrings("2e60523720eb74e290c18116401c2158ecfee547a4259e5093187dfb040f9bb5", &std.fmt.bytesToHex(final, .lower));
}

test "small complete RGB bytes expose intermediate uint8 rounding and clipping" {
    const rgb = [_]u8{ 255, 0, 17, 0, 255, 127, 7, 99, 255, 201, 81, 3 };
    const golden = "ff0000fc0c0a82834406f87f00ff93803383813a83746d6567a0466ea745006cff0667fc665785c7480dea4300";
    var expected: [golden.len / 2]u8 = undefined;
    _ = try std.fmt.hexToBytes(&expected, golden);
    var out = try resize.resize(testing.allocator, .{ .shape = .{ .height = 2, .width = 2 }, .bytes = &rgb }, .{ .height = 3, .width = 5 });
    defer out.deinit(testing.allocator);
    try testing.expectEqualSlices(u8, &expected, out.bytes);
}

// Actual PNG decoded offline by Pillow, then cropped without resampling.
// assets/tensorfold-hero.png SHA256 5ed0d5370027575cdcbfb927af2856d06de4278f62557b615fbaa28c8d24c161
// Crop box (2064,320,2112,352); embedded canonical RGB avoids any fixture skip/codec dependency.
const real_rgb =
    "adaeca5f669323266303064e00034e000c56020f5a06145d000047444b82dcdfedfffffdfaf8f5fbfbfbf9f9faf9f9fbf9f9fbfbfafbfdfcfafefdfafefcf9fbfaf9f7f7f9f5f5f8" ++
    "f4f5f9f6f6f9f4f4f4ffffffaab0c8080f50060a4b0a155309124e08124f061051030e53000c54000b55000b56000b56000b56000c56000c56000b56000c56000c56000c57000c57" ++
    "ffffffffffffdbd6e27375a025286201055000085709176200014c494b83e1e0ecfffffdfaf9f6fbfbfaf9f9faf9f9fbf9f9fbfbfbfbfdfbf9fefdf9fefcf9fbfaf9f7f7f8f5f5f8" ++
    "f4f5f9f6f6f8f5f4f3ffffffabb0c8091050060a4c0a155409124f08114f060f51030d54000c55000a55000a56000b56000b55000b56000c56000c56000c56000c56000c57000c57" ++
    "f5f4f3f4f4f2fcfdfdffffffdbdae55f62940f125b07125e00054f555486e7e5eefffffefcfaf7fdfcfcfafafbf9f9fbf9f9fbfbfbfafdfcf8fefcf8fdfcf8fbfaf7f8f7f7f5f5f8" ++
    "f4f5f9f6f7f9f5f5f4ffffffabb1c8091050060a4c0a155409124f071150050f52030d54000c54000a56000a56000b55000b56000c56000c56000c56000c56010d56010d57010d57" ++
    "fcfcfafafbf9f7f6f7f3f3f6fcfffeffffffa0a2bf2b327000004b585b8be8e7effffffffdfcf9fefdfcfbfbfbf9f9fbf9f9fafbfaf9fdfbf7fefcf6fdfbf6fbf9f6f8f7f6f5f5f7" ++
    "f4f5f9f6f7f9f5f5f4ffffffabb1c8091050060a4c0a155409124f071250051052030d53000c54000b56000b56000b55000b56010c56010c56010c56010c56010c56010d57010d57" ++
    "fbfbfaf9f9faf8f8fcf9f9fbf9f7f6f6f6f6ffffffcaccd82c306554537fe6e6eefffffffdfcf8fefdfbfbfbfcfafafcfafafafcfbf8fefbf6fefbf4fdfbf4fbf9f4f8f7f6f5f5f8" ++
    "f4f5f9f6f6f9f5f4f4ffffffabb0c9080f51060a4b0a155509124f081250061052030d53000c54000b55000b56000b56000b56010b56010c56010c56010c56010d57000d57000d57" ++
    "fcfcfbfbfbfbfbfbfcfcfcfdfefefefffefaf8f8f5ffffffc8c6d18a86a0e0dee5fffffffdfcf8fefefbfcfcf9fbfbf9fbfbf9fdfbf7fffcf5fffcf4fdfbf4fbf9f4f8f7f6f5f5f8" ++
    "f4f5f9f6f6f9f4f4f4ffffffabb0c8080f51060a4b0a145509114f081250061052030d53000c54000b56000b56000b55000c56010c56010c56010c56010c56010d56000d57010d57" ++
    "fefdfbfefdfbfefefdfffefefffffefffffffffffcfdfdfbfffefef6f1f3f6f4f5fffffefffdf9fefdfafefdf8fdfdf7fdfcf7fffcf5fffcf5fffcf5fdfbf5fbf9f5f7f7f6f5f5f7" ++
    "f5f5f9f6f6f9f4f4f2ffffffabb0c8081050060a4b0a155509114f081250061052030d53000b54000b56000b56000c55000c56000c56000d56010c56010c56010d56000d57000d57" ++
    "fffdf9fffdf9fefdf8fffefafffefbfefffbfefffcfefefbfcfbf9fdfefefefffdfefdf9fefdfafefdfafefdf8fefdf7fefcf7fffcf5fffcf5fefcf5fcfbf5faf9f5f7f7f6f5f5f7" ++
    "f5f5f9f6f6f9f4f4f4ffffffabb2c909115206094b0a155508124f081150060f52030d53000b54000b56000b56000b55000b56000c56000c56000c56000c56010d57000d57000d57" ++
    "fffdf7fffdf7fffdf7fefdf9fefdf9fefdfafdfcf9fdfcf9fdfcf8fdf9f7fdfaf8fefcf8fefcf8fefcf8fefdf7fffdf7fffdf7fffdf7fffdf6fffdf6fdfbf6fbf9f6f7f7f7f5f5f7" ++
    "f5f5f9f6f6f9f4f4f4ffffffabb2c909115206094b0a155408124e07124f051051030d54000c56000b56000b56000b56000b56000b56000c56000c56000c56000c56000c56000c56" ++
    "fefcf7fefcf7fefcf7fefcf7fefcf7fefcf7fdfcf7fdfbf5fdfbf6fefbf6fefbf6fefcf6fefcf6fefcf6fefcf6fffdf6fffdf6fffdf6fffdf6fffdf7fdfcf7fbfaf7f7f7f7f5f5f8" ++
    "f5f5f8f6f6f9f4f4f3ffffffacb2ca0a125205094b0a155308124e08114f060f52030d54000b55000b57000b57000b56000b56000c56000c56000c55000c55000c55000c56000c56" ++
    "fdfbf6fdfbf6fdfbf6fefbf5fefbf5fefcf5fdfcf5fdfbf4fefbf4fefbf5fefcf5fffcf5fffcf5fffcf5fffcf5fffdf6fffdf6fffdf6fffdf6fffdf6fdfcf7fbfaf7f7f7f7f5f5f8" ++
    "f5f5f8f6f6f9f4f4f3ffffffadb2ca0b125305094b0a155508124f081250061052030d54010b56000b56000b56000b56000b56000c56000c56000c55000c55000d56000d56000d56" ++
    "fefbf5fefcf5fefcf5fefbf4fefbf4fefbf4fefbf4fefbf4fefbf4fffbf4fffbf4fefcf4fefcf4fefcf4fefcf4fefcf4fefcf5fefcf5fefcf5fefcf6fdfbf6fbf9f6f7f7f8f5f5f8" ++
    "f5f5f8f6f6f9f5f4f3ffffffaeb4cb0c135405094b0a155508114f081150060f52030e54000c56000b56000b56000b56000b56000b56000c56000c56000c55000c55000c56000c56" ++
    "fefcf4fefcf4fefbf4fefbf4fefbf3fefaf3fefaf3fefbf3fefbf3fefbf3fffbf3fffcf3fefcf3fefbf3fefbf3fdfcf3fdfcf4fefcf4fefcf4fefcf5fcfbf5faf9f7f7f7f7f5f5f7" ++
    "f5f5f9f6f7f9f5f5f3ffffffafb4cb0d135405094b0a155508114f071150050f52030d54000b54000b56000b56000b55000b56000b56000c56000c55000c55000c55000c56000c56" ++
    "fcfaf2fdfbf2fefbf3fefaf2fefaf2fdfaf2fdfaf2fefbf3fefbf3fefbf3fefbf3fefcf4fefcf4fefbf3fefbf3fdfcf4fdfcf4fefcf4fefcf4fefcf5fcfbf5faf9f5f7f7f6f5f5f7" ++
    "f5f5f9f6f6f9f5f4f3ffffffafb4cb0d145405094c0a1556081250071150050f52030e54000c55000b55000b56000b56000b56000c56000c56000c55000c55000d55000d56000d56" ++
    "fdfaf3fcfaf0fdfaeefefbeefefaf0fefaf0fefaf1fefaf1fefaf1fefbf3fefbf3fefcf4fefcf4fefbf4fefbf4fdfbf3fdfbf4fefbf4fefcf4fefcf5fcfbf5faf9f5f7f7f6f5f5f7" ++
    "f5f5f9f6f6f9f5f4f3ffffffafb4cb0d1454050a4c0a1657081251071251051053030e55000c56000b56000b56000c56000c56000c56000c56000d56000c56000d56000d57000c57" ++
    "f4f4f4fffdf6fffcf2fbf8effcf9effdfaeffefaf1fefaf1fefbf1fefbf3fffbf3fffcf4fefbf4fefbf4fefbf4fefbf4fefbf4fefbf4fffcf4fffcf4fdfbf6fbf9f6f7f7f6f5f5f7" ++
    "f5f5f9f6f7f9f5f5f4ffffffafb4ca0d1453050a4d0a1657081251071251051053020d55000b56000b56000b57000c56000c56010d56010d56000d56000d56000d56000d56000c56" ++
    "817b99c5c2cbefeeeffdfcf8fcfcf6f9f7f3faf9f4fafaf5fcfaf5fefbf3fefbf3fefcf4fefcf4fefcf4fefcf4fefcf4fffcf4fffcf4fffcf4fffcf4fdfbf4fbf9f6f7f7f6f5f5f7" ++
    "f5f5f9f6f6f9f5f4f4ffffffaeb3ca0c1353050a4c0a1656081250071151050f53030d55000b56000b56000b57000c57000c57000c57000c56000c56000c56000d56000d56000c56" ++
    "000043000b544f548ac5c4d6f8fafcfafafdf6f7f8f6f7f8f8f7f9faf9f8fcfbf5fefbf4fefbf4fefcf4fdfbf4fdfbf4fefcf4fefcf4fefcf3fefcf3fcfbf5faf9f5f7f7f6f5f5f7" ++
    "f5f5f9f6f6f8f5f4f3ffffffadb2c70b1251050a4c0a1656081250071251051053030e55000c56000b56000b56000b56000b56000c56000c56000c56000c56000d56000d57000d57" ++
    "07125a070f5b000152010b5a7e81a7efedf3fffffffbfbf9fbfbfafbfbfafcfbf9fcfbf9fdfbf7fdfbf6fefbf6fefcf5fefcf5fffcf4fffcf4fffcf4fdfbf4fbf9f6f7f7f6f5f5f7" ++
    "f5f5f9f6f6f9f5f4f3ffffffadb1c70a1151050b4d0a1655081250071351051053030e55000c56000c56000c56000b56000b56000b56000c56000c56000c56000c57000d57000c57" ++
    "030d55030f5707145c0b1058000045505684e0dfe7fffffffafaf7f9f9faf9f9fbf8f9faf9faf8fbfaf6fdfaf6fefcf4fefcf4fffcf4fffcf4fffcf5fdfaf5fbf9f5f7f7f6f5f5f7" ++
    "f5f5f9f6f6f9f6f4f4ffffffacb0c70a1050060b4d0a1655091350081351061153030f55000e56000d56000c56000c56000c56000c56000c56000c56000c56010d56010d57000d57" ++
    "040f540610530812500917530f1b5e000045434679dbdbe7fffffff6f5f7f6f6faf5f6f9f7f6f7f9f8f6fbfaf6fdfcf5fefcf5fffcf4fffcf4fefaf5fcfaf5faf8f5f7f6f6f5f5f7" ++
    "f5f5f9f6f6f9f5f4f3ffffffacafc609104f070b4d0b1555091250081351061153030e55010d56000c56000c56000b56000b56000c56000c56000c56000c56000d57000d57000d57" ++
    "040f5207115009134f08134e0714510c185f000046595d8aefeef4fbfbfdf4f5f8f6f7faf8f7f8faf8f6fcfaf5fefbf5fffbf5fffcf5fffcf6fefcf6fcfbf6faf9f7f7f7f7f5f5f7" ++
    "f5f5f9f7f6f9f5f4f4ffffffacafc6090f4f070b4c0b155409124f071250051052030e54000d55000c55000b56000b56000b56000c56000c56010c56010c56010c56000d57000d57" ++
    "030e5405105107114f07114f05104e021258040c550004508c94b4fefefef7f7f6f9f8f5fbf9f7fcf9f6fcfbf6fefbf6fefbf6fefcf7fffcf7fffcf7fdfbf8fbf9f8f8f7f8f5f5f8" ++
    "f4f5f9f6f6f8f6f4f3ffffffaab0c609104f060b4c0a145409114f081250061052030d54000b56000b56000b56000b56000b56000b56000c56000c56000c56000d56010d57010d57" ++
    "020d56040f54040f52040f51030f51010d5204135f020245373a6ed1d5e0fffffdfaf8f2fdfbf5fcfaf6fcfaf6fcfbf6fcfbf8fdfbf8fffdf8fffdf8fdfcf9fbfaf9f7f7f8f5f5f8" ++
    "f5f5f9f6f6f8f5f4f3ffffffaab0c608104f060b4d0a145408114f08124f061052030d55010c56000b56000b57000b56000b56000c56000c56000c56000c56000c57010d57010d57" ++
    "020d58020d57020e57030e56030e54010d54020e58030c550005529096b6fffffff9f8f2fbfaf6fbfaf7fbfaf8fbfaf9fbfaf9fdfbf9fefdf8fefdf8fdfcf8fbfaf8f8f7f8f5f5f8" ++
    "f4f5f9f6f6f9f4f4f3ffffffabb0c608104e060a4c0a155309124e07124e051051030d54000b56000b56000b56000c56000c56000c56000d56010d56010d56010d56010d57010c57" ++
    "010d5b010d5b010e5b020e5a020e5a010d5a010d580613610100495c5e8df1f1f5fffffafaf9f5fbfbf9f9f9f9f9f9fafaf9fafcfbf8fefbf7fffdf7fefcf7fbfaf7f7f7f7f5f5f8" ++
    "f5f5f9f6f6f8f4f4f3ffffffabafc608104f060a4d0a155408124f081250061052030e54000b55000b56000b57000c57000c56010d56010d56000d56000d56000d57000c57000c57" ++
    "000d5d010e5d010e5d010e5f010e5f000e5f000d5e0414630000502c3675c8cfe0fffffffaf9f4fcfbfaf9f8f9f8f8faf9f8fafbf9f9fdfbf7fefcf6fdfcf6fbfaf6f7f7f7f5f5f7" ++
    "f5f5f9f6f6f8f5f4f3ffffffaab0c609104f060b4d0a165409134f071250051052030e55000c55000b57000b57000c56000c56000d56000d56000d56000d56000d57000c57000c57" ++
    "000e5d000e5f000e60000f61010f61010f61010e600413630109580b1462a9b0ccfffffffaf8f3fbfbf9f9f9faf7f8faf8f8f9faf9f7fcfbf6fdfbf4fdfbf4fbf9f6f8f7f6f5f5f7" ++
    "f5f5f9f7f6f9f5f4f4ffffffacb0c7091151060b4e0a1655091350071251051053030e55000c56000b56000b56000c56000c56000c56000d56000d56000d56010d57000d57000c57" ++
    "000d5d000d5f000d60010f61020f6103106003105f051360070f5a000b559aa2bffffffffaf9f5fbfafaf8f9faf7f7faf7f7f9f9f9f7fdfbf5fefcf5fefbf5fbf9f5f7f7f6f5f5f7" ++
    "f5f5f9f6f6f8f5f4f3ffffffabb0c609114f060b4d0a1555081250081351061153030e55010c57000c57000c57000b56000c56000c56000c56000c56000c56000c56010d57010d57" ++
    "000d5b000d5d000d5f010f5f01106003116005115f06135d09135a00064f939cb8fffffffaf9f6fbfaf9f8f9faf6f7faf7f7f9faf8f7fcfaf5fefbf4fdfbf4fbf9f6f7f7f6f5f5f7" ++
    "f5f5f9f6f6f8f4f4f3ffffffabb0c7091151060a4e0a1555081250081251061053030e55020e57010d57000c57000c57000c56000c56000c56000c56000c57010d57010d57010d57" ++
    "000c5a000d5c010e5d020f5e03105e04105e05105e06145c0a135a00054d929ab6fffffffafaf6fbf9f9f8f8faf6f7f9f7f7f8f9f9f7fcfaf5fdfaf4fdfaf4fbf8f4f7f6f6f5f5f8" ++
    "f5f5f9f6f6f9f4f4f4ffffffabb1c7091151060a4d0a1555081250081251061053030e55010c57000c57000c57000c57000c57000c57000c57000c57000c57000d57000c57000c58" ++
    "000c59010d59010d59010d5b030f5b04105b04105b05125a081058000450939bb8fffffff9f8f6fbf9faf8f9faf7f7f9f7f7f9f9f9f7fcfaf5fdfaf4fdfaf4fbf8f6f7f6f6f5f5f7" ++
    "f5f5f9f6f6f8f4f4f3ffffffabb0c6091150060b4d0a1655081250081351061053030e55000c56000b56000b57000b57000b57000b57000c57000c56000c57000d57000d57000c57";
test "decoded real PNG RGB fixture matches complete resized output hashes" {
    var bytes: [32 * 48 * 3]u8 = undefined;
    _ = try std.fmt.hexToBytes(&bytes, real_rgb);
    const cases = [_]struct { shape: pre.Shape, hash: []const u8 }{
        .{ .shape = .{ .height = 21, .width = 37 }, .hash = "73cb67658140037e3f5d170a801882bc1d616469deee1d64e115d573ab1fd916" },
        .{ .shape = .{ .height = 53, .width = 79 }, .hash = "395fd0873cd7e7e734826adef606cc4f547cb4cbcb1f3f910e3ffd3ac32a36fd" },
        .{ .shape = .{ .height = 32, .width = 19 }, .hash = "21e4a883f5b72b9cc9f63aae6f6d0a45575ccc853b47114a06e98b9c556ec363" },
        .{ .shape = .{ .height = 11, .width = 48 }, .hash = "9d2ae3c13efd8c7cfca5add8391d66fa21a2aa96f8dfd28bc47d47d144070c9b" },
    };
    for (cases) |cs| {
        var out = try resize.resize(testing.allocator, .{ .shape = .{ .height = 32, .width = 48 }, .bytes = &bytes }, cs.shape);
        defer out.deinit(testing.allocator);
        try hashEqual(cs.hash, out.bytes);
    }
}

test "resize owns its identity output and plugs into the frozen preprocessing callback" {
    const a = testing.allocator;
    const bytes = try pattern(a, .{ .height = 13, .width = 17 }, .ramp);
    defer a.free(bytes);
    const input: pre.Rgb = .{ .shape = .{ .height = 13, .width = 17 }, .bytes = bytes };
    var out = try resize.resize(a, input, input.shape);
    defer out.deinit(a);
    try testing.expectEqualSlices(u8, bytes, out.bytes);
    try testing.expect(bytes.ptr != out.bytes.ptr);
    const c = try @import("cuda_vision_config.zig").parse(a, @embedFile("fixtures_cuda_config_exl3.json"));
    var prepared = try pre.prepareRgb(a, c, input, .{}, .{}, resize.resizer);
    defer prepared.deinit(a);
    try prepared.check(c);
    try testing.expectEqualDeep(pre.Shape{ .height = 107, .width = 140 }, prepared.content);
    // Full FP32 output of archived fit + patches on the same offline CPU oracle.
    try hashEqual("cc5bd0dbb26cf21a6fb6f17f54df66b25a5d784e3be900126804634c5e69420a", std.mem.sliceAsBytes(prepared.pixels));
}

test "resize refuses malformed or excessive shapes without allocating" {
    const fail = testing.failing_allocator;
    try testing.expectError(error.InvalidImageShape, resize.resize(fail, .{ .shape = .{ .height = 1, .width = 1 }, .bytes = &.{} }, .{ .height = 1, .width = 1 }));
    const rgb: pre.Rgb = .{ .shape = .{ .height = 1, .width = 1 }, .bytes = &.{ 1, 2, 3 } };
    for ([_]pre.Shape{ .{ .height = 0, .width = 1 }, .{ .height = 1, .width = std.math.maxInt(u32) }, .{ .height = 2800, .width = 2800 } }) |shape| {
        try testing.expectError(error.UnsupportedResizeShape, resize.resize(fail, rgb, shape));
    }
    var tall: [8192 * 3]u8 = @splat(0);
    try testing.expectError(error.UnsupportedResizeShape, resize.resize(fail, .{ .shape = .{ .height = 8192, .width = 1 }, .bytes = &tall }, .{ .height = 1, .width = 224000 }));
}

test "all resize pass combinations and caller preparation balance allocation failures" {
    const a = testing.allocator;
    const bytes = try pattern(a, .{ .height = 13, .width = 17 }, .ramp);
    defer a.free(bytes);
    const input: pre.Rgb = .{ .shape = .{ .height = 13, .width = 17 }, .bytes = bytes };
    const Trial = struct {
        fn run(alloc: Allocator, rgb: pre.Rgb, shape: pre.Shape) !void {
            var out = try resize.resize(alloc, rgb, shape);
            defer out.deinit(alloc);
        }
        fn prepare(alloc: Allocator, rgb: pre.Rgb, c: @import("cuda_vision_config.zig").Config) !void {
            var out = try pre.prepareRgb(alloc, c, rgb, .{}, .{}, resize.resizer);
            defer out.deinit(alloc);
        }
    };
    for ([_]pre.Shape{ input.shape, .{ .height = 13, .width = 7 }, .{ .height = 7, .width = 17 }, .{ .height = 19, .width = 7 } }) |target| {
        try testing.checkAllAllocationFailures(a, Trial.run, .{ input, target });
    }
    const c = try @import("cuda_vision_config.zig").parse(a, @embedFile("fixtures_cuda_config_exl3.json"));
    try testing.checkAllAllocationFailures(a, Trial.prepare, .{ input, c });
}
