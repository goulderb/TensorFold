const std = @import("std");
pub fn partsInto(total: u32, world: u32, unit: u32, out: []u32) ![]u32 {
    if (world < 1 or unit < 1 or world > out.len) return error.BadSplit;
    if (total % unit != 0) return error.BadSplit;
    const n_units = total / unit;
    if (n_units < world) return error.BadSplit;
    const base = n_units / world;
    const extra = n_units % world;
    for (0..world) |r| out[r] = (base + @intFromBool(r < extra)) * unit;
    return out[0..world];
}
pub fn splitSizes(total: u32, world: u32, unit: u32, out: []u32) ![]u32 {
    if (world == 2 and total % 2 == 0) {
        if (out.len < 2) return error.BadSplit;
        out[0] = total / 2;
        out[1] = total / 2;
        return out[0..2];
    }
    return partsInto(total, world, unit, out);
}
pub fn shareOf(sizes: []const u32, rank: u32) struct { start: u32, stop: u32 } {
    var start: u32 = 0;
    for (sizes[0..rank]) |s| start += s;
    return .{ .start = start, .stop = start + sizes[rank] };
}
pub fn draftParts(heads: u32, kv_heads: u32, inter: u32, world: u32, q_out: []u32, kv_out: []u32, mlp_out: []u32) !void {
    if (kv_heads == 0 or heads % kv_heads != 0) return error.BadSplit;
    const group = heads / kv_heads;
    const kv = try splitSizes(kv_heads, world, 1, kv_out);
    const q = q_out[0..kv.len];
    for (kv, q) |k, *o| o.* = k * group;
    const mlp_unit: u32 = if (inter % 128 == 0) 128 else 1;
    _ = try splitSizes(inter, world, mlp_unit, mlp_out);
}
