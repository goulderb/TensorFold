// Mathematical reference adapted from Hugging Face Transformers revision
// e0eb2810ac1b2caf6d082f247d8d37162ecf67a0, modeling_qwen3_5_moe.py.
// Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// Local changes: small host oracles and persistent-state regression fixtures.
const std = @import("std");

pub fn bf16(x: f32) f32 {
    const bits: u32 = @bitCast(x);
    if ((bits & 0x7fffffff) > 0x7f800000) return @bitCast(bits | 0x00400000);
    const rounded = bits +% 0x7fff +% ((bits >> 16) & 1);
    return @bitCast(rounded & 0xffff0000);
}

fn sigmoid(x: f32) f32 {
    return 1 / (1 + @exp(-x));
}

pub const Route = struct { picks: [8]i32, weights: [9]f32 };

pub fn route(logits: [256]f32, shared: f32) Route {
    var result: Route = undefined;
    var selected: [256]bool = @splat(false);
    for (0..8) |slot| {
        var best: ?usize = null;
        for (logits, 0..) |logit, candidate| {
            if (selected[candidate]) continue;
            if (best == null or logit > logits[best.?]) best = candidate;
        }
        result.picks[slot] = @intCast(best.?);
        selected[best.?] = true;
    }
    var denominator: f32 = 0;
    const top = logits[@intCast(result.picks[0])];
    for (result.picks, 0..) |pick, slot| {
        result.weights[slot] = @exp(logits[@intCast(pick)] - top);
        denominator += result.weights[slot];
    }
    for (result.weights[0..8]) |*weight| weight.* /= denominator;
    result.weights[8] = sigmoid(shared);
    return result;
}

pub fn l2(x: [128]f32, query: bool) [128]f32 {
    var squared: f32 = 0;
    for (x) |v| squared += v * v;
    const norm = @sqrt(squared + 1e-6);
    var result: [128]f32 = undefined;
    for (x, &result) |v, *out| out.* = (v / norm) / (if (query) @sqrt(@as(f32, 128)) else 1);
    return result;
}

pub fn conv(qkv: []const f32, weights: []const f32, history: []f32, out: []f32) void {
    const width = qkv.len;
    std.debug.assert(weights.len == width * 4 and history.len == width * 3 and out.len == width);
    for (qkv, 0..) |x, ch| {
        var sum: f32 = 0;
        for (0..3) |row| sum += history[row * width + ch] * weights[ch * 4 + row];
        sum += x * weights[ch * 4 + 3];
        const rounded = bf16(sum);
        out[ch] = bf16(rounded * sigmoid(rounded));
        history[ch] = history[width + ch];
        history[width + ch] = history[2 * width + ch];
        history[2 * width + ch] = x;
    }
}

pub fn gates(a: f32, b: f32, a_log: f32, dt_bias: f32) struct { log_decay: f32, beta: f32 } {
    const dt = a + dt_bias;
    const softplus = if (dt > 20) dt else @log(1 + @exp(dt));
    return .{ .log_decay = -@exp(a_log) * softplus, .beta = bf16(sigmoid(b)) };
}

/// The CPU oracle uses key-major state, independently of CUDA's value-major storage.
pub fn gdn(q: [128]f32, k: [128]f32, v: [128]f32, log_decay: f32, beta: f32, state: *[128][128]f32) [128]f32 {
    const decay = @exp(log_decay);
    var memory: [128]f32 = @splat(0);
    for (state, k) |*row, key| {
        for (row, &memory) |*cell, *mem| {
            cell.* *= decay;
            mem.* += cell.* * key;
        }
    }
    var delta: [128]f32 = undefined;
    for (v, memory, &delta) |value, mem, *out| out.* = (value - mem) * beta;
    var result: [128]f32 = @splat(0);
    for (state, k, q) |*row, key, query| {
        for (row, delta, &result) |*cell, update, *out| {
            cell.* += key * update;
            out.* += cell.* * query;
        }
    }
    return result;
}

pub fn gatedNorm(y: [128]f32, z: [128]f32, weight: [128]f32, eps: f32) [128]f32 {
    var rounded: [128]f32 = undefined;
    var variance: f32 = 0;
    for (y, &rounded) |v, *r| {
        r.* = bf16(v);
        variance += r.* * r.*;
    }
    const inv = 1 / @sqrt(variance / 128 + eps);
    var result: [128]f32 = undefined;
    for (rounded, z, weight, &result) |v, gate, w, *out| out.* = bf16(bf16(bf16(v * inv) * w) * (gate * sigmoid(gate)));
    return result;
}

pub fn rotary(x: [256]f32, position: usize) [256]f32 {
    var out = x;
    for (0..32) |pair| {
        const exponent = -@as(f32, @floatFromInt(pair)) / 32;
        const angle = @as(f32, @floatFromInt(position)) * std.math.pow(f32, 10000000, exponent);
        const c = bf16(@cos(angle));
        const s = bf16(@sin(angle));
        out[pair] = bf16(bf16(x[pair] * c) - bf16(x[pair + 32] * s));
        out[pair + 32] = bf16(bf16(x[pair + 32] * c) + bf16(x[pair] * s));
    }
    return out;
}

pub fn attention(q_gate: [16][512]f32, q_norm: [256]f32, keys: []const f32, values: []const f32, position: usize) [16][256]f32 {
    std.debug.assert(keys.len >= (position + 1) * 512 and values.len >= (position + 1) * 512);
    var out: [16][256]f32 = undefined;
    for (q_gate, &out, 0..) |head, *result, h| {
        var squared: f32 = 0;
        for (head[0..256]) |x| squared += x * x;
        const inv = 1 / @sqrt(squared / 256 + 1e-6);
        var query: [256]f32 = undefined;
        for (head[0..256], q_norm, &query) |x, w, *q| q.* = bf16(x * inv * w);
        query = rotary(query, position);
        var maximum: f64 = -std.math.inf(f64);
        for (0..position + 1) |token| {
            var score: f64 = 0;
            for (query, 0..) |q, d| score += @as(f64, q) * keys[token * 512 + (h / 8) * 256 + d];
            maximum = @max(maximum, score / 16);
        }
        var denominator: f64 = 0;
        var value_sum: [256]f64 = @splat(0);
        for (0..position + 1) |token| {
            var score: f64 = 0;
            for (query, 0..) |q, d| score += @as(f64, q) * keys[token * 512 + (h / 8) * 256 + d];
            const probability = @exp(score / 16 - maximum);
            denominator += probability;
            for (&value_sum, 0..) |*acc, d| acc.* += probability * values[token * 512 + (h / 8) * 256 + d];
        }
        for (value_sum, head[256..], result) |value, gate, *r| r.* = bf16(bf16(@floatCast(value / denominator)) * bf16(sigmoid(gate)));
    }
    return out;
}

pub fn combine(pairs: [8][2048]f32, weights: [9]f32, shared: [2048]f32) [2048]f32 {
    var out: [2048]f32 = @splat(0);
    for (pairs, weights[0..8]) |pair, weight| {
        for (pair, &out) |value, *result| result.* += value * weight;
    }
    for (&out, shared) |*result, value| result.* = bf16(result.* + value * weights[8]);
    return out;
}

test "top eight is stable, excludes duplicates and normalizes only chosen logits" {
    var logits: [256]f32 = @splat(-1000);
    logits[200] = 1000;
    logits[2] = 1000;
    logits[35] = 999;
    const result = route(logits, 0);
    try std.testing.expectEqualSlices(i32, &.{ 2, 200, 35, 0, 1, 3, 4, 5 }, &result.picks);
    var total: f32 = 0;
    for (result.weights[0..8]) |weight| total += weight;
    try std.testing.expectApproxEqAbs(@as(f32, 1), total, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / (2.0 + @exp(-@as(f32, 1)))), result.weights[0], 1e-6);
    try std.testing.expectEqual(@as(f32, 0.5), result.weights[8]);
    const tied = route(@splat(0), 0);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, &tied.picks);
    for (tied.weights[0..8]) |weight| try std.testing.expectEqual(@as(f32, 0.125), weight);
}

test "convolution keeps the last three raw rows independently by channel" {
    var history = [_]f32{ 1, 10, 2, 20, 3, 30 };
    var output: [2]f32 = undefined;
    conv(&.{ 4, 40 }, &.{ 1, 0, 0, 0, 0, 0, 0, 1 }, &history, &output);
    try std.testing.expectEqualSlices(f32, &.{ 2, 20, 3, 30, 4, 40 }, &history);
    try std.testing.expectEqual(bf16(sigmoid(1)), output[0]);
    try std.testing.expectEqual(@as(f32, 40), output[1]);
    conv(&.{ 5, 50 }, &.{ 1, 0, 0, 0, 0, 0, 0, 1 }, &history, &output);
    try std.testing.expectEqualSlices(f32, &.{ 3, 30, 4, 40, 5, 50 }, &history);
}

test "qk normalization adds epsilon to the sum and keeps small norms finite" {
    var input: [128]f32 = @splat(0);
    const zero = l2(input, false);
    try std.testing.expectEqual(@as(f32, 0), zero[0]);
    input[0] = 0.001;
    const k = l2(input, false);
    const q = l2(input, true);
    try std.testing.expectApproxEqAbs(@as(f32, 1 / @sqrt(@as(f32, 2))), k[0], 1e-6);
    try std.testing.expectApproxEqAbs(k[0] / @sqrt(@as(f32, 128)), q[0], 1e-6);
}

test "GDN decays before delta update, persists state and reads the updated state" {
    var q: [128]f32 = @splat(0);
    var k: [128]f32 = @splat(0);
    var v: [128]f32 = @splat(0);
    var state: [128][128]f32 = @splat(@splat(0));
    q[0] = 0.25;
    q[1] = 0.5;
    k[0] = 1;
    v[3] = 4;
    const first = gdn(q, k, v, 0, 0.5, &state);
    try std.testing.expectEqual(@as(f32, 2), state[0][3]);
    try std.testing.expectEqual(@as(f32, 0.5), first[3]);
    k[0] = 0;
    k[1] = 1;
    v[3] = 2;
    const second = gdn(q, k, v, @log(@as(f32, 0.5)), 0.5, &state);
    try std.testing.expectEqual(@as(f32, 1), state[0][3]);
    try std.testing.expectEqual(@as(f32, 1), state[1][3]);
    try std.testing.expectEqual(@as(f32, 0.75), second[3]);
}

test "decay remains negative log and beta follows BF16 projection dtype" {
    const result = gates(0, 0, 0, 0);
    try std.testing.expectApproxEqAbs(-@log(@as(f32, 2)), result.log_decay, 1e-6);
    try std.testing.expectEqual(@as(f32, 0.5), result.beta);
    try std.testing.expect(std.math.isFinite(gates(100, 0, 0, 0).log_decay));
    try std.testing.expectEqual(@as(f32, -100), gates(100, 0, 0, 0).log_decay);
}

test "partial rotary pairs split halves and preserves all 192 pass-through dimensions" {
    var input: [256]f32 = @splat(3);
    input[0] = 1;
    input[32] = 2;
    try std.testing.expectEqualSlices(f32, &input, &rotary(input, 0));
    const out = rotary(input, 1);
    const c = bf16(@cos(@as(f32, 1)));
    const s = bf16(@sin(@as(f32, 1)));
    try std.testing.expectEqual(bf16(bf16(c) - bf16(2 * s)), out[0]);
    try std.testing.expectEqual(bf16(bf16(2 * c) + bf16(s)), out[32]);
    try std.testing.expectEqualSlices(f32, input[64..], out[64..]);
}

test "gated norm rounds recurrence and normalized values to model dtype" {
    const y: [128]f32 = @splat(1.0039);
    const out = gatedNorm(y, @splat(0), @splat(1), 1e-6);
    for (out) |value| try std.testing.expectEqual(@as(f32, 0), value);
    const active = gatedNorm(@splat(2), @splat(1), @splat(1), 1e-6);
    try std.testing.expectEqual(bf16(sigmoid(1)), active[0]);
}

test "full attention is causal, uses eight queries per KV head and interleaves each query gate" {
    var q_gate: [16][512]f32 = @splat(@splat(0));
    const keys: [3 * 512]f32 = @splat(0);
    var values: [3 * 512]f32 = undefined;
    @memset(values[0..256], 2);
    @memset(values[256..512], 10);
    @memset(values[512..768], 6);
    @memset(values[768..1024], 14);
    @memset(values[1024..], 10000);
    @memset(q_gate[3][256..], bf16(@log(@as(f32, 3))));
    const first = attention(q_gate, @splat(1), &keys, &values, 0);
    try std.testing.expectEqual(@as(f32, 1), first[0][0]);
    try std.testing.expectEqual(@as(f32, 1), first[7][255]);
    try std.testing.expectEqual(@as(f32, 5), first[8][0]);
    try std.testing.expectEqual(@as(f32, 1.5), first[3][0]);
    const second = attention(q_gate, @splat(1), &keys, &values, 1);
    try std.testing.expectEqual(@as(f32, 2), second[0][0]);
    try std.testing.expectEqual(@as(f32, 6), second[15][255]);
    try std.testing.expectEqual(@as(f32, 3), second[3][255]);
}

test "MoE combine weights eight FP32 pairs in pick order and applies independent shared gate" {
    var pairs: [8][2048]f32 = @splat(@splat(0));
    pairs[0][17] = 8;
    pairs[7][17] = 16;
    var weights: [9]f32 = @splat(0);
    weights[0] = 0.25;
    weights[7] = 0.75;
    weights[8] = 0.5;
    var shared: [2048]f32 = @splat(0);
    shared[17] = 10;
    const out = combine(pairs, weights, shared);
    try std.testing.expectEqual(@as(f32, 19), out[17]);
    try std.testing.expectEqual(@as(f32, 0), out[18]);
}
