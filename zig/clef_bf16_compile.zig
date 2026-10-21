//! Compile the explicit BF16 constructor and hidden transaction. This object never opens a device.
const qwen = @import("qwen35");
const Engine = qwen.engine.Engine;

export fn compiledBf16Init() usize {
    return @intFromPtr(&Engine.initBf16);
}

export fn compiledBf16Forward() usize {
    return @intFromPtr(&Engine.forward);
}

export fn compiledBf16Close() usize {
    return @intFromPtr(&Engine.deinit);
}

export fn compiledBf16Cache() usize {
    return @intFromPtr(&qwen.engine.Cache.init);
}

export fn compiledBf16Hidden() usize {
    return @intFromPtr(&Engine.hiddenRows);
}
