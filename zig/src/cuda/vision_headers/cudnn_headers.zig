const __root = @This();
pub const __builtin = @import("std").zig.c_translation.builtins;
pub const __helpers = @import("std").zig.c_translation.helpers;
pub const int_least64_t = i64;
pub const uint_least64_t = u64;
pub const int_fast64_t = i64;
pub const uint_fast64_t = u64;
pub const int_least32_t = i32;
pub const uint_least32_t = u32;
pub const int_fast32_t = i32;
pub const uint_fast32_t = u32;
pub const int_least16_t = i16;
pub const uint_least16_t = u16;
pub const int_fast16_t = i16;
pub const uint_fast16_t = u16;
pub const int_least8_t = i8;
pub const uint_least8_t = u8;
pub const int_fast8_t = i8;
pub const uint_fast8_t = u8;
pub const intmax_t = c_longlong;
pub const uintmax_t = c_ulonglong;
pub const cudaRoundNearest: c_int = 0;
pub const cudaRoundZero: c_int = 1;
pub const cudaRoundPosInf: c_int = 2;
pub const cudaRoundMinInf: c_int = 3;
pub const enum_cudaRoundMode = c_uint;
pub const struct_char1 = extern struct {
    x: i8 = 0,
};
pub const struct_uchar1 = extern struct {
    x: u8 = 0,
};
pub const struct_char2 = extern struct {
    x: i8 align(2) = 0,
    y: i8 = 0,
};
pub const struct_uchar2 = extern struct {
    x: u8 align(2) = 0,
    y: u8 = 0,
};
pub const struct_char3 = extern struct {
    x: i8 = 0,
    y: i8 = 0,
    z: i8 = 0,
};
pub const struct_uchar3 = extern struct {
    x: u8 = 0,
    y: u8 = 0,
    z: u8 = 0,
};
pub const struct_char4 = extern struct {
    x: i8 align(4) = 0,
    y: i8 = 0,
    z: i8 = 0,
    w: i8 = 0,
};
pub const struct_uchar4 = extern struct {
    x: u8 align(4) = 0,
    y: u8 = 0,
    z: u8 = 0,
    w: u8 = 0,
};
pub const struct_short1 = extern struct {
    x: c_short = 0,
};
pub const struct_ushort1 = extern struct {
    x: c_ushort = 0,
};
pub const struct_short2 = extern struct {
    x: c_short align(4) = 0,
    y: c_short = 0,
};
pub const struct_ushort2 = extern struct {
    x: c_ushort align(4) = 0,
    y: c_ushort = 0,
};
pub const struct_short3 = extern struct {
    x: c_short = 0,
    y: c_short = 0,
    z: c_short = 0,
};
pub const struct_ushort3 = extern struct {
    x: c_ushort = 0,
    y: c_ushort = 0,
    z: c_ushort = 0,
};
pub const struct_short4 = extern struct {
    x: c_short align(8) = 0,
    y: c_short = 0,
    z: c_short = 0,
    w: c_short = 0,
};
pub const struct_ushort4 = extern struct {
    x: c_ushort align(8) = 0,
    y: c_ushort = 0,
    z: c_ushort = 0,
    w: c_ushort = 0,
};
pub const struct_int1 = extern struct {
    x: c_int = 0,
};
pub const struct_uint1 = extern struct {
    x: c_uint = 0,
};
pub const struct_int2 = extern struct {
    x: c_int align(8) = 0,
    y: c_int = 0,
};
pub const struct_uint2 = extern struct {
    x: c_uint align(8) = 0,
    y: c_uint = 0,
};
pub const struct_int3 = extern struct {
    x: c_int = 0,
    y: c_int = 0,
    z: c_int = 0,
};
pub const struct_uint3 = extern struct {
    x: c_uint = 0,
    y: c_uint = 0,
    z: c_uint = 0,
};
pub const struct_int4 = extern struct {
    x: c_int align(16) = 0,
    y: c_int = 0,
    z: c_int = 0,
    w: c_int = 0,
};
pub const struct_uint4 = extern struct {
    x: c_uint align(16) = 0,
    y: c_uint = 0,
    z: c_uint = 0,
    w: c_uint = 0,
};
pub const struct_long1 = extern struct {
    x: c_long = 0,
};
pub const struct_ulong1 = extern struct {
    x: c_ulong = 0,
};
pub const struct_long2 = extern struct {
    x: c_long align(8) = 0,
    y: c_long = 0,
};
pub const struct_ulong2 = extern struct {
    x: c_ulong align(8) = 0,
    y: c_ulong = 0,
};
pub const struct_long3 = extern struct {
    x: c_long = 0,
    y: c_long = 0,
    z: c_long = 0,
};
pub const struct_ulong3 = extern struct {
    x: c_ulong = 0,
    y: c_ulong = 0,
    z: c_ulong = 0,
};
pub const struct_long4 = extern struct {
    x: c_long align(16) = 0,
    y: c_long = 0,
    z: c_long = 0,
    w: c_long = 0,
};
pub const struct_ulong4 = extern struct {
    x: c_ulong align(16) = 0,
    y: c_ulong = 0,
    z: c_ulong = 0,
    w: c_ulong = 0,
};
pub const struct_long4_16a = extern struct {
    x: c_long align(16) = 0,
    y: c_long = 0,
    z: c_long = 0,
    w: c_long = 0,
};
pub const struct_ulong4_16a = extern struct {
    x: c_ulong align(16) = 0,
    y: c_ulong = 0,
    z: c_ulong = 0,
    w: c_ulong = 0,
};
pub const struct_long4_32a = extern struct {
    x: c_long align(32) = 0,
    y: c_long = 0,
    z: c_long = 0,
    w: c_long = 0,
};
pub const struct_ulong4_32a = extern struct {
    x: c_ulong align(32) = 0,
    y: c_ulong = 0,
    z: c_ulong = 0,
    w: c_ulong = 0,
};
pub const struct_float1 = extern struct {
    x: f32 = 0,
};
pub const struct_float2 = extern struct {
    x: f32 align(8) = 0,
    y: f32 = 0,
};
pub const struct_float3 = extern struct {
    x: f32 = 0,
    y: f32 = 0,
    z: f32 = 0,
};
pub const struct_float4 = extern struct {
    x: f32 align(16) = 0,
    y: f32 = 0,
    z: f32 = 0,
    w: f32 = 0,
};
pub const struct_longlong1 = extern struct {
    x: c_longlong = 0,
};
pub const struct_ulonglong1 = extern struct {
    x: c_ulonglong = 0,
};
pub const struct_longlong2 = extern struct {
    x: c_longlong align(16) = 0,
    y: c_longlong = 0,
};
pub const struct_ulonglong2 = extern struct {
    x: c_ulonglong align(16) = 0,
    y: c_ulonglong = 0,
};
pub const struct_longlong3 = extern struct {
    x: c_longlong = 0,
    y: c_longlong = 0,
    z: c_longlong = 0,
};
pub const struct_ulonglong3 = extern struct {
    x: c_ulonglong = 0,
    y: c_ulonglong = 0,
    z: c_ulonglong = 0,
};
pub const struct_longlong4 = extern struct {
    x: c_longlong align(16) = 0,
    y: c_longlong = 0,
    z: c_longlong = 0,
    w: c_longlong = 0,
};
pub const struct_longlong4_16a = extern struct {
    x: c_longlong align(16) = 0,
    y: c_longlong = 0,
    z: c_longlong = 0,
    w: c_longlong = 0,
};
pub const struct_longlong4_32a = extern struct {
    x: c_longlong align(32) = 0,
    y: c_longlong = 0,
    z: c_longlong = 0,
    w: c_longlong = 0,
};
pub const struct_ulonglong4 = extern struct {
    x: c_ulonglong align(16) = 0,
    y: c_ulonglong = 0,
    z: c_ulonglong = 0,
    w: c_ulonglong = 0,
};
pub const struct_ulonglong4_16a = extern struct {
    x: c_ulonglong align(16) = 0,
    y: c_ulonglong = 0,
    z: c_ulonglong = 0,
    w: c_ulonglong = 0,
};
pub const struct_ulonglong4_32a = extern struct {
    x: c_ulonglong align(32) = 0,
    y: c_ulonglong = 0,
    z: c_ulonglong = 0,
    w: c_ulonglong = 0,
};
pub const struct_double1 = extern struct {
    x: f64 = 0,
};
pub const struct_double2 = extern struct {
    x: f64 align(16) = 0,
    y: f64 = 0,
};
pub const struct_double3 = extern struct {
    x: f64 = 0,
    y: f64 = 0,
    z: f64 = 0,
};
pub const struct_double4 = extern struct {
    x: f64 align(16) = 0,
    y: f64 = 0,
    z: f64 = 0,
    w: f64 = 0,
};
pub const struct_double4_16a = extern struct {
    x: f64 align(16) = 0,
    y: f64 = 0,
    z: f64 = 0,
    w: f64 = 0,
};
pub const struct_double4_32a = extern struct {
    x: f64 align(32) = 0,
    y: f64 = 0,
    z: f64 = 0,
    w: f64 = 0,
};
pub const char1 = struct_char1;
pub const uchar1 = struct_uchar1;
pub const char2 = struct_char2;
pub const uchar2 = struct_uchar2;
pub const char3 = struct_char3;
pub const uchar3 = struct_uchar3;
pub const char4 = struct_char4;
pub const uchar4 = struct_uchar4;
pub const short1 = struct_short1;
pub const ushort1 = struct_ushort1;
pub const short2 = struct_short2;
pub const ushort2 = struct_ushort2;
pub const short3 = struct_short3;
pub const ushort3 = struct_ushort3;
pub const short4 = struct_short4;
pub const ushort4 = struct_ushort4;
pub const int1 = struct_int1;
pub const uint1 = struct_uint1;
pub const int2 = struct_int2;
pub const uint2 = struct_uint2;
pub const int3 = struct_int3;
pub const uint3 = struct_uint3;
pub const int4 = struct_int4;
pub const uint4 = struct_uint4;
pub const long1 = struct_long1;
pub const ulong1 = struct_ulong1;
pub const long2 = struct_long2;
pub const ulong2 = struct_ulong2;
pub const long3 = struct_long3;
pub const ulong3 = struct_ulong3;
pub const long4 = struct_long4;
pub const ulong4 = struct_ulong4;
pub const long4_16a = struct_long4_16a;
pub const ulong4_16a = struct_ulong4_16a;
pub const long4_32a = struct_long4_32a;
pub const ulong4_32a = struct_ulong4_32a;
pub const float1 = struct_float1;
pub const float2 = struct_float2;
pub const float3 = struct_float3;
pub const float4 = struct_float4;
pub const longlong1 = struct_longlong1;
pub const ulonglong1 = struct_ulonglong1;
pub const longlong2 = struct_longlong2;
pub const ulonglong2 = struct_ulonglong2;
pub const longlong3 = struct_longlong3;
pub const ulonglong3 = struct_ulonglong3;
pub const longlong4 = struct_longlong4;
pub const ulonglong4 = struct_ulonglong4;
pub const longlong4_16a = struct_longlong4_16a;
pub const ulonglong4_16a = struct_ulonglong4_16a;
pub const longlong4_32a = struct_longlong4_32a;
pub const ulonglong4_32a = struct_ulonglong4_32a;
pub const double1 = struct_double1;
pub const double2 = struct_double2;
pub const double3 = struct_double3;
pub const double4 = struct_double4;
pub const double4_16a = struct_double4_16a;
pub const double4_32a = struct_double4_32a;
pub const struct_dim3 = extern struct {
    x: c_uint = 0,
    y: c_uint = 0,
    z: c_uint = 0,
};
pub const dim3 = struct_dim3;
pub const ptrdiff_t = c_longlong;
pub const wchar_t = c_ushort;
pub const max_align_t = extern struct {
    __aro_max_align_ll: c_longlong = 0,
    __aro_max_align_ld: c_longdouble = 0,
};
pub const cudaSuccess: c_int = 0;
pub const cudaErrorInvalidValue: c_int = 1;
pub const cudaErrorMemoryAllocation: c_int = 2;
pub const cudaErrorInitializationError: c_int = 3;
pub const cudaErrorCudartUnloading: c_int = 4;
pub const cudaErrorProfilerDisabled: c_int = 5;
pub const cudaErrorProfilerNotInitialized: c_int = 6;
pub const cudaErrorProfilerAlreadyStarted: c_int = 7;
pub const cudaErrorProfilerAlreadyStopped: c_int = 8;
pub const cudaErrorInvalidConfiguration: c_int = 9;
pub const cudaErrorVersionTranslation: c_int = 10;
pub const cudaErrorInvalidPitchValue: c_int = 12;
pub const cudaErrorInvalidSymbol: c_int = 13;
pub const cudaErrorInvalidHostPointer: c_int = 16;
pub const cudaErrorInvalidDevicePointer: c_int = 17;
pub const cudaErrorInvalidTexture: c_int = 18;
pub const cudaErrorInvalidTextureBinding: c_int = 19;
pub const cudaErrorInvalidChannelDescriptor: c_int = 20;
pub const cudaErrorInvalidMemcpyDirection: c_int = 21;
pub const cudaErrorAddressOfConstant: c_int = 22;
pub const cudaErrorTextureFetchFailed: c_int = 23;
pub const cudaErrorTextureNotBound: c_int = 24;
pub const cudaErrorSynchronizationError: c_int = 25;
pub const cudaErrorInvalidFilterSetting: c_int = 26;
pub const cudaErrorInvalidNormSetting: c_int = 27;
pub const cudaErrorMixedDeviceExecution: c_int = 28;
pub const cudaErrorNotYetImplemented: c_int = 31;
pub const cudaErrorMemoryValueTooLarge: c_int = 32;
pub const cudaErrorStubLibrary: c_int = 34;
pub const cudaErrorInsufficientDriver: c_int = 35;
pub const cudaErrorCallRequiresNewerDriver: c_int = 36;
pub const cudaErrorInvalidSurface: c_int = 37;
pub const cudaErrorDuplicateVariableName: c_int = 43;
pub const cudaErrorDuplicateTextureName: c_int = 44;
pub const cudaErrorDuplicateSurfaceName: c_int = 45;
pub const cudaErrorDevicesUnavailable: c_int = 46;
pub const cudaErrorIncompatibleDriverContext: c_int = 49;
pub const cudaErrorMissingConfiguration: c_int = 52;
pub const cudaErrorPriorLaunchFailure: c_int = 53;
pub const cudaErrorLaunchMaxDepthExceeded: c_int = 65;
pub const cudaErrorLaunchFileScopedTex: c_int = 66;
pub const cudaErrorLaunchFileScopedSurf: c_int = 67;
pub const cudaErrorSyncDepthExceeded: c_int = 68;
pub const cudaErrorLaunchPendingCountExceeded: c_int = 69;
pub const cudaErrorInvalidDeviceFunction: c_int = 98;
pub const cudaErrorNoDevice: c_int = 100;
pub const cudaErrorInvalidDevice: c_int = 101;
pub const cudaErrorDeviceNotLicensed: c_int = 102;
pub const cudaErrorSoftwareValidityNotEstablished: c_int = 103;
pub const cudaErrorStartupFailure: c_int = 127;
pub const cudaErrorInvalidKernelImage: c_int = 200;
pub const cudaErrorDeviceUninitialized: c_int = 201;
pub const cudaErrorMapBufferObjectFailed: c_int = 205;
pub const cudaErrorUnmapBufferObjectFailed: c_int = 206;
pub const cudaErrorArrayIsMapped: c_int = 207;
pub const cudaErrorAlreadyMapped: c_int = 208;
pub const cudaErrorNoKernelImageForDevice: c_int = 209;
pub const cudaErrorAlreadyAcquired: c_int = 210;
pub const cudaErrorNotMapped: c_int = 211;
pub const cudaErrorNotMappedAsArray: c_int = 212;
pub const cudaErrorNotMappedAsPointer: c_int = 213;
pub const cudaErrorECCUncorrectable: c_int = 214;
pub const cudaErrorUnsupportedLimit: c_int = 215;
pub const cudaErrorDeviceAlreadyInUse: c_int = 216;
pub const cudaErrorPeerAccessUnsupported: c_int = 217;
pub const cudaErrorInvalidPtx: c_int = 218;
pub const cudaErrorInvalidGraphicsContext: c_int = 219;
pub const cudaErrorNvlinkUncorrectable: c_int = 220;
pub const cudaErrorJitCompilerNotFound: c_int = 221;
pub const cudaErrorUnsupportedPtxVersion: c_int = 222;
pub const cudaErrorJitCompilationDisabled: c_int = 223;
pub const cudaErrorUnsupportedExecAffinity: c_int = 224;
pub const cudaErrorUnsupportedDevSideSync: c_int = 225;
pub const cudaErrorContained: c_int = 226;
pub const cudaErrorInvalidSource: c_int = 300;
pub const cudaErrorFileNotFound: c_int = 301;
pub const cudaErrorSharedObjectSymbolNotFound: c_int = 302;
pub const cudaErrorSharedObjectInitFailed: c_int = 303;
pub const cudaErrorOperatingSystem: c_int = 304;
pub const cudaErrorInvalidResourceHandle: c_int = 400;
pub const cudaErrorIllegalState: c_int = 401;
pub const cudaErrorLossyQuery: c_int = 402;
pub const cudaErrorSymbolNotFound: c_int = 500;
pub const cudaErrorNotReady: c_int = 600;
pub const cudaErrorIllegalAddress: c_int = 700;
pub const cudaErrorLaunchOutOfResources: c_int = 701;
pub const cudaErrorLaunchTimeout: c_int = 702;
pub const cudaErrorLaunchIncompatibleTexturing: c_int = 703;
pub const cudaErrorPeerAccessAlreadyEnabled: c_int = 704;
pub const cudaErrorPeerAccessNotEnabled: c_int = 705;
pub const cudaErrorSetOnActiveProcess: c_int = 708;
pub const cudaErrorContextIsDestroyed: c_int = 709;
pub const cudaErrorAssert: c_int = 710;
pub const cudaErrorTooManyPeers: c_int = 711;
pub const cudaErrorHostMemoryAlreadyRegistered: c_int = 712;
pub const cudaErrorHostMemoryNotRegistered: c_int = 713;
pub const cudaErrorHardwareStackError: c_int = 714;
pub const cudaErrorIllegalInstruction: c_int = 715;
pub const cudaErrorMisalignedAddress: c_int = 716;
pub const cudaErrorInvalidAddressSpace: c_int = 717;
pub const cudaErrorInvalidPc: c_int = 718;
pub const cudaErrorLaunchFailure: c_int = 719;
pub const cudaErrorCooperativeLaunchTooLarge: c_int = 720;
pub const cudaErrorTensorMemoryLeak: c_int = 721;
pub const cudaErrorNotPermitted: c_int = 800;
pub const cudaErrorNotSupported: c_int = 801;
pub const cudaErrorSystemNotReady: c_int = 802;
pub const cudaErrorSystemDriverMismatch: c_int = 803;
pub const cudaErrorCompatNotSupportedOnDevice: c_int = 804;
pub const cudaErrorMpsConnectionFailed: c_int = 805;
pub const cudaErrorMpsRpcFailure: c_int = 806;
pub const cudaErrorMpsServerNotReady: c_int = 807;
pub const cudaErrorMpsMaxClientsReached: c_int = 808;
pub const cudaErrorMpsMaxConnectionsReached: c_int = 809;
pub const cudaErrorMpsClientTerminated: c_int = 810;
pub const cudaErrorCdpNotSupported: c_int = 811;
pub const cudaErrorCdpVersionMismatch: c_int = 812;
pub const cudaErrorStreamCaptureUnsupported: c_int = 900;
pub const cudaErrorStreamCaptureInvalidated: c_int = 901;
pub const cudaErrorStreamCaptureMerge: c_int = 902;
pub const cudaErrorStreamCaptureUnmatched: c_int = 903;
pub const cudaErrorStreamCaptureUnjoined: c_int = 904;
pub const cudaErrorStreamCaptureIsolation: c_int = 905;
pub const cudaErrorStreamCaptureImplicit: c_int = 906;
pub const cudaErrorCapturedEvent: c_int = 907;
pub const cudaErrorStreamCaptureWrongThread: c_int = 908;
pub const cudaErrorTimeout: c_int = 909;
pub const cudaErrorGraphExecUpdateFailure: c_int = 910;
pub const cudaErrorExternalDevice: c_int = 911;
pub const cudaErrorInvalidClusterSize: c_int = 912;
pub const cudaErrorFunctionNotLoaded: c_int = 913;
pub const cudaErrorInvalidResourceType: c_int = 914;
pub const cudaErrorInvalidResourceConfiguration: c_int = 915;
pub const cudaErrorStreamDetached: c_int = 917;
pub const cudaErrorGraphRecaptureFailure: c_int = 918;
pub const cudaErrorUnknown: c_int = 999;
pub const cudaErrorApiFailureBase: c_int = 10000;
pub const enum_cudaError = c_uint;
pub const cudaChannelFormatKindSigned: c_int = 0;
pub const cudaChannelFormatKindUnsigned: c_int = 1;
pub const cudaChannelFormatKindFloat: c_int = 2;
pub const cudaChannelFormatKindNone: c_int = 3;
pub const cudaChannelFormatKindNV12: c_int = 4;
pub const cudaChannelFormatKindUnsignedNormalized8X1: c_int = 5;
pub const cudaChannelFormatKindUnsignedNormalized8X2: c_int = 6;
pub const cudaChannelFormatKindUnsignedNormalized8X4: c_int = 7;
pub const cudaChannelFormatKindUnsignedNormalized16X1: c_int = 8;
pub const cudaChannelFormatKindUnsignedNormalized16X2: c_int = 9;
pub const cudaChannelFormatKindUnsignedNormalized16X4: c_int = 10;
pub const cudaChannelFormatKindSignedNormalized8X1: c_int = 11;
pub const cudaChannelFormatKindSignedNormalized8X2: c_int = 12;
pub const cudaChannelFormatKindSignedNormalized8X4: c_int = 13;
pub const cudaChannelFormatKindSignedNormalized16X1: c_int = 14;
pub const cudaChannelFormatKindSignedNormalized16X2: c_int = 15;
pub const cudaChannelFormatKindSignedNormalized16X4: c_int = 16;
pub const cudaChannelFormatKindUnsignedBlockCompressed1: c_int = 17;
pub const cudaChannelFormatKindUnsignedBlockCompressed1SRGB: c_int = 18;
pub const cudaChannelFormatKindUnsignedBlockCompressed2: c_int = 19;
pub const cudaChannelFormatKindUnsignedBlockCompressed2SRGB: c_int = 20;
pub const cudaChannelFormatKindUnsignedBlockCompressed3: c_int = 21;
pub const cudaChannelFormatKindUnsignedBlockCompressed3SRGB: c_int = 22;
pub const cudaChannelFormatKindUnsignedBlockCompressed4: c_int = 23;
pub const cudaChannelFormatKindSignedBlockCompressed4: c_int = 24;
pub const cudaChannelFormatKindUnsignedBlockCompressed5: c_int = 25;
pub const cudaChannelFormatKindSignedBlockCompressed5: c_int = 26;
pub const cudaChannelFormatKindUnsignedBlockCompressed6H: c_int = 27;
pub const cudaChannelFormatKindSignedBlockCompressed6H: c_int = 28;
pub const cudaChannelFormatKindUnsignedBlockCompressed7: c_int = 29;
pub const cudaChannelFormatKindUnsignedBlockCompressed7SRGB: c_int = 30;
pub const cudaChannelFormatKindUnsignedNormalized1010102: c_int = 31;
pub const cudaChannelFormatKindUnsigned8Packed422: c_int = 32;
pub const cudaChannelFormatKindUnsigned8Packed444: c_int = 33;
pub const cudaChannelFormatKindUnsigned8SemiPlanar420: c_int = 34;
pub const cudaChannelFormatKindUnsigned16SemiPlanar420: c_int = 35;
pub const cudaChannelFormatKindUnsigned8SemiPlanar422: c_int = 36;
pub const cudaChannelFormatKindUnsigned16SemiPlanar422: c_int = 37;
pub const cudaChannelFormatKindUnsigned8SemiPlanar444: c_int = 38;
pub const cudaChannelFormatKindUnsigned16SemiPlanar444: c_int = 39;
pub const cudaChannelFormatKindUnsigned8Planar420: c_int = 40;
pub const cudaChannelFormatKindUnsigned16Planar420: c_int = 41;
pub const cudaChannelFormatKindUnsigned8Planar422: c_int = 42;
pub const cudaChannelFormatKindUnsigned16Planar422: c_int = 43;
pub const cudaChannelFormatKindUnsigned8Planar444: c_int = 44;
pub const cudaChannelFormatKindUnsigned16Planar444: c_int = 45;
pub const enum_cudaChannelFormatKind = c_uint;
pub const struct_CUdevResourceDesc_st = opaque {};
pub const cudaDevResourceDesc_t = ?*struct_CUdevResourceDesc_st;
pub const struct_cudaExecutionContext_st = opaque {
    pub const cudaExecutionCtxDestroy = __root.cudaExecutionCtxDestroy;
    pub const cudaExecutionCtxGetDevResource = __root.cudaExecutionCtxGetDevResource;
    pub const cudaExecutionCtxGetId = __root.cudaExecutionCtxGetId;
    pub const cudaExecutionCtxSynchronize = __root.cudaExecutionCtxSynchronize;
    pub const cudaExecutionCtxRecordEvent = __root.cudaExecutionCtxRecordEvent;
    pub const cudaExecutionCtxWaitEvent = __root.cudaExecutionCtxWaitEvent;
};
pub const cudaExecutionContext_t = ?*struct_cudaExecutionContext_st;
pub const struct_cudaChannelFormatDesc = extern struct {
    x: c_int = 0,
    y: c_int = 0,
    z: c_int = 0,
    w: c_int = 0,
    f: enum_cudaChannelFormatKind = @import("std").mem.zeroes(enum_cudaChannelFormatKind),
    pub const cudaArrayGetInfo = __root.cudaArrayGetInfo;
    pub const cudaGetChannelDesc = __root.cudaGetChannelDesc;
};
pub const struct_cudaArray = opaque {
    pub const cudaFreeArray = __root.cudaFreeArray;
    pub const cudaMemcpy2DToArray = __root.cudaMemcpy2DToArray;
    pub const cudaMemcpy2DArrayToArray = __root.cudaMemcpy2DArrayToArray;
    pub const cudaMemcpy2DToArrayAsync = __root.cudaMemcpy2DToArrayAsync;
    pub const cudaMemcpyToArray = __root.cudaMemcpyToArray;
    pub const cudaMemcpyArrayToArray = __root.cudaMemcpyArrayToArray;
    pub const cudaMemcpyToArrayAsync = __root.cudaMemcpyToArrayAsync;
};
pub const cudaArray_t = ?*struct_cudaArray;
pub const cudaArray_const_t = ?*const struct_cudaArray;
pub const struct_cudaMipmappedArray = opaque {
    pub const cudaFreeMipmappedArray = __root.cudaFreeMipmappedArray;
};
pub const cudaMipmappedArray_t = ?*struct_cudaMipmappedArray;
pub const cudaMipmappedArray_const_t = ?*const struct_cudaMipmappedArray;
const struct_unnamed_1 = extern struct {
    width: c_uint = 0,
    height: c_uint = 0,
    depth: c_uint = 0,
};
pub const struct_cudaArraySparseProperties = extern struct {
    tileExtent: struct_unnamed_1 = @import("std").mem.zeroes(struct_unnamed_1),
    miptailFirstLevel: c_uint = 0,
    miptailSize: c_ulonglong = 0,
    flags: c_uint = 0,
    reserved: [4]c_uint = @import("std").mem.zeroes([4]c_uint),
    pub const cudaArrayGetSparseProperties = __root.cudaArrayGetSparseProperties;
    pub const cudaMipmappedArrayGetSparseProperties = __root.cudaMipmappedArrayGetSparseProperties;
};
pub const struct_cudaArrayMemoryRequirements = extern struct {
    size: usize = 0,
    alignment: usize = 0,
    reserved: [4]c_uint = @import("std").mem.zeroes([4]c_uint),
    pub const cudaArrayGetMemoryRequirements = __root.cudaArrayGetMemoryRequirements;
    pub const cudaMipmappedArrayGetMemoryRequirements = __root.cudaMipmappedArrayGetMemoryRequirements;
};
pub const cudaMemoryTypeUnregistered: c_int = 0;
pub const cudaMemoryTypeHost: c_int = 1;
pub const cudaMemoryTypeDevice: c_int = 2;
pub const cudaMemoryTypeManaged: c_int = 3;
pub const enum_cudaMemoryType = c_uint;
pub const cudaMemcpyHostToHost: c_int = 0;
pub const cudaMemcpyHostToDevice: c_int = 1;
pub const cudaMemcpyDeviceToHost: c_int = 2;
pub const cudaMemcpyDeviceToDevice: c_int = 3;
pub const cudaMemcpyDefault: c_int = 4;
pub const enum_cudaMemcpyKind = c_uint;
pub const struct_cudaPitchedPtr = extern struct {
    ptr: ?*anyopaque = null,
    pitch: usize = 0,
    xsize: usize = 0,
    ysize: usize = 0,
    pub const cudaMalloc3D = __root.cudaMalloc3D;
    pub const cudaMemset3D = __root.cudaMemset3D;
    pub const cudaMemset3DAsync = __root.cudaMemset3DAsync;
};
pub const struct_cudaExtent = extern struct {
    width: usize = 0,
    height: usize = 0,
    depth: usize = 0,
};
pub const struct_cudaPos = extern struct {
    x: usize = 0,
    y: usize = 0,
    z: usize = 0,
};
pub const struct_cudaMemcpy3DParms = extern struct {
    srcArray: cudaArray_t = null,
    srcPos: struct_cudaPos = @import("std").mem.zeroes(struct_cudaPos),
    srcPtr: struct_cudaPitchedPtr = @import("std").mem.zeroes(struct_cudaPitchedPtr),
    dstArray: cudaArray_t = null,
    dstPos: struct_cudaPos = @import("std").mem.zeroes(struct_cudaPos),
    dstPtr: struct_cudaPitchedPtr = @import("std").mem.zeroes(struct_cudaPitchedPtr),
    extent: struct_cudaExtent = @import("std").mem.zeroes(struct_cudaExtent),
    kind: enum_cudaMemcpyKind = @import("std").mem.zeroes(enum_cudaMemcpyKind),
    pub const cudaMemcpy3D = __root.cudaMemcpy3D;
    pub const cudaMemcpy3DAsync = __root.cudaMemcpy3DAsync;
};
pub const struct_cudaMemcpyNodeParams = extern struct {
    flags: c_int = 0,
    reserved: c_int = 0,
    ctx: cudaExecutionContext_t = null,
    copyParams: struct_cudaMemcpy3DParms = @import("std").mem.zeroes(struct_cudaMemcpy3DParms),
};
pub const struct_cudaMemcpy3DPeerParms = extern struct {
    srcArray: cudaArray_t = null,
    srcPos: struct_cudaPos = @import("std").mem.zeroes(struct_cudaPos),
    srcPtr: struct_cudaPitchedPtr = @import("std").mem.zeroes(struct_cudaPitchedPtr),
    srcDevice: c_int = 0,
    dstArray: cudaArray_t = null,
    dstPos: struct_cudaPos = @import("std").mem.zeroes(struct_cudaPos),
    dstPtr: struct_cudaPitchedPtr = @import("std").mem.zeroes(struct_cudaPitchedPtr),
    dstDevice: c_int = 0,
    extent: struct_cudaExtent = @import("std").mem.zeroes(struct_cudaExtent),
    pub const cudaMemcpy3DPeer = __root.cudaMemcpy3DPeer;
    pub const cudaMemcpy3DPeerAsync = __root.cudaMemcpy3DPeerAsync;
};
pub const struct_cudaMemsetParams = extern struct {
    dst: ?*anyopaque = null,
    pitch: usize = 0,
    value: c_uint = 0,
    elementSize: c_uint = 0,
    width: usize = 0,
    height: usize = 0,
};
pub const struct_cudaMemsetParamsV2 = extern struct {
    dst: ?*anyopaque = null,
    pitch: usize = 0,
    value: c_uint = 0,
    elementSize: c_uint = 0,
    width: usize = 0,
    height: usize = 0,
    ctx: cudaExecutionContext_t = null,
};
pub const cudaAccessPropertyNormal: c_int = 0;
pub const cudaAccessPropertyStreaming: c_int = 1;
pub const cudaAccessPropertyPersisting: c_int = 2;
pub const enum_cudaAccessProperty = c_uint;
pub const struct_cudaAccessPolicyWindow = extern struct {
    base_ptr: ?*anyopaque = null,
    num_bytes: usize = 0,
    hitRatio: f32 = 0,
    hitProp: enum_cudaAccessProperty = @import("std").mem.zeroes(enum_cudaAccessProperty),
    missProp: enum_cudaAccessProperty = @import("std").mem.zeroes(enum_cudaAccessProperty),
};
pub const cudaHostFn_t = ?*const fn (userData: ?*anyopaque) callconv(.c) void;
pub const struct_cudaHostNodeParams = extern struct {
    @"fn": cudaHostFn_t = null,
    userData: ?*anyopaque = null,
};
pub const struct_cudaHostNodeParamsV2 = extern struct {
    @"fn": cudaHostFn_t = null,
    userData: ?*anyopaque = null,
    syncMode: c_uint = 0,
};
pub const cudaStreamCaptureStatusNone: c_int = 0;
pub const cudaStreamCaptureStatusActive: c_int = 1;
pub const cudaStreamCaptureStatusInvalidated: c_int = 2;
pub const enum_cudaStreamCaptureStatus = c_uint;
pub const cudaGraphRecaptureEligibleForUpdate: c_int = 0;
pub const cudaGraphRecaptureIneligibleForUpdate: c_int = 1;
pub const cudaGraphRecaptureError: c_int = 2;
pub const enum_cudaGraphRecaptureStatus = c_uint;
pub const cudaStreamCaptureModeGlobal: c_int = 0;
pub const cudaStreamCaptureModeThreadLocal: c_int = 1;
pub const cudaStreamCaptureModeRelaxed: c_int = 2;
pub const enum_cudaStreamCaptureMode = c_uint;
pub const cudaSyncPolicyAuto: c_int = 1;
pub const cudaSyncPolicySpin: c_int = 2;
pub const cudaSyncPolicyYield: c_int = 3;
pub const cudaSyncPolicyBlockingSync: c_int = 4;
pub const enum_cudaSynchronizationPolicy = c_uint;
pub const cudaClusterSchedulingPolicyDefault: c_int = 0;
pub const cudaClusterSchedulingPolicySpread: c_int = 1;
pub const cudaClusterSchedulingPolicyLoadBalancing: c_int = 2;
pub const enum_cudaClusterSchedulingPolicy = c_uint;
pub const cudaStreamAddCaptureDependencies: c_int = 0;
pub const cudaStreamSetCaptureDependencies: c_int = 1;
pub const enum_cudaStreamUpdateCaptureDependenciesFlags = c_uint;
pub const cudaUserObjectNoDestructorSync: c_int = 1;
pub const enum_cudaUserObjectFlags = c_uint;
pub const cudaGraphUserObjectMove: c_int = 1;
pub const enum_cudaUserObjectRetainFlags = c_uint;
pub const cudaHostTaskBlocking: c_int = 0;
pub const cudaHostTaskSpinWait: c_int = 1;
pub const enum_cudaHostTaskSyncMode = c_uint;
pub const struct_cudaGraphicsResource = opaque {
    pub const cudaGraphicsUnregisterResource = __root.cudaGraphicsUnregisterResource;
    pub const cudaGraphicsResourceSetMapFlags = __root.cudaGraphicsResourceSetMapFlags;
};
pub const cudaGraphicsRegisterFlagsNone: c_int = 0;
pub const cudaGraphicsRegisterFlagsReadOnly: c_int = 1;
pub const cudaGraphicsRegisterFlagsWriteDiscard: c_int = 2;
pub const cudaGraphicsRegisterFlagsSurfaceLoadStore: c_int = 4;
pub const cudaGraphicsRegisterFlagsTextureGather: c_int = 8;
pub const enum_cudaGraphicsRegisterFlags = c_uint;
pub const cudaGraphicsMapFlagsNone: c_int = 0;
pub const cudaGraphicsMapFlagsReadOnly: c_int = 1;
pub const cudaGraphicsMapFlagsWriteDiscard: c_int = 2;
pub const enum_cudaGraphicsMapFlags = c_uint;
pub const cudaGraphicsCubeFacePositiveX: c_int = 0;
pub const cudaGraphicsCubeFaceNegativeX: c_int = 1;
pub const cudaGraphicsCubeFacePositiveY: c_int = 2;
pub const cudaGraphicsCubeFaceNegativeY: c_int = 3;
pub const cudaGraphicsCubeFacePositiveZ: c_int = 4;
pub const cudaGraphicsCubeFaceNegativeZ: c_int = 5;
pub const enum_cudaGraphicsCubeFace = c_uint;
pub const cudaResourceTypeArray: c_int = 0;
pub const cudaResourceTypeMipmappedArray: c_int = 1;
pub const cudaResourceTypeLinear: c_int = 2;
pub const cudaResourceTypePitch2D: c_int = 3;
pub const enum_cudaResourceType = c_uint;
pub const cudaResViewFormatNone: c_int = 0;
pub const cudaResViewFormatUnsignedChar1: c_int = 1;
pub const cudaResViewFormatUnsignedChar2: c_int = 2;
pub const cudaResViewFormatUnsignedChar4: c_int = 3;
pub const cudaResViewFormatSignedChar1: c_int = 4;
pub const cudaResViewFormatSignedChar2: c_int = 5;
pub const cudaResViewFormatSignedChar4: c_int = 6;
pub const cudaResViewFormatUnsignedShort1: c_int = 7;
pub const cudaResViewFormatUnsignedShort2: c_int = 8;
pub const cudaResViewFormatUnsignedShort4: c_int = 9;
pub const cudaResViewFormatSignedShort1: c_int = 10;
pub const cudaResViewFormatSignedShort2: c_int = 11;
pub const cudaResViewFormatSignedShort4: c_int = 12;
pub const cudaResViewFormatUnsignedInt1: c_int = 13;
pub const cudaResViewFormatUnsignedInt2: c_int = 14;
pub const cudaResViewFormatUnsignedInt4: c_int = 15;
pub const cudaResViewFormatSignedInt1: c_int = 16;
pub const cudaResViewFormatSignedInt2: c_int = 17;
pub const cudaResViewFormatSignedInt4: c_int = 18;
pub const cudaResViewFormatHalf1: c_int = 19;
pub const cudaResViewFormatHalf2: c_int = 20;
pub const cudaResViewFormatHalf4: c_int = 21;
pub const cudaResViewFormatFloat1: c_int = 22;
pub const cudaResViewFormatFloat2: c_int = 23;
pub const cudaResViewFormatFloat4: c_int = 24;
pub const cudaResViewFormatUnsignedBlockCompressed1: c_int = 25;
pub const cudaResViewFormatUnsignedBlockCompressed2: c_int = 26;
pub const cudaResViewFormatUnsignedBlockCompressed3: c_int = 27;
pub const cudaResViewFormatUnsignedBlockCompressed4: c_int = 28;
pub const cudaResViewFormatSignedBlockCompressed4: c_int = 29;
pub const cudaResViewFormatUnsignedBlockCompressed5: c_int = 30;
pub const cudaResViewFormatSignedBlockCompressed5: c_int = 31;
pub const cudaResViewFormatUnsignedBlockCompressed6H: c_int = 32;
pub const cudaResViewFormatSignedBlockCompressed6H: c_int = 33;
pub const cudaResViewFormatUnsignedBlockCompressed7: c_int = 34;
pub const enum_cudaResourceViewFormat = c_uint;
const struct_unnamed_3 = extern struct {
    array: cudaArray_t = null,
};
const struct_unnamed_4 = extern struct {
    mipmap: cudaMipmappedArray_t = null,
};
const struct_unnamed_5 = extern struct {
    devPtr: ?*anyopaque = null,
    desc: struct_cudaChannelFormatDesc = @import("std").mem.zeroes(struct_cudaChannelFormatDesc),
    sizeInBytes: usize = 0,
};
const struct_unnamed_6 = extern struct {
    devPtr: ?*anyopaque = null,
    desc: struct_cudaChannelFormatDesc = @import("std").mem.zeroes(struct_cudaChannelFormatDesc),
    width: usize = 0,
    height: usize = 0,
    pitchInBytes: usize = 0,
};
const struct_unnamed_7 = extern struct {
    reserved: [32]c_int = @import("std").mem.zeroes([32]c_int),
};
const union_unnamed_2 = extern union {
    array: struct_unnamed_3,
    mipmap: struct_unnamed_4,
    linear: struct_unnamed_5,
    pitch2D: struct_unnamed_6,
    reserved: struct_unnamed_7,
};
pub const struct_cudaResourceDesc = extern struct {
    resType: enum_cudaResourceType = @import("std").mem.zeroes(enum_cudaResourceType),
    res: union_unnamed_2 = @import("std").mem.zeroes(union_unnamed_2),
    flags: c_uint = 0,
    pub const cudaGetTextureObjectResourceDesc = __root.cudaGetTextureObjectResourceDesc;
    pub const cudaGetSurfaceObjectResourceDesc = __root.cudaGetSurfaceObjectResourceDesc;
};
pub const struct_cudaResourceViewDesc = extern struct {
    format: enum_cudaResourceViewFormat = @import("std").mem.zeroes(enum_cudaResourceViewFormat),
    width: usize = 0,
    height: usize = 0,
    depth: usize = 0,
    firstMipmapLevel: c_uint = 0,
    lastMipmapLevel: c_uint = 0,
    firstLayer: c_uint = 0,
    lastLayer: c_uint = 0,
    reserved: [16]c_uint = @import("std").mem.zeroes([16]c_uint),
    pub const cudaGetTextureObjectResourceViewDesc = __root.cudaGetTextureObjectResourceViewDesc;
};
pub const cudaSharedMemoryModeDefault: c_int = 0;
pub const cudaSharedMemoryModeRequirePortable: c_int = 1;
pub const cudaSharedMemoryModeAllowNonPortable: c_int = 2;
pub const enum_cudaSharedMemoryMode = c_uint;
pub const cudaSharedMemoryMode = enum_cudaSharedMemoryMode;
pub const struct_cudaPointerAttributes = extern struct {
    type: enum_cudaMemoryType = @import("std").mem.zeroes(enum_cudaMemoryType),
    device: c_int = 0,
    devicePointer: ?*anyopaque = null,
    hostPointer: ?*anyopaque = null,
    reserved: [8]c_long = @import("std").mem.zeroes([8]c_long),
    pub const cudaPointerGetAttributes = __root.cudaPointerGetAttributes;
};
pub const struct_cudaFuncAttributes = extern struct {
    sharedSizeBytes: usize = 0,
    constSizeBytes: usize = 0,
    localSizeBytes: usize = 0,
    maxThreadsPerBlock: c_int = 0,
    numRegs: c_int = 0,
    ptxVersion: c_int = 0,
    binaryVersion: c_int = 0,
    cacheModeCA: c_int = 0,
    maxDynamicSharedSizeBytes: c_int = 0,
    preferredShmemCarveout: c_int = 0,
    clusterDimMustBeSet: c_int = 0,
    requiredClusterWidth: c_int = 0,
    requiredClusterHeight: c_int = 0,
    requiredClusterDepth: c_int = 0,
    clusterSchedulingPolicyPreference: c_int = 0,
    nonPortableClusterSizeAllowed: c_int = 0,
    deviceNodeUpdateStatus: c_int = 0,
    reserved1: c_int = 0,
    reserved: [14]c_int = @import("std").mem.zeroes([14]c_int),
    pub const cudaFuncGetAttributes = __root.cudaFuncGetAttributes;
};
pub const cudaFuncAttributeMaxDynamicSharedMemorySize: c_int = 8;
pub const cudaFuncAttributePreferredSharedMemoryCarveout: c_int = 9;
pub const cudaFuncAttributeClusterDimMustBeSet: c_int = 10;
pub const cudaFuncAttributeRequiredClusterWidth: c_int = 11;
pub const cudaFuncAttributeRequiredClusterHeight: c_int = 12;
pub const cudaFuncAttributeRequiredClusterDepth: c_int = 13;
pub const cudaFuncAttributeNonPortableClusterSizeAllowed: c_int = 14;
pub const cudaFuncAttributeClusterSchedulingPolicyPreference: c_int = 15;
pub const cudaFuncAttributeMax: c_int = 16;
pub const enum_cudaFuncAttribute = c_uint;
pub const cudaFuncCachePreferNone: c_int = 0;
pub const cudaFuncCachePreferShared: c_int = 1;
pub const cudaFuncCachePreferL1: c_int = 2;
pub const cudaFuncCachePreferEqual: c_int = 3;
pub const enum_cudaFuncCache = c_uint;
pub const cudaSharedMemBankSizeDefault: c_int = 0;
pub const cudaSharedMemBankSizeFourByte: c_int = 1;
pub const cudaSharedMemBankSizeEightByte: c_int = 2;
pub const enum_cudaSharedMemConfig = c_uint;
pub const cudaSharedmemCarveoutDefault: c_int = -1;
pub const cudaSharedmemCarveoutMaxShared: c_int = 100;
pub const cudaSharedmemCarveoutMaxL1: c_int = 0;
pub const enum_cudaSharedCarveout = c_int;
pub const cudaComputeModeDefault: c_int = 0;
pub const cudaComputeModeExclusive: c_int = 1;
pub const cudaComputeModeProhibited: c_int = 2;
pub const cudaComputeModeExclusiveProcess: c_int = 3;
pub const enum_cudaComputeMode = c_uint;
pub const cudaLimitStackSize: c_int = 0;
pub const cudaLimitPrintfFifoSize: c_int = 1;
pub const cudaLimitMallocHeapSize: c_int = 2;
pub const cudaLimitDevRuntimeSyncDepth: c_int = 3;
pub const cudaLimitDevRuntimePendingLaunchCount: c_int = 4;
pub const cudaLimitMaxL2FetchGranularity: c_int = 5;
pub const cudaLimitPersistingL2CacheSize: c_int = 6;
pub const enum_cudaLimit = c_uint;
pub const cudaMemAdviseSetReadMostly: c_int = 1;
pub const cudaMemAdviseUnsetReadMostly: c_int = 2;
pub const cudaMemAdviseSetPreferredLocation: c_int = 3;
pub const cudaMemAdviseUnsetPreferredLocation: c_int = 4;
pub const cudaMemAdviseSetAccessedBy: c_int = 5;
pub const cudaMemAdviseUnsetAccessedBy: c_int = 6;
pub const enum_cudaMemoryAdvise = c_uint;
pub const cudaMemRangeAttributeReadMostly: c_int = 1;
pub const cudaMemRangeAttributePreferredLocation: c_int = 2;
pub const cudaMemRangeAttributeAccessedBy: c_int = 3;
pub const cudaMemRangeAttributeLastPrefetchLocation: c_int = 4;
pub const cudaMemRangeAttributePreferredLocationType: c_int = 5;
pub const cudaMemRangeAttributePreferredLocationId: c_int = 6;
pub const cudaMemRangeAttributeLastPrefetchLocationType: c_int = 7;
pub const cudaMemRangeAttributeLastPrefetchLocationId: c_int = 8;
pub const enum_cudaMemRangeAttribute = c_uint;
pub const cudaFlushGPUDirectRDMAWritesOptionHost: c_int = 1;
pub const cudaFlushGPUDirectRDMAWritesOptionMemOps: c_int = 2;
pub const enum_cudaFlushGPUDirectRDMAWritesOptions = c_uint;
pub const cudaGPUDirectRDMAWritesOrderingNone: c_int = 0;
pub const cudaGPUDirectRDMAWritesOrderingOwner: c_int = 100;
pub const cudaGPUDirectRDMAWritesOrderingAllDevices: c_int = 200;
pub const enum_cudaGPUDirectRDMAWritesOrdering = c_uint;
pub const cudaFlushGPUDirectRDMAWritesToOwner: c_int = 100;
pub const cudaFlushGPUDirectRDMAWritesToAllDevices: c_int = 200;
pub const enum_cudaFlushGPUDirectRDMAWritesScope = c_uint;
pub const cudaFlushGPUDirectRDMAWritesTargetCurrentDevice: c_int = 0;
pub const enum_cudaFlushGPUDirectRDMAWritesTarget = c_uint;
pub const cudaDevAttrMaxThreadsPerBlock: c_int = 1;
pub const cudaDevAttrMaxBlockDimX: c_int = 2;
pub const cudaDevAttrMaxBlockDimY: c_int = 3;
pub const cudaDevAttrMaxBlockDimZ: c_int = 4;
pub const cudaDevAttrMaxGridDimX: c_int = 5;
pub const cudaDevAttrMaxGridDimY: c_int = 6;
pub const cudaDevAttrMaxGridDimZ: c_int = 7;
pub const cudaDevAttrMaxSharedMemoryPerBlock: c_int = 8;
pub const cudaDevAttrTotalConstantMemory: c_int = 9;
pub const cudaDevAttrWarpSize: c_int = 10;
pub const cudaDevAttrMaxPitch: c_int = 11;
pub const cudaDevAttrMaxRegistersPerBlock: c_int = 12;
pub const cudaDevAttrClockRate: c_int = 13;
pub const cudaDevAttrTextureAlignment: c_int = 14;
pub const cudaDevAttrGpuOverlap: c_int = 15;
pub const cudaDevAttrMultiProcessorCount: c_int = 16;
pub const cudaDevAttrKernelExecTimeout: c_int = 17;
pub const cudaDevAttrIntegrated: c_int = 18;
pub const cudaDevAttrCanMapHostMemory: c_int = 19;
pub const cudaDevAttrComputeMode: c_int = 20;
pub const cudaDevAttrMaxTexture1DWidth: c_int = 21;
pub const cudaDevAttrMaxTexture2DWidth: c_int = 22;
pub const cudaDevAttrMaxTexture2DHeight: c_int = 23;
pub const cudaDevAttrMaxTexture3DWidth: c_int = 24;
pub const cudaDevAttrMaxTexture3DHeight: c_int = 25;
pub const cudaDevAttrMaxTexture3DDepth: c_int = 26;
pub const cudaDevAttrMaxTexture2DLayeredWidth: c_int = 27;
pub const cudaDevAttrMaxTexture2DLayeredHeight: c_int = 28;
pub const cudaDevAttrMaxTexture2DLayeredLayers: c_int = 29;
pub const cudaDevAttrSurfaceAlignment: c_int = 30;
pub const cudaDevAttrConcurrentKernels: c_int = 31;
pub const cudaDevAttrEccEnabled: c_int = 32;
pub const cudaDevAttrPciBusId: c_int = 33;
pub const cudaDevAttrPciDeviceId: c_int = 34;
pub const cudaDevAttrTccDriver: c_int = 35;
pub const cudaDevAttrMemoryClockRate: c_int = 36;
pub const cudaDevAttrGlobalMemoryBusWidth: c_int = 37;
pub const cudaDevAttrL2CacheSize: c_int = 38;
pub const cudaDevAttrMaxThreadsPerMultiProcessor: c_int = 39;
pub const cudaDevAttrAsyncEngineCount: c_int = 40;
pub const cudaDevAttrUnifiedAddressing: c_int = 41;
pub const cudaDevAttrMaxTexture1DLayeredWidth: c_int = 42;
pub const cudaDevAttrMaxTexture1DLayeredLayers: c_int = 43;
pub const cudaDevAttrMaxTexture2DGatherWidth: c_int = 45;
pub const cudaDevAttrMaxTexture2DGatherHeight: c_int = 46;
pub const cudaDevAttrMaxTexture3DWidthAlt: c_int = 47;
pub const cudaDevAttrMaxTexture3DHeightAlt: c_int = 48;
pub const cudaDevAttrMaxTexture3DDepthAlt: c_int = 49;
pub const cudaDevAttrPciDomainId: c_int = 50;
pub const cudaDevAttrTexturePitchAlignment: c_int = 51;
pub const cudaDevAttrMaxTextureCubemapWidth: c_int = 52;
pub const cudaDevAttrMaxTextureCubemapLayeredWidth: c_int = 53;
pub const cudaDevAttrMaxTextureCubemapLayeredLayers: c_int = 54;
pub const cudaDevAttrMaxSurface1DWidth: c_int = 55;
pub const cudaDevAttrMaxSurface2DWidth: c_int = 56;
pub const cudaDevAttrMaxSurface2DHeight: c_int = 57;
pub const cudaDevAttrMaxSurface3DWidth: c_int = 58;
pub const cudaDevAttrMaxSurface3DHeight: c_int = 59;
pub const cudaDevAttrMaxSurface3DDepth: c_int = 60;
pub const cudaDevAttrMaxSurface1DLayeredWidth: c_int = 61;
pub const cudaDevAttrMaxSurface1DLayeredLayers: c_int = 62;
pub const cudaDevAttrMaxSurface2DLayeredWidth: c_int = 63;
pub const cudaDevAttrMaxSurface2DLayeredHeight: c_int = 64;
pub const cudaDevAttrMaxSurface2DLayeredLayers: c_int = 65;
pub const cudaDevAttrMaxSurfaceCubemapWidth: c_int = 66;
pub const cudaDevAttrMaxSurfaceCubemapLayeredWidth: c_int = 67;
pub const cudaDevAttrMaxSurfaceCubemapLayeredLayers: c_int = 68;
pub const cudaDevAttrMaxTexture1DLinearWidth: c_int = 69;
pub const cudaDevAttrMaxTexture2DLinearWidth: c_int = 70;
pub const cudaDevAttrMaxTexture2DLinearHeight: c_int = 71;
pub const cudaDevAttrMaxTexture2DLinearPitch: c_int = 72;
pub const cudaDevAttrMaxTexture2DMipmappedWidth: c_int = 73;
pub const cudaDevAttrMaxTexture2DMipmappedHeight: c_int = 74;
pub const cudaDevAttrComputeCapabilityMajor: c_int = 75;
pub const cudaDevAttrComputeCapabilityMinor: c_int = 76;
pub const cudaDevAttrMaxTexture1DMipmappedWidth: c_int = 77;
pub const cudaDevAttrStreamPrioritiesSupported: c_int = 78;
pub const cudaDevAttrGlobalL1CacheSupported: c_int = 79;
pub const cudaDevAttrLocalL1CacheSupported: c_int = 80;
pub const cudaDevAttrMaxSharedMemoryPerMultiprocessor: c_int = 81;
pub const cudaDevAttrMaxRegistersPerMultiprocessor: c_int = 82;
pub const cudaDevAttrManagedMemory: c_int = 83;
pub const cudaDevAttrIsMultiGpuBoard: c_int = 84;
pub const cudaDevAttrMultiGpuBoardGroupID: c_int = 85;
pub const cudaDevAttrHostNativeAtomicSupported: c_int = 86;
pub const cudaDevAttrSingleToDoublePrecisionPerfRatio: c_int = 87;
pub const cudaDevAttrPageableMemoryAccess: c_int = 88;
pub const cudaDevAttrConcurrentManagedAccess: c_int = 89;
pub const cudaDevAttrComputePreemptionSupported: c_int = 90;
pub const cudaDevAttrCanUseHostPointerForRegisteredMem: c_int = 91;
pub const cudaDevAttrReserved92: c_int = 92;
pub const cudaDevAttrReserved93: c_int = 93;
pub const cudaDevAttrReserved94: c_int = 94;
pub const cudaDevAttrCooperativeLaunch: c_int = 95;
pub const cudaDevAttrReserved96: c_int = 96;
pub const cudaDevAttrMaxSharedMemoryPerBlockOptin: c_int = 97;
pub const cudaDevAttrCanFlushRemoteWrites: c_int = 98;
pub const cudaDevAttrHostRegisterSupported: c_int = 99;
pub const cudaDevAttrPageableMemoryAccessUsesHostPageTables: c_int = 100;
pub const cudaDevAttrDirectManagedMemAccessFromHost: c_int = 101;
pub const cudaDevAttrMaxBlocksPerMultiprocessor: c_int = 106;
pub const cudaDevAttrMaxPersistingL2CacheSize: c_int = 108;
pub const cudaDevAttrMaxAccessPolicyWindowSize: c_int = 109;
pub const cudaDevAttrReservedSharedMemoryPerBlock: c_int = 111;
pub const cudaDevAttrSparseCudaArraySupported: c_int = 112;
pub const cudaDevAttrHostRegisterReadOnlySupported: c_int = 113;
pub const cudaDevAttrTimelineSemaphoreInteropSupported: c_int = 114;
pub const cudaDevAttrMemoryPoolsSupported: c_int = 115;
pub const cudaDevAttrGPUDirectRDMASupported: c_int = 116;
pub const cudaDevAttrGPUDirectRDMAFlushWritesOptions: c_int = 117;
pub const cudaDevAttrGPUDirectRDMAWritesOrdering: c_int = 118;
pub const cudaDevAttrMemoryPoolSupportedHandleTypes: c_int = 119;
pub const cudaDevAttrClusterLaunch: c_int = 120;
pub const cudaDevAttrDeferredMappingCudaArraySupported: c_int = 121;
pub const cudaDevAttrReserved122: c_int = 122;
pub const cudaDevAttrReserved123: c_int = 123;
pub const cudaDevAttrReserved124: c_int = 124;
pub const cudaDevAttrIpcEventSupport: c_int = 125;
pub const cudaDevAttrMemSyncDomainCount: c_int = 126;
pub const cudaDevAttrReserved127: c_int = 127;
pub const cudaDevAttrReserved128: c_int = 128;
pub const cudaDevAttrReserved129: c_int = 129;
pub const cudaDevAttrNumaConfig: c_int = 130;
pub const cudaDevAttrNumaId: c_int = 131;
pub const cudaDevAttrReserved132: c_int = 132;
pub const cudaDevAttrMpsEnabled: c_int = 133;
pub const cudaDevAttrHostNumaId: c_int = 134;
pub const cudaDevAttrD3D12CigSupported: c_int = 135;
pub const cudaDevAttrVulkanCigSupported: c_int = 138;
pub const cudaDevAttrGpuPciDeviceId: c_int = 139;
pub const cudaDevAttrGpuPciSubsystemId: c_int = 140;
pub const cudaDevAttrReserved141: c_int = 141;
pub const cudaDevAttrHostNumaMemoryPoolsSupported: c_int = 142;
pub const cudaDevAttrHostNumaMultinodeIpcSupported: c_int = 143;
pub const cudaDevAttrHostMemoryPoolsSupported: c_int = 144;
pub const cudaDevAttrReserved145: c_int = 145;
pub const cudaDevAttrOnlyPartialHostNativeAtomicSupported: c_int = 147;
pub const cudaDevAttrAtomicReductionSupported: c_int = 148;
pub const cudaDevAttrCigStreamsSupported: c_int = 151;
pub const cudaDevAttrMax: c_int = 152;
pub const enum_cudaDeviceAttr = c_uint;
pub const cudaMemPoolReuseFollowEventDependencies: c_int = 1;
pub const cudaMemPoolReuseAllowOpportunistic: c_int = 2;
pub const cudaMemPoolReuseAllowInternalDependencies: c_int = 3;
pub const cudaMemPoolAttrReleaseThreshold: c_int = 4;
pub const cudaMemPoolAttrReservedMemCurrent: c_int = 5;
pub const cudaMemPoolAttrReservedMemHigh: c_int = 6;
pub const cudaMemPoolAttrUsedMemCurrent: c_int = 7;
pub const cudaMemPoolAttrUsedMemHigh: c_int = 8;
pub const cudaMemPoolAttrAllocationType: c_int = 9;
pub const cudaMemPoolAttrExportHandleTypes: c_int = 10;
pub const cudaMemPoolAttrLocationId: c_int = 11;
pub const cudaMemPoolAttrLocationType: c_int = 12;
pub const cudaMemPoolAttrMaxPoolSize: c_int = 13;
pub const cudaMemPoolAttrHwDecompressEnabled: c_int = 14;
pub const enum_cudaMemPoolAttr = c_uint;
pub const cudaMemLocationTypeInvalid: c_int = 0;
pub const cudaMemLocationTypeNone: c_int = 0;
pub const cudaMemLocationTypeDevice: c_int = 1;
pub const cudaMemLocationTypeHost: c_int = 2;
pub const cudaMemLocationTypeHostNuma: c_int = 3;
pub const cudaMemLocationTypeHostNumaCurrent: c_int = 4;
pub const cudaMemLocationTypeInvisible: c_int = 5;
pub const enum_cudaMemLocationType = c_uint;
const union_unnamed_8 = extern union {
    id: c_int,
};
pub const struct_cudaMemLocation = extern struct {
    type: enum_cudaMemLocationType = @import("std").mem.zeroes(enum_cudaMemLocationType),
    unnamed_0: union_unnamed_8 = @import("std").mem.zeroes(union_unnamed_8),
    pub const cudaMemSetMemPool = __root.cudaMemSetMemPool;
};
pub const cudaMemAccessFlagsProtNone: c_int = 0;
pub const cudaMemAccessFlagsProtRead: c_int = 1;
pub const cudaMemAccessFlagsProtReadWrite: c_int = 3;
pub const enum_cudaMemAccessFlags = c_uint;
pub const struct_cudaMemAccessDesc = extern struct {
    location: struct_cudaMemLocation = @import("std").mem.zeroes(struct_cudaMemLocation),
    flags: enum_cudaMemAccessFlags = @import("std").mem.zeroes(enum_cudaMemAccessFlags),
};
pub const cudaMemAllocationTypeInvalid: c_int = 0;
pub const cudaMemAllocationTypePinned: c_int = 1;
pub const cudaMemAllocationTypeManaged: c_int = 2;
pub const cudaMemAllocationTypeMax: c_int = 2147483647;
pub const enum_cudaMemAllocationType = c_uint;
pub const cudaMemHandleTypeNone: c_int = 0;
pub const cudaMemHandleTypePosixFileDescriptor: c_int = 1;
pub const cudaMemHandleTypeWin32: c_int = 2;
pub const cudaMemHandleTypeWin32Kmt: c_int = 4;
pub const cudaMemHandleTypeFabric: c_int = 8;
pub const enum_cudaMemAllocationHandleType = c_uint;
pub const struct_cudaMemPoolProps = extern struct {
    allocType: enum_cudaMemAllocationType = @import("std").mem.zeroes(enum_cudaMemAllocationType),
    handleTypes: enum_cudaMemAllocationHandleType = @import("std").mem.zeroes(enum_cudaMemAllocationHandleType),
    location: struct_cudaMemLocation = @import("std").mem.zeroes(struct_cudaMemLocation),
    win32SecurityAttributes: ?*anyopaque = null,
    maxSize: usize = 0,
    usage: c_ushort = 0,
    reserved: [54]u8 = @import("std").mem.zeroes([54]u8),
};
pub const struct_cudaMemPoolPtrExportData = extern struct {
    reserved: [64]u8 = @import("std").mem.zeroes([64]u8),
    pub const cudaMemPoolExportPointer = __root.cudaMemPoolExportPointer;
};
pub const struct_cudaMemAllocNodeParams = extern struct {
    poolProps: struct_cudaMemPoolProps = @import("std").mem.zeroes(struct_cudaMemPoolProps),
    accessDescs: [*c]const struct_cudaMemAccessDesc = null,
    accessDescCount: usize = 0,
    bytesize: usize = 0,
    dptr: ?*anyopaque = null,
};
pub const struct_cudaMemAllocNodeParamsV2 = extern struct {
    poolProps: struct_cudaMemPoolProps = @import("std").mem.zeroes(struct_cudaMemPoolProps),
    accessDescs: [*c]const struct_cudaMemAccessDesc = null,
    accessDescCount: usize = 0,
    bytesize: usize = 0,
    dptr: ?*anyopaque = null,
};
pub const struct_cudaMemFreeNodeParams = extern struct {
    dptr: ?*anyopaque = null,
};
pub const cudaGraphMemAttrUsedMemCurrent: c_int = 0;
pub const cudaGraphMemAttrUsedMemHigh: c_int = 1;
pub const cudaGraphMemAttrReservedMemCurrent: c_int = 2;
pub const cudaGraphMemAttrReservedMemHigh: c_int = 3;
pub const enum_cudaGraphMemAttributeType = c_uint;
pub const cudaMemcpyFlagDefault: c_int = 0;
pub const cudaMemcpyFlagPreferOverlapWithCompute: c_int = 1;
pub const enum_cudaMemcpyFlags = c_uint;
pub const cudaMemcpySrcAccessOrderInvalid: c_int = 0;
pub const cudaMemcpySrcAccessOrderStream: c_int = 1;
pub const cudaMemcpySrcAccessOrderDuringApiCall: c_int = 2;
pub const cudaMemcpySrcAccessOrderAny: c_int = 3;
pub const cudaMemcpySrcAccessOrderMax: c_int = 2147483647;
pub const enum_cudaMemcpySrcAccessOrder = c_uint;
pub const struct_cudaMemcpyAttributes = extern struct {
    srcAccessOrder: enum_cudaMemcpySrcAccessOrder = @import("std").mem.zeroes(enum_cudaMemcpySrcAccessOrder),
    srcLocHint: struct_cudaMemLocation = @import("std").mem.zeroes(struct_cudaMemLocation),
    dstLocHint: struct_cudaMemLocation = @import("std").mem.zeroes(struct_cudaMemLocation),
    flags: c_uint = 0,
};
pub const cudaMemcpyOperandTypePointer: c_int = 1;
pub const cudaMemcpyOperandTypeArray: c_int = 2;
pub const cudaMemcpyOperandTypeMax: c_int = 2147483647;
pub const enum_cudaMemcpy3DOperandType = c_uint;
pub const struct_cudaOffset3D = extern struct {
    x: usize = 0,
    y: usize = 0,
    z: usize = 0,
};
const struct_unnamed_10 = extern struct {
    ptr: ?*anyopaque = null,
    rowLength: usize = 0,
    layerHeight: usize = 0,
    locHint: struct_cudaMemLocation = @import("std").mem.zeroes(struct_cudaMemLocation),
};
const struct_unnamed_11 = extern struct {
    array: cudaArray_t = null,
    offset: struct_cudaOffset3D = @import("std").mem.zeroes(struct_cudaOffset3D),
};
const union_unnamed_9 = extern union {
    ptr: struct_unnamed_10,
    array: struct_unnamed_11,
};
pub const struct_cudaMemcpy3DOperand = extern struct {
    type: enum_cudaMemcpy3DOperandType = @import("std").mem.zeroes(enum_cudaMemcpy3DOperandType),
    op: union_unnamed_9 = @import("std").mem.zeroes(union_unnamed_9),
};
pub const struct_cudaMemcpy3DBatchOp = extern struct {
    src: struct_cudaMemcpy3DOperand = @import("std").mem.zeroes(struct_cudaMemcpy3DOperand),
    dst: struct_cudaMemcpy3DOperand = @import("std").mem.zeroes(struct_cudaMemcpy3DOperand),
    extent: struct_cudaExtent = @import("std").mem.zeroes(struct_cudaExtent),
    srcAccessOrder: enum_cudaMemcpySrcAccessOrder = @import("std").mem.zeroes(enum_cudaMemcpySrcAccessOrder),
    flags: c_uint = 0,
    pub const cudaMemcpy3DWithAttributesAsync = __root.cudaMemcpy3DWithAttributesAsync;
};
pub const cudaDevP2PAttrPerformanceRank: c_int = 1;
pub const cudaDevP2PAttrAccessSupported: c_int = 2;
pub const cudaDevP2PAttrNativeAtomicSupported: c_int = 3;
pub const cudaDevP2PAttrCudaArrayAccessSupported: c_int = 4;
pub const cudaDevP2PAttrOnlyPartialNativeAtomicSupported: c_int = 5;
pub const enum_cudaDeviceP2PAttr = c_uint;
pub const cudaAtomicOperationIntegerAdd: c_int = 0;
pub const cudaAtomicOperationIntegerMin: c_int = 1;
pub const cudaAtomicOperationIntegerMax: c_int = 2;
pub const cudaAtomicOperationIntegerIncrement: c_int = 3;
pub const cudaAtomicOperationIntegerDecrement: c_int = 4;
pub const cudaAtomicOperationAnd: c_int = 5;
pub const cudaAtomicOperationOr: c_int = 6;
pub const cudaAtomicOperationXOR: c_int = 7;
pub const cudaAtomicOperationExchange: c_int = 8;
pub const cudaAtomicOperationCAS: c_int = 9;
pub const cudaAtomicOperationFloatAdd: c_int = 10;
pub const cudaAtomicOperationFloatMin: c_int = 11;
pub const cudaAtomicOperationFloatMax: c_int = 12;
pub const enum_cudaAtomicOperation = c_uint;
pub const cudaAtomicCapabilitySigned: c_int = 1;
pub const cudaAtomicCapabilityUnsigned: c_int = 2;
pub const cudaAtomicCapabilityReduction: c_int = 4;
pub const cudaAtomicCapabilityScalar32: c_int = 8;
pub const cudaAtomicCapabilityScalar64: c_int = 16;
pub const cudaAtomicCapabilityScalar128: c_int = 32;
pub const cudaAtomicCapabilityVector32x4: c_int = 64;
pub const enum_cudaAtomicOperationCapability = c_uint;
pub const struct_CUuuid_st = extern struct {
    bytes: [16]u8 = @import("std").mem.zeroes([16]u8),
};
pub const CUuuid = struct_CUuuid_st;
pub const cudaUUID_t = struct_CUuuid_st;
pub const struct_cudaDeviceProp = extern struct {
    name: [256]u8 = @import("std").mem.zeroes([256]u8),
    uuid: cudaUUID_t = @import("std").mem.zeroes(cudaUUID_t),
    luid: [8]u8 = @import("std").mem.zeroes([8]u8),
    luidDeviceNodeMask: c_uint = 0,
    totalGlobalMem: usize = 0,
    sharedMemPerBlock: usize = 0,
    regsPerBlock: c_int = 0,
    warpSize: c_int = 0,
    memPitch: usize = 0,
    maxThreadsPerBlock: c_int = 0,
    maxThreadsDim: [3]c_int = @import("std").mem.zeroes([3]c_int),
    maxGridSize: [3]c_int = @import("std").mem.zeroes([3]c_int),
    totalConstMem: usize = 0,
    major: c_int = 0,
    minor: c_int = 0,
    textureAlignment: usize = 0,
    texturePitchAlignment: usize = 0,
    multiProcessorCount: c_int = 0,
    integrated: c_int = 0,
    canMapHostMemory: c_int = 0,
    maxTexture1D: c_int = 0,
    maxTexture1DMipmap: c_int = 0,
    maxTexture2D: [2]c_int = @import("std").mem.zeroes([2]c_int),
    maxTexture2DMipmap: [2]c_int = @import("std").mem.zeroes([2]c_int),
    maxTexture2DLinear: [3]c_int = @import("std").mem.zeroes([3]c_int),
    maxTexture2DGather: [2]c_int = @import("std").mem.zeroes([2]c_int),
    maxTexture3D: [3]c_int = @import("std").mem.zeroes([3]c_int),
    maxTexture3DAlt: [3]c_int = @import("std").mem.zeroes([3]c_int),
    maxTextureCubemap: c_int = 0,
    maxTexture1DLayered: [2]c_int = @import("std").mem.zeroes([2]c_int),
    maxTexture2DLayered: [3]c_int = @import("std").mem.zeroes([3]c_int),
    maxTextureCubemapLayered: [2]c_int = @import("std").mem.zeroes([2]c_int),
    maxSurface1D: c_int = 0,
    maxSurface2D: [2]c_int = @import("std").mem.zeroes([2]c_int),
    maxSurface3D: [3]c_int = @import("std").mem.zeroes([3]c_int),
    maxSurface1DLayered: [2]c_int = @import("std").mem.zeroes([2]c_int),
    maxSurface2DLayered: [3]c_int = @import("std").mem.zeroes([3]c_int),
    maxSurfaceCubemap: c_int = 0,
    maxSurfaceCubemapLayered: [2]c_int = @import("std").mem.zeroes([2]c_int),
    surfaceAlignment: usize = 0,
    concurrentKernels: c_int = 0,
    ECCEnabled: c_int = 0,
    pciBusID: c_int = 0,
    pciDeviceID: c_int = 0,
    pciDomainID: c_int = 0,
    tccDriver: c_int = 0,
    asyncEngineCount: c_int = 0,
    unifiedAddressing: c_int = 0,
    memoryBusWidth: c_int = 0,
    l2CacheSize: c_int = 0,
    persistingL2CacheMaxSize: c_int = 0,
    maxThreadsPerMultiProcessor: c_int = 0,
    streamPrioritiesSupported: c_int = 0,
    globalL1CacheSupported: c_int = 0,
    localL1CacheSupported: c_int = 0,
    sharedMemPerMultiprocessor: usize = 0,
    regsPerMultiprocessor: c_int = 0,
    managedMemory: c_int = 0,
    isMultiGpuBoard: c_int = 0,
    multiGpuBoardGroupID: c_int = 0,
    hostNativeAtomicSupported: c_int = 0,
    pageableMemoryAccess: c_int = 0,
    concurrentManagedAccess: c_int = 0,
    computePreemptionSupported: c_int = 0,
    canUseHostPointerForRegisteredMem: c_int = 0,
    cooperativeLaunch: c_int = 0,
    sharedMemPerBlockOptin: usize = 0,
    pageableMemoryAccessUsesHostPageTables: c_int = 0,
    directManagedMemAccessFromHost: c_int = 0,
    maxBlocksPerMultiProcessor: c_int = 0,
    accessPolicyMaxWindowSize: c_int = 0,
    reservedSharedMemPerBlock: usize = 0,
    hostRegisterSupported: c_int = 0,
    sparseCudaArraySupported: c_int = 0,
    hostRegisterReadOnlySupported: c_int = 0,
    timelineSemaphoreInteropSupported: c_int = 0,
    memoryPoolsSupported: c_int = 0,
    gpuDirectRDMASupported: c_int = 0,
    gpuDirectRDMAFlushWritesOptions: c_uint = 0,
    gpuDirectRDMAWritesOrdering: c_int = 0,
    memoryPoolSupportedHandleTypes: c_uint = 0,
    deferredMappingCudaArraySupported: c_int = 0,
    ipcEventSupported: c_int = 0,
    clusterLaunch: c_int = 0,
    unifiedFunctionPointers: c_int = 0,
    deviceNumaConfig: c_int = 0,
    deviceNumaId: c_int = 0,
    mpsEnabled: c_int = 0,
    hostNumaId: c_int = 0,
    gpuPciDeviceID: c_uint = 0,
    gpuPciSubsystemID: c_uint = 0,
    hostNumaMultinodeIpcSupported: c_int = 0,
    reserved: [56]c_int = @import("std").mem.zeroes([56]c_int),
    pub const cudaGetDeviceProperties = __root.cudaGetDeviceProperties;
};
pub const struct_cudaIpcEventHandle_st = extern struct {
    reserved: [64]u8 = @import("std").mem.zeroes([64]u8),
    pub const cudaIpcGetEventHandle = __root.cudaIpcGetEventHandle;
};
pub const cudaIpcEventHandle_t = struct_cudaIpcEventHandle_st;
pub const struct_cudaIpcMemHandle_st = extern struct {
    reserved: [64]u8 = @import("std").mem.zeroes([64]u8),
    pub const cudaIpcGetMemHandle = __root.cudaIpcGetMemHandle;
};
pub const cudaIpcMemHandle_t = struct_cudaIpcMemHandle_st;
pub const struct_cudaMemFabricHandle_st = extern struct {
    reserved: [64]u8 = @import("std").mem.zeroes([64]u8),
};
pub const cudaMemFabricHandle_t = struct_cudaMemFabricHandle_st;
pub const cudaExternalMemoryHandleTypeOpaqueFd: c_int = 1;
pub const cudaExternalMemoryHandleTypeOpaqueWin32: c_int = 2;
pub const cudaExternalMemoryHandleTypeOpaqueWin32Kmt: c_int = 3;
pub const cudaExternalMemoryHandleTypeD3D12Heap: c_int = 4;
pub const cudaExternalMemoryHandleTypeD3D12Resource: c_int = 5;
pub const cudaExternalMemoryHandleTypeD3D11Resource: c_int = 6;
pub const cudaExternalMemoryHandleTypeD3D11ResourceKmt: c_int = 7;
pub const cudaExternalMemoryHandleTypeNvSciBuf: c_int = 8;
pub const enum_cudaExternalMemoryHandleType = c_uint;
const struct_unnamed_13 = extern struct {
    handle: ?*anyopaque = null,
    name: ?*const anyopaque = null,
};
const union_unnamed_12 = extern union {
    fd: c_int,
    win32: struct_unnamed_13,
    nvSciBufObject: ?*const anyopaque,
};
pub const struct_cudaExternalMemoryHandleDesc = extern struct {
    type: enum_cudaExternalMemoryHandleType = @import("std").mem.zeroes(enum_cudaExternalMemoryHandleType),
    handle: union_unnamed_12 = @import("std").mem.zeroes(union_unnamed_12),
    size: c_ulonglong = 0,
    flags: c_uint = 0,
    reserved: [16]c_uint = @import("std").mem.zeroes([16]c_uint),
};
pub const struct_cudaExternalMemoryBufferDesc = extern struct {
    offset: c_ulonglong = 0,
    size: c_ulonglong = 0,
    flags: c_uint = 0,
    reserved: [16]c_uint = @import("std").mem.zeroes([16]c_uint),
};
pub const struct_cudaExternalMemoryMipmappedArrayDesc = extern struct {
    offset: c_ulonglong = 0,
    formatDesc: struct_cudaChannelFormatDesc = @import("std").mem.zeroes(struct_cudaChannelFormatDesc),
    extent: struct_cudaExtent = @import("std").mem.zeroes(struct_cudaExtent),
    flags: c_uint = 0,
    numLevels: c_uint = 0,
    reserved: [16]c_uint = @import("std").mem.zeroes([16]c_uint),
};
pub const cudaExternalSemaphoreHandleTypeOpaqueFd: c_int = 1;
pub const cudaExternalSemaphoreHandleTypeOpaqueWin32: c_int = 2;
pub const cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt: c_int = 3;
pub const cudaExternalSemaphoreHandleTypeD3D12Fence: c_int = 4;
pub const cudaExternalSemaphoreHandleTypeD3D11Fence: c_int = 5;
pub const cudaExternalSemaphoreHandleTypeNvSciSync: c_int = 6;
pub const cudaExternalSemaphoreHandleTypeKeyedMutex: c_int = 7;
pub const cudaExternalSemaphoreHandleTypeKeyedMutexKmt: c_int = 8;
pub const cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd: c_int = 9;
pub const cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32: c_int = 10;
pub const enum_cudaExternalSemaphoreHandleType = c_uint;
const struct_unnamed_15 = extern struct {
    handle: ?*anyopaque = null,
    name: ?*const anyopaque = null,
};
const union_unnamed_14 = extern union {
    fd: c_int,
    win32: struct_unnamed_15,
    nvSciSyncObj: ?*const anyopaque,
};
pub const struct_cudaExternalSemaphoreHandleDesc = extern struct {
    type: enum_cudaExternalSemaphoreHandleType = @import("std").mem.zeroes(enum_cudaExternalSemaphoreHandleType),
    handle: union_unnamed_14 = @import("std").mem.zeroes(union_unnamed_14),
    flags: c_uint = 0,
    reserved: [16]c_uint = @import("std").mem.zeroes([16]c_uint),
};
const struct_unnamed_17 = extern struct {
    value: c_ulonglong = 0,
};
const union_unnamed_18 = extern union {
    fence: ?*anyopaque,
    reserved: c_ulonglong,
};
const struct_unnamed_19 = extern struct {
    key: c_ulonglong = 0,
};
const struct_unnamed_16 = extern struct {
    fence: struct_unnamed_17 = @import("std").mem.zeroes(struct_unnamed_17),
    nvSciSync: union_unnamed_18 = @import("std").mem.zeroes(union_unnamed_18),
    keyedMutex: struct_unnamed_19 = @import("std").mem.zeroes(struct_unnamed_19),
    reserved: [12]c_uint = @import("std").mem.zeroes([12]c_uint),
};
pub const struct_cudaExternalSemaphoreSignalParams = extern struct {
    params: struct_unnamed_16 = @import("std").mem.zeroes(struct_unnamed_16),
    flags: c_uint = 0,
    reserved: [16]c_uint = @import("std").mem.zeroes([16]c_uint),
};
const struct_unnamed_21 = extern struct {
    value: c_ulonglong = 0,
};
const union_unnamed_22 = extern union {
    fence: ?*anyopaque,
    reserved: c_ulonglong,
};
const struct_unnamed_23 = extern struct {
    key: c_ulonglong = 0,
    timeoutMs: c_uint = 0,
};
const struct_unnamed_20 = extern struct {
    fence: struct_unnamed_21 = @import("std").mem.zeroes(struct_unnamed_21),
    nvSciSync: union_unnamed_22 = @import("std").mem.zeroes(union_unnamed_22),
    keyedMutex: struct_unnamed_23 = @import("std").mem.zeroes(struct_unnamed_23),
    reserved: [10]c_uint = @import("std").mem.zeroes([10]c_uint),
};
pub const struct_cudaExternalSemaphoreWaitParams = extern struct {
    params: struct_unnamed_20 = @import("std").mem.zeroes(struct_unnamed_20),
    flags: c_uint = 0,
    reserved: [16]c_uint = @import("std").mem.zeroes([16]c_uint),
};
pub const cudaDevSmResourceGroupDefault: c_int = 0;
pub const cudaDevSmResourceGroupBackfill: c_int = 1;
pub const enum_cudaDevSmResourceGroup_flags = c_uint;
pub const cudaDevSmResourceSplitIgnoreSmCoscheduling: c_int = 1;
pub const cudaDevSmResourceSplitMaxPotentialClusterSize: c_int = 2;
pub const enum_cudaDevSmResourceSplitByCount_flags = c_uint;
pub const cudaDevResourceTypeInvalid: c_int = 0;
pub const cudaDevResourceTypeSm: c_int = 1;
pub const cudaDevResourceTypeWorkqueueConfig: c_int = 1000;
pub const cudaDevResourceTypeWorkqueue: c_int = 10000;
pub const enum_cudaDevResourceType = c_uint;
pub const struct_cudaDevSmResource = extern struct {
    smCount: c_uint = 0,
    minSmPartitionSize: c_uint = 0,
    smCoscheduledAlignment: c_uint = 0,
    flags: c_uint = 0,
};
pub const cudaDevWorkqueueConfigScopeDeviceCtx: c_int = 0;
pub const cudaDevWorkqueueConfigScopeGreenCtxBalanced: c_int = 1;
pub const enum_cudaDevWorkqueueConfigScope = c_uint;
pub const struct_cudaDevWorkqueueConfigResource = extern struct {
    device: c_int = 0,
    wqConcurrencyLimit: c_uint = 0,
    sharingScope: enum_cudaDevWorkqueueConfigScope = @import("std").mem.zeroes(enum_cudaDevWorkqueueConfigScope),
};
pub const struct_cudaDevWorkqueueResource = extern struct {
    reserved: [40]u8 = @import("std").mem.zeroes([40]u8),
};
pub const struct_cudaDevSmResourceGroupParams_st = extern struct {
    smCount: c_uint = 0,
    coscheduledSmCount: c_uint = 0,
    preferredCoscheduledSmCount: c_uint = 0,
    flags: c_uint = 0,
    reserved: [12]c_uint = @import("std").mem.zeroes([12]c_uint),
};
pub const cudaDevSmResourceGroupParams = struct_cudaDevSmResourceGroupParams_st;
const union_unnamed_24 = extern union {
    sm: struct_cudaDevSmResource,
    wqConfig: struct_cudaDevWorkqueueConfigResource,
    wq: struct_cudaDevWorkqueueResource,
    _oversize: [40]u8,
};
pub const struct_cudaDevResource_st = extern struct {
    type: enum_cudaDevResourceType = @import("std").mem.zeroes(enum_cudaDevResourceType),
    _internal_padding: [92]u8 = @import("std").mem.zeroes([92]u8),
    unnamed_0: union_unnamed_24 = @import("std").mem.zeroes(union_unnamed_24),
    nextResource: [*c]struct_cudaDevResource_st = null,
    pub const cudaDevSmResourceSplitByCount = __root.cudaDevSmResourceSplitByCount;
    pub const cudaDevSmResourceSplit = __root.cudaDevSmResourceSplit;
};
pub const cudaDevResource = struct_cudaDevResource_st;
pub const cudaError_t = enum_cudaError;
pub const struct_CUstream_st = opaque {
    pub const cudaStreamGetPriority = __root.cudaStreamGetPriority;
    pub const cudaStreamGetFlags = __root.cudaStreamGetFlags;
    pub const cudaStreamGetId = __root.cudaStreamGetId;
    pub const cudaStreamGetDevice = __root.cudaStreamGetDevice;
    pub const cudaStreamCopyAttributes = __root.cudaStreamCopyAttributes;
    pub const cudaStreamGetAttribute = __root.cudaStreamGetAttribute;
    pub const cudaStreamSetAttribute = __root.cudaStreamSetAttribute;
    pub const cudaStreamDestroy = __root.cudaStreamDestroy;
    pub const cudaStreamWaitEvent = __root.cudaStreamWaitEvent;
    pub const cudaStreamAddCallback = __root.cudaStreamAddCallback;
    pub const cudaStreamSynchronize = __root.cudaStreamSynchronize;
    pub const cudaStreamQuery = __root.cudaStreamQuery;
    pub const cudaStreamAttachMemAsync = __root.cudaStreamAttachMemAsync;
    pub const cudaStreamBeginCapture = __root.cudaStreamBeginCapture;
    pub const cudaStreamBeginRecaptureToGraph = __root.cudaStreamBeginRecaptureToGraph;
    pub const cudaStreamBeginCaptureToGraph = __root.cudaStreamBeginCaptureToGraph;
    pub const cudaStreamEndCapture = __root.cudaStreamEndCapture;
    pub const cudaStreamIsCapturing = __root.cudaStreamIsCapturing;
    pub const cudaStreamGetCaptureInfo = __root.cudaStreamGetCaptureInfo;
    pub const cudaStreamUpdateCaptureDependencies = __root.cudaStreamUpdateCaptureDependencies;
    pub const cudaLaunchHostFunc = __root.cudaLaunchHostFunc;
    pub const cudaLaunchHostFunc_v2 = __root.cudaLaunchHostFunc_v2;
    pub const cudaStreamGetDevResource = __root.cudaStreamGetDevResource;
    pub const v2 = __root.cudaLaunchHostFunc_v2;
};
pub const cudaStream_t = ?*struct_CUstream_st;
pub const struct_CUevent_st = opaque {
    pub const cudaEventRecord = __root.cudaEventRecord;
    pub const cudaEventRecordWithFlags = __root.cudaEventRecordWithFlags;
    pub const cudaEventQuery = __root.cudaEventQuery;
    pub const cudaEventSynchronize = __root.cudaEventSynchronize;
    pub const cudaEventDestroy = __root.cudaEventDestroy;
};
pub const cudaEvent_t = ?*struct_CUevent_st;
pub const cudaGraphicsResource_t = ?*struct_cudaGraphicsResource;
pub const struct_CUexternalMemory_st = opaque {
    pub const cudaDestroyExternalMemory = __root.cudaDestroyExternalMemory;
};
pub const cudaExternalMemory_t = ?*struct_CUexternalMemory_st;
pub const struct_CUexternalSemaphore_st = opaque {
    pub const cudaDestroyExternalSemaphore = __root.cudaDestroyExternalSemaphore;
};
pub const cudaExternalSemaphore_t = ?*struct_CUexternalSemaphore_st;
pub const struct_CUgraph_st = opaque {
    pub const cudaGraphGetId = __root.cudaGraphGetId;
    pub const cudaGraphGetNodes = __root.cudaGraphGetNodes;
    pub const cudaGraphGetRootNodes = __root.cudaGraphGetRootNodes;
    pub const cudaGraphGetEdges = __root.cudaGraphGetEdges;
    pub const cudaGraphAddDependencies = __root.cudaGraphAddDependencies;
    pub const cudaGraphRemoveDependencies = __root.cudaGraphRemoveDependencies;
    pub const cudaGraphDestroy = __root.cudaGraphDestroy;
    pub const cudaGraphDebugDotPrint = __root.cudaGraphDebugDotPrint;
    pub const cudaGraphRetainUserObject = __root.cudaGraphRetainUserObject;
    pub const cudaGraphReleaseUserObject = __root.cudaGraphReleaseUserObject;
};
pub const cudaGraph_t = ?*struct_CUgraph_st;
pub const struct_CUgraphNode_st = opaque {
    pub const cudaGraphKernelNodeGetParams = __root.cudaGraphKernelNodeGetParams;
    pub const cudaGraphKernelNodeSetParams = __root.cudaGraphKernelNodeSetParams;
    pub const cudaGraphKernelNodeCopyAttributes = __root.cudaGraphKernelNodeCopyAttributes;
    pub const cudaGraphKernelNodeGetAttribute = __root.cudaGraphKernelNodeGetAttribute;
    pub const cudaGraphKernelNodeSetAttribute = __root.cudaGraphKernelNodeSetAttribute;
    pub const cudaGraphMemcpyNodeGetParams = __root.cudaGraphMemcpyNodeGetParams;
    pub const cudaGraphMemcpyNodeSetParams = __root.cudaGraphMemcpyNodeSetParams;
    pub const cudaGraphMemcpyNodeSetParamsToSymbol = __root.cudaGraphMemcpyNodeSetParamsToSymbol;
    pub const cudaGraphMemcpyNodeSetParamsFromSymbol = __root.cudaGraphMemcpyNodeSetParamsFromSymbol;
    pub const cudaGraphMemcpyNodeSetParams1D = __root.cudaGraphMemcpyNodeSetParams1D;
    pub const cudaGraphMemsetNodeGetParams = __root.cudaGraphMemsetNodeGetParams;
    pub const cudaGraphMemsetNodeSetParams = __root.cudaGraphMemsetNodeSetParams;
    pub const cudaGraphHostNodeGetParams = __root.cudaGraphHostNodeGetParams;
    pub const cudaGraphHostNodeSetParams = __root.cudaGraphHostNodeSetParams;
    pub const cudaGraphChildGraphNodeGetGraph = __root.cudaGraphChildGraphNodeGetGraph;
    pub const cudaGraphEventRecordNodeGetEvent = __root.cudaGraphEventRecordNodeGetEvent;
    pub const cudaGraphEventRecordNodeSetEvent = __root.cudaGraphEventRecordNodeSetEvent;
    pub const cudaGraphEventWaitNodeGetEvent = __root.cudaGraphEventWaitNodeGetEvent;
    pub const cudaGraphEventWaitNodeSetEvent = __root.cudaGraphEventWaitNodeSetEvent;
    pub const cudaGraphExternalSemaphoresSignalNodeGetParams = __root.cudaGraphExternalSemaphoresSignalNodeGetParams;
    pub const cudaGraphExternalSemaphoresSignalNodeSetParams = __root.cudaGraphExternalSemaphoresSignalNodeSetParams;
    pub const cudaGraphExternalSemaphoresWaitNodeGetParams = __root.cudaGraphExternalSemaphoresWaitNodeGetParams;
    pub const cudaGraphExternalSemaphoresWaitNodeSetParams = __root.cudaGraphExternalSemaphoresWaitNodeSetParams;
    pub const cudaGraphMemAllocNodeGetParams = __root.cudaGraphMemAllocNodeGetParams;
    pub const cudaGraphMemFreeNodeGetParams = __root.cudaGraphMemFreeNodeGetParams;
    pub const cudaGraphNodeGetType = __root.cudaGraphNodeGetType;
    pub const cudaGraphNodeGetContainingGraph = __root.cudaGraphNodeGetContainingGraph;
    pub const cudaGraphNodeGetLocalId = __root.cudaGraphNodeGetLocalId;
    pub const cudaGraphNodeGetToolsId = __root.cudaGraphNodeGetToolsId;
    pub const cudaGraphNodeGetDependencies = __root.cudaGraphNodeGetDependencies;
    pub const cudaGraphNodeGetDependentNodes = __root.cudaGraphNodeGetDependentNodes;
    pub const cudaGraphDestroyNode = __root.cudaGraphDestroyNode;
    pub const cudaGraphNodeSetParams = __root.cudaGraphNodeSetParams;
    pub const cudaGraphNodeGetParams = __root.cudaGraphNodeGetParams;
};
pub const cudaGraphNode_t = ?*struct_CUgraphNode_st;
pub const struct_CUuserObject_st = opaque {
    pub const cudaUserObjectRetain = __root.cudaUserObjectRetain;
    pub const cudaUserObjectRelease = __root.cudaUserObjectRelease;
};
pub const cudaUserObject_t = ?*struct_CUuserObject_st;
pub const cudaGraphConditionalHandle = c_ulonglong;
pub const struct_CUfunc_st = opaque {};
pub const cudaFunction_t = ?*struct_CUfunc_st;
pub const struct_CUkern_st = opaque {
    pub const cudaKernelSetAttributeForDevice = __root.cudaKernelSetAttributeForDevice;
};
pub const cudaKernel_t = ?*struct_CUkern_st;
pub const cudaJitMaxRegisters: c_int = 0;
pub const cudaJitThreadsPerBlock: c_int = 1;
pub const cudaJitWallTime: c_int = 2;
pub const cudaJitInfoLogBuffer: c_int = 3;
pub const cudaJitInfoLogBufferSizeBytes: c_int = 4;
pub const cudaJitErrorLogBuffer: c_int = 5;
pub const cudaJitErrorLogBufferSizeBytes: c_int = 6;
pub const cudaJitOptimizationLevel: c_int = 7;
pub const cudaJitFallbackStrategy: c_int = 10;
pub const cudaJitGenerateDebugInfo: c_int = 11;
pub const cudaJitLogVerbose: c_int = 12;
pub const cudaJitGenerateLineInfo: c_int = 13;
pub const cudaJitCacheMode: c_int = 14;
pub const cudaJitPositionIndependentCode: c_int = 30;
pub const cudaJitMinCtaPerSm: c_int = 31;
pub const cudaJitMaxThreadsPerBlock: c_int = 32;
pub const cudaJitOverrideDirectiveValues: c_int = 33;
pub const enum_cudaJitOption = c_uint;
pub const cudaLibraryHostUniversalFunctionAndDataTable: c_int = 0;
pub const cudaLibraryBinaryIsPreserved: c_int = 1;
pub const enum_cudaLibraryOption = c_uint;
pub const struct_cudalibraryHostUniversalFunctionAndDataTable = extern struct {
    functionTable: ?*anyopaque = null,
    functionWindowSize: usize = 0,
    dataTable: ?*anyopaque = null,
    dataWindowSize: usize = 0,
};
pub const cudaJitCacheOptionNone: c_int = 0;
pub const cudaJitCacheOptionCG: c_int = 1;
pub const cudaJitCacheOptionCA: c_int = 2;
pub const enum_cudaJit_CacheMode = c_uint;
pub const cudaPreferPtx: c_int = 0;
pub const cudaPreferBinary: c_int = 1;
pub const enum_cudaJit_Fallback = c_uint;
pub const struct_CUlib_st = opaque {
    pub const cudaLibraryUnload = __root.cudaLibraryUnload;
};
pub const cudaLibrary_t = ?*struct_CUlib_st;
pub const struct_CUmemPoolHandle_st = opaque {
    pub const cudaMemPoolTrimTo = __root.cudaMemPoolTrimTo;
    pub const cudaMemPoolSetAttribute = __root.cudaMemPoolSetAttribute;
    pub const cudaMemPoolGetAttribute = __root.cudaMemPoolGetAttribute;
    pub const cudaMemPoolSetAccess = __root.cudaMemPoolSetAccess;
    pub const cudaMemPoolDestroy = __root.cudaMemPoolDestroy;
};
pub const cudaMemPool_t = ?*struct_CUmemPoolHandle_st;
pub const cudaCGScopeInvalid: c_int = 0;
pub const cudaCGScopeGrid: c_int = 1;
pub const cudaCGScopeReserved: c_int = 2;
pub const enum_cudaCGScope = c_uint;
pub const struct_cudaKernelNodeParams = extern struct {
    func: ?*anyopaque = null,
    gridDim: dim3 = @import("std").mem.zeroes(dim3),
    blockDim: dim3 = @import("std").mem.zeroes(dim3),
    sharedMemBytes: c_uint = 0,
    kernelParams: [*c]?*anyopaque = null,
    extra: [*c]?*anyopaque = null,
};
pub const cudaKernelFunctionTypeUnspecified: c_int = 0;
pub const cudaKernelFunctionTypeDeviceEntry: c_int = 1;
pub const cudaKernelFunctionTypeKernel: c_int = 2;
pub const cudaKernelFunctionTypeFunction: c_int = 3;
pub const enum_cudaKernelFunctionType = c_uint;
const union_unnamed_25 = extern union {
    func: ?*anyopaque,
    kern: cudaKernel_t,
    cuFunc: cudaFunction_t,
};
pub const struct_cudaKernelNodeParamsV2 = extern struct {
    unnamed_0: union_unnamed_25 = @import("std").mem.zeroes(union_unnamed_25),
    gridDim: dim3 = @import("std").mem.zeroes(dim3),
    blockDim: dim3 = @import("std").mem.zeroes(dim3),
    sharedMemBytes: c_uint = 0,
    kernelParams: [*c]?*anyopaque = null,
    extra: [*c]?*anyopaque = null,
    ctx: cudaExecutionContext_t = null,
    functionType: enum_cudaKernelFunctionType = @import("std").mem.zeroes(enum_cudaKernelFunctionType),
};
pub const struct_cudaExternalSemaphoreSignalNodeParams = extern struct {
    extSemArray: [*c]cudaExternalSemaphore_t = null,
    paramsArray: [*c]const struct_cudaExternalSemaphoreSignalParams = null,
    numExtSems: c_uint = 0,
};
pub const struct_cudaExternalSemaphoreSignalNodeParamsV2 = extern struct {
    extSemArray: [*c]cudaExternalSemaphore_t = null,
    paramsArray: [*c]const struct_cudaExternalSemaphoreSignalParams = null,
    numExtSems: c_uint = 0,
};
pub const struct_cudaExternalSemaphoreWaitNodeParams = extern struct {
    extSemArray: [*c]cudaExternalSemaphore_t = null,
    paramsArray: [*c]const struct_cudaExternalSemaphoreWaitParams = null,
    numExtSems: c_uint = 0,
};
pub const struct_cudaExternalSemaphoreWaitNodeParamsV2 = extern struct {
    extSemArray: [*c]cudaExternalSemaphore_t = null,
    paramsArray: [*c]const struct_cudaExternalSemaphoreWaitParams = null,
    numExtSems: c_uint = 0,
};
pub const cudaGraphCondAssignDefault: c_int = 1;
pub const enum_cudaGraphConditionalHandleFlags = c_uint;
pub const cudaGraphCondTypeIf: c_int = 0;
pub const cudaGraphCondTypeWhile: c_int = 1;
pub const cudaGraphCondTypeSwitch: c_int = 2;
pub const enum_cudaGraphConditionalNodeType = c_uint;
pub const struct_cudaConditionalNodeParams = extern struct {
    handle: cudaGraphConditionalHandle = 0,
    type: enum_cudaGraphConditionalNodeType = @import("std").mem.zeroes(enum_cudaGraphConditionalNodeType),
    size: c_uint = 0,
    phGraph_out: [*c]cudaGraph_t = null,
    ctx: cudaExecutionContext_t = null,
};
pub const cudaGraphNodeTypeKernel: c_int = 0;
pub const cudaGraphNodeTypeMemcpy: c_int = 1;
pub const cudaGraphNodeTypeMemset: c_int = 2;
pub const cudaGraphNodeTypeHost: c_int = 3;
pub const cudaGraphNodeTypeGraph: c_int = 4;
pub const cudaGraphNodeTypeEmpty: c_int = 5;
pub const cudaGraphNodeTypeWaitEvent: c_int = 6;
pub const cudaGraphNodeTypeEventRecord: c_int = 7;
pub const cudaGraphNodeTypeExtSemaphoreSignal: c_int = 8;
pub const cudaGraphNodeTypeExtSemaphoreWait: c_int = 9;
pub const cudaGraphNodeTypeMemAlloc: c_int = 10;
pub const cudaGraphNodeTypeMemFree: c_int = 11;
pub const cudaGraphNodeTypeConditional: c_int = 13;
pub const cudaGraphNodeTypeReserved16: c_int = 16;
pub const cudaGraphNodeTypeCount: c_int = 17;
pub const enum_cudaGraphNodeType = c_uint;
pub const cudaGraphChildGraphOwnershipClone: c_int = 0;
pub const cudaGraphChildGraphOwnershipMove: c_int = 1;
pub const cudaGraphChildGraphOwnershipInvalid: c_int = -1;
pub const enum_cudaGraphChildGraphNodeOwnership = c_int;
pub const struct_cudaChildGraphNodeParams = extern struct {
    graph: cudaGraph_t = null,
    ownership: enum_cudaGraphChildGraphNodeOwnership = @import("std").mem.zeroes(enum_cudaGraphChildGraphNodeOwnership),
};
pub const struct_cudaEventRecordNodeParams = extern struct {
    event: cudaEvent_t = null,
};
pub const struct_cudaEventWaitNodeParams = extern struct {
    event: cudaEvent_t = null,
};
const union_unnamed_26 = extern union {
    reserved1: [29]c_longlong,
    kernel: struct_cudaKernelNodeParamsV2,
    memcpy: struct_cudaMemcpyNodeParams,
    memset: struct_cudaMemsetParamsV2,
    host: struct_cudaHostNodeParamsV2,
    graph: struct_cudaChildGraphNodeParams,
    eventWait: struct_cudaEventWaitNodeParams,
    eventRecord: struct_cudaEventRecordNodeParams,
    extSemSignal: struct_cudaExternalSemaphoreSignalNodeParamsV2,
    extSemWait: struct_cudaExternalSemaphoreWaitNodeParamsV2,
    alloc: struct_cudaMemAllocNodeParamsV2,
    free: struct_cudaMemFreeNodeParams,
    conditional: struct_cudaConditionalNodeParams,
};
pub const struct_cudaGraphNodeParams = extern struct {
    type: enum_cudaGraphNodeType = @import("std").mem.zeroes(enum_cudaGraphNodeType),
    reserved0: [3]c_int = @import("std").mem.zeroes([3]c_int),
    unnamed_0: union_unnamed_26 = @import("std").mem.zeroes(union_unnamed_26),
    reserved2: c_longlong = 0,
};
pub const cudaGraphDependencyTypeDefault: c_int = 0;
pub const cudaGraphDependencyTypeProgrammatic: c_int = 1;
pub const enum_cudaGraphDependencyType_enum = c_uint;
pub const cudaGraphDependencyType = enum_cudaGraphDependencyType_enum;
pub const struct_cudaGraphEdgeData_st = extern struct {
    from_port: u8 = 0,
    to_port: u8 = 0,
    type: u8 = 0,
    reserved: [5]u8 = @import("std").mem.zeroes([5]u8),
};
pub const cudaGraphEdgeData = struct_cudaGraphEdgeData_st;
pub const struct_CUgraphExec_st = opaque {
    pub const cudaGraphExecGetId = __root.cudaGraphExecGetId;
    pub const cudaGraphExecGetFlags = __root.cudaGraphExecGetFlags;
    pub const cudaGraphExecKernelNodeSetParams = __root.cudaGraphExecKernelNodeSetParams;
    pub const cudaGraphExecMemcpyNodeSetParams = __root.cudaGraphExecMemcpyNodeSetParams;
    pub const cudaGraphExecMemcpyNodeSetParamsToSymbol = __root.cudaGraphExecMemcpyNodeSetParamsToSymbol;
    pub const cudaGraphExecMemcpyNodeSetParamsFromSymbol = __root.cudaGraphExecMemcpyNodeSetParamsFromSymbol;
    pub const cudaGraphExecMemcpyNodeSetParams1D = __root.cudaGraphExecMemcpyNodeSetParams1D;
    pub const cudaGraphExecMemsetNodeSetParams = __root.cudaGraphExecMemsetNodeSetParams;
    pub const cudaGraphExecHostNodeSetParams = __root.cudaGraphExecHostNodeSetParams;
    pub const cudaGraphExecChildGraphNodeSetParams = __root.cudaGraphExecChildGraphNodeSetParams;
    pub const cudaGraphExecEventRecordNodeSetEvent = __root.cudaGraphExecEventRecordNodeSetEvent;
    pub const cudaGraphExecEventWaitNodeSetEvent = __root.cudaGraphExecEventWaitNodeSetEvent;
    pub const cudaGraphExecExternalSemaphoresSignalNodeSetParams = __root.cudaGraphExecExternalSemaphoresSignalNodeSetParams;
    pub const cudaGraphExecExternalSemaphoresWaitNodeSetParams = __root.cudaGraphExecExternalSemaphoresWaitNodeSetParams;
    pub const cudaGraphNodeSetEnabled = __root.cudaGraphNodeSetEnabled;
    pub const cudaGraphNodeGetEnabled = __root.cudaGraphNodeGetEnabled;
    pub const cudaGraphExecUpdate = __root.cudaGraphExecUpdate;
    pub const cudaGraphUpload = __root.cudaGraphUpload;
    pub const cudaGraphLaunch = __root.cudaGraphLaunch;
    pub const cudaGraphExecDestroy = __root.cudaGraphExecDestroy;
    pub const cudaGraphExecNodeSetParams = __root.cudaGraphExecNodeSetParams;
};
pub const cudaGraphExec_t = ?*struct_CUgraphExec_st;
pub const cudaGraphExecUpdateSuccess: c_int = 0;
pub const cudaGraphExecUpdateError: c_int = 1;
pub const cudaGraphExecUpdateErrorTopologyChanged: c_int = 2;
pub const cudaGraphExecUpdateErrorNodeTypeChanged: c_int = 3;
pub const cudaGraphExecUpdateErrorFunctionChanged: c_int = 4;
pub const cudaGraphExecUpdateErrorParametersChanged: c_int = 5;
pub const cudaGraphExecUpdateErrorNotSupported: c_int = 6;
pub const cudaGraphExecUpdateErrorUnsupportedFunctionChange: c_int = 7;
pub const cudaGraphExecUpdateErrorAttributesChanged: c_int = 8;
pub const enum_cudaGraphExecUpdateResult = c_uint;
pub const cudaGraphInstantiateSuccess: c_int = 0;
pub const cudaGraphInstantiateError: c_int = 1;
pub const cudaGraphInstantiateInvalidStructure: c_int = 2;
pub const cudaGraphInstantiateNodeOperationNotSupported: c_int = 3;
pub const cudaGraphInstantiateMultipleDevicesNotSupported: c_int = 4;
pub const cudaGraphInstantiateConditionalHandleUnused: c_int = 5;
pub const enum_cudaGraphInstantiateResult = c_uint;
pub const cudaGraphInstantiateResult = enum_cudaGraphInstantiateResult;
pub const struct_cudaGraphInstantiateParams_st = extern struct {
    flags: c_ulonglong = 0,
    uploadStream: cudaStream_t = null,
    errNode_out: cudaGraphNode_t = null,
    result_out: cudaGraphInstantiateResult = @import("std").mem.zeroes(cudaGraphInstantiateResult),
};
pub const cudaGraphInstantiateParams = struct_cudaGraphInstantiateParams_st;
pub const struct_cudaGraphExecUpdateResultInfo_st = extern struct {
    result: enum_cudaGraphExecUpdateResult = @import("std").mem.zeroes(enum_cudaGraphExecUpdateResult),
    errorNode: cudaGraphNode_t = null,
    errorFromNode: cudaGraphNode_t = null,
};
pub const cudaGraphExecUpdateResultInfo = struct_cudaGraphExecUpdateResultInfo_st;
pub const struct_CUgraphDeviceUpdatableNode_st = opaque {};
pub const cudaGraphDeviceNode_t = ?*struct_CUgraphDeviceUpdatableNode_st;
pub const cudaGraphKernelNodeFieldInvalid: c_int = 0;
pub const cudaGraphKernelNodeFieldGridDim: c_int = 1;
pub const cudaGraphKernelNodeFieldParam: c_int = 2;
pub const cudaGraphKernelNodeFieldEnabled: c_int = 3;
pub const enum_cudaGraphKernelNodeField = c_uint;
const struct_unnamed_28 = extern struct {
    pValue: ?*const anyopaque = null,
    offset: usize = 0,
    size: usize = 0,
};
const union_unnamed_27 = extern union {
    gridDim: dim3,
    param: struct_unnamed_28,
    isEnabled: c_uint,
};
pub const struct_cudaGraphKernelNodeUpdate = extern struct {
    node: cudaGraphDeviceNode_t = null,
    field: enum_cudaGraphKernelNodeField = @import("std").mem.zeroes(enum_cudaGraphKernelNodeField),
    updateData: union_unnamed_27 = @import("std").mem.zeroes(union_unnamed_27),
};
pub const cudaEnableDefault: c_int = 0;
pub const cudaEnableLegacyStream: c_int = 1;
pub const cudaEnablePerThreadDefaultStream: c_int = 2;
pub const enum_cudaGetDriverEntryPointFlags = c_uint;
pub const cudaDriverEntryPointSuccess: c_int = 0;
pub const cudaDriverEntryPointSymbolNotFound: c_int = 1;
pub const cudaDriverEntryPointVersionNotSufficent: c_int = 2;
pub const enum_cudaDriverEntryPointQueryResult = c_uint;
pub const cudaGraphDebugDotFlagsVerbose: c_int = 1;
pub const cudaGraphDebugDotFlagsKernelNodeParams: c_int = 4;
pub const cudaGraphDebugDotFlagsMemcpyNodeParams: c_int = 8;
pub const cudaGraphDebugDotFlagsMemsetNodeParams: c_int = 16;
pub const cudaGraphDebugDotFlagsHostNodeParams: c_int = 32;
pub const cudaGraphDebugDotFlagsEventNodeParams: c_int = 64;
pub const cudaGraphDebugDotFlagsExtSemasSignalNodeParams: c_int = 128;
pub const cudaGraphDebugDotFlagsExtSemasWaitNodeParams: c_int = 256;
pub const cudaGraphDebugDotFlagsKernelNodeAttributes: c_int = 512;
pub const cudaGraphDebugDotFlagsHandles: c_int = 1024;
pub const cudaGraphDebugDotFlagsConditionalNodeParams: c_int = 32768;
pub const enum_cudaGraphDebugDotFlags = c_uint;
pub const cudaGraphInstantiateFlagAutoFreeOnLaunch: c_int = 1;
pub const cudaGraphInstantiateFlagUpload: c_int = 2;
pub const cudaGraphInstantiateFlagDeviceLaunch: c_int = 4;
pub const cudaGraphInstantiateFlagUseNodePriority: c_int = 8;
pub const enum_cudaGraphInstantiateFlags = c_uint;
pub const cudaLaunchMemSyncDomainDefault: c_int = 0;
pub const cudaLaunchMemSyncDomainRemote: c_int = 1;
pub const enum_cudaLaunchMemSyncDomain = c_uint;
pub const cudaLaunchMemSyncDomain = enum_cudaLaunchMemSyncDomain;
pub const struct_cudaLaunchMemSyncDomainMap_st = extern struct {
    default_: u8 = 0,
    remote: u8 = 0,
};
pub const cudaLaunchMemSyncDomainMap = struct_cudaLaunchMemSyncDomainMap_st;
pub const cudaLaunchPortableClusterModeDefault: c_int = 0;
pub const cudaLaunchPortableClusterModeRequirePortable: c_int = 1;
pub const cudaLaunchPortableClusterModeAllowNonPortable: c_int = 2;
pub const enum_cudaLaunchAttributePortableClusterMode = c_uint;
pub const cudaLaunchAttributePortableClusterMode = enum_cudaLaunchAttributePortableClusterMode;
pub const cudaLaunchAttributeIgnore: c_int = 0;
pub const cudaLaunchAttributeAccessPolicyWindow: c_int = 1;
pub const cudaLaunchAttributeCooperative: c_int = 2;
pub const cudaLaunchAttributeSynchronizationPolicy: c_int = 3;
pub const cudaLaunchAttributeClusterDimension: c_int = 4;
pub const cudaLaunchAttributeClusterSchedulingPolicyPreference: c_int = 5;
pub const cudaLaunchAttributeProgrammaticStreamSerialization: c_int = 6;
pub const cudaLaunchAttributeProgrammaticEvent: c_int = 7;
pub const cudaLaunchAttributePriority: c_int = 8;
pub const cudaLaunchAttributeMemSyncDomainMap: c_int = 9;
pub const cudaLaunchAttributeMemSyncDomain: c_int = 10;
pub const cudaLaunchAttributePreferredClusterDimension: c_int = 11;
pub const cudaLaunchAttributeLaunchCompletionEvent: c_int = 12;
pub const cudaLaunchAttributeDeviceUpdatableKernelNode: c_int = 13;
pub const cudaLaunchAttributePreferredSharedMemoryCarveout: c_int = 14;
pub const cudaLaunchAttributeNvlinkUtilCentricScheduling: c_int = 16;
pub const cudaLaunchAttributePortableClusterSizeMode: c_int = 17;
pub const cudaLaunchAttributeSharedMemoryMode: c_int = 18;
pub const enum_cudaLaunchAttributeID = c_uint;
pub const cudaLaunchAttributeID = enum_cudaLaunchAttributeID;
const struct_unnamed_29 = extern struct {
    x: c_uint = 0,
    y: c_uint = 0,
    z: c_uint = 0,
};
const struct_unnamed_30 = extern struct {
    event: cudaEvent_t = null,
    flags: c_int = 0,
    triggerAtBlockStart: c_int = 0,
};
const struct_unnamed_31 = extern struct {
    x: c_uint = 0,
    y: c_uint = 0,
    z: c_uint = 0,
};
const struct_unnamed_32 = extern struct {
    event: cudaEvent_t = null,
    flags: c_int = 0,
};
const struct_unnamed_33 = extern struct {
    deviceUpdatable: c_int = 0,
    devNode: cudaGraphDeviceNode_t = null,
};
pub const union_cudaLaunchAttributeValue = extern union {
    pad: [64]u8,
    accessPolicyWindow: struct_cudaAccessPolicyWindow,
    cooperative: c_int,
    syncPolicy: enum_cudaSynchronizationPolicy,
    clusterDim: struct_unnamed_29,
    clusterSchedulingPolicyPreference: enum_cudaClusterSchedulingPolicy,
    programmaticStreamSerializationAllowed: c_int,
    programmaticEvent: struct_unnamed_30,
    priority: c_int,
    memSyncDomainMap: cudaLaunchMemSyncDomainMap,
    memSyncDomain: cudaLaunchMemSyncDomain,
    preferredClusterDim: struct_unnamed_31,
    launchCompletionEvent: struct_unnamed_32,
    deviceUpdatableKernelNode: struct_unnamed_33,
    sharedMemCarveout: c_uint,
    nvlinkUtilCentricScheduling: c_uint,
    portableClusterSizeMode: cudaLaunchAttributePortableClusterMode,
    sharedMemoryMode: cudaSharedMemoryMode,
};
pub const cudaLaunchAttributeValue = union_cudaLaunchAttributeValue;
pub const struct_cudaLaunchAttribute_st = extern struct {
    id: cudaLaunchAttributeID = @import("std").mem.zeroes(cudaLaunchAttributeID),
    pad: [4]u8 = @import("std").mem.zeroes([4]u8),
    val: cudaLaunchAttributeValue = @import("std").mem.zeroes(cudaLaunchAttributeValue),
};
pub const cudaLaunchAttribute = struct_cudaLaunchAttribute_st;
pub const struct_cudaLaunchConfig_st = extern struct {
    gridDim: dim3 = @import("std").mem.zeroes(dim3),
    blockDim: dim3 = @import("std").mem.zeroes(dim3),
    dynamicSmemBytes: usize = 0,
    stream: cudaStream_t = null,
    attrs: [*c]cudaLaunchAttribute = null,
    numAttrs: c_uint = 0,
    pub const cudaLaunchKernelExC = __root.cudaLaunchKernelExC;
};
pub const cudaLaunchConfig_t = struct_cudaLaunchConfig_st;
pub const cudaDeviceNumaConfigNone: c_int = 0;
pub const cudaDeviceNumaConfigNumaNode: c_int = 1;
pub const enum_cudaDeviceNumaConfig = c_uint;
pub const struct_cudaAsyncCallbackEntry = opaque {};
pub const cudaAsyncCallbackHandle_t = ?*struct_cudaAsyncCallbackEntry;
pub const cudaAsyncNotificationTypeOverBudget: c_int = 1;
pub const enum_cudaAsyncNotificationType_enum = c_uint;
pub const cudaAsyncNotificationType = enum_cudaAsyncNotificationType_enum;
const struct_unnamed_35 = extern struct {
    bytesOverBudget: c_ulonglong = 0,
};
const union_unnamed_34 = extern union {
    overBudget: struct_unnamed_35,
};
pub const struct_cudaAsyncNotificationInfo = extern struct {
    type: cudaAsyncNotificationType = @import("std").mem.zeroes(cudaAsyncNotificationType),
    info: union_unnamed_34 = @import("std").mem.zeroes(union_unnamed_34),
};
pub const cudaAsyncNotificationInfo_t = struct_cudaAsyncNotificationInfo;
pub const cudaAsyncCallback = ?*const fn ([*c]cudaAsyncNotificationInfo_t, ?*anyopaque, cudaAsyncCallbackHandle_t) callconv(.c) void;
pub const cudaLogLevelError: c_int = 0;
pub const cudaLogLevelWarning: c_int = 1;
pub const enum_CUDAlogLevel_enum = c_uint;
pub const cudaLogLevel = enum_CUDAlogLevel_enum;
pub const struct_CUlogsCallbackEntry_st = opaque {
    pub const cudaLogsUnregisterCallback = __root.cudaLogsUnregisterCallback;
};
pub const cudaLogsCallbackHandle = ?*struct_CUlogsCallbackEntry_st;
pub const cudaLogIterator = c_uint;
pub const cudaFabricOpStatusSourceMbarrierV1: c_int = 0;
pub const cudaFabricOpStatusSourceMax: c_int = 2147483647;
pub const enum_cudaFabricOpStatusSource = c_uint;
pub const cudaFabricOpStatusInfoSuccess: c_int = 0;
pub const cudaFabricOpStatusInfoLast: c_int = 0;
pub const cudaFabricOpStatusInfoMax: c_int = 2147483647;
pub const enum_cudaFabricOpStatusInfo = c_uint;
pub const cudaBoundaryModeZero: c_int = 0;
pub const cudaBoundaryModeClamp: c_int = 1;
pub const cudaBoundaryModeTrap: c_int = 2;
pub const enum_cudaSurfaceBoundaryMode = c_uint;
pub const cudaFormatModeForced: c_int = 0;
pub const cudaFormatModeAuto: c_int = 1;
pub const enum_cudaSurfaceFormatMode = c_uint;
pub const cudaSurfaceObject_t = c_ulonglong;
pub const cudaAddressModeWrap: c_int = 0;
pub const cudaAddressModeClamp: c_int = 1;
pub const cudaAddressModeMirror: c_int = 2;
pub const cudaAddressModeBorder: c_int = 3;
pub const enum_cudaTextureAddressMode = c_uint;
pub const cudaFilterModePoint: c_int = 0;
pub const cudaFilterModeLinear: c_int = 1;
pub const enum_cudaTextureFilterMode = c_uint;
pub const cudaReadModeElementType: c_int = 0;
pub const cudaReadModeNormalizedFloat: c_int = 1;
pub const enum_cudaTextureReadMode = c_uint;
pub const struct_cudaTextureDesc = extern struct {
    addressMode: [3]enum_cudaTextureAddressMode = @import("std").mem.zeroes([3]enum_cudaTextureAddressMode),
    filterMode: enum_cudaTextureFilterMode = @import("std").mem.zeroes(enum_cudaTextureFilterMode),
    readMode: enum_cudaTextureReadMode = @import("std").mem.zeroes(enum_cudaTextureReadMode),
    sRGB: c_int = 0,
    borderColor: [4]f32 = @import("std").mem.zeroes([4]f32),
    normalizedCoords: c_int = 0,
    maxAnisotropy: c_uint = 0,
    mipmapFilterMode: enum_cudaTextureFilterMode = @import("std").mem.zeroes(enum_cudaTextureFilterMode),
    mipmapLevelBias: f32 = 0,
    minMipmapLevelClamp: f32 = 0,
    maxMipmapLevelClamp: f32 = 0,
    disableTrilinearOptimization: c_int = 0,
    seamlessCubemap: c_int = 0,
    pub const cudaGetTextureObjectTextureDesc = __root.cudaGetTextureObjectTextureDesc;
};
pub const cudaTextureObject_t = c_ulonglong;
pub extern fn cudaDeviceReset() cudaError_t;
pub extern fn cudaDeviceSynchronize() cudaError_t;
pub extern fn cudaDeviceSetLimit(limit: enum_cudaLimit, value: usize) cudaError_t;
pub extern fn cudaDeviceGetLimit(pValue: [*c]usize, limit: enum_cudaLimit) cudaError_t;
pub extern fn cudaDeviceGetTexture1DLinearMaxWidth(maxWidthInElements: [*c]usize, fmtDesc: [*c]const struct_cudaChannelFormatDesc, device: c_int) cudaError_t;
pub extern fn cudaDeviceGetCacheConfig(pCacheConfig: [*c]enum_cudaFuncCache) cudaError_t;
pub extern fn cudaDeviceGetStreamPriorityRange(leastPriority: [*c]c_int, greatestPriority: [*c]c_int) cudaError_t;
pub extern fn cudaDeviceSetCacheConfig(cacheConfig: enum_cudaFuncCache) cudaError_t;
pub extern fn cudaDeviceGetByPCIBusId(device: [*c]c_int, pciBusId: [*c]const u8) cudaError_t;
pub extern fn cudaDeviceGetPCIBusId(pciBusId: [*c]u8, len: c_int, device: c_int) cudaError_t;
pub extern fn cudaIpcGetEventHandle(handle: [*c]cudaIpcEventHandle_t, event: cudaEvent_t) cudaError_t;
pub extern fn cudaIpcOpenEventHandle(event: [*c]cudaEvent_t, handle: cudaIpcEventHandle_t) cudaError_t;
pub extern fn cudaIpcGetMemHandle(handle: [*c]cudaIpcMemHandle_t, devPtr: ?*anyopaque) cudaError_t;
pub extern fn cudaIpcOpenMemHandle(devPtr: [*c]?*anyopaque, handle: cudaIpcMemHandle_t, flags: c_uint) cudaError_t;
pub extern fn cudaIpcCloseMemHandle(devPtr: ?*anyopaque) cudaError_t;
pub extern fn cudaDeviceFlushGPUDirectRDMAWrites(target: enum_cudaFlushGPUDirectRDMAWritesTarget, scope: enum_cudaFlushGPUDirectRDMAWritesScope) cudaError_t;
pub extern fn cudaDeviceRegisterAsyncNotification(device: c_int, callbackFunc: cudaAsyncCallback, userData: ?*anyopaque, callback: [*c]cudaAsyncCallbackHandle_t) cudaError_t;
pub extern fn cudaDeviceUnregisterAsyncNotification(device: c_int, callback: cudaAsyncCallbackHandle_t) cudaError_t;
pub extern fn cudaDeviceGetSharedMemConfig(pConfig: [*c]enum_cudaSharedMemConfig) cudaError_t;
pub extern fn cudaDeviceSetSharedMemConfig(config: enum_cudaSharedMemConfig) cudaError_t;
pub extern fn cudaGetLastError() cudaError_t;
pub extern fn cudaPeekAtLastError() cudaError_t;
pub extern fn cudaGetErrorName(@"error": cudaError_t) [*c]const u8;
pub extern fn cudaGetErrorString(@"error": cudaError_t) [*c]const u8;
pub extern fn cudaGetDeviceCount(count: [*c]c_int) cudaError_t;
pub extern fn cudaGetDeviceProperties(prop: [*c]struct_cudaDeviceProp, device: c_int) cudaError_t;
pub extern fn cudaDeviceGetAttribute(value: [*c]c_int, attr: enum_cudaDeviceAttr, device: c_int) cudaError_t;
pub extern fn cudaDeviceGetHostAtomicCapabilities(capabilities: [*c]c_uint, operations: [*c]const enum_cudaAtomicOperation, count: c_uint, device: c_int) cudaError_t;
pub extern fn cudaDeviceGetDefaultMemPool(memPool: [*c]cudaMemPool_t, device: c_int) cudaError_t;
pub extern fn cudaDeviceSetMemPool(device: c_int, memPool: cudaMemPool_t) cudaError_t;
pub extern fn cudaDeviceGetMemPool(memPool: [*c]cudaMemPool_t, device: c_int) cudaError_t;
pub extern fn cudaDeviceGetNvSciSyncAttributes(nvSciSyncAttrList: ?*anyopaque, device: c_int, flags: c_int) cudaError_t;
pub extern fn cudaDeviceGetP2PAttribute(value: [*c]c_int, attr: enum_cudaDeviceP2PAttr, srcDevice: c_int, dstDevice: c_int) cudaError_t;
pub extern fn cudaDeviceGetP2PAtomicCapabilities(capabilities: [*c]c_uint, operations: [*c]const enum_cudaAtomicOperation, count: c_uint, srcDevice: c_int, dstDevice: c_int) cudaError_t;
pub extern fn cudaChooseDevice(device: [*c]c_int, prop: [*c]const struct_cudaDeviceProp) cudaError_t;
pub extern fn cudaInitDevice(device: c_int, deviceFlags: c_uint, flags: c_uint) cudaError_t;
pub extern fn cudaSetDevice(device: c_int) cudaError_t;
pub extern fn cudaGetDevice(device: [*c]c_int) cudaError_t;
pub extern fn cudaSetValidDevices(device_arr: [*c]c_int, len: c_int) cudaError_t;
pub extern fn cudaSetDeviceFlags(flags: c_uint) cudaError_t;
pub extern fn cudaGetDeviceFlags(flags: [*c]c_uint) cudaError_t;
pub extern fn cudaStreamCreate(pStream: [*c]cudaStream_t) cudaError_t;
pub extern fn cudaStreamCreateWithFlags(pStream: [*c]cudaStream_t, flags: c_uint) cudaError_t;
pub extern fn cudaStreamCreateWithPriority(pStream: [*c]cudaStream_t, flags: c_uint, priority: c_int) cudaError_t;
pub extern fn cudaStreamGetPriority(hStream: cudaStream_t, priority: [*c]c_int) cudaError_t;
pub extern fn cudaStreamGetFlags(hStream: cudaStream_t, flags: [*c]c_uint) cudaError_t;
pub extern fn cudaStreamGetId(hStream: cudaStream_t, streamId: [*c]c_ulonglong) cudaError_t;
pub extern fn cudaStreamGetDevice(hStream: cudaStream_t, device: [*c]c_int) cudaError_t;
pub extern fn cudaCtxResetPersistingL2Cache() cudaError_t;
pub extern fn cudaStreamCopyAttributes(dst: cudaStream_t, src: cudaStream_t) cudaError_t;
pub extern fn cudaStreamGetAttribute(hStream: cudaStream_t, attr: cudaLaunchAttributeID, value_out: [*c]cudaLaunchAttributeValue) cudaError_t;
pub extern fn cudaStreamSetAttribute(hStream: cudaStream_t, attr: cudaLaunchAttributeID, value: [*c]const cudaLaunchAttributeValue) cudaError_t;
pub extern fn cudaStreamDestroy(stream: cudaStream_t) cudaError_t;
pub extern fn cudaStreamWaitEvent(stream: cudaStream_t, event: cudaEvent_t, flags: c_uint) cudaError_t;
pub const cudaStreamCallback_t = ?*const fn (stream: cudaStream_t, status: cudaError_t, userData: ?*anyopaque) callconv(.c) void;
pub extern fn cudaStreamAddCallback(stream: cudaStream_t, callback: cudaStreamCallback_t, userData: ?*anyopaque, flags: c_uint) cudaError_t;
pub extern fn cudaStreamSynchronize(stream: cudaStream_t) cudaError_t;
pub extern fn cudaStreamQuery(stream: cudaStream_t) cudaError_t;
pub extern fn cudaStreamAttachMemAsync(stream: cudaStream_t, devPtr: ?*anyopaque, length: usize, flags: c_uint) cudaError_t;
pub extern fn cudaStreamBeginCapture(stream: cudaStream_t, mode: enum_cudaStreamCaptureMode) cudaError_t;
pub const cudaGraphRecaptureCallback_t = ?*const fn (data: ?*anyopaque, node: cudaGraphNode_t, originalParams: [*c]const struct_cudaGraphNodeParams, recaptureParams: [*c]const struct_cudaGraphNodeParams, status: enum_cudaGraphRecaptureStatus) callconv(.c) cudaError_t;
pub const struct_cudaGraphRecaptureCallbackData = extern struct {
    callbackFunc: cudaGraphRecaptureCallback_t = null,
    userData: ?*anyopaque = null,
};
pub extern fn cudaStreamBeginRecaptureToGraph(stream: cudaStream_t, mode: enum_cudaStreamCaptureMode, graph: cudaGraph_t, callbackData: [*c]struct_cudaGraphRecaptureCallbackData) cudaError_t;
pub extern fn cudaStreamBeginCaptureToGraph(stream: cudaStream_t, graph: cudaGraph_t, dependencies: [*c]const cudaGraphNode_t, dependencyData: [*c]const cudaGraphEdgeData, numDependencies: usize, mode: enum_cudaStreamCaptureMode) cudaError_t;
pub extern fn cudaThreadExchangeStreamCaptureMode(mode: [*c]enum_cudaStreamCaptureMode) cudaError_t;
pub extern fn cudaStreamEndCapture(stream: cudaStream_t, pGraph: [*c]cudaGraph_t) cudaError_t;
pub extern fn cudaStreamIsCapturing(stream: cudaStream_t, pCaptureStatus: [*c]enum_cudaStreamCaptureStatus) cudaError_t;
pub extern fn cudaStreamGetCaptureInfo(stream: cudaStream_t, captureStatus_out: [*c]enum_cudaStreamCaptureStatus, id_out: [*c]c_ulonglong, graph_out: [*c]cudaGraph_t, dependencies_out: [*c][*c]const cudaGraphNode_t, edgeData_out: [*c][*c]const cudaGraphEdgeData, numDependencies_out: [*c]usize) cudaError_t;
pub extern fn cudaStreamUpdateCaptureDependencies(stream: cudaStream_t, dependencies: [*c]cudaGraphNode_t, dependencyData: [*c]const cudaGraphEdgeData, numDependencies: usize, flags: c_uint) cudaError_t;
pub extern fn cudaEventCreate(event: [*c]cudaEvent_t) cudaError_t;
pub extern fn cudaEventCreateWithFlags(event: [*c]cudaEvent_t, flags: c_uint) cudaError_t;
pub extern fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) cudaError_t;
pub extern fn cudaEventRecordWithFlags(event: cudaEvent_t, stream: cudaStream_t, flags: c_uint) cudaError_t;
pub extern fn cudaEventQuery(event: cudaEvent_t) cudaError_t;
pub extern fn cudaEventSynchronize(event: cudaEvent_t) cudaError_t;
pub extern fn cudaEventDestroy(event: cudaEvent_t) cudaError_t;
pub extern fn cudaEventElapsedTime(ms: [*c]f32, start: cudaEvent_t, end: cudaEvent_t) cudaError_t;
pub extern fn cudaImportExternalMemory(extMem_out: [*c]cudaExternalMemory_t, memHandleDesc: [*c]const struct_cudaExternalMemoryHandleDesc) cudaError_t;
pub extern fn cudaExternalMemoryGetMappedBuffer(devPtr: [*c]?*anyopaque, extMem: cudaExternalMemory_t, bufferDesc: [*c]const struct_cudaExternalMemoryBufferDesc) cudaError_t;
pub extern fn cudaExternalMemoryGetMappedMipmappedArray(mipmap: [*c]cudaMipmappedArray_t, extMem: cudaExternalMemory_t, mipmapDesc: [*c]const struct_cudaExternalMemoryMipmappedArrayDesc) cudaError_t;
pub extern fn cudaDestroyExternalMemory(extMem: cudaExternalMemory_t) cudaError_t;
pub extern fn cudaImportExternalSemaphore(extSem_out: [*c]cudaExternalSemaphore_t, semHandleDesc: [*c]const struct_cudaExternalSemaphoreHandleDesc) cudaError_t;
pub extern fn cudaSignalExternalSemaphoresAsync(extSemArray: [*c]const cudaExternalSemaphore_t, paramsArray: [*c]const struct_cudaExternalSemaphoreSignalParams, numExtSems: c_uint, stream: cudaStream_t) cudaError_t;
pub extern fn cudaWaitExternalSemaphoresAsync(extSemArray: [*c]const cudaExternalSemaphore_t, paramsArray: [*c]const struct_cudaExternalSemaphoreWaitParams, numExtSems: c_uint, stream: cudaStream_t) cudaError_t;
pub extern fn cudaDestroyExternalSemaphore(extSem: cudaExternalSemaphore_t) cudaError_t;
pub extern fn cudaLaunchKernel(func: ?*const anyopaque, gridDim: dim3, blockDim: dim3, args: [*c]?*anyopaque, sharedMem: usize, stream: cudaStream_t) cudaError_t;
pub extern fn cudaLaunchKernelExC(config: [*c]const cudaLaunchConfig_t, func: ?*const anyopaque, args: [*c]?*anyopaque) cudaError_t;
pub extern fn cudaLaunchCooperativeKernel(func: ?*const anyopaque, gridDim: dim3, blockDim: dim3, args: [*c]?*anyopaque, sharedMem: usize, stream: cudaStream_t) cudaError_t;
pub extern fn cudaFuncSetCacheConfig(func: ?*const anyopaque, cacheConfig: enum_cudaFuncCache) cudaError_t;
pub extern fn cudaFuncGetAttributes(attr: [*c]struct_cudaFuncAttributes, func: ?*const anyopaque) cudaError_t;
pub extern fn cudaFuncSetAttribute(func: ?*const anyopaque, attr: enum_cudaFuncAttribute, value: c_int) cudaError_t;
pub extern fn cudaFuncGetName(name: [*c][*c]const u8, func: ?*const anyopaque) cudaError_t;
pub extern fn cudaFuncGetParamInfo(func: ?*const anyopaque, paramIndex: usize, paramOffset: [*c]usize, paramSize: [*c]usize) cudaError_t;
pub extern fn cudaFuncGetParamCount(func: ?*const anyopaque, paramCount: [*c]usize) cudaError_t;
pub extern fn cudaLaunchHostFunc(stream: cudaStream_t, @"fn": cudaHostFn_t, userData: ?*anyopaque) cudaError_t;
pub extern fn cudaLaunchHostFunc_v2(stream: cudaStream_t, @"fn": cudaHostFn_t, userData: ?*anyopaque, syncMode: c_uint) cudaError_t;
pub extern fn cudaFuncSetSharedMemConfig(func: ?*const anyopaque, config: enum_cudaSharedMemConfig) cudaError_t;
pub extern fn cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks: [*c]c_int, func: ?*const anyopaque, blockSize: c_int, dynamicSMemSize: usize) cudaError_t;
pub extern fn cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize: [*c]usize, func: ?*const anyopaque, numBlocks: c_int, blockSize: c_int) cudaError_t;
pub extern fn cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks: [*c]c_int, func: ?*const anyopaque, blockSize: c_int, dynamicSMemSize: usize, flags: c_uint) cudaError_t;
pub extern fn cudaOccupancyMaxPotentialClusterSize(clusterSize: [*c]c_int, func: ?*const anyopaque, launchConfig: [*c]const cudaLaunchConfig_t) cudaError_t;
pub extern fn cudaOccupancyMaxActiveClusters(numClusters: [*c]c_int, func: ?*const anyopaque, launchConfig: [*c]const cudaLaunchConfig_t) cudaError_t;
pub extern fn cudaMallocManaged(devPtr: [*c]?*anyopaque, size: usize, flags: c_uint) cudaError_t;
pub extern fn cudaMalloc(devPtr: [*c]?*anyopaque, size: usize) cudaError_t;
pub extern fn cudaMallocHost(ptr: [*c]?*anyopaque, size: usize) cudaError_t;
pub extern fn cudaMallocPitch(devPtr: [*c]?*anyopaque, pitch: [*c]usize, width: usize, height: usize) cudaError_t;
pub extern fn cudaMallocArray(array: [*c]cudaArray_t, desc: [*c]const struct_cudaChannelFormatDesc, width: usize, height: usize, flags: c_uint) cudaError_t;
pub extern fn cudaFree(devPtr: ?*anyopaque) cudaError_t;
pub extern fn cudaFreeHost(ptr: ?*anyopaque) cudaError_t;
pub extern fn cudaFreeArray(array: cudaArray_t) cudaError_t;
pub extern fn cudaFreeMipmappedArray(mipmappedArray: cudaMipmappedArray_t) cudaError_t;
pub extern fn cudaHostAlloc(pHost: [*c]?*anyopaque, size: usize, flags: c_uint) cudaError_t;
pub extern fn cudaHostRegister(ptr: ?*anyopaque, size: usize, flags: c_uint) cudaError_t;
pub extern fn cudaHostUnregister(ptr: ?*anyopaque) cudaError_t;
pub extern fn cudaHostGetDevicePointer(pDevice: [*c]?*anyopaque, pHost: ?*anyopaque, flags: c_uint) cudaError_t;
pub extern fn cudaHostGetFlags(pFlags: [*c]c_uint, pHost: ?*anyopaque) cudaError_t;
pub extern fn cudaMalloc3D(pitchedDevPtr: [*c]struct_cudaPitchedPtr, extent: struct_cudaExtent) cudaError_t;
pub extern fn cudaMalloc3DArray(array: [*c]cudaArray_t, desc: [*c]const struct_cudaChannelFormatDesc, extent: struct_cudaExtent, flags: c_uint) cudaError_t;
pub extern fn cudaMallocMipmappedArray(mipmappedArray: [*c]cudaMipmappedArray_t, desc: [*c]const struct_cudaChannelFormatDesc, extent: struct_cudaExtent, numLevels: c_uint, flags: c_uint) cudaError_t;
pub extern fn cudaGetMipmappedArrayLevel(levelArray: [*c]cudaArray_t, mipmappedArray: cudaMipmappedArray_const_t, level: c_uint) cudaError_t;
pub extern fn cudaMemcpy3D(p: [*c]const struct_cudaMemcpy3DParms) cudaError_t;
pub extern fn cudaMemcpy3DPeer(p: [*c]const struct_cudaMemcpy3DPeerParms) cudaError_t;
pub extern fn cudaMemcpy3DAsync(p: [*c]const struct_cudaMemcpy3DParms, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemcpy3DPeerAsync(p: [*c]const struct_cudaMemcpy3DPeerParms, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemGetInfo(free: [*c]usize, total: [*c]usize) cudaError_t;
pub extern fn cudaArrayGetInfo(desc: [*c]struct_cudaChannelFormatDesc, extent: [*c]struct_cudaExtent, flags: [*c]c_uint, array: cudaArray_t) cudaError_t;
pub extern fn cudaArrayGetPlane(pPlaneArray: [*c]cudaArray_t, hArray: cudaArray_t, planeIdx: c_uint) cudaError_t;
pub extern fn cudaArrayGetMemoryRequirements(memoryRequirements: [*c]struct_cudaArrayMemoryRequirements, array: cudaArray_t, device: c_int) cudaError_t;
pub extern fn cudaMipmappedArrayGetMemoryRequirements(memoryRequirements: [*c]struct_cudaArrayMemoryRequirements, mipmap: cudaMipmappedArray_t, device: c_int) cudaError_t;
pub extern fn cudaArrayGetSparseProperties(sparseProperties: [*c]struct_cudaArraySparseProperties, array: cudaArray_t) cudaError_t;
pub extern fn cudaMipmappedArrayGetSparseProperties(sparseProperties: [*c]struct_cudaArraySparseProperties, mipmap: cudaMipmappedArray_t) cudaError_t;
pub extern fn cudaMemcpy(dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaMemcpyPeer(dst: ?*anyopaque, dstDevice: c_int, src: ?*const anyopaque, srcDevice: c_int, count: usize) cudaError_t;
pub extern fn cudaMemcpy2D(dst: ?*anyopaque, dpitch: usize, src: ?*const anyopaque, spitch: usize, width: usize, height: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaMemcpy2DToArray(dst: cudaArray_t, wOffset: usize, hOffset: usize, src: ?*const anyopaque, spitch: usize, width: usize, height: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaMemcpy2DFromArray(dst: ?*anyopaque, dpitch: usize, src: cudaArray_const_t, wOffset: usize, hOffset: usize, width: usize, height: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaMemcpy2DArrayToArray(dst: cudaArray_t, wOffsetDst: usize, hOffsetDst: usize, src: cudaArray_const_t, wOffsetSrc: usize, hOffsetSrc: usize, width: usize, height: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaMemcpyToSymbol(symbol: ?*const anyopaque, src: ?*const anyopaque, count: usize, offset: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaMemcpyFromSymbol(dst: ?*anyopaque, symbol: ?*const anyopaque, count: usize, offset: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaMemcpyAsync(dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: enum_cudaMemcpyKind, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemcpyPeerAsync(dst: ?*anyopaque, dstDevice: c_int, src: ?*const anyopaque, srcDevice: c_int, count: usize, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemcpyBatchAsync(dsts: [*c]const ?*anyopaque, srcs: [*c]const ?*const anyopaque, sizes: [*c]const usize, count: usize, attrs: [*c]struct_cudaMemcpyAttributes, attrsIdxs: [*c]usize, numAttrs: usize, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemcpy3DBatchAsync(numOps: usize, opList: [*c]struct_cudaMemcpy3DBatchOp, flags: c_ulonglong, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemcpyWithAttributesAsync(dst: ?*anyopaque, src: ?*const anyopaque, size: usize, attr: [*c]struct_cudaMemcpyAttributes, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemcpy3DWithAttributesAsync(op: [*c]struct_cudaMemcpy3DBatchOp, flags: c_ulonglong, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemcpy2DAsync(dst: ?*anyopaque, dpitch: usize, src: ?*const anyopaque, spitch: usize, width: usize, height: usize, kind: enum_cudaMemcpyKind, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemcpy2DToArrayAsync(dst: cudaArray_t, wOffset: usize, hOffset: usize, src: ?*const anyopaque, spitch: usize, width: usize, height: usize, kind: enum_cudaMemcpyKind, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemcpy2DFromArrayAsync(dst: ?*anyopaque, dpitch: usize, src: cudaArray_const_t, wOffset: usize, hOffset: usize, width: usize, height: usize, kind: enum_cudaMemcpyKind, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemcpyToSymbolAsync(symbol: ?*const anyopaque, src: ?*const anyopaque, count: usize, offset: usize, kind: enum_cudaMemcpyKind, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemcpyFromSymbolAsync(dst: ?*anyopaque, symbol: ?*const anyopaque, count: usize, offset: usize, kind: enum_cudaMemcpyKind, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemset(devPtr: ?*anyopaque, value: c_int, count: usize) cudaError_t;
pub extern fn cudaMemset2D(devPtr: ?*anyopaque, pitch: usize, value: c_int, width: usize, height: usize) cudaError_t;
pub extern fn cudaMemset3D(pitchedDevPtr: struct_cudaPitchedPtr, value: c_int, extent: struct_cudaExtent) cudaError_t;
pub extern fn cudaMemsetAsync(devPtr: ?*anyopaque, value: c_int, count: usize, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemset2DAsync(devPtr: ?*anyopaque, pitch: usize, value: c_int, width: usize, height: usize, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemset3DAsync(pitchedDevPtr: struct_cudaPitchedPtr, value: c_int, extent: struct_cudaExtent, stream: cudaStream_t) cudaError_t;
pub extern fn cudaGetSymbolAddress(devPtr: [*c]?*anyopaque, symbol: ?*const anyopaque) cudaError_t;
pub extern fn cudaGetSymbolSize(size: [*c]usize, symbol: ?*const anyopaque) cudaError_t;
pub extern fn cudaMemPrefetchAsync(devPtr: ?*const anyopaque, count: usize, location: struct_cudaMemLocation, flags: c_uint, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemPrefetchBatchAsync(dptrs: [*c]?*anyopaque, sizes: [*c]usize, count: usize, prefetchLocs: [*c]struct_cudaMemLocation, prefetchLocIdxs: [*c]usize, numPrefetchLocs: usize, flags: c_ulonglong, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemDiscardBatchAsync(dptrs: [*c]?*anyopaque, sizes: [*c]usize, count: usize, flags: c_ulonglong, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemDiscardAndPrefetchBatchAsync(dptrs: [*c]?*anyopaque, sizes: [*c]usize, count: usize, prefetchLocs: [*c]struct_cudaMemLocation, prefetchLocIdxs: [*c]usize, numPrefetchLocs: usize, flags: c_ulonglong, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemAdvise(devPtr: ?*const anyopaque, count: usize, advice: enum_cudaMemoryAdvise, location: struct_cudaMemLocation) cudaError_t;
pub extern fn cudaMemRangeGetAttribute(data: ?*anyopaque, dataSize: usize, attribute: enum_cudaMemRangeAttribute, devPtr: ?*const anyopaque, count: usize) cudaError_t;
pub extern fn cudaMemRangeGetAttributes(data: [*c]?*anyopaque, dataSizes: [*c]usize, attributes: [*c]enum_cudaMemRangeAttribute, numAttributes: usize, devPtr: ?*const anyopaque, count: usize) cudaError_t;
pub extern fn cudaMemcpyToArray(dst: cudaArray_t, wOffset: usize, hOffset: usize, src: ?*const anyopaque, count: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaMemcpyFromArray(dst: ?*anyopaque, src: cudaArray_const_t, wOffset: usize, hOffset: usize, count: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaMemcpyArrayToArray(dst: cudaArray_t, wOffsetDst: usize, hOffsetDst: usize, src: cudaArray_const_t, wOffsetSrc: usize, hOffsetSrc: usize, count: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaMemcpyToArrayAsync(dst: cudaArray_t, wOffset: usize, hOffset: usize, src: ?*const anyopaque, count: usize, kind: enum_cudaMemcpyKind, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemcpyFromArrayAsync(dst: ?*anyopaque, src: cudaArray_const_t, wOffset: usize, hOffset: usize, count: usize, kind: enum_cudaMemcpyKind, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMallocAsync(devPtr: [*c]?*anyopaque, size: usize, hStream: cudaStream_t) cudaError_t;
pub extern fn cudaFreeAsync(devPtr: ?*anyopaque, hStream: cudaStream_t) cudaError_t;
pub extern fn cudaMemPoolTrimTo(memPool: cudaMemPool_t, minBytesToKeep: usize) cudaError_t;
pub extern fn cudaMemPoolSetAttribute(memPool: cudaMemPool_t, attr: enum_cudaMemPoolAttr, value: ?*anyopaque) cudaError_t;
pub extern fn cudaMemPoolGetAttribute(memPool: cudaMemPool_t, attr: enum_cudaMemPoolAttr, value: ?*anyopaque) cudaError_t;
pub extern fn cudaMemPoolSetAccess(memPool: cudaMemPool_t, descList: [*c]const struct_cudaMemAccessDesc, count: usize) cudaError_t;
pub extern fn cudaMemPoolGetAccess(flags: [*c]enum_cudaMemAccessFlags, memPool: cudaMemPool_t, location: [*c]struct_cudaMemLocation) cudaError_t;
pub extern fn cudaMemPoolCreate(memPool: [*c]cudaMemPool_t, poolProps: [*c]const struct_cudaMemPoolProps) cudaError_t;
pub extern fn cudaMemPoolDestroy(memPool: cudaMemPool_t) cudaError_t;
pub extern fn cudaMemGetDefaultMemPool(memPool: [*c]cudaMemPool_t, location: [*c]struct_cudaMemLocation, @"type": enum_cudaMemAllocationType) cudaError_t;
pub extern fn cudaMemGetMemPool(memPool: [*c]cudaMemPool_t, location: [*c]struct_cudaMemLocation, @"type": enum_cudaMemAllocationType) cudaError_t;
pub extern fn cudaMemSetMemPool(location: [*c]struct_cudaMemLocation, @"type": enum_cudaMemAllocationType, memPool: cudaMemPool_t) cudaError_t;
pub extern fn cudaMallocFromPoolAsync(ptr: [*c]?*anyopaque, size: usize, memPool: cudaMemPool_t, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemPoolExportToShareableHandle(shareableHandle: ?*anyopaque, memPool: cudaMemPool_t, handleType: enum_cudaMemAllocationHandleType, flags: c_uint) cudaError_t;
pub extern fn cudaMemPoolImportFromShareableHandle(memPool: [*c]cudaMemPool_t, shareableHandle: ?*anyopaque, handleType: enum_cudaMemAllocationHandleType, flags: c_uint) cudaError_t;
pub extern fn cudaMemPoolExportPointer(exportData: [*c]struct_cudaMemPoolPtrExportData, ptr: ?*anyopaque) cudaError_t;
pub extern fn cudaMemPoolImportPointer(ptr: [*c]?*anyopaque, memPool: cudaMemPool_t, exportData: [*c]struct_cudaMemPoolPtrExportData) cudaError_t;
pub extern fn cudaPointerGetAttributes(attributes: [*c]struct_cudaPointerAttributes, ptr: ?*const anyopaque) cudaError_t;
pub extern fn cudaDeviceCanAccessPeer(canAccessPeer: [*c]c_int, device: c_int, peerDevice: c_int) cudaError_t;
pub extern fn cudaDeviceEnablePeerAccess(peerDevice: c_int, flags: c_uint) cudaError_t;
pub extern fn cudaDeviceDisablePeerAccess(peerDevice: c_int) cudaError_t;
pub extern fn cudaGraphicsUnregisterResource(resource: cudaGraphicsResource_t) cudaError_t;
pub extern fn cudaGraphicsResourceSetMapFlags(resource: cudaGraphicsResource_t, flags: c_uint) cudaError_t;
pub extern fn cudaGraphicsMapResources(count: c_int, resources: [*c]cudaGraphicsResource_t, stream: cudaStream_t) cudaError_t;
pub extern fn cudaGraphicsUnmapResources(count: c_int, resources: [*c]cudaGraphicsResource_t, stream: cudaStream_t) cudaError_t;
pub extern fn cudaGraphicsResourceGetMappedPointer(devPtr: [*c]?*anyopaque, size: [*c]usize, resource: cudaGraphicsResource_t) cudaError_t;
pub extern fn cudaGraphicsSubResourceGetMappedArray(array: [*c]cudaArray_t, resource: cudaGraphicsResource_t, arrayIndex: c_uint, mipLevel: c_uint) cudaError_t;
pub extern fn cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray: [*c]cudaMipmappedArray_t, resource: cudaGraphicsResource_t) cudaError_t;
pub extern fn cudaGetChannelDesc(desc: [*c]struct_cudaChannelFormatDesc, array: cudaArray_const_t) cudaError_t;
pub extern fn cudaCreateChannelDesc(x: c_int, y: c_int, z: c_int, w: c_int, f: enum_cudaChannelFormatKind) struct_cudaChannelFormatDesc;
pub extern fn cudaCreateTextureObject(pTexObject: [*c]cudaTextureObject_t, pResDesc: [*c]const struct_cudaResourceDesc, pTexDesc: [*c]const struct_cudaTextureDesc, pResViewDesc: [*c]const struct_cudaResourceViewDesc) cudaError_t;
pub extern fn cudaDestroyTextureObject(texObject: cudaTextureObject_t) cudaError_t;
pub extern fn cudaGetTextureObjectResourceDesc(pResDesc: [*c]struct_cudaResourceDesc, texObject: cudaTextureObject_t) cudaError_t;
pub extern fn cudaGetTextureObjectTextureDesc(pTexDesc: [*c]struct_cudaTextureDesc, texObject: cudaTextureObject_t) cudaError_t;
pub extern fn cudaGetTextureObjectResourceViewDesc(pResViewDesc: [*c]struct_cudaResourceViewDesc, texObject: cudaTextureObject_t) cudaError_t;
pub extern fn cudaCreateSurfaceObject(pSurfObject: [*c]cudaSurfaceObject_t, pResDesc: [*c]const struct_cudaResourceDesc) cudaError_t;
pub extern fn cudaDestroySurfaceObject(surfObject: cudaSurfaceObject_t) cudaError_t;
pub extern fn cudaGetSurfaceObjectResourceDesc(pResDesc: [*c]struct_cudaResourceDesc, surfObject: cudaSurfaceObject_t) cudaError_t;
pub extern fn cudaDriverGetVersion(driverVersion: [*c]c_int) cudaError_t;
pub extern fn cudaRuntimeGetVersion(runtimeVersion: [*c]c_int) cudaError_t;
pub const cudaLogsCallback_t = ?*const fn (data: ?*anyopaque, logLevel: cudaLogLevel, message: [*c]u8, length: usize) callconv(.c) void;
pub extern fn cudaLogsRegisterCallback(callbackFunc: cudaLogsCallback_t, userData: ?*anyopaque, callback_out: [*c]cudaLogsCallbackHandle) cudaError_t;
pub extern fn cudaLogsUnregisterCallback(callback: cudaLogsCallbackHandle) cudaError_t;
pub extern fn cudaLogsCurrent(iterator_out: [*c]cudaLogIterator, flags: c_uint) cudaError_t;
pub extern fn cudaLogsDumpToFile(iterator: [*c]cudaLogIterator, pathToFile: [*c]const u8, flags: c_uint) cudaError_t;
pub extern fn cudaLogsDumpToMemory(iterator: [*c]cudaLogIterator, buffer: [*c]u8, size: [*c]usize, flags: c_uint) cudaError_t;
pub extern fn cudaGraphCreate(pGraph: [*c]cudaGraph_t, flags: c_uint) cudaError_t;
pub extern fn cudaGraphAddKernelNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, pNodeParams: [*c]const struct_cudaKernelNodeParams) cudaError_t;
pub extern fn cudaGraphKernelNodeGetParams(node: cudaGraphNode_t, pNodeParams: [*c]struct_cudaKernelNodeParams) cudaError_t;
pub extern fn cudaGraphKernelNodeSetParams(node: cudaGraphNode_t, pNodeParams: [*c]const struct_cudaKernelNodeParams) cudaError_t;
pub extern fn cudaGraphKernelNodeCopyAttributes(hDst: cudaGraphNode_t, hSrc: cudaGraphNode_t) cudaError_t;
pub extern fn cudaGraphKernelNodeGetAttribute(hNode: cudaGraphNode_t, attr: cudaLaunchAttributeID, value_out: [*c]cudaLaunchAttributeValue) cudaError_t;
pub extern fn cudaGraphKernelNodeSetAttribute(hNode: cudaGraphNode_t, attr: cudaLaunchAttributeID, value: [*c]const cudaLaunchAttributeValue) cudaError_t;
pub extern fn cudaGraphAddMemcpyNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, pCopyParams: [*c]const struct_cudaMemcpy3DParms) cudaError_t;
pub extern fn cudaGraphAddMemcpyNodeToSymbol(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, symbol: ?*const anyopaque, src: ?*const anyopaque, count: usize, offset: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaGraphAddMemcpyNodeFromSymbol(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, dst: ?*anyopaque, symbol: ?*const anyopaque, count: usize, offset: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaGraphAddMemcpyNode1D(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaGraphMemcpyNodeGetParams(node: cudaGraphNode_t, pNodeParams: [*c]struct_cudaMemcpy3DParms) cudaError_t;
pub extern fn cudaGraphMemcpyNodeSetParams(node: cudaGraphNode_t, pNodeParams: [*c]const struct_cudaMemcpy3DParms) cudaError_t;
pub extern fn cudaGraphMemcpyNodeSetParamsToSymbol(node: cudaGraphNode_t, symbol: ?*const anyopaque, src: ?*const anyopaque, count: usize, offset: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaGraphMemcpyNodeSetParamsFromSymbol(node: cudaGraphNode_t, dst: ?*anyopaque, symbol: ?*const anyopaque, count: usize, offset: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaGraphMemcpyNodeSetParams1D(node: cudaGraphNode_t, dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaGraphAddMemsetNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, pMemsetParams: [*c]const struct_cudaMemsetParams) cudaError_t;
pub extern fn cudaGraphMemsetNodeGetParams(node: cudaGraphNode_t, pNodeParams: [*c]struct_cudaMemsetParams) cudaError_t;
pub extern fn cudaGraphMemsetNodeSetParams(node: cudaGraphNode_t, pNodeParams: [*c]const struct_cudaMemsetParams) cudaError_t;
pub extern fn cudaGraphAddHostNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, pNodeParams: [*c]const struct_cudaHostNodeParams) cudaError_t;
pub extern fn cudaGraphHostNodeGetParams(node: cudaGraphNode_t, pNodeParams: [*c]struct_cudaHostNodeParams) cudaError_t;
pub extern fn cudaGraphHostNodeSetParams(node: cudaGraphNode_t, pNodeParams: [*c]const struct_cudaHostNodeParams) cudaError_t;
pub extern fn cudaGraphAddChildGraphNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, childGraph: cudaGraph_t) cudaError_t;
pub extern fn cudaGraphChildGraphNodeGetGraph(node: cudaGraphNode_t, pGraph: [*c]cudaGraph_t) cudaError_t;
pub extern fn cudaGraphAddEmptyNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize) cudaError_t;
pub extern fn cudaGraphAddEventRecordNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, event: cudaEvent_t) cudaError_t;
pub extern fn cudaGraphEventRecordNodeGetEvent(node: cudaGraphNode_t, event_out: [*c]cudaEvent_t) cudaError_t;
pub extern fn cudaGraphEventRecordNodeSetEvent(node: cudaGraphNode_t, event: cudaEvent_t) cudaError_t;
pub extern fn cudaGraphAddEventWaitNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, event: cudaEvent_t) cudaError_t;
pub extern fn cudaGraphEventWaitNodeGetEvent(node: cudaGraphNode_t, event_out: [*c]cudaEvent_t) cudaError_t;
pub extern fn cudaGraphEventWaitNodeSetEvent(node: cudaGraphNode_t, event: cudaEvent_t) cudaError_t;
pub extern fn cudaGraphAddExternalSemaphoresSignalNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, nodeParams: [*c]const struct_cudaExternalSemaphoreSignalNodeParams) cudaError_t;
pub extern fn cudaGraphExternalSemaphoresSignalNodeGetParams(hNode: cudaGraphNode_t, params_out: [*c]struct_cudaExternalSemaphoreSignalNodeParams) cudaError_t;
pub extern fn cudaGraphExternalSemaphoresSignalNodeSetParams(hNode: cudaGraphNode_t, nodeParams: [*c]const struct_cudaExternalSemaphoreSignalNodeParams) cudaError_t;
pub extern fn cudaGraphAddExternalSemaphoresWaitNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, nodeParams: [*c]const struct_cudaExternalSemaphoreWaitNodeParams) cudaError_t;
pub extern fn cudaGraphExternalSemaphoresWaitNodeGetParams(hNode: cudaGraphNode_t, params_out: [*c]struct_cudaExternalSemaphoreWaitNodeParams) cudaError_t;
pub extern fn cudaGraphExternalSemaphoresWaitNodeSetParams(hNode: cudaGraphNode_t, nodeParams: [*c]const struct_cudaExternalSemaphoreWaitNodeParams) cudaError_t;
pub extern fn cudaGraphAddMemAllocNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, nodeParams: [*c]struct_cudaMemAllocNodeParams) cudaError_t;
pub extern fn cudaGraphMemAllocNodeGetParams(node: cudaGraphNode_t, params_out: [*c]struct_cudaMemAllocNodeParams) cudaError_t;
pub extern fn cudaGraphAddMemFreeNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, dptr: ?*anyopaque) cudaError_t;
pub extern fn cudaGraphMemFreeNodeGetParams(node: cudaGraphNode_t, dptr_out: ?*anyopaque) cudaError_t;
pub extern fn cudaDeviceGraphMemTrim(device: c_int) cudaError_t;
pub extern fn cudaDeviceGetGraphMemAttribute(device: c_int, attr: enum_cudaGraphMemAttributeType, value: ?*anyopaque) cudaError_t;
pub extern fn cudaDeviceSetGraphMemAttribute(device: c_int, attr: enum_cudaGraphMemAttributeType, value: ?*anyopaque) cudaError_t;
pub extern fn cudaGraphClone(pGraphClone: [*c]cudaGraph_t, originalGraph: cudaGraph_t) cudaError_t;
pub extern fn cudaGraphNodeFindInClone(pNode: [*c]cudaGraphNode_t, originalNode: cudaGraphNode_t, clonedGraph: cudaGraph_t) cudaError_t;
pub extern fn cudaGraphNodeGetType(node: cudaGraphNode_t, pType: [*c]enum_cudaGraphNodeType) cudaError_t;
pub extern fn cudaGraphNodeGetContainingGraph(hNode: cudaGraphNode_t, phGraph: [*c]cudaGraph_t) cudaError_t;
pub extern fn cudaGraphNodeGetLocalId(hNode: cudaGraphNode_t, nodeId: [*c]c_uint) cudaError_t;
pub extern fn cudaGraphNodeGetToolsId(hNode: cudaGraphNode_t, toolsNodeId: [*c]c_ulonglong) cudaError_t;
pub extern fn cudaGraphGetId(hGraph: cudaGraph_t, graphID: [*c]c_uint) cudaError_t;
pub extern fn cudaGraphExecGetId(hGraphExec: cudaGraphExec_t, graphID: [*c]c_uint) cudaError_t;
pub extern fn cudaGraphGetNodes(graph: cudaGraph_t, nodes: [*c]cudaGraphNode_t, numNodes: [*c]usize) cudaError_t;
pub extern fn cudaGraphGetRootNodes(graph: cudaGraph_t, pRootNodes: [*c]cudaGraphNode_t, pNumRootNodes: [*c]usize) cudaError_t;
pub extern fn cudaGraphGetEdges(graph: cudaGraph_t, from: [*c]cudaGraphNode_t, to: [*c]cudaGraphNode_t, edgeData: [*c]cudaGraphEdgeData, numEdges: [*c]usize) cudaError_t;
pub extern fn cudaGraphNodeGetDependencies(node: cudaGraphNode_t, pDependencies: [*c]cudaGraphNode_t, edgeData: [*c]cudaGraphEdgeData, pNumDependencies: [*c]usize) cudaError_t;
pub extern fn cudaGraphNodeGetDependentNodes(node: cudaGraphNode_t, pDependentNodes: [*c]cudaGraphNode_t, edgeData: [*c]cudaGraphEdgeData, pNumDependentNodes: [*c]usize) cudaError_t;
pub extern fn cudaGraphAddDependencies(graph: cudaGraph_t, from: [*c]const cudaGraphNode_t, to: [*c]const cudaGraphNode_t, edgeData: [*c]const cudaGraphEdgeData, numDependencies: usize) cudaError_t;
pub extern fn cudaGraphRemoveDependencies(graph: cudaGraph_t, from: [*c]const cudaGraphNode_t, to: [*c]const cudaGraphNode_t, edgeData: [*c]const cudaGraphEdgeData, numDependencies: usize) cudaError_t;
pub extern fn cudaGraphDestroyNode(node: cudaGraphNode_t) cudaError_t;
pub extern fn cudaGraphInstantiate(pGraphExec: [*c]cudaGraphExec_t, graph: cudaGraph_t, flags: c_ulonglong) cudaError_t;
pub extern fn cudaGraphInstantiateWithFlags(pGraphExec: [*c]cudaGraphExec_t, graph: cudaGraph_t, flags: c_ulonglong) cudaError_t;
pub extern fn cudaGraphInstantiateWithParams(pGraphExec: [*c]cudaGraphExec_t, graph: cudaGraph_t, instantiateParams: [*c]cudaGraphInstantiateParams) cudaError_t;
pub extern fn cudaGraphExecGetFlags(graphExec: cudaGraphExec_t, flags: [*c]c_ulonglong) cudaError_t;
pub extern fn cudaGraphExecKernelNodeSetParams(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, pNodeParams: [*c]const struct_cudaKernelNodeParams) cudaError_t;
pub extern fn cudaGraphExecMemcpyNodeSetParams(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, pNodeParams: [*c]const struct_cudaMemcpy3DParms) cudaError_t;
pub extern fn cudaGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, symbol: ?*const anyopaque, src: ?*const anyopaque, count: usize, offset: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, dst: ?*anyopaque, symbol: ?*const anyopaque, count: usize, offset: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaGraphExecMemcpyNodeSetParams1D(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaGraphExecMemsetNodeSetParams(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, pNodeParams: [*c]const struct_cudaMemsetParams) cudaError_t;
pub extern fn cudaGraphExecHostNodeSetParams(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, pNodeParams: [*c]const struct_cudaHostNodeParams) cudaError_t;
pub extern fn cudaGraphExecChildGraphNodeSetParams(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, childGraph: cudaGraph_t) cudaError_t;
pub extern fn cudaGraphExecEventRecordNodeSetEvent(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, event: cudaEvent_t) cudaError_t;
pub extern fn cudaGraphExecEventWaitNodeSetEvent(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, event: cudaEvent_t) cudaError_t;
pub extern fn cudaGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, nodeParams: [*c]const struct_cudaExternalSemaphoreSignalNodeParams) cudaError_t;
pub extern fn cudaGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, nodeParams: [*c]const struct_cudaExternalSemaphoreWaitNodeParams) cudaError_t;
pub extern fn cudaGraphNodeSetEnabled(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, isEnabled: c_uint) cudaError_t;
pub extern fn cudaGraphNodeGetEnabled(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, isEnabled: [*c]c_uint) cudaError_t;
pub extern fn cudaGraphExecUpdate(hGraphExec: cudaGraphExec_t, hGraph: cudaGraph_t, resultInfo: [*c]cudaGraphExecUpdateResultInfo) cudaError_t;
pub extern fn cudaGraphUpload(graphExec: cudaGraphExec_t, stream: cudaStream_t) cudaError_t;
pub extern fn cudaGraphLaunch(graphExec: cudaGraphExec_t, stream: cudaStream_t) cudaError_t;
pub extern fn cudaGraphExecDestroy(graphExec: cudaGraphExec_t) cudaError_t;
pub extern fn cudaGraphDestroy(graph: cudaGraph_t) cudaError_t;
pub extern fn cudaGraphDebugDotPrint(graph: cudaGraph_t, path: [*c]const u8, flags: c_uint) cudaError_t;
pub extern fn cudaUserObjectCreate(object_out: [*c]cudaUserObject_t, ptr: ?*anyopaque, destroy: cudaHostFn_t, initialRefcount: c_uint, flags: c_uint) cudaError_t;
pub extern fn cudaUserObjectRetain(object: cudaUserObject_t, count: c_uint) cudaError_t;
pub extern fn cudaUserObjectRelease(object: cudaUserObject_t, count: c_uint) cudaError_t;
pub extern fn cudaGraphRetainUserObject(graph: cudaGraph_t, object: cudaUserObject_t, count: c_uint, flags: c_uint) cudaError_t;
pub extern fn cudaGraphReleaseUserObject(graph: cudaGraph_t, object: cudaUserObject_t, count: c_uint) cudaError_t;
pub extern fn cudaGraphAddNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, dependencyData: [*c]const cudaGraphEdgeData, numDependencies: usize, nodeParams: [*c]struct_cudaGraphNodeParams) cudaError_t;
pub extern fn cudaGraphNodeSetParams(node: cudaGraphNode_t, nodeParams: [*c]struct_cudaGraphNodeParams) cudaError_t;
pub extern fn cudaGraphNodeGetParams(node: cudaGraphNode_t, nodeParams: [*c]struct_cudaGraphNodeParams) cudaError_t;
pub extern fn cudaGraphExecNodeSetParams(graphExec: cudaGraphExec_t, node: cudaGraphNode_t, nodeParams: [*c]struct_cudaGraphNodeParams) cudaError_t;
pub extern fn cudaGraphConditionalHandleCreate(pHandle_out: [*c]cudaGraphConditionalHandle, graph: cudaGraph_t, defaultLaunchValue: c_uint, flags: c_uint) cudaError_t;
pub extern fn cudaGraphConditionalHandleCreate_v2(pHandle_out: [*c]cudaGraphConditionalHandle, graph: cudaGraph_t, ctx: cudaExecutionContext_t, defaultLaunchValue: c_uint, flags: c_uint) cudaError_t;
pub extern fn cudaGetDriverEntryPoint(symbol: [*c]const u8, funcPtr: [*c]?*anyopaque, flags: c_ulonglong, driverStatus: [*c]enum_cudaDriverEntryPointQueryResult) cudaError_t;
pub extern fn cudaGetDriverEntryPointByVersion(symbol: [*c]const u8, funcPtr: [*c]?*anyopaque, cudaVersion: c_uint, flags: c_ulonglong, driverStatus: [*c]enum_cudaDriverEntryPointQueryResult) cudaError_t;
pub extern fn cudaLibraryLoadData(library: [*c]cudaLibrary_t, code: ?*const anyopaque, jitOptions: [*c]enum_cudaJitOption, jitOptionsValues: [*c]?*anyopaque, numJitOptions: c_uint, libraryOptions: [*c]enum_cudaLibraryOption, libraryOptionValues: [*c]?*anyopaque, numLibraryOptions: c_uint) cudaError_t;
pub extern fn cudaLibraryLoadFromFile(library: [*c]cudaLibrary_t, fileName: [*c]const u8, jitOptions: [*c]enum_cudaJitOption, jitOptionsValues: [*c]?*anyopaque, numJitOptions: c_uint, libraryOptions: [*c]enum_cudaLibraryOption, libraryOptionValues: [*c]?*anyopaque, numLibraryOptions: c_uint) cudaError_t;
pub extern fn cudaLibraryUnload(library: cudaLibrary_t) cudaError_t;
pub extern fn cudaLibraryGetKernel(pKernel: [*c]cudaKernel_t, library: cudaLibrary_t, name: [*c]const u8) cudaError_t;
pub extern fn cudaLibraryGetGlobal(dptr: [*c]?*anyopaque, bytes: [*c]usize, library: cudaLibrary_t, name: [*c]const u8) cudaError_t;
pub extern fn cudaLibraryGetManaged(dptr: [*c]?*anyopaque, bytes: [*c]usize, library: cudaLibrary_t, name: [*c]const u8) cudaError_t;
pub extern fn cudaLibraryGetUnifiedFunction(fptr: [*c]?*anyopaque, library: cudaLibrary_t, symbol: [*c]const u8) cudaError_t;
pub extern fn cudaLibraryGetKernelCount(count: [*c]c_uint, lib: cudaLibrary_t) cudaError_t;
pub extern fn cudaLibraryEnumerateKernels(kernels: [*c]cudaKernel_t, numKernels: c_uint, lib: cudaLibrary_t) cudaError_t;
pub extern fn cudaKernelSetAttributeForDevice(kernel: cudaKernel_t, attr: enum_cudaFuncAttribute, value: c_int, device: c_int) cudaError_t;
pub extern fn cudaDeviceGetDevResource(device: c_int, resource: [*c]cudaDevResource, @"type": enum_cudaDevResourceType) cudaError_t;
pub extern fn cudaDevSmResourceSplitByCount(result: [*c]cudaDevResource, nbGroups: [*c]c_uint, input: [*c]const cudaDevResource, remaining: [*c]cudaDevResource, flags: c_uint, minCount: c_uint) cudaError_t;
pub extern fn cudaDevSmResourceSplit(result: [*c]cudaDevResource, nbGroups: c_uint, input: [*c]const cudaDevResource, remainder: [*c]cudaDevResource, flags: c_uint, groupParams: [*c]cudaDevSmResourceGroupParams) cudaError_t;
pub extern fn cudaDevResourceGenerateDesc(phDesc: [*c]cudaDevResourceDesc_t, resources: [*c]cudaDevResource, nbResources: c_uint) cudaError_t;
pub extern fn cudaGreenCtxCreate(phCtx: [*c]cudaExecutionContext_t, desc: cudaDevResourceDesc_t, device: c_int, flags: c_uint) cudaError_t;
pub extern fn cudaExecutionCtxDestroy(ctx: cudaExecutionContext_t) cudaError_t;
pub extern fn cudaExecutionCtxGetDevResource(ctx: cudaExecutionContext_t, resource: [*c]cudaDevResource, @"type": enum_cudaDevResourceType) cudaError_t;
pub extern fn cudaExecutionCtxGetDevice(device: [*c]c_int, ctx: cudaExecutionContext_t) cudaError_t;
pub extern fn cudaExecutionCtxGetId(ctx: cudaExecutionContext_t, ctxId: [*c]c_ulonglong) cudaError_t;
pub extern fn cudaExecutionCtxStreamCreate(phStream: [*c]cudaStream_t, ctx: cudaExecutionContext_t, flags: c_uint, priority: c_int) cudaError_t;
pub extern fn cudaExecutionCtxSynchronize(ctx: cudaExecutionContext_t) cudaError_t;
pub extern fn cudaStreamGetDevResource(hStream: cudaStream_t, resource: [*c]cudaDevResource, @"type": enum_cudaDevResourceType) cudaError_t;
pub extern fn cudaExecutionCtxRecordEvent(ctx: cudaExecutionContext_t, event: cudaEvent_t) cudaError_t;
pub extern fn cudaExecutionCtxWaitEvent(ctx: cudaExecutionContext_t, event: cudaEvent_t) cudaError_t;
pub extern fn cudaDeviceGetExecutionCtx(ctx: [*c]cudaExecutionContext_t, device: c_int) cudaError_t;
pub extern fn cudaGetExportTable(ppExportTable: [*c]?*const anyopaque, pExportTableId: [*c]const cudaUUID_t) cudaError_t;
pub extern fn cudaGetFuncBySymbol(functionPtr: [*c]cudaFunction_t, symbolPtr: ?*const anyopaque) cudaError_t;
pub extern fn cudaGetKernel(kernelPtr: [*c]cudaKernel_t, entryFuncAddr: ?*const anyopaque) cudaError_t;
pub const CUDA_R_16F: c_int = 2;
pub const CUDA_C_16F: c_int = 6;
pub const CUDA_R_16BF: c_int = 14;
pub const CUDA_C_16BF: c_int = 15;
pub const CUDA_R_32F: c_int = 0;
pub const CUDA_C_32F: c_int = 4;
pub const CUDA_R_64F: c_int = 1;
pub const CUDA_C_64F: c_int = 5;
pub const CUDA_R_4I: c_int = 16;
pub const CUDA_C_4I: c_int = 17;
pub const CUDA_R_4U: c_int = 18;
pub const CUDA_C_4U: c_int = 19;
pub const CUDA_R_8I: c_int = 3;
pub const CUDA_C_8I: c_int = 7;
pub const CUDA_R_8U: c_int = 8;
pub const CUDA_C_8U: c_int = 9;
pub const CUDA_R_16I: c_int = 20;
pub const CUDA_C_16I: c_int = 21;
pub const CUDA_R_16U: c_int = 22;
pub const CUDA_C_16U: c_int = 23;
pub const CUDA_R_32I: c_int = 10;
pub const CUDA_C_32I: c_int = 11;
pub const CUDA_R_32U: c_int = 12;
pub const CUDA_C_32U: c_int = 13;
pub const CUDA_R_64I: c_int = 24;
pub const CUDA_C_64I: c_int = 25;
pub const CUDA_R_64U: c_int = 26;
pub const CUDA_C_64U: c_int = 27;
pub const CUDA_R_8F_E4M3: c_int = 28;
pub const CUDA_R_8F_UE4M3: c_int = 28;
pub const CUDA_R_8F_E5M2: c_int = 29;
pub const CUDA_R_8F_UE8M0: c_int = 30;
pub const CUDA_R_6F_E2M3: c_int = 31;
pub const CUDA_R_6F_E3M2: c_int = 32;
pub const CUDA_R_4F_E2M1: c_int = 33;
pub const enum_cudaDataType_t = c_uint;
pub const cudaDataType = enum_cudaDataType_t;
pub const CUDA_EMULATION_STRATEGY_DEFAULT: c_int = 0;
pub const CUDA_EMULATION_STRATEGY_PERFORMANT: c_int = 1;
pub const CUDA_EMULATION_STRATEGY_EAGER: c_int = 2;
pub const enum_cudaEmulationStrategy_t = c_uint;
pub const cudaEmulationStrategy = enum_cudaEmulationStrategy_t;
pub const CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC: c_int = 0;
pub const CUDA_EMULATION_MANTISSA_CONTROL_FIXED: c_int = 1;
pub const enum_cudaEmulationMantissaControl_t = c_uint;
pub const cudaEmulationMantissaControl = enum_cudaEmulationMantissaControl_t;
pub const CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT: c_int = 65535;
pub const CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NONE: c_int = 0;
pub const CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY: c_int = 1;
pub const CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NAN: c_int = 2;
pub const enum_cudaEmulationSpecialValuesSupport_t = c_uint;
pub const cudaEmulationSpecialValuesSupport = enum_cudaEmulationSpecialValuesSupport_t;
pub const MAJOR_VERSION: c_int = 0;
pub const MINOR_VERSION: c_int = 1;
pub const PATCH_LEVEL: c_int = 2;
pub const enum_libraryPropertyType_t = c_uint;
pub const libraryPropertyType = enum_libraryPropertyType_t;
pub const cudaDataType_t = enum_cudaDataType_t;
pub const cudaEmulationStrategy_t = enum_cudaEmulationStrategy_t;
pub const cudaEmulationMantissaControl_t = enum_cudaEmulationMantissaControl_t;
pub const cudaEmulationSpecialValuesSupport_t = enum_cudaEmulationSpecialValuesSupport_t;
pub const libraryPropertyType_t = enum_libraryPropertyType_t;
pub const struct_cudnnContext = opaque {
    pub const cudnnQueryRuntimeError = __root.cudnnQueryRuntimeError;
    pub const cudnnDestroy = __root.cudnnDestroy;
    pub const cudnnSetStream = __root.cudnnSetStream;
    pub const cudnnGetStream = __root.cudnnGetStream;
    pub const cudnnGetExecutionPlanWorkspaceSize = __root.cudnnGetExecutionPlanWorkspaceSize;
    pub const cudnnBackendExecute = __root.cudnnBackendExecute;
    pub const cudnnBackendPopulateCudaGraph = __root.cudnnBackendPopulateCudaGraph;
    pub const cudnnBackendUpdateCudaGraph = __root.cudnnBackendUpdateCudaGraph;
    pub const cudnnTransformTensor = __root.cudnnTransformTensor;
    pub const cudnnTransformTensorEx = __root.cudnnTransformTensorEx;
    pub const cudnnAddTensor = __root.cudnnAddTensor;
    pub const cudnnOpTensor = __root.cudnnOpTensor;
    pub const cudnnGetReductionIndicesSize = __root.cudnnGetReductionIndicesSize;
    pub const cudnnGetReductionWorkspaceSize = __root.cudnnGetReductionWorkspaceSize;
    pub const cudnnReduceTensor = __root.cudnnReduceTensor;
    pub const cudnnSetTensor = __root.cudnnSetTensor;
    pub const cudnnScaleTensor = __root.cudnnScaleTensor;
    pub const cudnnTransformFilter = __root.cudnnTransformFilter;
    pub const cudnnSoftmaxForward = __root.cudnnSoftmaxForward;
    pub const cudnnPoolingForward = __root.cudnnPoolingForward;
    pub const cudnnActivationForward = __root.cudnnActivationForward;
    pub const cudnnLRNCrossChannelForward = __root.cudnnLRNCrossChannelForward;
    pub const cudnnDivisiveNormalizationForward = __root.cudnnDivisiveNormalizationForward;
    pub const cudnnBatchNormalizationForwardInference = __root.cudnnBatchNormalizationForwardInference;
    pub const cudnnNormalizationForwardInference = __root.cudnnNormalizationForwardInference;
    pub const cudnnSpatialTfGridGeneratorForward = __root.cudnnSpatialTfGridGeneratorForward;
    pub const cudnnSpatialTfSamplerForward = __root.cudnnSpatialTfSamplerForward;
    pub const cudnnDropoutGetStatesSize = __root.cudnnDropoutGetStatesSize;
    pub const cudnnDropoutForward = __root.cudnnDropoutForward;
    pub const cudnnSoftmaxBackward = __root.cudnnSoftmaxBackward;
    pub const cudnnPoolingBackward = __root.cudnnPoolingBackward;
    pub const cudnnActivationBackward = __root.cudnnActivationBackward;
    pub const cudnnLRNCrossChannelBackward = __root.cudnnLRNCrossChannelBackward;
    pub const cudnnDivisiveNormalizationBackward = __root.cudnnDivisiveNormalizationBackward;
    pub const cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize = __root.cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize;
    pub const cudnnGetBatchNormalizationBackwardExWorkspaceSize = __root.cudnnGetBatchNormalizationBackwardExWorkspaceSize;
    pub const cudnnGetBatchNormalizationTrainingExReserveSpaceSize = __root.cudnnGetBatchNormalizationTrainingExReserveSpaceSize;
    pub const cudnnBatchNormalizationForwardTraining = __root.cudnnBatchNormalizationForwardTraining;
    pub const cudnnBatchNormalizationForwardTrainingEx = __root.cudnnBatchNormalizationForwardTrainingEx;
    pub const cudnnBatchNormalizationBackward = __root.cudnnBatchNormalizationBackward;
    pub const cudnnBatchNormalizationBackwardEx = __root.cudnnBatchNormalizationBackwardEx;
    pub const cudnnGetNormalizationForwardTrainingWorkspaceSize = __root.cudnnGetNormalizationForwardTrainingWorkspaceSize;
    pub const cudnnGetNormalizationBackwardWorkspaceSize = __root.cudnnGetNormalizationBackwardWorkspaceSize;
    pub const cudnnGetNormalizationTrainingReserveSpaceSize = __root.cudnnGetNormalizationTrainingReserveSpaceSize;
    pub const cudnnNormalizationForwardTraining = __root.cudnnNormalizationForwardTraining;
    pub const cudnnNormalizationBackward = __root.cudnnNormalizationBackward;
    pub const cudnnSpatialTfGridGeneratorBackward = __root.cudnnSpatialTfGridGeneratorBackward;
    pub const cudnnSpatialTfSamplerBackward = __root.cudnnSpatialTfSamplerBackward;
    pub const cudnnDropoutBackward = __root.cudnnDropoutBackward;
    pub const cudnnGetConvolutionForwardAlgorithmMaxCount = __root.cudnnGetConvolutionForwardAlgorithmMaxCount;
    pub const cudnnGetConvolutionForwardAlgorithm_v7 = __root.cudnnGetConvolutionForwardAlgorithm_v7;
    pub const cudnnFindConvolutionForwardAlgorithm = __root.cudnnFindConvolutionForwardAlgorithm;
    pub const cudnnFindConvolutionForwardAlgorithmEx = __root.cudnnFindConvolutionForwardAlgorithmEx;
    pub const cudnnIm2Col = __root.cudnnIm2Col;
    pub const cudnnReorderFilterAndBias = __root.cudnnReorderFilterAndBias;
    pub const cudnnGetConvolutionForwardWorkspaceSize = __root.cudnnGetConvolutionForwardWorkspaceSize;
    pub const cudnnConvolutionForward = __root.cudnnConvolutionForward;
    pub const cudnnConvolutionBiasActivationForward = __root.cudnnConvolutionBiasActivationForward;
    pub const cudnnGetConvolutionBackwardDataAlgorithmMaxCount = __root.cudnnGetConvolutionBackwardDataAlgorithmMaxCount;
    pub const cudnnFindConvolutionBackwardDataAlgorithm = __root.cudnnFindConvolutionBackwardDataAlgorithm;
    pub const cudnnFindConvolutionBackwardDataAlgorithmEx = __root.cudnnFindConvolutionBackwardDataAlgorithmEx;
    pub const cudnnGetConvolutionBackwardDataAlgorithm_v7 = __root.cudnnGetConvolutionBackwardDataAlgorithm_v7;
    pub const cudnnGetConvolutionBackwardDataWorkspaceSize = __root.cudnnGetConvolutionBackwardDataWorkspaceSize;
    pub const cudnnConvolutionBackwardData = __root.cudnnConvolutionBackwardData;
    pub const cudnnGetFoldedConvBackwardDataDescriptors = __root.cudnnGetFoldedConvBackwardDataDescriptors;
    pub const cudnnGetConvolutionBackwardFilterAlgorithmMaxCount = __root.cudnnGetConvolutionBackwardFilterAlgorithmMaxCount;
    pub const cudnnFindConvolutionBackwardFilterAlgorithm = __root.cudnnFindConvolutionBackwardFilterAlgorithm;
    pub const cudnnFindConvolutionBackwardFilterAlgorithmEx = __root.cudnnFindConvolutionBackwardFilterAlgorithmEx;
    pub const cudnnGetConvolutionBackwardFilterAlgorithm_v7 = __root.cudnnGetConvolutionBackwardFilterAlgorithm_v7;
    pub const cudnnGetConvolutionBackwardFilterWorkspaceSize = __root.cudnnGetConvolutionBackwardFilterWorkspaceSize;
    pub const cudnnConvolutionBackwardFilter = __root.cudnnConvolutionBackwardFilter;
    pub const cudnnConvolutionBackwardBias = __root.cudnnConvolutionBackwardBias;
    pub const cudnnMakeFusedOpsPlan = __root.cudnnMakeFusedOpsPlan;
    pub const cudnnFusedOpsExecute = __root.cudnnFusedOpsExecute;
    pub const v7 = __root.cudnnGetConvolutionForwardAlgorithm_v7;
};
pub const cudnnHandle_t = ?*struct_cudnnContext;
pub extern fn cudnnGetVersion() usize;
pub extern fn cudnnGetMaxDeviceVersion() usize;
pub extern fn cudnnGetCudartVersion() usize;
pub const CUDNN_STATUS_SUCCESS: c_int = 0;
pub const CUDNN_STATUS_NOT_INITIALIZED: c_int = 1001;
pub const CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH: c_int = 1002;
pub const CUDNN_STATUS_SERIALIZATION_VERSION_MISMATCH: c_int = 1003;
pub const CUDNN_STATUS_DEPRECATED: c_int = 1004;
pub const CUDNN_STATUS_LICENSE_ERROR: c_int = 1005;
pub const CUDNN_STATUS_RUNTIME_IN_PROGRESS: c_int = 1006;
pub const CUDNN_STATUS_RUNTIME_FP_OVERFLOW: c_int = 1007;
pub const CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED: c_int = 1008;
pub const CUDNN_STATUS_BAD_PARAM: c_int = 2000;
pub const CUDNN_STATUS_BAD_PARAM_NULL_POINTER: c_int = 2002;
pub const CUDNN_STATUS_BAD_PARAM_MISALIGNED_POINTER: c_int = 2003;
pub const CUDNN_STATUS_BAD_PARAM_NOT_FINALIZED: c_int = 2004;
pub const CUDNN_STATUS_BAD_PARAM_OUT_OF_BOUND: c_int = 2005;
pub const CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT: c_int = 2006;
pub const CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH: c_int = 2007;
pub const CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH: c_int = 2008;
pub const CUDNN_STATUS_BAD_PARAM_DUPLICATED_ENTRIES: c_int = 2009;
pub const CUDNN_STATUS_BAD_PARAM_ATTRIBUTE_TYPE: c_int = 2010;
pub const CUDNN_STATUS_BAD_PARAM_CUDA_GRAPH_MISMATCH: c_int = 2011;
pub const CUDNN_STATUS_BAD_PARAM_DESCRIPTOR_TYPE: c_int = 2012;
pub const CUDNN_STATUS_NOT_SUPPORTED: c_int = 3000;
pub const CUDNN_STATUS_NOT_SUPPORTED_GRAPH_PATTERN: c_int = 3001;
pub const CUDNN_STATUS_NOT_SUPPORTED_SHAPE: c_int = 3002;
pub const CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE: c_int = 3003;
pub const CUDNN_STATUS_NOT_SUPPORTED_LAYOUT: c_int = 3004;
pub const CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDA_DRIVER: c_int = 3005;
pub const CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDART: c_int = 3006;
pub const CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH: c_int = 3007;
pub const CUDNN_STATUS_NOT_SUPPORTED_RUNTIME_PREREQUISITE_MISSING: c_int = 3008;
pub const CUDNN_STATUS_NOT_SUPPORTED_SUBLIBRARY_UNAVAILABLE: c_int = 3009;
pub const CUDNN_STATUS_NOT_SUPPORTED_SHARED_MEMORY_INSUFFICIENT: c_int = 3010;
pub const CUDNN_STATUS_NOT_SUPPORTED_PADDING: c_int = 3011;
pub const CUDNN_STATUS_NOT_SUPPORTED_BAD_LAUNCH_PARAM: c_int = 3012;
pub const CUDNN_STATUS_NOT_SUPPORTED_CUDA_GRAPH_NATIVE_API: c_int = 3013;
pub const CUDNN_STATUS_NOT_SUPPORTED_INVALID_DYNAMIC_SHAPE: c_int = 3014;
pub const CUDNN_STATUS_INTERNAL_ERROR: c_int = 4000;
pub const CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED: c_int = 4001;
pub const CUDNN_STATUS_INTERNAL_ERROR_UNEXPECTED_VALUE: c_int = 4002;
pub const CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED: c_int = 4003;
pub const CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED: c_int = 4004;
pub const CUDNN_STATUS_INTERNAL_ERROR_BAD_LAUNCH_PARAM: c_int = 4005;
pub const CUDNN_STATUS_INTERNAL_ERROR_TEXTURE_CREATION_FAILED: c_int = 4006;
pub const CUDNN_STATUS_EXECUTION_FAILED: c_int = 5000;
pub const CUDNN_STATUS_EXECUTION_FAILED_CUDA_DRIVER: c_int = 5001;
pub const CUDNN_STATUS_EXECUTION_FAILED_CUBLAS: c_int = 5002;
pub const CUDNN_STATUS_EXECUTION_FAILED_CUDART: c_int = 5003;
pub const CUDNN_STATUS_EXECUTION_FAILED_CURAND: c_int = 5004;
pub const CUDNN_STATUS_ALLOC_FAILED: c_int = 4003;
pub const CUDNN_STATUS_INVALID_VALUE: c_int = 2001;
pub const CUDNN_STATUS_ARCH_MISMATCH: c_int = 3007;
pub const CUDNN_STATUS_MAPPING_ERROR: c_int = 4006;
pub const CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING: c_int = 3008;
pub const CUDNN_STATUS_VERSION_MISMATCH: c_int = 1002;
pub const cudnnStatus_t = c_uint;
pub extern fn cudnnGetErrorString(status: cudnnStatus_t) [*c]const u8;
pub extern fn cudnnGetLastErrorString(message: [*c]u8, max_size: usize) void;
pub const struct_cudnnRuntimeTag_t = opaque {};
pub const cudnnRuntimeTag_t = struct_cudnnRuntimeTag_t;
pub const CUDNN_ERRQUERY_RAWCODE: c_int = 0;
pub const CUDNN_ERRQUERY_NONBLOCKING: c_int = 1;
pub const CUDNN_ERRQUERY_BLOCKING: c_int = 2;
pub const cudnnErrQueryMode_t = c_uint;
pub extern fn cudnnQueryRuntimeError(handle: cudnnHandle_t, rstatus: [*c]cudnnStatus_t, mode: cudnnErrQueryMode_t, tag: ?*cudnnRuntimeTag_t) cudnnStatus_t;
pub extern fn cudnnGetProperty(@"type": libraryPropertyType, value: [*c]c_int) cudnnStatus_t;
pub extern fn cudnnCreate(handle: [*c]cudnnHandle_t) cudnnStatus_t;
pub extern fn cudnnDestroy(handle: cudnnHandle_t) cudnnStatus_t;
pub extern fn cudnnSetStream(handle: cudnnHandle_t, streamId: cudaStream_t) cudnnStatus_t;
pub extern fn cudnnGetStream(handle: cudnnHandle_t, streamId: [*c]cudaStream_t) cudnnStatus_t;
pub const CUDNN_DATA_FLOAT: c_int = 0;
pub const CUDNN_DATA_DOUBLE: c_int = 1;
pub const CUDNN_DATA_HALF: c_int = 2;
pub const CUDNN_DATA_INT8: c_int = 3;
pub const CUDNN_DATA_INT32: c_int = 4;
pub const CUDNN_DATA_INT8x4: c_int = 5;
pub const CUDNN_DATA_UINT8: c_int = 6;
pub const CUDNN_DATA_UINT8x4: c_int = 7;
pub const CUDNN_DATA_INT8x32: c_int = 8;
pub const CUDNN_DATA_BFLOAT16: c_int = 9;
pub const CUDNN_DATA_INT64: c_int = 10;
pub const CUDNN_DATA_BOOLEAN: c_int = 11;
pub const CUDNN_DATA_FP8_E4M3: c_int = 12;
pub const CUDNN_DATA_FP8_E5M2: c_int = 13;
pub const CUDNN_DATA_FAST_FLOAT_FOR_FP8: c_int = 14;
pub const CUDNN_DATA_FP8_E8M0: c_int = 15;
pub const CUDNN_DATA_FP4_E2M1: c_int = 16;
pub const CUDNN_DATA_INT4: c_int = 17;
pub const CUDNN_DATA_UINT4: c_int = 18;
pub const CUDNN_DATA_UINT32: c_int = 19;
pub const CUDNN_DATA_COMPLEX_FP32: c_int = 20;
pub const CUDNN_DATA_COMPLEX_FP64: c_int = 21;
pub const cudnnDataType_t = c_uint;
pub const CUDNN_DEFAULT_MATH: c_int = 0;
pub const CUDNN_TENSOR_OP_MATH: c_int = 1;
pub const CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION: c_int = 2;
pub const CUDNN_FMA_MATH: c_int = 3;
pub const cudnnMathType_t = c_uint;
pub const CUDNN_NOT_PROPAGATE_NAN: c_int = 0;
pub const CUDNN_PROPAGATE_NAN: c_int = 1;
pub const cudnnNanPropagation_t = c_uint;
pub const CUDNN_CTC_ZERO_OOB_GRADIENTS: c_int = 0;
pub const CUDNN_CTC_SKIP_OOB_GRADIENTS: c_int = 1;
pub const cudnnCTCGradMode_t = c_uint;
pub const CUDNN_TENSOR_NCHW: c_int = 0;
pub const CUDNN_TENSOR_NHWC: c_int = 1;
pub const CUDNN_TENSOR_NCHW_VECT_C: c_int = 2;
pub const cudnnTensorFormat_t = c_uint;
pub const CUDNN_REDUCE_TENSOR_ADD: c_int = 0;
pub const CUDNN_REDUCE_TENSOR_MUL: c_int = 1;
pub const CUDNN_REDUCE_TENSOR_MIN: c_int = 2;
pub const CUDNN_REDUCE_TENSOR_MAX: c_int = 3;
pub const CUDNN_REDUCE_TENSOR_AMAX: c_int = 4;
pub const CUDNN_REDUCE_TENSOR_AVG: c_int = 5;
pub const CUDNN_REDUCE_TENSOR_NORM1: c_int = 6;
pub const CUDNN_REDUCE_TENSOR_NORM2: c_int = 7;
pub const CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS: c_int = 8;
pub const cudnnReduceTensorOp_t = c_uint;
pub const CUDNN_ACTIVATION_SIGMOID: c_int = 0;
pub const CUDNN_ACTIVATION_RELU: c_int = 1;
pub const CUDNN_ACTIVATION_TANH: c_int = 2;
pub const CUDNN_ACTIVATION_CLIPPED_RELU: c_int = 3;
pub const CUDNN_ACTIVATION_ELU: c_int = 4;
pub const CUDNN_ACTIVATION_IDENTITY: c_int = 5;
pub const CUDNN_ACTIVATION_SWISH: c_int = 6;
pub const cudnnActivationMode_t = c_uint;
pub const CUDNN_SEV_FATAL: c_int = 0;
pub const CUDNN_SEV_ERROR: c_int = 1;
pub const CUDNN_SEV_WARNING: c_int = 2;
pub const CUDNN_SEV_INFO: c_int = 3;
pub const cudnnSeverity_t = c_uint;
pub const struct_cudnnDebugStruct = extern struct {
    cudnn_version: c_uint = 0,
    cudnnStatus: cudnnStatus_t = @import("std").mem.zeroes(cudnnStatus_t),
    time_sec: c_uint = 0,
    time_usec: c_uint = 0,
    time_delta: c_uint = 0,
    handle: cudnnHandle_t = null,
    stream: cudaStream_t = null,
    pid: c_ulonglong = 0,
    tid: c_ulonglong = 0,
    cudaDeviceId: c_int = 0,
    reserved: [15]c_int = @import("std").mem.zeroes([15]c_int),
};
pub const cudnnDebug_t = struct_cudnnDebugStruct;
pub const cudnnCallback_t = ?*const fn (sev: cudnnSeverity_t, udata: ?*anyopaque, dbg: [*c]const cudnnDebug_t, msg: [*c]const u8) callconv(.c) void;
pub extern fn cudnnSetCallback(mask: c_uint, udata: ?*anyopaque, fptr: cudnnCallback_t) cudnnStatus_t;
pub extern fn cudnnGetCallback(mask: [*c]c_uint, udata: [*c]?*anyopaque, fptr: [*c]cudnnCallback_t) cudnnStatus_t;
pub extern fn cudnnGraphVersionCheck() cudnnStatus_t;
pub const CUDNN_CONVOLUTION: c_int = 0;
pub const CUDNN_CROSS_CORRELATION: c_int = 1;
pub const cudnnConvolutionMode_t = c_uint;
pub const CUDNN_DEFAULT_REORDER: c_int = 0;
pub const CUDNN_NO_REORDER: c_int = 1;
pub const cudnnReorderType_t = c_uint;
pub const cudnnBackendDescriptor_t = ?*anyopaque;
pub const struct_cudnnFractionStruct = extern struct {
    numerator: i64 = 0,
    denominator: i64 = 0,
};
pub const cudnnFraction_t = struct_cudnnFractionStruct;
pub const CUDNN_POINTWISE_ADD: c_int = 0;
pub const CUDNN_POINTWISE_ADD_SQUARE: c_int = 5;
pub const CUDNN_POINTWISE_DIV: c_int = 6;
pub const CUDNN_POINTWISE_MAX: c_int = 3;
pub const CUDNN_POINTWISE_MIN: c_int = 2;
pub const CUDNN_POINTWISE_MOD: c_int = 7;
pub const CUDNN_POINTWISE_MUL: c_int = 1;
pub const CUDNN_POINTWISE_POW: c_int = 8;
pub const CUDNN_POINTWISE_SUB: c_int = 9;
pub const CUDNN_POINTWISE_ABS: c_int = 10;
pub const CUDNN_POINTWISE_CEIL: c_int = 11;
pub const CUDNN_POINTWISE_COS: c_int = 12;
pub const CUDNN_POINTWISE_EXP: c_int = 13;
pub const CUDNN_POINTWISE_FLOOR: c_int = 14;
pub const CUDNN_POINTWISE_LOG: c_int = 15;
pub const CUDNN_POINTWISE_NEG: c_int = 16;
pub const CUDNN_POINTWISE_RSQRT: c_int = 17;
pub const CUDNN_POINTWISE_SIN: c_int = 18;
pub const CUDNN_POINTWISE_SQRT: c_int = 4;
pub const CUDNN_POINTWISE_TAN: c_int = 19;
pub const CUDNN_POINTWISE_ERF: c_int = 20;
pub const CUDNN_POINTWISE_IDENTITY: c_int = 21;
pub const CUDNN_POINTWISE_RECIPROCAL: c_int = 22;
pub const CUDNN_POINTWISE_ATAN2: c_int = 23;
pub const CUDNN_POINTWISE_RELU_FWD: c_int = 100;
pub const CUDNN_POINTWISE_TANH_FWD: c_int = 101;
pub const CUDNN_POINTWISE_SIGMOID_FWD: c_int = 102;
pub const CUDNN_POINTWISE_ELU_FWD: c_int = 103;
pub const CUDNN_POINTWISE_GELU_FWD: c_int = 104;
pub const CUDNN_POINTWISE_SOFTPLUS_FWD: c_int = 105;
pub const CUDNN_POINTWISE_SWISH_FWD: c_int = 106;
pub const CUDNN_POINTWISE_GELU_APPROX_TANH_FWD: c_int = 107;
pub const CUDNN_POINTWISE_RELU_BWD: c_int = 200;
pub const CUDNN_POINTWISE_TANH_BWD: c_int = 201;
pub const CUDNN_POINTWISE_SIGMOID_BWD: c_int = 202;
pub const CUDNN_POINTWISE_ELU_BWD: c_int = 203;
pub const CUDNN_POINTWISE_GELU_BWD: c_int = 204;
pub const CUDNN_POINTWISE_SOFTPLUS_BWD: c_int = 205;
pub const CUDNN_POINTWISE_SWISH_BWD: c_int = 206;
pub const CUDNN_POINTWISE_GELU_APPROX_TANH_BWD: c_int = 207;
pub const CUDNN_POINTWISE_CMP_EQ: c_int = 300;
pub const CUDNN_POINTWISE_CMP_NEQ: c_int = 301;
pub const CUDNN_POINTWISE_CMP_GT: c_int = 302;
pub const CUDNN_POINTWISE_CMP_GE: c_int = 303;
pub const CUDNN_POINTWISE_CMP_LT: c_int = 304;
pub const CUDNN_POINTWISE_CMP_LE: c_int = 305;
pub const CUDNN_POINTWISE_LOGICAL_AND: c_int = 400;
pub const CUDNN_POINTWISE_LOGICAL_OR: c_int = 401;
pub const CUDNN_POINTWISE_LOGICAL_NOT: c_int = 402;
pub const CUDNN_POINTWISE_GEN_INDEX: c_int = 501;
pub const CUDNN_POINTWISE_BINARY_SELECT: c_int = 601;
pub const cudnnPointwiseMode_t = c_uint;
pub const CUDNN_RESAMPLE_NEAREST: c_int = 0;
pub const CUDNN_RESAMPLE_BILINEAR: c_int = 1;
pub const CUDNN_RESAMPLE_AVGPOOL: c_int = 2;
pub const CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING: c_int = 2;
pub const CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING: c_int = 4;
pub const CUDNN_RESAMPLE_MAXPOOL: c_int = 3;
pub const cudnnResampleMode_t = c_uint;
pub const CUDNN_SIGNAL_SET: c_int = 0;
pub const CUDNN_SIGNAL_WAIT: c_int = 1;
pub const cudnnSignalMode_t = c_uint;
pub const CUDNN_GENSTATS_SUM_SQSUM: c_int = 0;
pub const cudnnGenStatsMode_t = c_uint;
pub const CUDNN_BN_FINALIZE_STATISTICS_TRAINING: c_int = 0;
pub const CUDNN_BN_FINALIZE_STATISTICS_INFERENCE: c_int = 1;
pub const cudnnBnFinalizeStatsMode_t = c_uint;
pub const CUDNN_RNG_DISTRIBUTION_BERNOULLI: c_int = 0;
pub const CUDNN_RNG_DISTRIBUTION_UNIFORM: c_int = 1;
pub const CUDNN_RNG_DISTRIBUTION_NORMAL: c_int = 2;
pub const cudnnRngDistribution_t = c_uint;
pub const CUDNN_MOE_GROUPED_MATMUL_MODE_NONE: c_int = 0;
pub const CUDNN_MOE_GROUPED_MATMUL_MODE_GATHER: c_int = 1;
pub const CUDNN_MOE_GROUPED_MATMUL_MODE_SCATTER: c_int = 2;
pub const cudnnMoeGroupedMatmulMode_t = c_uint;
pub const CUDNN_ATTR_POINTWISE_MODE: c_int = 0;
pub const CUDNN_ATTR_POINTWISE_MATH_PREC: c_int = 1;
pub const CUDNN_ATTR_POINTWISE_NAN_PROPAGATION: c_int = 2;
pub const CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP: c_int = 3;
pub const CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP: c_int = 4;
pub const CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE: c_int = 5;
pub const CUDNN_ATTR_POINTWISE_ELU_ALPHA: c_int = 6;
pub const CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA: c_int = 7;
pub const CUDNN_ATTR_POINTWISE_SWISH_BETA: c_int = 8;
pub const CUDNN_ATTR_POINTWISE_AXIS: c_int = 9;
pub const CUDNN_ATTR_CONVOLUTION_COMP_TYPE: c_int = 100;
pub const CUDNN_ATTR_CONVOLUTION_CONV_MODE: c_int = 101;
pub const CUDNN_ATTR_CONVOLUTION_DILATIONS: c_int = 102;
pub const CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES: c_int = 103;
pub const CUDNN_ATTR_CONVOLUTION_POST_PADDINGS: c_int = 104;
pub const CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS: c_int = 105;
pub const CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS: c_int = 106;
pub const CUDNN_ATTR_ENGINEHEUR_MODE: c_int = 200;
pub const CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH: c_int = 201;
pub const CUDNN_ATTR_ENGINEHEUR_RESULTS: c_int = 202;
pub const CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET: c_int = 203;
pub const CUDNN_ATTR_ENGINEHEUR_DEVICEPROP: c_int = 204;
pub const CUDNN_ATTR_ENGINECFG_ENGINE: c_int = 300;
pub const CUDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO: c_int = 301;
pub const CUDNN_ATTR_ENGINECFG_KNOB_CHOICES: c_int = 302;
pub const CUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE: c_int = 303;
pub const CUDNN_ATTR_ENGINECFG_SHARED_MEMORY_USED: c_int = 304;
pub const CUDNN_ATTR_EXECUTION_PLAN_HANDLE: c_int = 400;
pub const CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG: c_int = 401;
pub const CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE: c_int = 402;
pub const CUDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS: c_int = 403;
pub const CUDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS: c_int = 404;
pub const CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION: c_int = 405;
pub const CUDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE: c_int = 406;
pub const CUDNN_ATTR_EXECUTION_PLAN_DEVICEPROP: c_int = 407;
pub const CUDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID: c_int = 500;
pub const CUDNN_ATTR_INTERMEDIATE_INFO_SIZE: c_int = 501;
pub const CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS: c_int = 502;
pub const CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES: c_int = 503;
pub const CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE: c_int = 600;
pub const CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE: c_int = 601;
pub const CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA: c_int = 700;
pub const CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA: c_int = 701;
pub const CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC: c_int = 702;
pub const CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W: c_int = 703;
pub const CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X: c_int = 704;
pub const CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y: c_int = 705;
pub const CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA: c_int = 706;
pub const CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA: c_int = 707;
pub const CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC: c_int = 708;
pub const CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W: c_int = 709;
pub const CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX: c_int = 710;
pub const CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY: c_int = 711;
pub const CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA: c_int = 712;
pub const CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA: c_int = 713;
pub const CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC: c_int = 714;
pub const CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW: c_int = 715;
pub const CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X: c_int = 716;
pub const CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY: c_int = 717;
pub const CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR: c_int = 750;
pub const CUDNN_ATTR_OPERATION_POINTWISE_XDESC: c_int = 751;
pub const CUDNN_ATTR_OPERATION_POINTWISE_BDESC: c_int = 752;
pub const CUDNN_ATTR_OPERATION_POINTWISE_YDESC: c_int = 753;
pub const CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1: c_int = 754;
pub const CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2: c_int = 755;
pub const CUDNN_ATTR_OPERATION_POINTWISE_DXDESC: c_int = 756;
pub const CUDNN_ATTR_OPERATION_POINTWISE_DYDESC: c_int = 757;
pub const CUDNN_ATTR_OPERATION_POINTWISE_TDESC: c_int = 758;
pub const CUDNN_ATTR_OPERATION_GENSTATS_MODE: c_int = 770;
pub const CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC: c_int = 771;
pub const CUDNN_ATTR_OPERATION_GENSTATS_XDESC: c_int = 772;
pub const CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC: c_int = 773;
pub const CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC: c_int = 774;
pub const CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE: c_int = 780;
pub const CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC: c_int = 781;
pub const CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC: c_int = 782;
pub const CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC: c_int = 783;
pub const CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC: c_int = 784;
pub const CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC: c_int = 785;
pub const CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC: c_int = 786;
pub const CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC: c_int = 787;
pub const CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC: c_int = 788;
pub const CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC: c_int = 789;
pub const CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC: c_int = 790;
pub const CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC: c_int = 791;
pub const CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC: c_int = 792;
pub const CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC: c_int = 793;
pub const CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC: c_int = 794;
pub const CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC: c_int = 795;
pub const CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC: c_int = 796;
pub const CUDNN_ATTR_OPERATIONGRAPH_HANDLE: c_int = 800;
pub const CUDNN_ATTR_OPERATIONGRAPH_OPS: c_int = 801;
pub const CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT: c_int = 802;
pub const CUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED: c_int = 803;
pub const CUDNN_ATTR_OPERATIONGRAPH_IS_SAME_TOPOLOGY: c_int = 804;
pub const CUDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED: c_int = 805;
pub const CUDNN_ATTR_OPERATIONGRAPH_MODE: c_int = 806;
pub const CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT: c_int = 900;
pub const CUDNN_ATTR_TENSOR_DATA_TYPE: c_int = 901;
pub const CUDNN_ATTR_TENSOR_DIMENSIONS: c_int = 902;
pub const CUDNN_ATTR_TENSOR_STRIDES: c_int = 903;
pub const CUDNN_ATTR_TENSOR_VECTOR_COUNT: c_int = 904;
pub const CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION: c_int = 905;
pub const CUDNN_ATTR_TENSOR_UNIQUE_ID: c_int = 906;
pub const CUDNN_ATTR_TENSOR_IS_VIRTUAL: c_int = 907;
pub const CUDNN_ATTR_TENSOR_IS_BY_VALUE: c_int = 908;
pub const CUDNN_ATTR_TENSOR_REORDERING_MODE: c_int = 909;
pub const CUDNN_ATTR_TENSOR_CONSTANT_VALUE: c_int = 910;
pub const CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC: c_int = 913;
pub const CUDNN_ATTR_TENSOR_RAGGED_OFFSET_MULTIPLIER: c_int = 914;
pub const CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS: c_int = 1000;
pub const CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS: c_int = 1001;
pub const CUDNN_ATTR_VARIANT_PACK_INTERMEDIATES: c_int = 1002;
pub const CUDNN_ATTR_VARIANT_PACK_WORKSPACE: c_int = 1003;
pub const CUDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS: c_int = 1010;
pub const CUDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES: c_int = 1011;
pub const CUDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES: c_int = 1012;
pub const CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID: c_int = 1100;
pub const CUDNN_ATTR_LAYOUT_INFO_TYPES: c_int = 1101;
pub const CUDNN_ATTR_KNOB_INFO_TYPE: c_int = 1200;
pub const CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE: c_int = 1201;
pub const CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE: c_int = 1202;
pub const CUDNN_ATTR_KNOB_INFO_STRIDE: c_int = 1203;
pub const CUDNN_ATTR_ENGINE_OPERATION_GRAPH: c_int = 1300;
pub const CUDNN_ATTR_ENGINE_GLOBAL_INDEX: c_int = 1301;
pub const CUDNN_ATTR_ENGINE_KNOB_INFO: c_int = 1302;
pub const CUDNN_ATTR_ENGINE_NUMERICAL_NOTE: c_int = 1303;
pub const CUDNN_ATTR_ENGINE_LAYOUT_INFO: c_int = 1304;
pub const CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE: c_int = 1305;
pub const CUDNN_ATTR_ENGINE_SM_COUNT_TARGET: c_int = 1306;
pub const CUDNN_ATTR_ENGINE_DEVICEPROP: c_int = 1307;
pub const CUDNN_ATTR_ENGINE_DISABLE_CLUSTER_COOPERATIVE: c_int = 1308;
pub const CUDNN_ATTR_MATMUL_COMP_TYPE: c_int = 1500;
pub const CUDNN_ATTR_MATMUL_PADDING_VALUE: c_int = 1503;
pub const CUDNN_ATTR_OPERATION_MATMUL_ADESC: c_int = 1520;
pub const CUDNN_ATTR_OPERATION_MATMUL_BDESC: c_int = 1521;
pub const CUDNN_ATTR_OPERATION_MATMUL_CDESC: c_int = 1522;
pub const CUDNN_ATTR_OPERATION_MATMUL_DESC: c_int = 1523;
pub const CUDNN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT: c_int = 1524;
pub const CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC: c_int = 1525;
pub const CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC: c_int = 1526;
pub const CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC: c_int = 1527;
pub const CUDNN_ATTR_REDUCTION_OPERATOR: c_int = 1600;
pub const CUDNN_ATTR_REDUCTION_COMP_TYPE: c_int = 1601;
pub const CUDNN_ATTR_REDUCTION_IS_DETERMINISTIC: c_int = 1602;
pub const CUDNN_ATTR_OPERATION_REDUCTION_XDESC: c_int = 1610;
pub const CUDNN_ATTR_OPERATION_REDUCTION_YDESC: c_int = 1611;
pub const CUDNN_ATTR_OPERATION_REDUCTION_DESC: c_int = 1612;
pub const CUDNN_ATTR_OPERATION_REDUCTION_GROUP_OFFSET_DESC: c_int = 1613;
pub const CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC: c_int = 1620;
pub const CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC: c_int = 1621;
pub const CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC: c_int = 1622;
pub const CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC: c_int = 1623;
pub const CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC: c_int = 1624;
pub const CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC: c_int = 1625;
pub const CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC: c_int = 1626;
pub const CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC: c_int = 1627;
pub const CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC: c_int = 1628;
pub const CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC: c_int = 1629;
pub const CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS: c_int = 1630;
pub const CUDNN_ATTR_RESAMPLE_MODE: c_int = 1700;
pub const CUDNN_ATTR_RESAMPLE_COMP_TYPE: c_int = 1701;
pub const CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS: c_int = 1702;
pub const CUDNN_ATTR_RESAMPLE_POST_PADDINGS: c_int = 1703;
pub const CUDNN_ATTR_RESAMPLE_PRE_PADDINGS: c_int = 1704;
pub const CUDNN_ATTR_RESAMPLE_STRIDES: c_int = 1705;
pub const CUDNN_ATTR_RESAMPLE_WINDOW_DIMS: c_int = 1706;
pub const CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION: c_int = 1707;
pub const CUDNN_ATTR_RESAMPLE_PADDING_MODE: c_int = 1708;
pub const CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC: c_int = 1710;
pub const CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC: c_int = 1711;
pub const CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC: c_int = 1712;
pub const CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA: c_int = 1713;
pub const CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA: c_int = 1714;
pub const CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC: c_int = 1716;
pub const CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC: c_int = 1720;
pub const CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC: c_int = 1721;
pub const CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC: c_int = 1722;
pub const CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA: c_int = 1723;
pub const CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA: c_int = 1724;
pub const CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC: c_int = 1725;
pub const CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC: c_int = 1726;
pub const CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC: c_int = 1727;
pub const CUDNN_ATTR_OPERATION_CONCAT_AXIS: c_int = 1800;
pub const CUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS: c_int = 1801;
pub const CUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX: c_int = 1802;
pub const CUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC: c_int = 1803;
pub const CUDNN_ATTR_OPERATION_SIGNAL_MODE: c_int = 1900;
pub const CUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC: c_int = 1901;
pub const CUDNN_ATTR_OPERATION_SIGNAL_VALUE: c_int = 1902;
pub const CUDNN_ATTR_OPERATION_SIGNAL_XDESC: c_int = 1903;
pub const CUDNN_ATTR_OPERATION_SIGNAL_YDESC: c_int = 1904;
pub const CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_CONTAINER_DESC: c_int = 1950;
pub const CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_YDESC: c_int = 1951;
pub const CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_SEQUENCE_DESC: c_int = 1952;
pub const CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_PAGE_TABLE_DESC: c_int = 1953;
pub const CUDNN_ATTR_OPERATION_NORM_FWD_MODE: c_int = 2000;
pub const CUDNN_ATTR_OPERATION_NORM_FWD_PHASE: c_int = 2001;
pub const CUDNN_ATTR_OPERATION_NORM_FWD_XDESC: c_int = 2002;
pub const CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC: c_int = 2003;
pub const CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC: c_int = 2004;
pub const CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC: c_int = 2005;
pub const CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC: c_int = 2006;
pub const CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC: c_int = 2007;
pub const CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC: c_int = 2008;
pub const CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC: c_int = 2009;
pub const CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC: c_int = 2010;
pub const CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC: c_int = 2011;
pub const CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC: c_int = 2012;
pub const CUDNN_ATTR_OPERATION_NORM_FWD_YDESC: c_int = 2013;
pub const CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS: c_int = 2014;
pub const CUDNN_ATTR_OPERATION_NORM_BWD_MODE: c_int = 2100;
pub const CUDNN_ATTR_OPERATION_NORM_BWD_XDESC: c_int = 2101;
pub const CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC: c_int = 2102;
pub const CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC: c_int = 2103;
pub const CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC: c_int = 2104;
pub const CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC: c_int = 2105;
pub const CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC: c_int = 2106;
pub const CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC: c_int = 2107;
pub const CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC: c_int = 2108;
pub const CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC: c_int = 2109;
pub const CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS: c_int = 2110;
pub const CUDNN_ATTR_OPERATION_RESHAPE_XDESC: c_int = 2200;
pub const CUDNN_ATTR_OPERATION_RESHAPE_YDESC: c_int = 2201;
pub const CUDNN_ATTR_OPERATION_RESHAPE_MODE: c_int = 2202;
pub const CUDNN_ATTR_OPERATION_TRANSPOSE_XDESC: c_int = 3200;
pub const CUDNN_ATTR_OPERATION_TRANSPOSE_YDESC: c_int = 3201;
pub const CUDNN_ATTR_OPERATION_TRANSPOSE_PERMUTATION: c_int = 3202;
pub const CUDNN_ATTR_OPERATION_SLICE_XDESC: c_int = 3300;
pub const CUDNN_ATTR_OPERATION_SLICE_YDESC: c_int = 3301;
pub const CUDNN_ATTR_OPERATION_SLICE_START_INDICES: c_int = 3302;
pub const CUDNN_ATTR_OPERATION_SLICE_LIMIT_INDICES: c_int = 3303;
pub const CUDNN_ATTR_OPERATION_SLICE_STRIDES: c_int = 3304;
pub const CUDNN_ATTR_OPERATION_ROPE_FWD_XDESC: c_int = 3400;
pub const CUDNN_ATTR_OPERATION_ROPE_FWD_YDESC: c_int = 3401;
pub const CUDNN_ATTR_OPERATION_ROPE_FWD_FREQSDESC: c_int = 3402;
pub const CUDNN_ATTR_OPERATION_ROPE_FWD_OUTPUT_SCALE: c_int = 3403;
pub const CUDNN_ATTR_OPERATION_ROPE_FWD_ROPE_DIM: c_int = 3404;
pub const CUDNN_ATTR_OPERATION_ROPE_BWD_DYDESC: c_int = 3410;
pub const CUDNN_ATTR_OPERATION_ROPE_BWD_DXDESC: c_int = 3411;
pub const CUDNN_ATTR_OPERATION_ROPE_BWD_FREQSDESC: c_int = 3412;
pub const CUDNN_ATTR_OPERATION_ROPE_BWD_OUTPUT_SCALE: c_int = 3413;
pub const CUDNN_ATTR_OPERATION_ROPE_BWD_ROPE_DIM: c_int = 3414;
pub const CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_XDESC: c_int = 2250;
pub const CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_YDESC: c_int = 2251;
pub const CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_LOWER_BANDWIDTH: c_int = 2252;
pub const CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_UPPER_BANDWIDTH: c_int = 2253;
pub const CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_AXIS: c_int = 2254;
pub const CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_PAD_VALUE: c_int = 2255;
pub const CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_KV_TOKEN_OFFSET_DESC: c_int = 2256;
pub const CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_SPECULATIVE_MASK_DESC: c_int = 2257;
pub const CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_XDESC: c_int = 2270;
pub const CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_YDESC: c_int = 2271;
pub const CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_LOWER_BANDWIDTH: c_int = 2272;
pub const CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_UPPER_BANDWIDTH: c_int = 2273;
pub const CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_AXIS: c_int = 2274;
pub const CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_PAD_VALUE: c_int = 2275;
pub const CUDNN_ATTR_OPERATION_CONTRACT_BAND_MAX_TOKEN_VALUE: c_int = 2276;
pub const CUDNN_ATTR_RNG_DISTRIBUTION: c_int = 2300;
pub const CUDNN_ATTR_RNG_NORMAL_DIST_MEAN: c_int = 2301;
pub const CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION: c_int = 2302;
pub const CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM: c_int = 2303;
pub const CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM: c_int = 2304;
pub const CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY: c_int = 2305;
pub const CUDNN_ATTR_OPERATION_RNG_YDESC: c_int = 2310;
pub const CUDNN_ATTR_OPERATION_RNG_SEED: c_int = 2311;
pub const CUDNN_ATTR_OPERATION_RNG_DESC: c_int = 2312;
pub const CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC: c_int = 2313;
pub const CUDNN_ATTR_KERNEL_CACHE_OPERATION_GRAPH: c_int = 2400;
pub const CUDNN_ATTR_KERNEL_CACHE_IS_ENGINECFG_KERNEL_CACHED: c_int = 2401;
pub const CUDNN_ATTR_KERNEL_CACHE_JSON_REPRESENTATION: c_int = 2402;
pub const CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC: c_int = 2500;
pub const CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC: c_int = 2501;
pub const CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC: c_int = 2502;
pub const CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC: c_int = 2503;
pub const CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE: c_int = 2504;
pub const CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC: c_int = 2600;
pub const CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC: c_int = 2601;
pub const CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC: c_int = 2602;
pub const CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC: c_int = 2603;
pub const CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE: c_int = 2604;
pub const CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE: c_int = 2605;
pub const CUDNN_ATTR_DEVICEPROP_DEVICE_ID: c_int = 2700;
pub const CUDNN_ATTR_DEVICEPROP_HANDLE: c_int = 2701;
pub const CUDNN_ATTR_DEVICEPROP_JSON_REPRESENTATION: c_int = 2702;
pub const CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC: c_int = 2800;
pub const CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC: c_int = 2801;
pub const CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC: c_int = 2802;
pub const CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC: c_int = 2803;
pub const CUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC: c_int = 2804;
pub const CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC: c_int = 2805;
pub const CUDNN_ATTR_OPERATION_SDPA_FWD_BLOCK_MASK_DESC: c_int = 2806;
pub const CUDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_KDESC: c_int = 2807;
pub const CUDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_VDESC: c_int = 2808;
pub const CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC: c_int = 2809;
pub const CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC: c_int = 2810;
pub const CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH: c_int = 2811;
pub const CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_INPUT_UID: c_int = 2812;
pub const CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_OUTPUT_UID: c_int = 2813;
pub const CUDNN_ATTR_OPERATION_SDPA_FWD_SOFTMAX_DESC: c_int = 2814;
pub const CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_SEED_DESC: c_int = 2815;
pub const CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_OFFSET_DESC: c_int = 2816;
pub const CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_RNG_DUMP_DESC: c_int = 2817;
pub const CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_PROBABILITY: c_int = 2818;
pub const CUDNN_ATTR_OPERATION_SDPA_FWD_UNFUSE_FMA: c_int = 2819;
pub const CUDNN_ATTR_OPERATION_SDPA_FWD_CU_SEQ_LEN_QDESC: c_int = 2820;
pub const CUDNN_ATTR_OPERATION_SDPA_FWD_CU_SEQ_LEN_KVDESC: c_int = 2821;
pub const CUDNN_ATTR_OPERATION_SDPA_BWD_QDESC: c_int = 2851;
pub const CUDNN_ATTR_OPERATION_SDPA_BWD_KDESC: c_int = 2852;
pub const CUDNN_ATTR_OPERATION_SDPA_BWD_VDESC: c_int = 2853;
pub const CUDNN_ATTR_OPERATION_SDPA_BWD_ODESC: c_int = 2854;
pub const CUDNN_ATTR_OPERATION_SDPA_BWD_STATSDESC: c_int = 2855;
pub const CUDNN_ATTR_OPERATION_SDPA_BWD_SCALEDESC: c_int = 2856;
pub const CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_QDESC: c_int = 2857;
pub const CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_KVDESC: c_int = 2858;
pub const CUDNN_ATTR_OPERATION_SDPA_BWD_DQDESC: c_int = 2859;
pub const CUDNN_ATTR_OPERATION_SDPA_BWD_DKDESC: c_int = 2860;
pub const CUDNN_ATTR_OPERATION_SDPA_BWD_DVDESC: c_int = 2861;
pub const CUDNN_ATTR_OPERATION_SDPA_BWD_DODDESC: c_int = 2862;
pub const CUDNN_ATTR_OPERATION_SDPA_BWD_SINK_DESC: c_int = 2863;
pub const CUDNN_ATTR_OPERATION_SDPA_BWD_DSINK_DESC: c_int = 2864;
pub const CUDNN_ATTR_OPERATION_SDPA_BWD_MAX_TOTAL_SEQ_LEN_Q: c_int = 2865;
pub const CUDNN_ATTR_OPERATION_SDPA_BWD_MAX_TOTAL_SEQ_LEN_KV: c_int = 2866;
pub const CUDNN_ATTR_OPERATION_SDPA_BWD_SUBGRAPH: c_int = 2867;
pub const CUDNN_ATTR_OPERATION_SDPA_BWD_SUBGRAPH_INPUT_UID: c_int = 2868;
pub const CUDNN_ATTR_OPERATION_SDPA_BWD_SUBGRAPH_OUTPUT_UID: c_int = 2869;
pub const CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE: c_int = 2900;
pub const CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC: c_int = 2901;
pub const CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC: c_int = 2902;
pub const CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC: c_int = 2903;
pub const CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC: c_int = 2904;
pub const CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC: c_int = 2905;
pub const CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC: c_int = 2906;
pub const CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC: c_int = 2907;
pub const CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K: c_int = 2908;
pub const CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC: c_int = 2951;
pub const CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_TOKEN_DESC: c_int = 2952;
pub const CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DWEIGHT_DESC: c_int = 2953;
pub const CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC: c_int = 2954;
pub const CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC: c_int = 2955;
pub const CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_XDESC: c_int = 3000;
pub const CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_KVDESC: c_int = 3001;
pub const CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_QDESC: c_int = 3002;
pub const CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_LEFT_BOUND_DESC: c_int = 3003;
pub const CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SHIFT_RIGHT_BOUND_DESC: c_int = 3004;
pub const CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_BDESC: c_int = 3005;
pub const CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_YDESC: c_int = 3006;
pub const CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_COMPARISON_MODE: c_int = 3007;
pub const CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_CU_SEQ_LEN_KVDESC: c_int = 3008;
pub const CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_CU_SEQ_LEN_QDESC: c_int = 3009;
pub const CUDNN_ATTR_OPERATION_SOFTMAX_XDESC: c_int = 3100;
pub const CUDNN_ATTR_OPERATION_SOFTMAX_YDESC: c_int = 3101;
pub const CUDNN_ATTR_OPERATION_SOFTMAX_STATS_DESC: c_int = 3102;
pub const CUDNN_ATTR_OPERATION_SOFTMAX_MAX_DESC: c_int = 3103;
pub const CUDNN_ATTR_OPERATION_SOFTMAX_SUM_EXP_DESC: c_int = 3104;
pub const CUDNN_ATTR_OPERATION_SOFTMAX_SINK_DESC: c_int = 3105;
pub const cudnnBackendAttributeName_t = c_uint;
pub const CUDNN_TYPE_HANDLE: c_int = 0;
pub const CUDNN_TYPE_DATA_TYPE: c_int = 1;
pub const CUDNN_TYPE_BOOLEAN: c_int = 2;
pub const CUDNN_TYPE_INT64: c_int = 3;
pub const CUDNN_TYPE_FLOAT: c_int = 4;
pub const CUDNN_TYPE_DOUBLE: c_int = 5;
pub const CUDNN_TYPE_VOID_PTR: c_int = 6;
pub const CUDNN_TYPE_CONVOLUTION_MODE: c_int = 7;
pub const CUDNN_TYPE_HEUR_MODE: c_int = 8;
pub const CUDNN_TYPE_KNOB_TYPE: c_int = 9;
pub const CUDNN_TYPE_NAN_PROPOGATION: c_int = 10;
pub const CUDNN_TYPE_NUMERICAL_NOTE: c_int = 11;
pub const CUDNN_TYPE_LAYOUT_TYPE: c_int = 12;
pub const CUDNN_TYPE_ATTRIB_NAME: c_int = 13;
pub const CUDNN_TYPE_POINTWISE_MODE: c_int = 14;
pub const CUDNN_TYPE_BACKEND_DESCRIPTOR: c_int = 15;
pub const CUDNN_TYPE_GENSTATS_MODE: c_int = 16;
pub const CUDNN_TYPE_BN_FINALIZE_STATS_MODE: c_int = 17;
pub const CUDNN_TYPE_REDUCTION_OPERATOR_TYPE: c_int = 18;
pub const CUDNN_TYPE_BEHAVIOR_NOTE: c_int = 19;
pub const CUDNN_TYPE_TENSOR_REORDERING_MODE: c_int = 20;
pub const CUDNN_TYPE_RESAMPLE_MODE: c_int = 21;
pub const CUDNN_TYPE_PADDING_MODE: c_int = 22;
pub const CUDNN_TYPE_INT32: c_int = 23;
pub const CUDNN_TYPE_CHAR: c_int = 24;
pub const CUDNN_TYPE_SIGNAL_MODE: c_int = 25;
pub const CUDNN_TYPE_FRACTION: c_int = 26;
pub const CUDNN_TYPE_NORM_MODE: c_int = 27;
pub const CUDNN_TYPE_NORM_FWD_PHASE: c_int = 28;
pub const CUDNN_TYPE_RNG_DISTRIBUTION: c_int = 29;
pub const CUDNN_TYPE_MOE_GROUPED_MATMUL_MODE: c_int = 30;
pub const CUDNN_TYPE_RESHAPE_MODE: c_int = 31;
pub const CUDNN_TYPE_OPERATIONGRAPH_MODE: c_int = 32;
pub const cudnnBackendAttributeType_t = c_uint;
pub const CUDNN_BACKEND_POINTWISE_DESCRIPTOR: c_int = 0;
pub const CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: c_int = 1;
pub const CUDNN_BACKEND_ENGINE_DESCRIPTOR: c_int = 2;
pub const CUDNN_BACKEND_ENGINECFG_DESCRIPTOR: c_int = 3;
pub const CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: c_int = 4;
pub const CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: c_int = 5;
pub const CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR: c_int = 6;
pub const CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR: c_int = 7;
pub const CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR: c_int = 8;
pub const CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR: c_int = 9;
pub const CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR: c_int = 10;
pub const CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR: c_int = 11;
pub const CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR: c_int = 12;
pub const CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR: c_int = 13;
pub const CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR: c_int = 14;
pub const CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: c_int = 15;
pub const CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR: c_int = 16;
pub const CUDNN_BACKEND_TENSOR_DESCRIPTOR: c_int = 17;
pub const CUDNN_BACKEND_MATMUL_DESCRIPTOR: c_int = 18;
pub const CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR: c_int = 19;
pub const CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR: c_int = 20;
pub const CUDNN_BACKEND_REDUCTION_DESCRIPTOR: c_int = 21;
pub const CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR: c_int = 22;
pub const CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR: c_int = 23;
pub const CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: c_int = 24;
pub const CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR: c_int = 25;
pub const CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR: c_int = 26;
pub const CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR: c_int = 27;
pub const CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR: c_int = 28;
pub const CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR: c_int = 29;
pub const CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR: c_int = 30;
pub const CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR: c_int = 31;
pub const CUDNN_BACKEND_RNG_DESCRIPTOR: c_int = 32;
pub const CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR: c_int = 33;
pub const CUDNN_BACKEND_KERNEL_CACHE_DESCRIPTOR: c_int = 34;
pub const CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR: c_int = 35;
pub const CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR: c_int = 36;
pub const CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR: c_int = 37;
pub const CUDNN_BACKEND_DEVICEPROP_DESCRIPTOR: c_int = 38;
pub const CUDNN_BACKEND_OPERATION_EXPAND_BAND_MATRIX_DESCRIPTOR: c_int = 39;
pub const CUDNN_BACKEND_OPERATION_CONTRACT_BAND_MATRIX_DESCRIPTOR: c_int = 40;
pub const CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR: c_int = 41;
pub const CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR: c_int = 42;
pub const CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR: c_int = 43;
pub const CUDNN_BACKEND_OPERATION_DIAGONAL_BAND_MASK_DESCRIPTOR: c_int = 44;
pub const CUDNN_BACKEND_OPERATION_SOFTMAX_DESCRIPTOR: c_int = 45;
pub const CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR: c_int = 46;
pub const CUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR: c_int = 47;
pub const CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR: c_int = 48;
pub const CUDNN_BACKEND_OPERATION_ROPE_FWD_DESCRIPTOR: c_int = 49;
pub const CUDNN_BACKEND_OPERATION_ROPE_BWD_DESCRIPTOR: c_int = 50;
pub const cudnnBackendDescriptorType_t = c_uint;
pub const CUDNN_NUMERICAL_NOTE_TENSOR_CORE: c_int = 0;
pub const CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS: c_int = 1;
pub const CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION: c_int = 2;
pub const CUDNN_NUMERICAL_NOTE_FFT: c_int = 3;
pub const CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC: c_int = 4;
pub const CUDNN_NUMERICAL_NOTE_WINOGRAD: c_int = 5;
pub const CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4: c_int = 6;
pub const CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6: c_int = 7;
pub const CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13: c_int = 8;
pub const CUDNN_NUMERICAL_NOTE_STRICT_NAN_PROP: c_int = 9;
pub const CUDNN_NUMERICAL_NOTE_TYPE_COUNT: c_int = 10;
pub const cudnnBackendNumericalNote_t = c_uint;
pub const CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION: c_int = 0;
pub const CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER: c_int = 1;
pub const CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER: c_int = 2;
pub const CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API: c_int = 3;
pub const CUDNN_BEHAVIOR_NOTE_CUBLASLT_DEPENDENCY: c_int = 4;
pub const CUDNN_BEHAVIOR_NOTE_TYPE_COUNT: c_int = 5;
pub const cudnnBackendBehaviorNote_t = c_uint;
pub const CUDNN_KNOB_TYPE_SPLIT_K: c_int = 0;
pub const CUDNN_KNOB_TYPE_SWIZZLE: c_int = 1;
pub const CUDNN_KNOB_TYPE_TILE_SIZE: c_int = 2;
pub const CUDNN_KNOB_TYPE_USE_TEX: c_int = 3;
pub const CUDNN_KNOB_TYPE_EDGE: c_int = 4;
pub const CUDNN_KNOB_TYPE_KBLOCK: c_int = 5;
pub const CUDNN_KNOB_TYPE_LDGA: c_int = 6;
pub const CUDNN_KNOB_TYPE_LDGB: c_int = 7;
pub const CUDNN_KNOB_TYPE_CHUNK_K: c_int = 8;
pub const CUDNN_KNOB_TYPE_SPLIT_H: c_int = 9;
pub const CUDNN_KNOB_TYPE_WINO_TILE: c_int = 10;
pub const CUDNN_KNOB_TYPE_MULTIPLY: c_int = 11;
pub const CUDNN_KNOB_TYPE_SPLIT_K_BUF: c_int = 12;
pub const CUDNN_KNOB_TYPE_TILEK: c_int = 13;
pub const CUDNN_KNOB_TYPE_STAGES: c_int = 14;
pub const CUDNN_KNOB_TYPE_REDUCTION_MODE: c_int = 15;
pub const CUDNN_KNOB_TYPE_CTA_SPLIT_K_MODE: c_int = 16;
pub const CUDNN_KNOB_TYPE_SPLIT_K_SLC: c_int = 17;
pub const CUDNN_KNOB_TYPE_IDX_MODE: c_int = 18;
pub const CUDNN_KNOB_TYPE_SLICED: c_int = 19;
pub const CUDNN_KNOB_TYPE_SPLIT_RS: c_int = 20;
pub const CUDNN_KNOB_TYPE_SINGLEBUFFER: c_int = 21;
pub const CUDNN_KNOB_TYPE_LDGC: c_int = 22;
pub const CUDNN_KNOB_TYPE_SPECFILT: c_int = 23;
pub const CUDNN_KNOB_TYPE_KERNEL_CFG: c_int = 24;
pub const CUDNN_KNOB_TYPE_WORKSPACE: c_int = 25;
pub const CUDNN_KNOB_TYPE_TILE_CGA: c_int = 26;
pub const CUDNN_KNOB_TYPE_TILE_CGA_M: c_int = 27;
pub const CUDNN_KNOB_TYPE_TILE_CGA_N: c_int = 28;
pub const CUDNN_KNOB_TYPE_BLOCK_SIZE: c_int = 29;
pub const CUDNN_KNOB_TYPE_OCCUPANCY: c_int = 30;
pub const CUDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD: c_int = 31;
pub const CUDNN_KNOB_TYPE_NUM_C_PER_BLOCK: c_int = 32;
pub const CUDNN_KNOB_TYPE_SPLIT_COLS: c_int = 33;
pub const CUDNN_KNOB_TYPE_TILE_ROWS: c_int = 34;
pub const CUDNN_KNOB_TYPE_TILE_COLS: c_int = 35;
pub const CUDNN_KNOB_TYPE_LOAD_SIZE: c_int = 36;
pub const CUDNN_KNOB_TYPE_CTA_COUNT: c_int = 37;
pub const CUDNN_KNOB_TYPE_STREAM_K: c_int = 38;
pub const CUDNN_KNOB_TYPE_SPLIT_P_SLC: c_int = 39;
pub const CUDNN_KNOB_TYPE_TILE_M: c_int = 40;
pub const CUDNN_KNOB_TYPE_TILE_N: c_int = 41;
pub const CUDNN_KNOB_TYPE_WARP_SPEC_CFG: c_int = 42;
pub const CUDNN_KNOB_TYPE_SWAP_AB: c_int = 43;
pub const CUDNN_KNOB_TYPE_INPUT_TMA_ENABLE: c_int = 44;
pub const CUDNN_KNOB_TYPE_OUTPUT_TMA_ENABLE: c_int = 45;
pub const CUDNN_KNOB_TYPE_COUNTS: c_int = 46;
pub const cudnnBackendKnobType_t = c_uint;
pub const CUDNN_LAYOUT_TYPE_PREFERRED_NCHW: c_int = 0;
pub const CUDNN_LAYOUT_TYPE_PREFERRED_NHWC: c_int = 1;
pub const CUDNN_LAYOUT_TYPE_PREFERRED_PAD4CK: c_int = 2;
pub const CUDNN_LAYOUT_TYPE_PREFERRED_PAD8CK: c_int = 3;
pub const CUDNN_LAYOUT_TYPE_COUNT: c_int = 4;
pub const cudnnBackendLayoutType_t = c_uint;
pub const CUDNN_HEUR_MODE_INSTANT: c_int = 0;
pub const CUDNN_HEUR_MODE_B: c_int = 1;
pub const CUDNN_HEUR_MODE_FALLBACK: c_int = 2;
pub const CUDNN_HEUR_MODE_A: c_int = 3;
pub const CUDNN_HEUR_MODES_COUNT: c_int = 4;
pub const cudnnBackendHeurMode_t = c_uint;
pub const CUDNN_TENSOR_REORDERING_NONE: c_int = 0;
pub const CUDNN_TENSOR_REORDERING_INT8x32: c_int = 1;
pub const CUDNN_TENSOR_REORDERING_F16x16: c_int = 2;
pub const CUDNN_TENSOR_REORDERING_F8_128x4: c_int = 3;
pub const cudnnBackendTensorReordering_t = c_uint;
pub const CUDNN_ZERO_PAD: c_int = 0;
pub const CUDNN_NEG_INF_PAD: c_int = 1;
pub const CUDNN_EDGE_VAL_PAD: c_int = 2;
pub const cudnnPaddingMode_t = c_uint;
pub const CUDNN_LAYER_NORM: c_int = 0;
pub const CUDNN_INSTANCE_NORM: c_int = 1;
pub const CUDNN_BATCH_NORM: c_int = 2;
pub const CUDNN_GROUP_NORM: c_int = 3;
pub const CUDNN_RMS_NORM: c_int = 4;
pub const CUDNN_ADA_LAYER_NORM: c_int = 5;
pub const cudnnBackendNormMode_t = c_uint;
pub const CUDNN_NORM_FWD_INFERENCE: c_int = 0;
pub const CUDNN_NORM_FWD_TRAINING: c_int = 1;
pub const cudnnBackendNormFwdPhase_t = c_uint;
pub const CUDNN_RESHAPE_VIEW_ONLY: c_int = 0;
pub const CUDNN_RESHAPE_LOGICAL: c_int = 1;
pub const cudnnBackendReshapeMode_t = c_uint;
pub const CUDNN_OPERATIONGRAPH_MODE_AUTO: c_int = -1;
pub const CUDNN_OPERATIONGRAPH_MODE_CONV_FORWARD: c_int = 0;
pub const CUDNN_OPERATIONGRAPH_MODE_CONV_BWD_DATA: c_int = 1;
pub const CUDNN_OPERATIONGRAPH_MODE_CONV_BWD_FILTER: c_int = 2;
pub const CUDNN_OPERATIONGRAPH_MODE_NORM_FWD_INFER: c_int = 17;
pub const CUDNN_OPERATIONGRAPH_MODE_NORM_FWD_TRAIN: c_int = 18;
pub const CUDNN_OPERATIONGRAPH_MODE_NORM_BWD: c_int = 19;
pub const CUDNN_OPERATIONGRAPH_MODE_GENERIC_CONV_FUSION: c_int = 20;
pub const CUDNN_OPERATIONGRAPH_MODE_RESAMPLE_FWD: c_int = 22;
pub const CUDNN_OPERATIONGRAPH_MODE_RESAMPLE_BWD: c_int = 23;
pub const CUDNN_OPERATIONGRAPH_MODE_BLAS_MATMUL: c_int = 25;
pub const CUDNN_OPERATIONGRAPH_MODE_GENERIC_MATMUL_FUSION: c_int = 26;
pub const CUDNN_OPERATIONGRAPH_MODE_GENERIC_POINTWISE_FUSION: c_int = 27;
pub const CUDNN_OPERATIONGRAPH_MODE_SDPA_FWD: c_int = 28;
pub const CUDNN_OPERATIONGRAPH_MODE_SDPA_BWD: c_int = 29;
pub const CUDNN_OPERATIONGRAPH_MODE_UNIFIED_SDPA_FWD: c_int = 30;
pub const CUDNN_OPERATIONGRAPH_MODE_UNIFIED_SDPA_BWD: c_int = 31;
pub const cudnnBackendOperationGraphMode_t = c_int;
pub extern fn cudnnBackendCreateDescriptor(descriptorType: cudnnBackendDescriptorType_t, descriptor: [*c]cudnnBackendDescriptor_t) cudnnStatus_t;
pub extern fn cudnnBackendDestroyDescriptor(descriptor: cudnnBackendDescriptor_t) cudnnStatus_t;
pub extern fn cudnnBackendInitialize(descriptor: cudnnBackendDescriptor_t) cudnnStatus_t;
pub extern fn cudnnBackendFinalize(descriptor: cudnnBackendDescriptor_t) cudnnStatus_t;
pub extern fn cudnnBackendSetAttribute(descriptor: cudnnBackendDescriptor_t, attributeName: cudnnBackendAttributeName_t, attributeType: cudnnBackendAttributeType_t, elementCount: i64, arrayOfElements: ?*const anyopaque) cudnnStatus_t;
pub extern fn cudnnBackendGetAttribute(descriptor: cudnnBackendDescriptor_t, attributeName: cudnnBackendAttributeName_t, attributeType: cudnnBackendAttributeType_t, requestedElementCount: i64, elementCount: [*c]i64, arrayOfElements: ?*anyopaque) cudnnStatus_t;
pub extern fn cudnnGetExecutionPlanWorkspaceSize(handle: cudnnHandle_t, executionPlan: cudnnBackendDescriptor_t, variantPack: cudnnBackendDescriptor_t, workspaceSizeInBytes: [*c]usize) cudnnStatus_t;
pub extern fn cudnnBackendExecute(handle: cudnnHandle_t, executionPlan: cudnnBackendDescriptor_t, variantPack: cudnnBackendDescriptor_t) cudnnStatus_t;
pub extern fn cudnnBackendPopulateCudaGraph(handle: cudnnHandle_t, executionPlan: cudnnBackendDescriptor_t, variantPack: cudnnBackendDescriptor_t, graph: cudaGraph_t) cudnnStatus_t;
pub extern fn cudnnBackendUpdateCudaGraph(handle: cudnnHandle_t, executionPlan: cudnnBackendDescriptor_t, variantPack: cudnnBackendDescriptor_t, graph: cudaGraph_t) cudnnStatus_t;
pub const struct_cudnnTensorStruct = opaque {
    pub const cudnnSetTensor4dDescriptor = __root.cudnnSetTensor4dDescriptor;
    pub const cudnnSetTensor4dDescriptorEx = __root.cudnnSetTensor4dDescriptorEx;
    pub const cudnnGetTensor4dDescriptor = __root.cudnnGetTensor4dDescriptor;
    pub const cudnnSetTensorNdDescriptor = __root.cudnnSetTensorNdDescriptor;
    pub const cudnnSetTensorNdDescriptorEx = __root.cudnnSetTensorNdDescriptorEx;
    pub const cudnnGetTensorNdDescriptor = __root.cudnnGetTensorNdDescriptor;
    pub const cudnnGetTensorSizeInBytes = __root.cudnnGetTensorSizeInBytes;
    pub const cudnnDestroyTensorDescriptor = __root.cudnnDestroyTensorDescriptor;
    pub const cudnnDeriveBNTensorDescriptor = __root.cudnnDeriveBNTensorDescriptor;
    pub const cudnnDeriveNormTensorDescriptor = __root.cudnnDeriveNormTensorDescriptor;
    pub const cudnnDropoutGetReserveSpaceSize = __root.cudnnDropoutGetReserveSpaceSize;
};
pub const cudnnTensorDescriptor_t = ?*struct_cudnnTensorStruct;
pub const struct_cudnnPoolingStruct = opaque {
    pub const cudnnSetPooling2dDescriptor = __root.cudnnSetPooling2dDescriptor;
    pub const cudnnGetPooling2dDescriptor = __root.cudnnGetPooling2dDescriptor;
    pub const cudnnSetPoolingNdDescriptor = __root.cudnnSetPoolingNdDescriptor;
    pub const cudnnGetPoolingNdDescriptor = __root.cudnnGetPoolingNdDescriptor;
    pub const cudnnGetPoolingNdForwardOutputDim = __root.cudnnGetPoolingNdForwardOutputDim;
    pub const cudnnGetPooling2dForwardOutputDim = __root.cudnnGetPooling2dForwardOutputDim;
    pub const cudnnDestroyPoolingDescriptor = __root.cudnnDestroyPoolingDescriptor;
};
pub const cudnnPoolingDescriptor_t = ?*struct_cudnnPoolingStruct;
pub const struct_cudnnFilterStruct = opaque {
    pub const cudnnSetFilter4dDescriptor = __root.cudnnSetFilter4dDescriptor;
    pub const cudnnGetFilter4dDescriptor = __root.cudnnGetFilter4dDescriptor;
    pub const cudnnSetFilterNdDescriptor = __root.cudnnSetFilterNdDescriptor;
    pub const cudnnGetFilterNdDescriptor = __root.cudnnGetFilterNdDescriptor;
    pub const cudnnGetFilterSizeInBytes = __root.cudnnGetFilterSizeInBytes;
    pub const cudnnDestroyFilterDescriptor = __root.cudnnDestroyFilterDescriptor;
};
pub const cudnnFilterDescriptor_t = ?*struct_cudnnFilterStruct;
pub const struct_cudnnLRNStruct = opaque {
    pub const cudnnSetLRNDescriptor = __root.cudnnSetLRNDescriptor;
    pub const cudnnGetLRNDescriptor = __root.cudnnGetLRNDescriptor;
    pub const cudnnDestroyLRNDescriptor = __root.cudnnDestroyLRNDescriptor;
};
pub const cudnnLRNDescriptor_t = ?*struct_cudnnLRNStruct;
pub const struct_cudnnActivationStruct = opaque {
    pub const cudnnSetActivationDescriptor = __root.cudnnSetActivationDescriptor;
    pub const cudnnGetActivationDescriptor = __root.cudnnGetActivationDescriptor;
    pub const cudnnSetActivationDescriptorSwishBeta = __root.cudnnSetActivationDescriptorSwishBeta;
    pub const cudnnGetActivationDescriptorSwishBeta = __root.cudnnGetActivationDescriptorSwishBeta;
    pub const cudnnDestroyActivationDescriptor = __root.cudnnDestroyActivationDescriptor;
};
pub const cudnnActivationDescriptor_t = ?*struct_cudnnActivationStruct;
pub const struct_cudnnSpatialTransformerStruct = opaque {
    pub const cudnnSetSpatialTransformerNdDescriptor = __root.cudnnSetSpatialTransformerNdDescriptor;
    pub const cudnnDestroySpatialTransformerDescriptor = __root.cudnnDestroySpatialTransformerDescriptor;
};
pub const cudnnSpatialTransformerDescriptor_t = ?*struct_cudnnSpatialTransformerStruct;
pub const struct_cudnnOpTensorStruct = opaque {
    pub const cudnnSetOpTensorDescriptor = __root.cudnnSetOpTensorDescriptor;
    pub const cudnnGetOpTensorDescriptor = __root.cudnnGetOpTensorDescriptor;
    pub const cudnnDestroyOpTensorDescriptor = __root.cudnnDestroyOpTensorDescriptor;
};
pub const cudnnOpTensorDescriptor_t = ?*struct_cudnnOpTensorStruct;
pub const struct_cudnnReduceTensorStruct = opaque {
    pub const cudnnSetReduceTensorDescriptor = __root.cudnnSetReduceTensorDescriptor;
    pub const cudnnGetReduceTensorDescriptor = __root.cudnnGetReduceTensorDescriptor;
    pub const cudnnDestroyReduceTensorDescriptor = __root.cudnnDestroyReduceTensorDescriptor;
};
pub const cudnnReduceTensorDescriptor_t = ?*struct_cudnnReduceTensorStruct;
pub const struct_cudnnCTCLossStruct = opaque {};
pub const cudnnCTCLossDescriptor_t = ?*struct_cudnnCTCLossStruct;
pub const struct_cudnnTensorTransformStruct = opaque {
    pub const cudnnInitTransformDest = __root.cudnnInitTransformDest;
    pub const cudnnSetTensorTransformDescriptor = __root.cudnnSetTensorTransformDescriptor;
    pub const cudnnGetTensorTransformDescriptor = __root.cudnnGetTensorTransformDescriptor;
    pub const cudnnDestroyTensorTransformDescriptor = __root.cudnnDestroyTensorTransformDescriptor;
};
pub const cudnnTensorTransformDescriptor_t = ?*struct_cudnnTensorTransformStruct;
pub const CUDNN_NON_DETERMINISTIC: c_int = 0;
pub const CUDNN_DETERMINISTIC: c_int = 1;
pub const cudnnDeterminism_t = c_uint;
pub extern fn cudnnCreateTensorDescriptor(tensorDesc: [*c]cudnnTensorDescriptor_t) cudnnStatus_t;
pub extern fn cudnnSetTensor4dDescriptor(tensorDesc: cudnnTensorDescriptor_t, format: cudnnTensorFormat_t, dataType: cudnnDataType_t, n: c_int, c: c_int, h: c_int, w: c_int) cudnnStatus_t;
pub extern fn cudnnSetTensor4dDescriptorEx(tensorDesc: cudnnTensorDescriptor_t, dataType: cudnnDataType_t, n: c_int, c: c_int, h: c_int, w: c_int, nStride: c_int, cStride: c_int, hStride: c_int, wStride: c_int) cudnnStatus_t;
pub extern fn cudnnGetTensor4dDescriptor(tensorDesc: cudnnTensorDescriptor_t, dataType: [*c]cudnnDataType_t, n: [*c]c_int, c: [*c]c_int, h: [*c]c_int, w: [*c]c_int, nStride: [*c]c_int, cStride: [*c]c_int, hStride: [*c]c_int, wStride: [*c]c_int) cudnnStatus_t;
pub extern fn cudnnSetTensorNdDescriptor(tensorDesc: cudnnTensorDescriptor_t, dataType: cudnnDataType_t, nbDims: c_int, dimA: [*c]const c_int, strideA: [*c]const c_int) cudnnStatus_t;
pub extern fn cudnnSetTensorNdDescriptorEx(tensorDesc: cudnnTensorDescriptor_t, format: cudnnTensorFormat_t, dataType: cudnnDataType_t, nbDims: c_int, dimA: [*c]const c_int) cudnnStatus_t;
pub extern fn cudnnGetTensorNdDescriptor(tensorDesc: cudnnTensorDescriptor_t, nbDimsRequested: c_int, dataType: [*c]cudnnDataType_t, nbDims: [*c]c_int, dimA: [*c]c_int, strideA: [*c]c_int) cudnnStatus_t;
pub extern fn cudnnGetTensorSizeInBytes(tensorDesc: cudnnTensorDescriptor_t, size: [*c]usize) cudnnStatus_t;
pub extern fn cudnnDestroyTensorDescriptor(tensorDesc: cudnnTensorDescriptor_t) cudnnStatus_t;
pub const CUDNN_TRANSFORM_FOLD: c_int = 0;
pub const CUDNN_TRANSFORM_UNFOLD: c_int = 1;
pub const cudnnFoldingDirection_t = c_uint;
pub extern fn cudnnInitTransformDest(transformDesc: cudnnTensorTransformDescriptor_t, srcDesc: cudnnTensorDescriptor_t, destDesc: cudnnTensorDescriptor_t, destSizeInBytes: [*c]usize) cudnnStatus_t;
pub extern fn cudnnCreateTensorTransformDescriptor(transformDesc: [*c]cudnnTensorTransformDescriptor_t) cudnnStatus_t;
pub extern fn cudnnSetTensorTransformDescriptor(transformDesc: cudnnTensorTransformDescriptor_t, nbDims: u32, destFormat: cudnnTensorFormat_t, padBeforeA: [*c]const i32, padAfterA: [*c]const i32, foldA: [*c]const u32, direction: cudnnFoldingDirection_t) cudnnStatus_t;
pub extern fn cudnnGetTensorTransformDescriptor(transformDesc: cudnnTensorTransformDescriptor_t, nbDimsRequested: u32, destFormat: [*c]cudnnTensorFormat_t, padBeforeA: [*c]i32, padAfterA: [*c]i32, foldA: [*c]u32, direction: [*c]cudnnFoldingDirection_t) cudnnStatus_t;
pub extern fn cudnnDestroyTensorTransformDescriptor(transformDesc: cudnnTensorTransformDescriptor_t) cudnnStatus_t;
pub extern fn cudnnTransformTensor(handle: cudnnHandle_t, alpha: ?*const anyopaque, xDesc: cudnnTensorDescriptor_t, x: ?*const anyopaque, beta: ?*const anyopaque, yDesc: cudnnTensorDescriptor_t, y: ?*anyopaque) cudnnStatus_t;
pub extern fn cudnnTransformTensorEx(handle: cudnnHandle_t, transDesc: cudnnTensorTransformDescriptor_t, alpha: ?*const anyopaque, srcDesc: cudnnTensorDescriptor_t, srcData: ?*const anyopaque, beta: ?*const anyopaque, destDesc: cudnnTensorDescriptor_t, destData: ?*anyopaque) cudnnStatus_t;
pub extern fn cudnnAddTensor(handle: cudnnHandle_t, alpha: ?*const anyopaque, aDesc: cudnnTensorDescriptor_t, A: ?*const anyopaque, beta: ?*const anyopaque, cDesc: cudnnTensorDescriptor_t, C: ?*anyopaque) cudnnStatus_t;
pub const CUDNN_OP_TENSOR_ADD: c_int = 0;
pub const CUDNN_OP_TENSOR_MUL: c_int = 1;
pub const CUDNN_OP_TENSOR_MIN: c_int = 2;
pub const CUDNN_OP_TENSOR_MAX: c_int = 3;
pub const CUDNN_OP_TENSOR_SQRT: c_int = 4;
pub const CUDNN_OP_TENSOR_NOT: c_int = 5;
pub const cudnnOpTensorOp_t = c_uint;
pub extern fn cudnnCreateOpTensorDescriptor(opTensorDesc: [*c]cudnnOpTensorDescriptor_t) cudnnStatus_t;
pub extern fn cudnnSetOpTensorDescriptor(opTensorDesc: cudnnOpTensorDescriptor_t, opTensorOp: cudnnOpTensorOp_t, opTensorCompType: cudnnDataType_t, opTensorNanOpt: cudnnNanPropagation_t) cudnnStatus_t;
pub extern fn cudnnGetOpTensorDescriptor(opTensorDesc: cudnnOpTensorDescriptor_t, opTensorOp: [*c]cudnnOpTensorOp_t, opTensorCompType: [*c]cudnnDataType_t, opTensorNanOpt: [*c]cudnnNanPropagation_t) cudnnStatus_t;
pub extern fn cudnnDestroyOpTensorDescriptor(opTensorDesc: cudnnOpTensorDescriptor_t) cudnnStatus_t;
pub extern fn cudnnOpTensor(handle: cudnnHandle_t, opTensorDesc: cudnnOpTensorDescriptor_t, alpha1: ?*const anyopaque, aDesc: cudnnTensorDescriptor_t, A: ?*const anyopaque, alpha2: ?*const anyopaque, bDesc: cudnnTensorDescriptor_t, B: ?*const anyopaque, beta: ?*const anyopaque, cDesc: cudnnTensorDescriptor_t, C: ?*anyopaque) cudnnStatus_t;
pub const CUDNN_REDUCE_TENSOR_NO_INDICES: c_int = 0;
pub const CUDNN_REDUCE_TENSOR_FLATTENED_INDICES: c_int = 1;
pub const cudnnReduceTensorIndices_t = c_uint;
pub const CUDNN_32BIT_INDICES: c_int = 0;
pub const CUDNN_64BIT_INDICES: c_int = 1;
pub const CUDNN_16BIT_INDICES: c_int = 2;
pub const CUDNN_8BIT_INDICES: c_int = 3;
pub const cudnnIndicesType_t = c_uint;
pub extern fn cudnnCreateReduceTensorDescriptor(reduceTensorDesc: [*c]cudnnReduceTensorDescriptor_t) cudnnStatus_t;
pub extern fn cudnnSetReduceTensorDescriptor(reduceTensorDesc: cudnnReduceTensorDescriptor_t, reduceTensorOp: cudnnReduceTensorOp_t, reduceTensorCompType: cudnnDataType_t, reduceTensorNanOpt: cudnnNanPropagation_t, reduceTensorIndices: cudnnReduceTensorIndices_t, reduceTensorIndicesType: cudnnIndicesType_t) cudnnStatus_t;
pub extern fn cudnnGetReduceTensorDescriptor(reduceTensorDesc: cudnnReduceTensorDescriptor_t, reduceTensorOp: [*c]cudnnReduceTensorOp_t, reduceTensorCompType: [*c]cudnnDataType_t, reduceTensorNanOpt: [*c]cudnnNanPropagation_t, reduceTensorIndices: [*c]cudnnReduceTensorIndices_t, reduceTensorIndicesType: [*c]cudnnIndicesType_t) cudnnStatus_t;
pub extern fn cudnnDestroyReduceTensorDescriptor(reduceTensorDesc: cudnnReduceTensorDescriptor_t) cudnnStatus_t;
pub extern fn cudnnGetReductionIndicesSize(handle: cudnnHandle_t, reduceTensorDesc: cudnnReduceTensorDescriptor_t, aDesc: cudnnTensorDescriptor_t, cDesc: cudnnTensorDescriptor_t, sizeInBytes: [*c]usize) cudnnStatus_t;
pub extern fn cudnnGetReductionWorkspaceSize(handle: cudnnHandle_t, reduceTensorDesc: cudnnReduceTensorDescriptor_t, aDesc: cudnnTensorDescriptor_t, cDesc: cudnnTensorDescriptor_t, sizeInBytes: [*c]usize) cudnnStatus_t;
pub extern fn cudnnReduceTensor(handle: cudnnHandle_t, reduceTensorDesc: cudnnReduceTensorDescriptor_t, indices: ?*anyopaque, indicesSizeInBytes: usize, workspace: ?*anyopaque, workspaceSizeInBytes: usize, alpha: ?*const anyopaque, aDesc: cudnnTensorDescriptor_t, A: ?*const anyopaque, beta: ?*const anyopaque, cDesc: cudnnTensorDescriptor_t, C: ?*anyopaque) cudnnStatus_t;
pub extern fn cudnnSetTensor(handle: cudnnHandle_t, yDesc: cudnnTensorDescriptor_t, y: ?*anyopaque, valuePtr: ?*const anyopaque) cudnnStatus_t;
pub extern fn cudnnScaleTensor(handle: cudnnHandle_t, yDesc: cudnnTensorDescriptor_t, y: ?*anyopaque, alpha: ?*const anyopaque) cudnnStatus_t;
pub extern fn cudnnCreateFilterDescriptor(filterDesc: [*c]cudnnFilterDescriptor_t) cudnnStatus_t;
pub extern fn cudnnSetFilter4dDescriptor(filterDesc: cudnnFilterDescriptor_t, dataType: cudnnDataType_t, format: cudnnTensorFormat_t, k: c_int, c: c_int, h: c_int, w: c_int) cudnnStatus_t;
pub extern fn cudnnGetFilter4dDescriptor(filterDesc: cudnnFilterDescriptor_t, dataType: [*c]cudnnDataType_t, format: [*c]cudnnTensorFormat_t, k: [*c]c_int, c: [*c]c_int, h: [*c]c_int, w: [*c]c_int) cudnnStatus_t;
pub extern fn cudnnSetFilterNdDescriptor(filterDesc: cudnnFilterDescriptor_t, dataType: cudnnDataType_t, format: cudnnTensorFormat_t, nbDims: c_int, filterDimA: [*c]const c_int) cudnnStatus_t;
pub extern fn cudnnGetFilterNdDescriptor(filterDesc: cudnnFilterDescriptor_t, nbDimsRequested: c_int, dataType: [*c]cudnnDataType_t, format: [*c]cudnnTensorFormat_t, nbDims: [*c]c_int, filterDimA: [*c]c_int) cudnnStatus_t;
pub extern fn cudnnGetFilterSizeInBytes(filterDesc: cudnnFilterDescriptor_t, size: [*c]usize) cudnnStatus_t;
pub extern fn cudnnTransformFilter(handle: cudnnHandle_t, transDesc: cudnnTensorTransformDescriptor_t, alpha: ?*const anyopaque, srcDesc: cudnnFilterDescriptor_t, srcData: ?*const anyopaque, beta: ?*const anyopaque, destDesc: cudnnFilterDescriptor_t, destData: ?*anyopaque) cudnnStatus_t;
pub extern fn cudnnDestroyFilterDescriptor(filterDesc: cudnnFilterDescriptor_t) cudnnStatus_t;
pub const CUDNN_SOFTMAX_FAST: c_int = 0;
pub const CUDNN_SOFTMAX_ACCURATE: c_int = 1;
pub const CUDNN_SOFTMAX_LOG: c_int = 2;
pub const cudnnSoftmaxAlgorithm_t = c_uint;
pub const CUDNN_SOFTMAX_MODE_INSTANCE: c_int = 0;
pub const CUDNN_SOFTMAX_MODE_CHANNEL: c_int = 1;
pub const cudnnSoftmaxMode_t = c_uint;
pub extern fn cudnnSoftmaxForward(handle: cudnnHandle_t, algo: cudnnSoftmaxAlgorithm_t, mode: cudnnSoftmaxMode_t, alpha: ?*const anyopaque, xDesc: cudnnTensorDescriptor_t, x: ?*const anyopaque, beta: ?*const anyopaque, yDesc: cudnnTensorDescriptor_t, y: ?*anyopaque) cudnnStatus_t;
pub const CUDNN_POOLING_MAX: c_int = 0;
pub const CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING: c_int = 1;
pub const CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING: c_int = 2;
pub const CUDNN_POOLING_MAX_DETERMINISTIC: c_int = 3;
pub const cudnnPoolingMode_t = c_uint;
pub extern fn cudnnCreatePoolingDescriptor(poolingDesc: [*c]cudnnPoolingDescriptor_t) cudnnStatus_t;
pub extern fn cudnnSetPooling2dDescriptor(poolingDesc: cudnnPoolingDescriptor_t, mode: cudnnPoolingMode_t, maxpoolingNanOpt: cudnnNanPropagation_t, windowHeight: c_int, windowWidth: c_int, verticalPadding: c_int, horizontalPadding: c_int, verticalStride: c_int, horizontalStride: c_int) cudnnStatus_t;
pub extern fn cudnnGetPooling2dDescriptor(poolingDesc: cudnnPoolingDescriptor_t, mode: [*c]cudnnPoolingMode_t, maxpoolingNanOpt: [*c]cudnnNanPropagation_t, windowHeight: [*c]c_int, windowWidth: [*c]c_int, verticalPadding: [*c]c_int, horizontalPadding: [*c]c_int, verticalStride: [*c]c_int, horizontalStride: [*c]c_int) cudnnStatus_t;
pub extern fn cudnnSetPoolingNdDescriptor(poolingDesc: cudnnPoolingDescriptor_t, mode: cudnnPoolingMode_t, maxpoolingNanOpt: cudnnNanPropagation_t, nbDims: c_int, windowDimA: [*c]const c_int, paddingA: [*c]const c_int, strideA: [*c]const c_int) cudnnStatus_t;
pub extern fn cudnnGetPoolingNdDescriptor(poolingDesc: cudnnPoolingDescriptor_t, nbDimsRequested: c_int, mode: [*c]cudnnPoolingMode_t, maxpoolingNanOpt: [*c]cudnnNanPropagation_t, nbDims: [*c]c_int, windowDimA: [*c]c_int, paddingA: [*c]c_int, strideA: [*c]c_int) cudnnStatus_t;
pub extern fn cudnnGetPoolingNdForwardOutputDim(poolingDesc: cudnnPoolingDescriptor_t, inputTensorDesc: cudnnTensorDescriptor_t, nbDims: c_int, outputTensorDimA: [*c]c_int) cudnnStatus_t;
pub extern fn cudnnGetPooling2dForwardOutputDim(poolingDesc: cudnnPoolingDescriptor_t, inputTensorDesc: cudnnTensorDescriptor_t, n: [*c]c_int, c: [*c]c_int, h: [*c]c_int, w: [*c]c_int) cudnnStatus_t;
pub extern fn cudnnDestroyPoolingDescriptor(poolingDesc: cudnnPoolingDescriptor_t) cudnnStatus_t;
pub extern fn cudnnPoolingForward(handle: cudnnHandle_t, poolingDesc: cudnnPoolingDescriptor_t, alpha: ?*const anyopaque, xDesc: cudnnTensorDescriptor_t, x: ?*const anyopaque, beta: ?*const anyopaque, yDesc: cudnnTensorDescriptor_t, y: ?*anyopaque) cudnnStatus_t;
pub extern fn cudnnCreateActivationDescriptor(activationDesc: [*c]cudnnActivationDescriptor_t) cudnnStatus_t;
pub extern fn cudnnSetActivationDescriptor(activationDesc: cudnnActivationDescriptor_t, mode: cudnnActivationMode_t, reluNanOpt: cudnnNanPropagation_t, coef: f64) cudnnStatus_t;
pub extern fn cudnnGetActivationDescriptor(activationDesc: cudnnActivationDescriptor_t, mode: [*c]cudnnActivationMode_t, reluNanOpt: [*c]cudnnNanPropagation_t, coef: [*c]f64) cudnnStatus_t;
pub extern fn cudnnSetActivationDescriptorSwishBeta(activationDesc: cudnnActivationDescriptor_t, swish_beta: f64) cudnnStatus_t;
pub extern fn cudnnGetActivationDescriptorSwishBeta(activationDesc: cudnnActivationDescriptor_t, swish_beta: [*c]f64) cudnnStatus_t;
pub extern fn cudnnDestroyActivationDescriptor(activationDesc: cudnnActivationDescriptor_t) cudnnStatus_t;
pub extern fn cudnnActivationForward(handle: cudnnHandle_t, activationDesc: cudnnActivationDescriptor_t, alpha: ?*const anyopaque, xDesc: cudnnTensorDescriptor_t, x: ?*const anyopaque, beta: ?*const anyopaque, yDesc: cudnnTensorDescriptor_t, y: ?*anyopaque) cudnnStatus_t;
pub extern fn cudnnCreateLRNDescriptor(normDesc: [*c]cudnnLRNDescriptor_t) cudnnStatus_t;
pub const CUDNN_LRN_CROSS_CHANNEL_DIM1: c_int = 0;
pub const cudnnLRNMode_t = c_uint;
pub extern fn cudnnSetLRNDescriptor(normDesc: cudnnLRNDescriptor_t, lrnN: c_uint, lrnAlpha: f64, lrnBeta: f64, lrnK: f64) cudnnStatus_t;
pub extern fn cudnnGetLRNDescriptor(normDesc: cudnnLRNDescriptor_t, lrnN: [*c]c_uint, lrnAlpha: [*c]f64, lrnBeta: [*c]f64, lrnK: [*c]f64) cudnnStatus_t;
pub extern fn cudnnDestroyLRNDescriptor(lrnDesc: cudnnLRNDescriptor_t) cudnnStatus_t;
pub extern fn cudnnLRNCrossChannelForward(handle: cudnnHandle_t, normDesc: cudnnLRNDescriptor_t, lrnMode: cudnnLRNMode_t, alpha: ?*const anyopaque, xDesc: cudnnTensorDescriptor_t, x: ?*const anyopaque, beta: ?*const anyopaque, yDesc: cudnnTensorDescriptor_t, y: ?*anyopaque) cudnnStatus_t;
pub const CUDNN_DIVNORM_PRECOMPUTED_MEANS: c_int = 0;
pub const cudnnDivNormMode_t = c_uint;
pub extern fn cudnnDivisiveNormalizationForward(handle: cudnnHandle_t, normDesc: cudnnLRNDescriptor_t, mode: cudnnDivNormMode_t, alpha: ?*const anyopaque, xDesc: cudnnTensorDescriptor_t, x: ?*const anyopaque, means: ?*const anyopaque, temp: ?*anyopaque, temp2: ?*anyopaque, beta: ?*const anyopaque, yDesc: cudnnTensorDescriptor_t, y: ?*anyopaque) cudnnStatus_t;
pub const CUDNN_BATCHNORM_PER_ACTIVATION: c_int = 0;
pub const CUDNN_BATCHNORM_SPATIAL: c_int = 1;
pub const CUDNN_BATCHNORM_SPATIAL_PERSISTENT: c_int = 2;
pub const cudnnBatchNormMode_t = c_uint;
pub extern fn cudnnDeriveBNTensorDescriptor(derivedBnDesc: cudnnTensorDescriptor_t, xDesc: cudnnTensorDescriptor_t, mode: cudnnBatchNormMode_t) cudnnStatus_t;
pub const CUDNN_BATCHNORM_OPS_BN: c_int = 0;
pub const CUDNN_BATCHNORM_OPS_BN_ACTIVATION: c_int = 1;
pub const CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION: c_int = 2;
pub const cudnnBatchNormOps_t = c_uint;
pub extern fn cudnnBatchNormalizationForwardInference(handle: cudnnHandle_t, mode: cudnnBatchNormMode_t, alpha: ?*const anyopaque, beta: ?*const anyopaque, xDesc: cudnnTensorDescriptor_t, x: ?*const anyopaque, yDesc: cudnnTensorDescriptor_t, y: ?*anyopaque, bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t, bnScale: ?*const anyopaque, bnBias: ?*const anyopaque, estimatedMean: ?*const anyopaque, estimatedVariance: ?*const anyopaque, epsilon: f64) cudnnStatus_t;
pub const CUDNN_NORM_PER_ACTIVATION: c_int = 0;
pub const CUDNN_NORM_PER_CHANNEL: c_int = 1;
pub const cudnnNormMode_t = c_uint;
pub const CUDNN_NORM_ALGO_STANDARD: c_int = 0;
pub const CUDNN_NORM_ALGO_PERSIST: c_int = 1;
pub const cudnnNormAlgo_t = c_uint;
pub extern fn cudnnDeriveNormTensorDescriptor(derivedNormScaleBiasDesc: cudnnTensorDescriptor_t, derivedNormMeanVarDesc: cudnnTensorDescriptor_t, xDesc: cudnnTensorDescriptor_t, mode: cudnnNormMode_t, groupCnt: c_int) cudnnStatus_t;
pub const CUDNN_NORM_OPS_NORM: c_int = 0;
pub const CUDNN_NORM_OPS_NORM_ACTIVATION: c_int = 1;
pub const CUDNN_NORM_OPS_NORM_ADD_ACTIVATION: c_int = 2;
pub const cudnnNormOps_t = c_uint;
pub extern fn cudnnNormalizationForwardInference(handle: cudnnHandle_t, mode: cudnnNormMode_t, normOps: cudnnNormOps_t, algo: cudnnNormAlgo_t, alpha: ?*const anyopaque, beta: ?*const anyopaque, xDesc: cudnnTensorDescriptor_t, x: ?*const anyopaque, normScaleBiasDesc: cudnnTensorDescriptor_t, normScale: ?*const anyopaque, normBias: ?*const anyopaque, normMeanVarDesc: cudnnTensorDescriptor_t, estimatedMean: ?*const anyopaque, estimatedVariance: ?*const anyopaque, zDesc: cudnnTensorDescriptor_t, z: ?*const anyopaque, activationDesc: cudnnActivationDescriptor_t, yDesc: cudnnTensorDescriptor_t, y: ?*anyopaque, epsilon: f64, groupCnt: c_int) cudnnStatus_t;
pub const CUDNN_SAMPLER_BILINEAR: c_int = 0;
pub const cudnnSamplerType_t = c_uint;
pub extern fn cudnnCreateSpatialTransformerDescriptor(stDesc: [*c]cudnnSpatialTransformerDescriptor_t) cudnnStatus_t;
pub extern fn cudnnSetSpatialTransformerNdDescriptor(stDesc: cudnnSpatialTransformerDescriptor_t, samplerType: cudnnSamplerType_t, dataType: cudnnDataType_t, nbDims: c_int, dimA: [*c]const c_int) cudnnStatus_t;
pub extern fn cudnnDestroySpatialTransformerDescriptor(stDesc: cudnnSpatialTransformerDescriptor_t) cudnnStatus_t;
pub extern fn cudnnSpatialTfGridGeneratorForward(handle: cudnnHandle_t, stDesc: cudnnSpatialTransformerDescriptor_t, theta: ?*const anyopaque, grid: ?*anyopaque) cudnnStatus_t;
pub extern fn cudnnSpatialTfSamplerForward(handle: cudnnHandle_t, stDesc: cudnnSpatialTransformerDescriptor_t, alpha: ?*const anyopaque, xDesc: cudnnTensorDescriptor_t, x: ?*const anyopaque, grid: ?*const anyopaque, beta: ?*const anyopaque, yDesc: cudnnTensorDescriptor_t, y: ?*anyopaque) cudnnStatus_t;
pub const struct_cudnnDropoutStruct = opaque {
    pub const cudnnDestroyDropoutDescriptor = __root.cudnnDestroyDropoutDescriptor;
    pub const cudnnSetDropoutDescriptor = __root.cudnnSetDropoutDescriptor;
    pub const cudnnRestoreDropoutDescriptor = __root.cudnnRestoreDropoutDescriptor;
    pub const cudnnGetDropoutDescriptor = __root.cudnnGetDropoutDescriptor;
};
pub const cudnnDropoutDescriptor_t = ?*struct_cudnnDropoutStruct;
pub extern fn cudnnCreateDropoutDescriptor(dropoutDesc: [*c]cudnnDropoutDescriptor_t) cudnnStatus_t;
pub extern fn cudnnDestroyDropoutDescriptor(dropoutDesc: cudnnDropoutDescriptor_t) cudnnStatus_t;
pub extern fn cudnnDropoutGetStatesSize(handle: cudnnHandle_t, sizeInBytes: [*c]usize) cudnnStatus_t;
pub extern fn cudnnDropoutGetReserveSpaceSize(xdesc: cudnnTensorDescriptor_t, sizeInBytes: [*c]usize) cudnnStatus_t;
pub extern fn cudnnSetDropoutDescriptor(dropoutDesc: cudnnDropoutDescriptor_t, handle: cudnnHandle_t, dropout: f32, states: ?*anyopaque, stateSizeInBytes: usize, seed: c_ulonglong) cudnnStatus_t;
pub extern fn cudnnRestoreDropoutDescriptor(dropoutDesc: cudnnDropoutDescriptor_t, handle: cudnnHandle_t, dropout: f32, states: ?*anyopaque, stateSizeInBytes: usize, seed: c_ulonglong) cudnnStatus_t;
pub extern fn cudnnGetDropoutDescriptor(dropoutDesc: cudnnDropoutDescriptor_t, handle: cudnnHandle_t, dropout: [*c]f32, states: [*c]?*anyopaque, seed: [*c]c_ulonglong) cudnnStatus_t;
pub extern fn cudnnDropoutForward(handle: cudnnHandle_t, dropoutDesc: cudnnDropoutDescriptor_t, xdesc: cudnnTensorDescriptor_t, x: ?*const anyopaque, ydesc: cudnnTensorDescriptor_t, y: ?*anyopaque, reserveSpace: ?*anyopaque, reserveSpaceSizeInBytes: usize) cudnnStatus_t;
pub const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM: c_int = 0;
pub const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM: c_int = 1;
pub const CUDNN_CONVOLUTION_FWD_ALGO_GEMM: c_int = 2;
pub const CUDNN_CONVOLUTION_FWD_ALGO_DIRECT: c_int = 3;
pub const CUDNN_CONVOLUTION_FWD_ALGO_FFT: c_int = 4;
pub const CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING: c_int = 5;
pub const CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD: c_int = 6;
pub const CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED: c_int = 7;
pub const CUDNN_CONVOLUTION_FWD_ALGO_COUNT: c_int = 8;
pub const cudnnConvolutionFwdAlgo_t = c_uint;
pub const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0: c_int = 0;
pub const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1: c_int = 1;
pub const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT: c_int = 2;
pub const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3: c_int = 3;
pub const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD: c_int = 4;
pub const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED: c_int = 5;
pub const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING: c_int = 6;
pub const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT: c_int = 7;
pub const cudnnConvolutionBwdFilterAlgo_t = c_uint;
pub const CUDNN_CONVOLUTION_BWD_DATA_ALGO_0: c_int = 0;
pub const CUDNN_CONVOLUTION_BWD_DATA_ALGO_1: c_int = 1;
pub const CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT: c_int = 2;
pub const CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING: c_int = 3;
pub const CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD: c_int = 4;
pub const CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED: c_int = 5;
pub const CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT: c_int = 6;
pub const cudnnConvolutionBwdDataAlgo_t = c_uint;
pub const CUDNN_CTC_LOSS_ALGO_DETERMINISTIC: c_int = 0;
pub const CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC: c_int = 1;
pub const cudnnCTCLossAlgo_t = c_uint;
pub extern fn cudnnOpsVersionCheck() cudnnStatus_t;
pub extern fn cudnnSoftmaxBackward(handle: cudnnHandle_t, algo: cudnnSoftmaxAlgorithm_t, mode: cudnnSoftmaxMode_t, alpha: ?*const anyopaque, yDesc: cudnnTensorDescriptor_t, y: ?*const anyopaque, dyDesc: cudnnTensorDescriptor_t, dy: ?*const anyopaque, beta: ?*const anyopaque, dxDesc: cudnnTensorDescriptor_t, dx: ?*anyopaque) cudnnStatus_t;
pub extern fn cudnnPoolingBackward(handle: cudnnHandle_t, poolingDesc: cudnnPoolingDescriptor_t, alpha: ?*const anyopaque, yDesc: cudnnTensorDescriptor_t, y: ?*const anyopaque, dyDesc: cudnnTensorDescriptor_t, dy: ?*const anyopaque, xDesc: cudnnTensorDescriptor_t, x: ?*const anyopaque, beta: ?*const anyopaque, dxDesc: cudnnTensorDescriptor_t, dx: ?*anyopaque) cudnnStatus_t;
pub extern fn cudnnActivationBackward(handle: cudnnHandle_t, activationDesc: cudnnActivationDescriptor_t, alpha: ?*const anyopaque, yDesc: cudnnTensorDescriptor_t, y: ?*const anyopaque, dyDesc: cudnnTensorDescriptor_t, dy: ?*const anyopaque, xDesc: cudnnTensorDescriptor_t, x: ?*const anyopaque, beta: ?*const anyopaque, dxDesc: cudnnTensorDescriptor_t, dx: ?*anyopaque) cudnnStatus_t;
pub extern fn cudnnLRNCrossChannelBackward(handle: cudnnHandle_t, normDesc: cudnnLRNDescriptor_t, lrnMode: cudnnLRNMode_t, alpha: ?*const anyopaque, yDesc: cudnnTensorDescriptor_t, y: ?*const anyopaque, dyDesc: cudnnTensorDescriptor_t, dy: ?*const anyopaque, xDesc: cudnnTensorDescriptor_t, x: ?*const anyopaque, beta: ?*const anyopaque, dxDesc: cudnnTensorDescriptor_t, dx: ?*anyopaque) cudnnStatus_t;
pub extern fn cudnnDivisiveNormalizationBackward(handle: cudnnHandle_t, normDesc: cudnnLRNDescriptor_t, mode: cudnnDivNormMode_t, alpha: ?*const anyopaque, xDesc: cudnnTensorDescriptor_t, x: ?*const anyopaque, means: ?*const anyopaque, dy: ?*const anyopaque, temp: ?*anyopaque, temp2: ?*anyopaque, beta: ?*const anyopaque, dXdMeansDesc: cudnnTensorDescriptor_t, dx: ?*anyopaque, dMeans: ?*anyopaque) cudnnStatus_t;
pub extern fn cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(handle: cudnnHandle_t, mode: cudnnBatchNormMode_t, bnOps: cudnnBatchNormOps_t, xDesc: cudnnTensorDescriptor_t, zDesc: cudnnTensorDescriptor_t, yDesc: cudnnTensorDescriptor_t, bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t, activationDesc: cudnnActivationDescriptor_t, sizeInBytes: [*c]usize) cudnnStatus_t;
pub extern fn cudnnGetBatchNormalizationBackwardExWorkspaceSize(handle: cudnnHandle_t, mode: cudnnBatchNormMode_t, bnOps: cudnnBatchNormOps_t, xDesc: cudnnTensorDescriptor_t, yDesc: cudnnTensorDescriptor_t, dyDesc: cudnnTensorDescriptor_t, dzDesc: cudnnTensorDescriptor_t, dxDesc: cudnnTensorDescriptor_t, dBnScaleBiasDesc: cudnnTensorDescriptor_t, activationDesc: cudnnActivationDescriptor_t, sizeInBytes: [*c]usize) cudnnStatus_t;
pub extern fn cudnnGetBatchNormalizationTrainingExReserveSpaceSize(handle: cudnnHandle_t, mode: cudnnBatchNormMode_t, bnOps: cudnnBatchNormOps_t, activationDesc: cudnnActivationDescriptor_t, xDesc: cudnnTensorDescriptor_t, sizeInBytes: [*c]usize) cudnnStatus_t;
pub extern fn cudnnBatchNormalizationForwardTraining(handle: cudnnHandle_t, mode: cudnnBatchNormMode_t, alpha: ?*const anyopaque, beta: ?*const anyopaque, xDesc: cudnnTensorDescriptor_t, x: ?*const anyopaque, yDesc: cudnnTensorDescriptor_t, y: ?*anyopaque, bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t, bnScale: ?*const anyopaque, bnBias: ?*const anyopaque, exponentialAverageFactor: f64, resultRunningMean: ?*anyopaque, resultRunningVariance: ?*anyopaque, epsilon: f64, resultSaveMean: ?*anyopaque, resultSaveInvVariance: ?*anyopaque) cudnnStatus_t;
pub extern fn cudnnBatchNormalizationForwardTrainingEx(handle: cudnnHandle_t, mode: cudnnBatchNormMode_t, bnOps: cudnnBatchNormOps_t, alpha: ?*const anyopaque, beta: ?*const anyopaque, xDesc: cudnnTensorDescriptor_t, xData: ?*const anyopaque, zDesc: cudnnTensorDescriptor_t, zData: ?*const anyopaque, yDesc: cudnnTensorDescriptor_t, yData: ?*anyopaque, bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t, bnScale: ?*const anyopaque, bnBias: ?*const anyopaque, exponentialAverageFactor: f64, resultRunningMean: ?*anyopaque, resultRunningVariance: ?*anyopaque, epsilon: f64, resultSaveMean: ?*anyopaque, resultSaveInvVariance: ?*anyopaque, activationDesc: cudnnActivationDescriptor_t, workspace: ?*anyopaque, workSpaceSizeInBytes: usize, reserveSpace: ?*anyopaque, reserveSpaceSizeInBytes: usize) cudnnStatus_t;
pub extern fn cudnnBatchNormalizationBackward(handle: cudnnHandle_t, mode: cudnnBatchNormMode_t, alphaDataDiff: ?*const anyopaque, betaDataDiff: ?*const anyopaque, alphaParamDiff: ?*const anyopaque, betaParamDiff: ?*const anyopaque, xDesc: cudnnTensorDescriptor_t, x: ?*const anyopaque, dyDesc: cudnnTensorDescriptor_t, dy: ?*const anyopaque, dxDesc: cudnnTensorDescriptor_t, dx: ?*anyopaque, dBnScaleBiasDesc: cudnnTensorDescriptor_t, bnScale: ?*const anyopaque, dBnScaleResult: ?*anyopaque, dBnBiasResult: ?*anyopaque, epsilon: f64, savedMean: ?*const anyopaque, savedInvVariance: ?*const anyopaque) cudnnStatus_t;
pub extern fn cudnnBatchNormalizationBackwardEx(handle: cudnnHandle_t, mode: cudnnBatchNormMode_t, bnOps: cudnnBatchNormOps_t, alphaDataDiff: ?*const anyopaque, betaDataDiff: ?*const anyopaque, alphaParamDiff: ?*const anyopaque, betaParamDiff: ?*const anyopaque, xDesc: cudnnTensorDescriptor_t, xData: ?*const anyopaque, yDesc: cudnnTensorDescriptor_t, yData: ?*const anyopaque, dyDesc: cudnnTensorDescriptor_t, dyData: ?*const anyopaque, dzDesc: cudnnTensorDescriptor_t, dzData: ?*anyopaque, dxDesc: cudnnTensorDescriptor_t, dxData: ?*anyopaque, dBnScaleBiasDesc: cudnnTensorDescriptor_t, bnScaleData: ?*const anyopaque, bnBiasData: ?*const anyopaque, dBnScaleData: ?*anyopaque, dBnBiasData: ?*anyopaque, epsilon: f64, savedMean: ?*const anyopaque, savedInvVariance: ?*const anyopaque, activationDesc: cudnnActivationDescriptor_t, workSpace: ?*anyopaque, workSpaceSizeInBytes: usize, reserveSpace: ?*anyopaque, reserveSpaceSizeInBytes: usize) cudnnStatus_t;
pub extern fn cudnnGetNormalizationForwardTrainingWorkspaceSize(handle: cudnnHandle_t, mode: cudnnNormMode_t, normOps: cudnnNormOps_t, algo: cudnnNormAlgo_t, xDesc: cudnnTensorDescriptor_t, zDesc: cudnnTensorDescriptor_t, yDesc: cudnnTensorDescriptor_t, normScaleBiasDesc: cudnnTensorDescriptor_t, activationDesc: cudnnActivationDescriptor_t, normMeanVarDesc: cudnnTensorDescriptor_t, sizeInBytes: [*c]usize, groupCnt: c_int) cudnnStatus_t;
pub extern fn cudnnGetNormalizationBackwardWorkspaceSize(handle: cudnnHandle_t, mode: cudnnNormMode_t, normOps: cudnnNormOps_t, algo: cudnnNormAlgo_t, xDesc: cudnnTensorDescriptor_t, yDesc: cudnnTensorDescriptor_t, dyDesc: cudnnTensorDescriptor_t, dzDesc: cudnnTensorDescriptor_t, dxDesc: cudnnTensorDescriptor_t, dNormScaleBiasDesc: cudnnTensorDescriptor_t, activationDesc: cudnnActivationDescriptor_t, normMeanVarDesc: cudnnTensorDescriptor_t, sizeInBytes: [*c]usize, groupCnt: c_int) cudnnStatus_t;
pub extern fn cudnnGetNormalizationTrainingReserveSpaceSize(handle: cudnnHandle_t, mode: cudnnNormMode_t, normOps: cudnnNormOps_t, algo: cudnnNormAlgo_t, activationDesc: cudnnActivationDescriptor_t, xDesc: cudnnTensorDescriptor_t, sizeInBytes: [*c]usize, groupCnt: c_int) cudnnStatus_t;
pub extern fn cudnnNormalizationForwardTraining(handle: cudnnHandle_t, mode: cudnnNormMode_t, normOps: cudnnNormOps_t, algo: cudnnNormAlgo_t, alpha: ?*const anyopaque, beta: ?*const anyopaque, xDesc: cudnnTensorDescriptor_t, xData: ?*const anyopaque, normScaleBiasDesc: cudnnTensorDescriptor_t, normScale: ?*const anyopaque, normBias: ?*const anyopaque, exponentialAverageFactor: f64, normMeanVarDesc: cudnnTensorDescriptor_t, resultRunningMean: ?*anyopaque, resultRunningVariance: ?*anyopaque, epsilon: f64, resultSaveMean: ?*anyopaque, resultSaveInvVariance: ?*anyopaque, activationDesc: cudnnActivationDescriptor_t, zDesc: cudnnTensorDescriptor_t, zData: ?*const anyopaque, yDesc: cudnnTensorDescriptor_t, yData: ?*anyopaque, workspace: ?*anyopaque, workSpaceSizeInBytes: usize, reserveSpace: ?*anyopaque, reserveSpaceSizeInBytes: usize, groupCnt: c_int) cudnnStatus_t;
pub extern fn cudnnNormalizationBackward(handle: cudnnHandle_t, mode: cudnnNormMode_t, normOps: cudnnNormOps_t, algo: cudnnNormAlgo_t, alphaDataDiff: ?*const anyopaque, betaDataDiff: ?*const anyopaque, alphaParamDiff: ?*const anyopaque, betaParamDiff: ?*const anyopaque, xDesc: cudnnTensorDescriptor_t, xData: ?*const anyopaque, yDesc: cudnnTensorDescriptor_t, yData: ?*const anyopaque, dyDesc: cudnnTensorDescriptor_t, dyData: ?*const anyopaque, dzDesc: cudnnTensorDescriptor_t, dzData: ?*anyopaque, dxDesc: cudnnTensorDescriptor_t, dxData: ?*anyopaque, dNormScaleBiasDesc: cudnnTensorDescriptor_t, normScaleData: ?*const anyopaque, normBiasData: ?*const anyopaque, dNormScaleData: ?*anyopaque, dNormBiasData: ?*anyopaque, epsilon: f64, normMeanVarDesc: cudnnTensorDescriptor_t, savedMean: ?*const anyopaque, savedInvVariance: ?*const anyopaque, activationDesc: cudnnActivationDescriptor_t, workSpace: ?*anyopaque, workSpaceSizeInBytes: usize, reserveSpace: ?*anyopaque, reserveSpaceSizeInBytes: usize, groupCnt: c_int) cudnnStatus_t;
pub extern fn cudnnSpatialTfGridGeneratorBackward(handle: cudnnHandle_t, stDesc: cudnnSpatialTransformerDescriptor_t, dgrid: ?*const anyopaque, dtheta: ?*anyopaque) cudnnStatus_t;
pub extern fn cudnnSpatialTfSamplerBackward(handle: cudnnHandle_t, stDesc: cudnnSpatialTransformerDescriptor_t, alpha: ?*const anyopaque, xDesc: cudnnTensorDescriptor_t, x: ?*const anyopaque, beta: ?*const anyopaque, dxDesc: cudnnTensorDescriptor_t, dx: ?*anyopaque, alphaDgrid: ?*const anyopaque, dyDesc: cudnnTensorDescriptor_t, dy: ?*const anyopaque, grid: ?*const anyopaque, betaDgrid: ?*const anyopaque, dgrid: ?*anyopaque) cudnnStatus_t;
pub extern fn cudnnDropoutBackward(handle: cudnnHandle_t, dropoutDesc: cudnnDropoutDescriptor_t, dydesc: cudnnTensorDescriptor_t, dy: ?*const anyopaque, dxdesc: cudnnTensorDescriptor_t, dx: ?*anyopaque, reserveSpace: ?*anyopaque, reserveSpaceSizeInBytes: usize) cudnnStatus_t;
pub const struct_cudnnConvolutionStruct = opaque {
    pub const cudnnDestroyConvolutionDescriptor = __root.cudnnDestroyConvolutionDescriptor;
    pub const cudnnSetConvolutionMathType = __root.cudnnSetConvolutionMathType;
    pub const cudnnGetConvolutionMathType = __root.cudnnGetConvolutionMathType;
    pub const cudnnSetConvolutionGroupCount = __root.cudnnSetConvolutionGroupCount;
    pub const cudnnGetConvolutionGroupCount = __root.cudnnGetConvolutionGroupCount;
    pub const cudnnSetConvolutionReorderType = __root.cudnnSetConvolutionReorderType;
    pub const cudnnGetConvolutionReorderType = __root.cudnnGetConvolutionReorderType;
    pub const cudnnSetConvolution2dDescriptor = __root.cudnnSetConvolution2dDescriptor;
    pub const cudnnGetConvolution2dDescriptor = __root.cudnnGetConvolution2dDescriptor;
    pub const cudnnSetConvolutionNdDescriptor = __root.cudnnSetConvolutionNdDescriptor;
    pub const cudnnGetConvolutionNdDescriptor = __root.cudnnGetConvolutionNdDescriptor;
    pub const cudnnGetConvolution2dForwardOutputDim = __root.cudnnGetConvolution2dForwardOutputDim;
    pub const cudnnGetConvolutionNdForwardOutputDim = __root.cudnnGetConvolutionNdForwardOutputDim;
};
pub const cudnnConvolutionDescriptor_t = ?*struct_cudnnConvolutionStruct;
pub const struct_cudnnConvolutionFwdAlgoPerfStruct = extern struct {
    algo: cudnnConvolutionFwdAlgo_t = @import("std").mem.zeroes(cudnnConvolutionFwdAlgo_t),
    status: cudnnStatus_t = @import("std").mem.zeroes(cudnnStatus_t),
    time: f32 = 0,
    memory: usize = 0,
    determinism: cudnnDeterminism_t = @import("std").mem.zeroes(cudnnDeterminism_t),
    mathType: cudnnMathType_t = @import("std").mem.zeroes(cudnnMathType_t),
    reserved: [3]c_int = @import("std").mem.zeroes([3]c_int),
};
pub const cudnnConvolutionFwdAlgoPerf_t = struct_cudnnConvolutionFwdAlgoPerfStruct;
pub extern fn cudnnCreateConvolutionDescriptor(convDesc: [*c]cudnnConvolutionDescriptor_t) cudnnStatus_t;
pub extern fn cudnnDestroyConvolutionDescriptor(convDesc: cudnnConvolutionDescriptor_t) cudnnStatus_t;
pub extern fn cudnnSetConvolutionMathType(convDesc: cudnnConvolutionDescriptor_t, mathType: cudnnMathType_t) cudnnStatus_t;
pub extern fn cudnnGetConvolutionMathType(convDesc: cudnnConvolutionDescriptor_t, mathType: [*c]cudnnMathType_t) cudnnStatus_t;
pub extern fn cudnnSetConvolutionGroupCount(convDesc: cudnnConvolutionDescriptor_t, groupCount: c_int) cudnnStatus_t;
pub extern fn cudnnGetConvolutionGroupCount(convDesc: cudnnConvolutionDescriptor_t, groupCount: [*c]c_int) cudnnStatus_t;
pub extern fn cudnnSetConvolutionReorderType(convDesc: cudnnConvolutionDescriptor_t, reorderType: cudnnReorderType_t) cudnnStatus_t;
pub extern fn cudnnGetConvolutionReorderType(convDesc: cudnnConvolutionDescriptor_t, reorderType: [*c]cudnnReorderType_t) cudnnStatus_t;
pub extern fn cudnnSetConvolution2dDescriptor(convDesc: cudnnConvolutionDescriptor_t, pad_h: c_int, pad_w: c_int, u: c_int, v: c_int, dilation_h: c_int, dilation_w: c_int, mode: cudnnConvolutionMode_t, computeType: cudnnDataType_t) cudnnStatus_t;
pub extern fn cudnnGetConvolution2dDescriptor(convDesc: cudnnConvolutionDescriptor_t, pad_h: [*c]c_int, pad_w: [*c]c_int, u: [*c]c_int, v: [*c]c_int, dilation_h: [*c]c_int, dilation_w: [*c]c_int, mode: [*c]cudnnConvolutionMode_t, computeType: [*c]cudnnDataType_t) cudnnStatus_t;
pub extern fn cudnnSetConvolutionNdDescriptor(convDesc: cudnnConvolutionDescriptor_t, arrayLength: c_int, padA: [*c]const c_int, filterStrideA: [*c]const c_int, dilationA: [*c]const c_int, mode: cudnnConvolutionMode_t, computeType: cudnnDataType_t) cudnnStatus_t;
pub extern fn cudnnGetConvolutionNdDescriptor(convDesc: cudnnConvolutionDescriptor_t, arrayLengthRequested: c_int, arrayLength: [*c]c_int, padA: [*c]c_int, strideA: [*c]c_int, dilationA: [*c]c_int, mode: [*c]cudnnConvolutionMode_t, computeType: [*c]cudnnDataType_t) cudnnStatus_t;
pub extern fn cudnnGetConvolution2dForwardOutputDim(convDesc: cudnnConvolutionDescriptor_t, inputTensorDesc: cudnnTensorDescriptor_t, filterDesc: cudnnFilterDescriptor_t, n: [*c]c_int, c: [*c]c_int, h: [*c]c_int, w: [*c]c_int) cudnnStatus_t;
pub extern fn cudnnGetConvolutionNdForwardOutputDim(convDesc: cudnnConvolutionDescriptor_t, inputTensorDesc: cudnnTensorDescriptor_t, filterDesc: cudnnFilterDescriptor_t, nbDims: c_int, tensorOuputDimA: [*c]c_int) cudnnStatus_t;
pub extern fn cudnnGetConvolutionForwardAlgorithmMaxCount(handle: cudnnHandle_t, count: [*c]c_int) cudnnStatus_t;
pub extern fn cudnnGetConvolutionForwardAlgorithm_v7(handle: cudnnHandle_t, srcDesc: cudnnTensorDescriptor_t, filterDesc: cudnnFilterDescriptor_t, convDesc: cudnnConvolutionDescriptor_t, destDesc: cudnnTensorDescriptor_t, requestedAlgoCount: c_int, returnedAlgoCount: [*c]c_int, perfResults: [*c]cudnnConvolutionFwdAlgoPerf_t) cudnnStatus_t;
pub extern fn cudnnFindConvolutionForwardAlgorithm(handle: cudnnHandle_t, xDesc: cudnnTensorDescriptor_t, wDesc: cudnnFilterDescriptor_t, convDesc: cudnnConvolutionDescriptor_t, yDesc: cudnnTensorDescriptor_t, requestedAlgoCount: c_int, returnedAlgoCount: [*c]c_int, perfResults: [*c]cudnnConvolutionFwdAlgoPerf_t) cudnnStatus_t;
pub extern fn cudnnFindConvolutionForwardAlgorithmEx(handle: cudnnHandle_t, xDesc: cudnnTensorDescriptor_t, x: ?*const anyopaque, wDesc: cudnnFilterDescriptor_t, w: ?*const anyopaque, convDesc: cudnnConvolutionDescriptor_t, yDesc: cudnnTensorDescriptor_t, y: ?*anyopaque, requestedAlgoCount: c_int, returnedAlgoCount: [*c]c_int, perfResults: [*c]cudnnConvolutionFwdAlgoPerf_t, workSpace: ?*anyopaque, workSpaceSizeInBytes: usize) cudnnStatus_t;
pub extern fn cudnnIm2Col(handle: cudnnHandle_t, xDesc: cudnnTensorDescriptor_t, x: ?*const anyopaque, wDesc: cudnnFilterDescriptor_t, convDesc: cudnnConvolutionDescriptor_t, colBuffer: ?*anyopaque) cudnnStatus_t;
pub extern fn cudnnReorderFilterAndBias(handle: cudnnHandle_t, filterDesc: cudnnFilterDescriptor_t, reorderType: cudnnReorderType_t, filterData: ?*const anyopaque, reorderedFilterData: ?*anyopaque, reorderBias: c_int, biasData: ?*const anyopaque, reorderedBiasData: ?*anyopaque) cudnnStatus_t;
pub extern fn cudnnGetConvolutionForwardWorkspaceSize(handle: cudnnHandle_t, xDesc: cudnnTensorDescriptor_t, wDesc: cudnnFilterDescriptor_t, convDesc: cudnnConvolutionDescriptor_t, yDesc: cudnnTensorDescriptor_t, algo: cudnnConvolutionFwdAlgo_t, sizeInBytes: [*c]usize) cudnnStatus_t;
pub extern fn cudnnConvolutionForward(handle: cudnnHandle_t, alpha: ?*const anyopaque, xDesc: cudnnTensorDescriptor_t, x: ?*const anyopaque, wDesc: cudnnFilterDescriptor_t, w: ?*const anyopaque, convDesc: cudnnConvolutionDescriptor_t, algo: cudnnConvolutionFwdAlgo_t, workSpace: ?*anyopaque, workSpaceSizeInBytes: usize, beta: ?*const anyopaque, yDesc: cudnnTensorDescriptor_t, y: ?*anyopaque) cudnnStatus_t;
pub extern fn cudnnConvolutionBiasActivationForward(handle: cudnnHandle_t, alpha1: ?*const anyopaque, xDesc: cudnnTensorDescriptor_t, x: ?*const anyopaque, wDesc: cudnnFilterDescriptor_t, w: ?*const anyopaque, convDesc: cudnnConvolutionDescriptor_t, algo: cudnnConvolutionFwdAlgo_t, workSpace: ?*anyopaque, workSpaceSizeInBytes: usize, alpha2: ?*const anyopaque, zDesc: cudnnTensorDescriptor_t, z: ?*const anyopaque, biasDesc: cudnnTensorDescriptor_t, bias: ?*const anyopaque, activationDesc: cudnnActivationDescriptor_t, yDesc: cudnnTensorDescriptor_t, y: ?*anyopaque) cudnnStatus_t;
pub const struct_cudnnConvolutionBwdDataAlgoPerfStruct = extern struct {
    algo: cudnnConvolutionBwdDataAlgo_t = @import("std").mem.zeroes(cudnnConvolutionBwdDataAlgo_t),
    status: cudnnStatus_t = @import("std").mem.zeroes(cudnnStatus_t),
    time: f32 = 0,
    memory: usize = 0,
    determinism: cudnnDeterminism_t = @import("std").mem.zeroes(cudnnDeterminism_t),
    mathType: cudnnMathType_t = @import("std").mem.zeroes(cudnnMathType_t),
    reserved: [3]c_int = @import("std").mem.zeroes([3]c_int),
};
pub const cudnnConvolutionBwdDataAlgoPerf_t = struct_cudnnConvolutionBwdDataAlgoPerfStruct;
pub extern fn cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle: cudnnHandle_t, count: [*c]c_int) cudnnStatus_t;
pub extern fn cudnnFindConvolutionBackwardDataAlgorithm(handle: cudnnHandle_t, wDesc: cudnnFilterDescriptor_t, dyDesc: cudnnTensorDescriptor_t, convDesc: cudnnConvolutionDescriptor_t, dxDesc: cudnnTensorDescriptor_t, requestedAlgoCount: c_int, returnedAlgoCount: [*c]c_int, perfResults: [*c]cudnnConvolutionBwdDataAlgoPerf_t) cudnnStatus_t;
pub extern fn cudnnFindConvolutionBackwardDataAlgorithmEx(handle: cudnnHandle_t, wDesc: cudnnFilterDescriptor_t, w: ?*const anyopaque, dyDesc: cudnnTensorDescriptor_t, dy: ?*const anyopaque, convDesc: cudnnConvolutionDescriptor_t, dxDesc: cudnnTensorDescriptor_t, dx: ?*anyopaque, requestedAlgoCount: c_int, returnedAlgoCount: [*c]c_int, perfResults: [*c]cudnnConvolutionBwdDataAlgoPerf_t, workSpace: ?*anyopaque, workSpaceSizeInBytes: usize) cudnnStatus_t;
pub extern fn cudnnGetConvolutionBackwardDataAlgorithm_v7(handle: cudnnHandle_t, filterDesc: cudnnFilterDescriptor_t, diffDesc: cudnnTensorDescriptor_t, convDesc: cudnnConvolutionDescriptor_t, gradDesc: cudnnTensorDescriptor_t, requestedAlgoCount: c_int, returnedAlgoCount: [*c]c_int, perfResults: [*c]cudnnConvolutionBwdDataAlgoPerf_t) cudnnStatus_t;
pub extern fn cudnnGetConvolutionBackwardDataWorkspaceSize(handle: cudnnHandle_t, wDesc: cudnnFilterDescriptor_t, dyDesc: cudnnTensorDescriptor_t, convDesc: cudnnConvolutionDescriptor_t, dxDesc: cudnnTensorDescriptor_t, algo: cudnnConvolutionBwdDataAlgo_t, sizeInBytes: [*c]usize) cudnnStatus_t;
pub extern fn cudnnConvolutionBackwardData(handle: cudnnHandle_t, alpha: ?*const anyopaque, wDesc: cudnnFilterDescriptor_t, w: ?*const anyopaque, dyDesc: cudnnTensorDescriptor_t, dy: ?*const anyopaque, convDesc: cudnnConvolutionDescriptor_t, algo: cudnnConvolutionBwdDataAlgo_t, workSpace: ?*anyopaque, workSpaceSizeInBytes: usize, beta: ?*const anyopaque, dxDesc: cudnnTensorDescriptor_t, dx: ?*anyopaque) cudnnStatus_t;
pub extern fn cudnnGetFoldedConvBackwardDataDescriptors(handle: cudnnHandle_t, filterDesc: cudnnFilterDescriptor_t, diffDesc: cudnnTensorDescriptor_t, convDesc: cudnnConvolutionDescriptor_t, gradDesc: cudnnTensorDescriptor_t, transformFormat: cudnnTensorFormat_t, foldedFilterDesc: cudnnFilterDescriptor_t, paddedDiffDesc: cudnnTensorDescriptor_t, foldedConvDesc: cudnnConvolutionDescriptor_t, foldedGradDesc: cudnnTensorDescriptor_t, filterFoldTransDesc: cudnnTensorTransformDescriptor_t, diffPadTransDesc: cudnnTensorTransformDescriptor_t, gradFoldTransDesc: cudnnTensorTransformDescriptor_t, gradUnfoldTransDesc: cudnnTensorTransformDescriptor_t) cudnnStatus_t;
pub const struct_cudnnFusedOpsConstParamStruct = opaque {
    pub const cudnnDestroyFusedOpsConstParamPack = __root.cudnnDestroyFusedOpsConstParamPack;
    pub const cudnnSetFusedOpsConstParamPackAttribute = __root.cudnnSetFusedOpsConstParamPackAttribute;
    pub const cudnnGetFusedOpsConstParamPackAttribute = __root.cudnnGetFusedOpsConstParamPackAttribute;
};
pub const cudnnFusedOpsConstParamPack_t = ?*struct_cudnnFusedOpsConstParamStruct;
pub const struct_cudnnFusedOpsVariantParamStruct = opaque {
    pub const cudnnDestroyFusedOpsVariantParamPack = __root.cudnnDestroyFusedOpsVariantParamPack;
    pub const cudnnSetFusedOpsVariantParamPackAttribute = __root.cudnnSetFusedOpsVariantParamPackAttribute;
    pub const cudnnGetFusedOpsVariantParamPackAttribute = __root.cudnnGetFusedOpsVariantParamPackAttribute;
};
pub const cudnnFusedOpsVariantParamPack_t = ?*struct_cudnnFusedOpsVariantParamStruct;
pub const struct_cudnnFusedOpsPlanStruct = opaque {
    pub const cudnnDestroyFusedOpsPlan = __root.cudnnDestroyFusedOpsPlan;
};
pub const cudnnFusedOpsPlan_t = ?*struct_cudnnFusedOpsPlanStruct;
pub const CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS: c_int = 0;
pub const CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD: c_int = 1;
pub const CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING: c_int = 2;
pub const CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE: c_int = 3;
pub const CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION: c_int = 4;
pub const CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK: c_int = 5;
pub const CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM: c_int = 6;
pub const cudnnFusedOps_t = c_uint;
pub const CUDNN_PARAM_XDESC: c_int = 0;
pub const CUDNN_PARAM_XDATA_PLACEHOLDER: c_int = 1;
pub const CUDNN_PARAM_BN_MODE: c_int = 2;
pub const CUDNN_PARAM_BN_EQSCALEBIAS_DESC: c_int = 3;
pub const CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER: c_int = 4;
pub const CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER: c_int = 5;
pub const CUDNN_PARAM_ACTIVATION_DESC: c_int = 6;
pub const CUDNN_PARAM_CONV_DESC: c_int = 7;
pub const CUDNN_PARAM_WDESC: c_int = 8;
pub const CUDNN_PARAM_WDATA_PLACEHOLDER: c_int = 9;
pub const CUDNN_PARAM_DWDESC: c_int = 10;
pub const CUDNN_PARAM_DWDATA_PLACEHOLDER: c_int = 11;
pub const CUDNN_PARAM_YDESC: c_int = 12;
pub const CUDNN_PARAM_YDATA_PLACEHOLDER: c_int = 13;
pub const CUDNN_PARAM_DYDESC: c_int = 14;
pub const CUDNN_PARAM_DYDATA_PLACEHOLDER: c_int = 15;
pub const CUDNN_PARAM_YSTATS_DESC: c_int = 16;
pub const CUDNN_PARAM_YSUM_PLACEHOLDER: c_int = 17;
pub const CUDNN_PARAM_YSQSUM_PLACEHOLDER: c_int = 18;
pub const CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC: c_int = 19;
pub const CUDNN_PARAM_BN_SCALE_PLACEHOLDER: c_int = 20;
pub const CUDNN_PARAM_BN_BIAS_PLACEHOLDER: c_int = 21;
pub const CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER: c_int = 22;
pub const CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER: c_int = 23;
pub const CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER: c_int = 24;
pub const CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER: c_int = 25;
pub const CUDNN_PARAM_ZDESC: c_int = 26;
pub const CUDNN_PARAM_ZDATA_PLACEHOLDER: c_int = 27;
pub const CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC: c_int = 28;
pub const CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER: c_int = 29;
pub const CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER: c_int = 30;
pub const CUDNN_PARAM_ACTIVATION_BITMASK_DESC: c_int = 31;
pub const CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER: c_int = 32;
pub const CUDNN_PARAM_DXDESC: c_int = 33;
pub const CUDNN_PARAM_DXDATA_PLACEHOLDER: c_int = 34;
pub const CUDNN_PARAM_DZDESC: c_int = 35;
pub const CUDNN_PARAM_DZDATA_PLACEHOLDER: c_int = 36;
pub const CUDNN_PARAM_BN_DSCALE_PLACEHOLDER: c_int = 37;
pub const CUDNN_PARAM_BN_DBIAS_PLACEHOLDER: c_int = 38;
pub const cudnnFusedOpsConstParamLabel_t = c_uint;
pub const CUDNN_PTR_NULL: c_int = 0;
pub const CUDNN_PTR_ELEM_ALIGNED: c_int = 1;
pub const CUDNN_PTR_16B_ALIGNED: c_int = 2;
pub const cudnnFusedOpsPointerPlaceHolder_t = c_uint;
pub const CUDNN_PTR_XDATA: c_int = 0;
pub const CUDNN_PTR_BN_EQSCALE: c_int = 1;
pub const CUDNN_PTR_BN_EQBIAS: c_int = 2;
pub const CUDNN_PTR_WDATA: c_int = 3;
pub const CUDNN_PTR_DWDATA: c_int = 4;
pub const CUDNN_PTR_YDATA: c_int = 5;
pub const CUDNN_PTR_DYDATA: c_int = 6;
pub const CUDNN_PTR_YSUM: c_int = 7;
pub const CUDNN_PTR_YSQSUM: c_int = 8;
pub const CUDNN_PTR_WORKSPACE: c_int = 9;
pub const CUDNN_PTR_BN_SCALE: c_int = 10;
pub const CUDNN_PTR_BN_BIAS: c_int = 11;
pub const CUDNN_PTR_BN_SAVED_MEAN: c_int = 12;
pub const CUDNN_PTR_BN_SAVED_INVSTD: c_int = 13;
pub const CUDNN_PTR_BN_RUNNING_MEAN: c_int = 14;
pub const CUDNN_PTR_BN_RUNNING_VAR: c_int = 15;
pub const CUDNN_PTR_ZDATA: c_int = 16;
pub const CUDNN_PTR_BN_Z_EQSCALE: c_int = 17;
pub const CUDNN_PTR_BN_Z_EQBIAS: c_int = 18;
pub const CUDNN_PTR_ACTIVATION_BITMASK: c_int = 19;
pub const CUDNN_PTR_DXDATA: c_int = 20;
pub const CUDNN_PTR_DZDATA: c_int = 21;
pub const CUDNN_PTR_BN_DSCALE: c_int = 22;
pub const CUDNN_PTR_BN_DBIAS: c_int = 23;
pub const CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES: c_int = 100;
pub const CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT: c_int = 101;
pub const CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR: c_int = 102;
pub const CUDNN_SCALAR_DOUBLE_BN_EPSILON: c_int = 103;
pub const cudnnFusedOpsVariantParamLabel_t = c_uint;
pub extern fn cudnnCnnVersionCheck() cudnnStatus_t;
pub const struct_cudnnConvolutionBwdFilterAlgoPerfStruct = extern struct {
    algo: cudnnConvolutionBwdFilterAlgo_t = @import("std").mem.zeroes(cudnnConvolutionBwdFilterAlgo_t),
    status: cudnnStatus_t = @import("std").mem.zeroes(cudnnStatus_t),
    time: f32 = 0,
    memory: usize = 0,
    determinism: cudnnDeterminism_t = @import("std").mem.zeroes(cudnnDeterminism_t),
    mathType: cudnnMathType_t = @import("std").mem.zeroes(cudnnMathType_t),
    reserved: [3]c_int = @import("std").mem.zeroes([3]c_int),
};
pub const cudnnConvolutionBwdFilterAlgoPerf_t = struct_cudnnConvolutionBwdFilterAlgoPerfStruct;
pub extern fn cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle: cudnnHandle_t, count: [*c]c_int) cudnnStatus_t;
pub extern fn cudnnFindConvolutionBackwardFilterAlgorithm(handle: cudnnHandle_t, xDesc: cudnnTensorDescriptor_t, dyDesc: cudnnTensorDescriptor_t, convDesc: cudnnConvolutionDescriptor_t, dwDesc: cudnnFilterDescriptor_t, requestedAlgoCount: c_int, returnedAlgoCount: [*c]c_int, perfResults: [*c]cudnnConvolutionBwdFilterAlgoPerf_t) cudnnStatus_t;
pub extern fn cudnnFindConvolutionBackwardFilterAlgorithmEx(handle: cudnnHandle_t, xDesc: cudnnTensorDescriptor_t, x: ?*const anyopaque, dyDesc: cudnnTensorDescriptor_t, y: ?*const anyopaque, convDesc: cudnnConvolutionDescriptor_t, dwDesc: cudnnFilterDescriptor_t, dw: ?*anyopaque, requestedAlgoCount: c_int, returnedAlgoCount: [*c]c_int, perfResults: [*c]cudnnConvolutionBwdFilterAlgoPerf_t, workSpace: ?*anyopaque, workSpaceSizeInBytes: usize) cudnnStatus_t;
pub extern fn cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle: cudnnHandle_t, srcDesc: cudnnTensorDescriptor_t, diffDesc: cudnnTensorDescriptor_t, convDesc: cudnnConvolutionDescriptor_t, gradDesc: cudnnFilterDescriptor_t, requestedAlgoCount: c_int, returnedAlgoCount: [*c]c_int, perfResults: [*c]cudnnConvolutionBwdFilterAlgoPerf_t) cudnnStatus_t;
pub extern fn cudnnGetConvolutionBackwardFilterWorkspaceSize(handle: cudnnHandle_t, xDesc: cudnnTensorDescriptor_t, dyDesc: cudnnTensorDescriptor_t, convDesc: cudnnConvolutionDescriptor_t, gradDesc: cudnnFilterDescriptor_t, algo: cudnnConvolutionBwdFilterAlgo_t, sizeInBytes: [*c]usize) cudnnStatus_t;
pub extern fn cudnnConvolutionBackwardFilter(handle: cudnnHandle_t, alpha: ?*const anyopaque, xDesc: cudnnTensorDescriptor_t, x: ?*const anyopaque, dyDesc: cudnnTensorDescriptor_t, dy: ?*const anyopaque, convDesc: cudnnConvolutionDescriptor_t, algo: cudnnConvolutionBwdFilterAlgo_t, workSpace: ?*anyopaque, workSpaceSizeInBytes: usize, beta: ?*const anyopaque, dwDesc: cudnnFilterDescriptor_t, dw: ?*anyopaque) cudnnStatus_t;
pub extern fn cudnnConvolutionBackwardBias(handle: cudnnHandle_t, alpha: ?*const anyopaque, dyDesc: cudnnTensorDescriptor_t, dy: ?*const anyopaque, beta: ?*const anyopaque, dbDesc: cudnnTensorDescriptor_t, db: ?*anyopaque) cudnnStatus_t;
pub extern fn cudnnCreateFusedOpsConstParamPack(constPack: [*c]cudnnFusedOpsConstParamPack_t, ops: cudnnFusedOps_t) cudnnStatus_t;
pub extern fn cudnnDestroyFusedOpsConstParamPack(constPack: cudnnFusedOpsConstParamPack_t) cudnnStatus_t;
pub extern fn cudnnSetFusedOpsConstParamPackAttribute(constPack: cudnnFusedOpsConstParamPack_t, paramLabel: cudnnFusedOpsConstParamLabel_t, param: ?*const anyopaque) cudnnStatus_t;
pub extern fn cudnnGetFusedOpsConstParamPackAttribute(constPack: cudnnFusedOpsConstParamPack_t, paramLabel: cudnnFusedOpsConstParamLabel_t, param: ?*anyopaque, isNULL: [*c]c_int) cudnnStatus_t;
pub extern fn cudnnCreateFusedOpsVariantParamPack(varPack: [*c]cudnnFusedOpsVariantParamPack_t, ops: cudnnFusedOps_t) cudnnStatus_t;
pub extern fn cudnnDestroyFusedOpsVariantParamPack(varPack: cudnnFusedOpsVariantParamPack_t) cudnnStatus_t;
pub extern fn cudnnSetFusedOpsVariantParamPackAttribute(varPack: cudnnFusedOpsVariantParamPack_t, paramLabel: cudnnFusedOpsVariantParamLabel_t, ptr: ?*anyopaque) cudnnStatus_t;
pub extern fn cudnnGetFusedOpsVariantParamPackAttribute(varPack: cudnnFusedOpsVariantParamPack_t, paramLabel: cudnnFusedOpsVariantParamLabel_t, ptr: ?*anyopaque) cudnnStatus_t;
pub extern fn cudnnCreateFusedOpsPlan(plan: [*c]cudnnFusedOpsPlan_t, ops: cudnnFusedOps_t) cudnnStatus_t;
pub extern fn cudnnDestroyFusedOpsPlan(plan: cudnnFusedOpsPlan_t) cudnnStatus_t;
pub extern fn cudnnMakeFusedOpsPlan(handle: cudnnHandle_t, plan: cudnnFusedOpsPlan_t, constPack: cudnnFusedOpsConstParamPack_t, workspaceSizeInBytes: [*c]usize) cudnnStatus_t;
pub extern fn cudnnFusedOpsExecute(handle: cudnnHandle_t, plan: cudnnFusedOpsPlan_t, varPack: cudnnFusedOpsVariantParamPack_t) cudnnStatus_t;

pub const __VERSION__ = "Aro aro-zig";
pub const __Aro__ = "";
pub const __STDC__ = @as(c_int, 1);
pub const __STDC_HOSTED__ = @as(c_int, 1);
pub const __STDC_UTF_16__ = @as(c_int, 1);
pub const __STDC_UTF_32__ = @as(c_int, 1);
pub const __STDC_EMBED_NOT_FOUND__ = @as(c_int, 0);
pub const __STDC_EMBED_FOUND__ = @as(c_int, 1);
pub const __STDC_EMBED_EMPTY__ = @as(c_int, 2);
pub const __STDC_VERSION__ = @as(c_long, 201710);
pub const __GNUC__ = @as(c_int, 7);
pub const __GNUC_MINOR__ = @as(c_int, 1);
pub const __GNUC_PATCHLEVEL__ = @as(c_int, 0);
pub const __ARO_EMULATE_NO__ = @as(c_int, 0);
pub const __ARO_EMULATE_CLANG__ = @as(c_int, 1);
pub const __ARO_EMULATE_GCC__ = @as(c_int, 2);
pub const __ARO_EMULATE_MSVC__ = @as(c_int, 3);
pub const __ARO_EMULATE__ = __ARO_EMULATE_GCC__;
pub inline fn __building_module(x: anytype) @TypeOf(@as(c_int, 0)) {
    _ = &x;
    return @as(c_int, 0);
}
pub const _WIN32 = @as(c_int, 1);
pub const _WIN64 = @as(c_int, 1);
pub const WIN32 = @as(c_int, 1);
pub const __WIN32 = @as(c_int, 1);
pub const __WIN32__ = @as(c_int, 1);
pub const WINNT = @as(c_int, 1);
pub const __WINNT = @as(c_int, 1);
pub const __WINNT__ = @as(c_int, 1);
pub const WIN64 = @as(c_int, 1);
pub const __WIN64 = @as(c_int, 1);
pub const __WIN64__ = @as(c_int, 1);
pub const __MINGW64__ = @as(c_int, 1);
pub const __MSVCRT__ = @as(c_int, 1);
pub const __MINGW32__ = @as(c_int, 1);
pub const __declspec = @compileError("unable to translate C expr: unexpected token '__attribute__'");
// <builtin>:34:9
pub const _cdecl = @compileError("unable to translate macro: undefined identifier `__cdecl__`");
// <builtin>:35:9
pub const __cdecl = "";
pub const _stdcall = @compileError("unable to translate macro: undefined identifier `__stdcall__`");
// <builtin>:37:9
pub const __stdcall = @compileError("unable to translate macro: undefined identifier `__stdcall__`");
// <builtin>:38:9
pub const _fastcall = @compileError("unable to translate macro: undefined identifier `__fastcall__`");
// <builtin>:39:9
pub const __fastcall = @compileError("unable to translate macro: undefined identifier `__fastcall__`");
// <builtin>:40:9
pub const _thiscall = @compileError("unable to translate macro: undefined identifier `__thiscall__`");
// <builtin>:41:9
pub const __thiscall = @compileError("unable to translate macro: undefined identifier `__thiscall__`");
// <builtin>:42:9
pub const __code_model_small__ = @as(c_int, 1);
pub const __amd64__ = @as(c_int, 1);
pub const __amd64 = @as(c_int, 1);
pub const __x86_64__ = @as(c_int, 1);
pub const __x86_64 = @as(c_int, 1);
pub const __SEG_GS = @as(c_int, 1);
pub const __SEG_FS = @as(c_int, 1);
pub const __seg_gs = @compileError("unable to translate macro: undefined identifier `address_space`");
// <builtin>:50:9
pub const __seg_fs = @compileError("unable to translate macro: undefined identifier `address_space`");
// <builtin>:51:9
pub const __LAHF_SAHF__ = @as(c_int, 1);
pub const __AES__ = @as(c_int, 1);
pub const __VAES__ = @as(c_int, 1);
pub const __PCLMUL__ = @as(c_int, 1);
pub const __VPCLMULQDQ__ = @as(c_int, 1);
pub const __LZCNT__ = @as(c_int, 1);
pub const __RDRND__ = @as(c_int, 1);
pub const __FSGSBASE__ = @as(c_int, 1);
pub const __BMI__ = @as(c_int, 1);
pub const __BMI2__ = @as(c_int, 1);
pub const __POPCNT__ = @as(c_int, 1);
pub const __PRFCHW__ = @as(c_int, 1);
pub const __RDSEED__ = @as(c_int, 1);
pub const __ADX__ = @as(c_int, 1);
pub const __MOVBE__ = @as(c_int, 1);
pub const __FMA__ = @as(c_int, 1);
pub const __F16C__ = @as(c_int, 1);
pub const __GFNI__ = @as(c_int, 1);
pub const __SHA__ = @as(c_int, 1);
pub const __SHA512__ = @as(c_int, 1);
pub const __FXSR__ = @as(c_int, 1);
pub const __XSAVE__ = @as(c_int, 1);
pub const __XSAVEOPT__ = @as(c_int, 1);
pub const __XSAVEC__ = @as(c_int, 1);
pub const __XSAVES__ = @as(c_int, 1);
pub const __CLFLUSHOPT__ = @as(c_int, 1);
pub const __CLWB__ = @as(c_int, 1);
pub const __SHSTK__ = @as(c_int, 1);
pub const __SM3__ = @as(c_int, 1);
pub const __SM4__ = @as(c_int, 1);
pub const __RDPID__ = @as(c_int, 1);
pub const __WAITPKG__ = @as(c_int, 1);
pub const __MOVDIRI__ = @as(c_int, 1);
pub const __MOVDIR64B__ = @as(c_int, 1);
pub const __PTWRITE__ = @as(c_int, 1);
pub const __INVPCID__ = @as(c_int, 1);
pub const __HRESET__ = @as(c_int, 1);
pub const __CMPCCXADD__ = @as(c_int, 1);
pub const __AVXIFMA__ = @as(c_int, 1);
pub const __AVXNECONVERT__ = @as(c_int, 1);
pub const __AVXVNNI__ = @as(c_int, 1);
pub const __AVXVNNIINT16__ = @as(c_int, 1);
pub const __AVXVNNIINT8__ = @as(c_int, 1);
pub const __SERIALIZE__ = @as(c_int, 1);
pub const __CRC32__ = @as(c_int, 1);
pub const __AVX2__ = @as(c_int, 1);
pub const __AVX__ = @as(c_int, 1);
pub const __SSE4_2__ = @as(c_int, 1);
pub const __SSE4_1__ = @as(c_int, 1);
pub const __SSSE3__ = @as(c_int, 1);
pub const __SSE3__ = @as(c_int, 1);
pub const __SSE2__ = @as(c_int, 1);
pub const __SSE__ = @as(c_int, 1);
pub const __SSE_MATH__ = @as(c_int, 1);
pub const __MMX__ = @as(c_int, 1);
pub const __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 = @as(c_int, 1);
pub const __ORDER_LITTLE_ENDIAN__ = @as(c_int, 1234);
pub const __ORDER_BIG_ENDIAN__ = @as(c_int, 4321);
pub const __ORDER_PDP_ENDIAN__ = @as(c_int, 3412);
pub const __BYTE_ORDER__ = __ORDER_LITTLE_ENDIAN__;
pub const __LITTLE_ENDIAN__ = @as(c_int, 1);
pub const __ATOMIC_RELAXED = @as(c_int, 0);
pub const __ATOMIC_CONSUME = @as(c_int, 1);
pub const __ATOMIC_ACQUIRE = @as(c_int, 2);
pub const __ATOMIC_RELEASE = @as(c_int, 3);
pub const __ATOMIC_ACQ_REL = @as(c_int, 4);
pub const __ATOMIC_SEQ_CST = @as(c_int, 5);
pub const __ATOMIC_BOOL_LOCK_FREE = @as(c_int, 1);
pub const __ATOMIC_CHAR_LOCK_FREE = @as(c_int, 1);
pub const __ATOMIC_CHAR16_T_LOCK_FREE = @as(c_int, 1);
pub const __ATOMIC_CHAR32_T_LOCK_FREE = @as(c_int, 1);
pub const __ATOMIC_WCHAR_T_LOCK_FREE = @as(c_int, 1);
pub const __ATOMIC_WINT_T_LOCK_FREE = @as(c_int, 1);
pub const __ATOMIC_SHORT_LOCK_FREE = @as(c_int, 1);
pub const __ATOMIC_INT_LOCK_FREE = @as(c_int, 1);
pub const __ATOMIC_LONG_LOCK_FREE = @as(c_int, 1);
pub const __ATOMIC_LLONG_LOCK_FREE = @as(c_int, 1);
pub const __ATOMIC_POINTER_LOCK_FREE = @as(c_int, 1);
pub const __WCHAR_UNSIGNED__ = @as(c_int, 1);
pub const __WINT_UNSIGNED__ = @as(c_int, 1);
pub const __CHAR_BIT__ = @as(c_int, 8);
pub const __BOOL_WIDTH__ = @as(c_int, 8);
pub const __SCHAR_MAX__ = @as(c_int, 127);
pub const __SCHAR_WIDTH__ = @as(c_int, 8);
pub const __SHRT_MAX__ = @as(c_int, 32767);
pub const __SHRT_WIDTH__ = @as(c_int, 16);
pub const __INT_MAX__ = __helpers.promoteIntLiteral(c_int, 2147483647, .decimal);
pub const __INT_WIDTH__ = @as(c_int, 32);
pub const __LONG_MAX__ = @as(c_long, 2147483647);
pub const __LONG_WIDTH__ = @as(c_int, 32);
pub const __LONG_LONG_MAX__ = @as(c_longlong, 9223372036854775807);
pub const __LONG_LONG_WIDTH__ = @as(c_int, 64);
pub const __WCHAR_MAX__ = __helpers.promoteIntLiteral(c_int, 65535, .decimal);
pub const __WCHAR_WIDTH__ = @as(c_int, 16);
pub const __WINT_MAX__ = __helpers.promoteIntLiteral(c_int, 65535, .decimal);
pub const __WINT_WIDTH__ = @as(c_int, 16);
pub const __INTMAX_MAX__ = @as(c_longlong, 9223372036854775807);
pub const __INTMAX_WIDTH__ = @as(c_int, 64);
pub const __SIZE_MAX__ = @as(c_ulonglong, 18446744073709551615);
pub const __SIZE_WIDTH__ = @as(c_int, 64);
pub const __UINTMAX_MAX__ = @as(c_ulonglong, 18446744073709551615);
pub const __UINTMAX_WIDTH__ = @as(c_int, 64);
pub const __PTRDIFF_MAX__ = @as(c_longlong, 9223372036854775807);
pub const __PTRDIFF_WIDTH__ = @as(c_int, 64);
pub const __INTPTR_MAX__ = @as(c_longlong, 9223372036854775807);
pub const __INTPTR_WIDTH__ = @as(c_int, 64);
pub const __UINTPTR_MAX__ = @as(c_ulonglong, 18446744073709551615);
pub const __UINTPTR_WIDTH__ = @as(c_int, 64);
pub const __SIG_ATOMIC_MAX__ = __helpers.promoteIntLiteral(c_int, 2147483647, .decimal);
pub const __SIG_ATOMIC_WIDTH__ = @as(c_int, 32);
pub const __BITINT_MAXWIDTH__ = __helpers.promoteIntLiteral(c_int, 65535, .decimal);
pub const __SIZEOF_FLOAT__ = @as(c_int, 4);
pub const __SIZEOF_DOUBLE__ = @as(c_int, 8);
pub const __SIZEOF_LONG_DOUBLE__ = @as(c_int, 10);
pub const __SIZEOF_SHORT__ = @as(c_int, 2);
pub const __SIZEOF_INT__ = @as(c_int, 4);
pub const __SIZEOF_LONG__ = @as(c_int, 4);
pub const __SIZEOF_LONG_LONG__ = @as(c_int, 8);
pub const __SIZEOF_POINTER__ = @as(c_int, 8);
pub const __SIZEOF_PTRDIFF_T__ = @as(c_int, 8);
pub const __SIZEOF_SIZE_T__ = @as(c_int, 8);
pub const __SIZEOF_WCHAR_T__ = @as(c_int, 2);
pub const __SIZEOF_WINT_T__ = @as(c_int, 2);
pub const __SIZEOF_INT128__ = @as(c_int, 16);
pub const __INTPTR_TYPE__ = c_longlong;
pub const __UINTPTR_TYPE__ = c_ulonglong;
pub const __INTMAX_TYPE__ = c_longlong;
pub const __INTMAX_C_SUFFIX__ = @compileError("unable to translate macro: undefined identifier `LL`");
// <builtin>:180:9
pub const __INTMAX_C = __helpers.LL_SUFFIX;
pub const __UINTMAX_TYPE__ = c_ulonglong;
pub const __UINTMAX_C_SUFFIX__ = @compileError("unable to translate macro: undefined identifier `ULL`");
// <builtin>:183:9
pub const __UINTMAX_C = __helpers.ULL_SUFFIX;
pub const __PTRDIFF_TYPE__ = c_longlong;
pub const __SIZE_TYPE__ = c_ulonglong;
pub const __WCHAR_TYPE__ = c_ushort;
pub const __WINT_TYPE__ = c_ushort;
pub const __CHAR16_TYPE__ = c_ushort;
pub const __CHAR32_TYPE__ = c_uint;
pub const __INT8_TYPE__ = i8;
pub const __INT8_FMTd__ = "hhd";
pub const __INT8_FMTi__ = "hhi";
pub const __INT8_C_SUFFIX__ = "";
pub inline fn __INT8_C(c: anytype) @TypeOf(c) {
    _ = &c;
    return c;
}
pub const __INT16_TYPE__ = c_short;
pub const __INT16_FMTd__ = "hd";
pub const __INT16_FMTi__ = "hi";
pub const __INT16_C_SUFFIX__ = "";
pub inline fn __INT16_C(c: anytype) @TypeOf(c) {
    _ = &c;
    return c;
}
pub const __INT32_TYPE__ = c_int;
pub const __INT32_FMTd__ = "d";
pub const __INT32_FMTi__ = "i";
pub const __INT32_C_SUFFIX__ = "";
pub inline fn __INT32_C(c: anytype) @TypeOf(c) {
    _ = &c;
    return c;
}
pub const __INT64_TYPE__ = c_longlong;
pub const __INT64_FMTd__ = "lld";
pub const __INT64_FMTi__ = "lli";
pub const __INT64_C_SUFFIX__ = @compileError("unable to translate macro: undefined identifier `LL`");
// <builtin>:209:9
pub const __INT64_C = __helpers.LL_SUFFIX;
pub const __UINT8_TYPE__ = u8;
pub const __UINT8_FMTo__ = "hho";
pub const __UINT8_FMTu__ = "hhu";
pub const __UINT8_FMTx__ = "hhx";
pub const __UINT8_FMTX__ = "hhX";
pub const __UINT8_C_SUFFIX__ = "";
pub inline fn __UINT8_C(c: anytype) @TypeOf(c) {
    _ = &c;
    return c;
}
pub const __UINT8_MAX__ = @as(c_int, 255);
pub const __INT8_MAX__ = @as(c_int, 127);
pub const __UINT16_TYPE__ = c_ushort;
pub const __UINT16_FMTo__ = "ho";
pub const __UINT16_FMTu__ = "hu";
pub const __UINT16_FMTx__ = "hx";
pub const __UINT16_FMTX__ = "hX";
pub const __UINT16_C_SUFFIX__ = "";
pub inline fn __UINT16_C(c: anytype) @TypeOf(c) {
    _ = &c;
    return c;
}
pub const __UINT16_MAX__ = __helpers.promoteIntLiteral(c_int, 65535, .decimal);
pub const __INT16_MAX__ = @as(c_int, 32767);
pub const __UINT32_TYPE__ = c_uint;
pub const __UINT32_FMTo__ = "o";
pub const __UINT32_FMTu__ = "u";
pub const __UINT32_FMTx__ = "x";
pub const __UINT32_FMTX__ = "X";
pub const __UINT32_C_SUFFIX__ = @compileError("unable to translate macro: undefined identifier `U`");
// <builtin>:234:9
pub const __UINT32_C = __helpers.U_SUFFIX;
pub const __UINT32_MAX__ = __helpers.promoteIntLiteral(c_uint, 4294967295, .decimal);
pub const __INT32_MAX__ = __helpers.promoteIntLiteral(c_int, 2147483647, .decimal);
pub const __UINT64_TYPE__ = c_ulonglong;
pub const __UINT64_FMTo__ = "llo";
pub const __UINT64_FMTu__ = "llu";
pub const __UINT64_FMTx__ = "llx";
pub const __UINT64_FMTX__ = "llX";
pub const __UINT64_C_SUFFIX__ = @compileError("unable to translate macro: undefined identifier `ULL`");
// <builtin>:243:9
pub const __UINT64_C = __helpers.ULL_SUFFIX;
pub const __UINT64_MAX__ = @as(c_ulonglong, 18446744073709551615);
pub const __INT64_MAX__ = @as(c_longlong, 9223372036854775807);
pub const __INT_LEAST8_TYPE__ = i8;
pub const __INT_LEAST8_MAX__ = @as(c_int, 127);
pub const __INT_LEAST8_WIDTH__ = @as(c_int, 8);
pub const INT_LEAST8_FMTd__ = "hhd";
pub const INT_LEAST8_FMTi__ = "hhi";
pub const __UINT_LEAST8_TYPE__ = u8;
pub const __UINT_LEAST8_MAX__ = @as(c_int, 255);
pub const UINT_LEAST8_FMTo__ = "hho";
pub const UINT_LEAST8_FMTu__ = "hhu";
pub const UINT_LEAST8_FMTx__ = "hhx";
pub const UINT_LEAST8_FMTX__ = "hhX";
pub const __INT_FAST8_TYPE__ = i8;
pub const __INT_FAST8_MAX__ = @as(c_int, 127);
pub const __INT_FAST8_WIDTH__ = @as(c_int, 8);
pub const INT_FAST8_FMTd__ = "hhd";
pub const INT_FAST8_FMTi__ = "hhi";
pub const __UINT_FAST8_TYPE__ = u8;
pub const __UINT_FAST8_MAX__ = @as(c_int, 255);
pub const UINT_FAST8_FMTo__ = "hho";
pub const UINT_FAST8_FMTu__ = "hhu";
pub const UINT_FAST8_FMTx__ = "hhx";
pub const UINT_FAST8_FMTX__ = "hhX";
pub const __INT_LEAST16_TYPE__ = c_short;
pub const __INT_LEAST16_MAX__ = @as(c_int, 32767);
pub const __INT_LEAST16_WIDTH__ = @as(c_int, 16);
pub const INT_LEAST16_FMTd__ = "hd";
pub const INT_LEAST16_FMTi__ = "hi";
pub const __UINT_LEAST16_TYPE__ = c_ushort;
pub const __UINT_LEAST16_MAX__ = __helpers.promoteIntLiteral(c_int, 65535, .decimal);
pub const UINT_LEAST16_FMTo__ = "ho";
pub const UINT_LEAST16_FMTu__ = "hu";
pub const UINT_LEAST16_FMTx__ = "hx";
pub const UINT_LEAST16_FMTX__ = "hX";
pub const __INT_FAST16_TYPE__ = c_short;
pub const __INT_FAST16_MAX__ = @as(c_int, 32767);
pub const __INT_FAST16_WIDTH__ = @as(c_int, 16);
pub const INT_FAST16_FMTd__ = "hd";
pub const INT_FAST16_FMTi__ = "hi";
pub const __UINT_FAST16_TYPE__ = c_ushort;
pub const __UINT_FAST16_MAX__ = __helpers.promoteIntLiteral(c_int, 65535, .decimal);
pub const UINT_FAST16_FMTo__ = "ho";
pub const UINT_FAST16_FMTu__ = "hu";
pub const UINT_FAST16_FMTx__ = "hx";
pub const UINT_FAST16_FMTX__ = "hX";
pub const __INT_LEAST32_TYPE__ = c_int;
pub const __INT_LEAST32_MAX__ = __helpers.promoteIntLiteral(c_int, 2147483647, .decimal);
pub const __INT_LEAST32_WIDTH__ = @as(c_int, 32);
pub const INT_LEAST32_FMTd__ = "d";
pub const INT_LEAST32_FMTi__ = "i";
pub const __UINT_LEAST32_TYPE__ = c_uint;
pub const __UINT_LEAST32_MAX__ = __helpers.promoteIntLiteral(c_uint, 4294967295, .decimal);
pub const UINT_LEAST32_FMTo__ = "o";
pub const UINT_LEAST32_FMTu__ = "u";
pub const UINT_LEAST32_FMTx__ = "x";
pub const UINT_LEAST32_FMTX__ = "X";
pub const __INT_FAST32_TYPE__ = c_int;
pub const __INT_FAST32_MAX__ = __helpers.promoteIntLiteral(c_int, 2147483647, .decimal);
pub const __INT_FAST32_WIDTH__ = @as(c_int, 32);
pub const INT_FAST32_FMTd__ = "d";
pub const INT_FAST32_FMTi__ = "i";
pub const __UINT_FAST32_TYPE__ = c_uint;
pub const __UINT_FAST32_MAX__ = __helpers.promoteIntLiteral(c_uint, 4294967295, .decimal);
pub const UINT_FAST32_FMTo__ = "o";
pub const UINT_FAST32_FMTu__ = "u";
pub const UINT_FAST32_FMTx__ = "x";
pub const UINT_FAST32_FMTX__ = "X";
pub const __INT_LEAST64_TYPE__ = c_longlong;
pub const __INT_LEAST64_MAX__ = @as(c_longlong, 9223372036854775807);
pub const __INT_LEAST64_WIDTH__ = @as(c_int, 64);
pub const INT_LEAST64_FMTd__ = "lld";
pub const INT_LEAST64_FMTi__ = "lli";
pub const __UINT_LEAST64_TYPE__ = c_ulonglong;
pub const __UINT_LEAST64_MAX__ = @as(c_ulonglong, 18446744073709551615);
pub const UINT_LEAST64_FMTo__ = "llo";
pub const UINT_LEAST64_FMTu__ = "llu";
pub const UINT_LEAST64_FMTx__ = "llx";
pub const UINT_LEAST64_FMTX__ = "llX";
pub const __INT_FAST64_TYPE__ = c_longlong;
pub const __INT_FAST64_MAX__ = @as(c_longlong, 9223372036854775807);
pub const __INT_FAST64_WIDTH__ = @as(c_int, 64);
pub const INT_FAST64_FMTd__ = "lld";
pub const INT_FAST64_FMTi__ = "lli";
pub const __UINT_FAST64_TYPE__ = c_ulonglong;
pub const __UINT_FAST64_MAX__ = @as(c_ulonglong, 18446744073709551615);
pub const UINT_FAST64_FMTo__ = "llo";
pub const UINT_FAST64_FMTu__ = "llu";
pub const UINT_FAST64_FMTx__ = "llx";
pub const UINT_FAST64_FMTX__ = "llX";
pub const __FLT16_DENORM_MIN__ = @as(f16, 5.9604644775390625e-8);
pub const __FLT16_HAS_DENORM__ = "";
pub const __FLT16_DIG__ = @as(c_int, 3);
pub const __FLT16_DECIMAL_DIG__ = @as(c_int, 5);
pub const __FLT16_EPSILON__ = @as(f16, 9.765625e-4);
pub const __FLT16_HAS_INFINITY__ = "";
pub const __FLT16_HAS_QUIET_NAN__ = "";
pub const __FLT16_MANT_DIG__ = @as(c_int, 11);
pub const __FLT16_MAX_10_EXP__ = @as(c_int, 4);
pub const __FLT16_MAX_EXP__ = @as(c_int, 16);
pub const __FLT16_MAX__ = @as(f16, 6.5504e+4);
pub const __FLT16_MIN_10_EXP__ = -@as(c_int, 4);
pub const __FLT16_MIN_EXP__ = -@as(c_int, 13);
pub const __FLT16_MIN__ = @as(f16, 6.103515625e-5);
pub const __FLT_DENORM_MIN__ = @as(f32, 1.40129846e-45);
pub const __FLT_HAS_DENORM__ = "";
pub const __FLT_DIG__ = @as(c_int, 6);
pub const __FLT_DECIMAL_DIG__ = @as(c_int, 9);
pub const __FLT_EPSILON__ = @as(f32, 1.19209290e-7);
pub const __FLT_HAS_INFINITY__ = "";
pub const __FLT_HAS_QUIET_NAN__ = "";
pub const __FLT_MANT_DIG__ = @as(c_int, 24);
pub const __FLT_MAX_10_EXP__ = @as(c_int, 38);
pub const __FLT_MAX_EXP__ = @as(c_int, 128);
pub const __FLT_MAX__ = @as(f32, 3.40282347e+38);
pub const __FLT_MIN_10_EXP__ = -@as(c_int, 37);
pub const __FLT_MIN_EXP__ = -@as(c_int, 125);
pub const __FLT_MIN__ = @as(f32, 1.17549435e-38);
pub const __DBL_DENORM_MIN__ = @as(f64, 4.9406564584124654e-324);
pub const __DBL_HAS_DENORM__ = "";
pub const __DBL_DIG__ = @as(c_int, 15);
pub const __DBL_DECIMAL_DIG__ = @as(c_int, 17);
pub const __DBL_EPSILON__ = @as(f64, 2.2204460492503131e-16);
pub const __DBL_HAS_INFINITY__ = "";
pub const __DBL_HAS_QUIET_NAN__ = "";
pub const __DBL_MANT_DIG__ = @as(c_int, 53);
pub const __DBL_MAX_10_EXP__ = @as(c_int, 308);
pub const __DBL_MAX_EXP__ = @as(c_int, 1024);
pub const __DBL_MAX__ = @as(f64, 1.7976931348623157e+308);
pub const __DBL_MIN_10_EXP__ = -@as(c_int, 307);
pub const __DBL_MIN_EXP__ = -@as(c_int, 1021);
pub const __DBL_MIN__ = @as(f64, 2.2250738585072014e-308);
pub const __LDBL_DENORM_MIN__ = @as(c_longdouble, 3.64519953188247460253e-4951);
pub const __LDBL_HAS_DENORM__ = "";
pub const __LDBL_DIG__ = @as(c_int, 18);
pub const __LDBL_DECIMAL_DIG__ = @as(c_int, 21);
pub const __LDBL_EPSILON__ = @as(c_longdouble, 1.08420217248550443401e-19);
pub const __LDBL_HAS_INFINITY__ = "";
pub const __LDBL_HAS_QUIET_NAN__ = "";
pub const __LDBL_MANT_DIG__ = @as(c_int, 64);
pub const __LDBL_MAX_10_EXP__ = @as(c_int, 4932);
pub const __LDBL_MAX_EXP__ = @as(c_int, 16384);
pub const __LDBL_MAX__ = @as(c_longdouble, 1.18973149535723176502e+4932);
pub const __LDBL_MIN_10_EXP__ = -@as(c_int, 4931);
pub const __LDBL_MIN_EXP__ = -@as(c_int, 16381);
pub const __LDBL_MIN__ = @as(c_longdouble, 3.36210314311209350626e-4932);
pub const __FLT_EVAL_METHOD__ = @as(c_int, 0);
pub const __FLT_RADIX__ = @as(c_int, 2);
pub const __DECIMAL_DIG__ = __LDBL_DECIMAL_DIG__;
pub const __pic__ = @as(c_int, 2);
pub const __PIC__ = @as(c_int, 2);
pub const CUDNN_CNN_H_ = "";
pub const __CLANG_STDINT_H = "";
pub const __int_least64_t = i64;
pub const __uint_least64_t = u64;
pub const __uint32_t_defined = "";
pub const __int_least32_t = i32;
pub const __uint_least32_t = u32;
pub const __int_least16_t = i16;
pub const __uint_least16_t = u16;
pub const __int_least8_t = i8;
pub const __uint_least8_t = u8;
pub const __int8_t_defined = "";
pub const __stdint_join3 = @compileError("unable to translate C expr: unexpected token '##'");
// C:/Users/developer\tools\zig-0.17.0\zig-x86_64-windows-0.17.0\lib\include\stdint.h:291:9
pub const __intptr_t_defined = "";
pub const _INTPTR_T = "";
pub const _UINTPTR_T = "";
pub inline fn INT64_C(v: anytype) @TypeOf(__INT64_C(v)) {
    _ = &v;
    return __INT64_C(v);
}
pub inline fn UINT64_C(v: anytype) @TypeOf(__UINT64_C(v)) {
    _ = &v;
    return __UINT64_C(v);
}
pub inline fn INT32_C(v: anytype) @TypeOf(__INT32_C(v)) {
    _ = &v;
    return __INT32_C(v);
}
pub inline fn UINT32_C(v: anytype) @TypeOf(__UINT32_C(v)) {
    _ = &v;
    return __UINT32_C(v);
}
pub inline fn INT16_C(v: anytype) @TypeOf(__INT16_C(v)) {
    _ = &v;
    return __INT16_C(v);
}
pub inline fn UINT16_C(v: anytype) @TypeOf(__UINT16_C(v)) {
    _ = &v;
    return __UINT16_C(v);
}
pub inline fn INT8_C(v: anytype) @TypeOf(__INT8_C(v)) {
    _ = &v;
    return __INT8_C(v);
}
pub inline fn UINT8_C(v: anytype) @TypeOf(__UINT8_C(v)) {
    _ = &v;
    return __UINT8_C(v);
}
pub const INT64_MAX = INT64_C(__helpers.promoteIntLiteral(c_int, 9223372036854775807, .decimal));
pub const INT64_MIN = -INT64_C(__helpers.promoteIntLiteral(c_int, 9223372036854775807, .decimal)) - @as(c_int, 1);
pub const UINT64_MAX = UINT64_C(__helpers.promoteIntLiteral(c_int, 18446744073709551615, .decimal));
pub const __INT_LEAST64_MIN = INT64_MIN;
pub const __INT_LEAST64_MAX = INT64_MAX;
pub const __UINT_LEAST64_MAX = UINT64_MAX;
pub const INT_LEAST64_MIN = __INT_LEAST64_MIN;
pub const INT_LEAST64_MAX = __INT_LEAST64_MAX;
pub const UINT_LEAST64_MAX = __UINT_LEAST64_MAX;
pub const INT_FAST64_MIN = __INT_LEAST64_MIN;
pub const INT_FAST64_MAX = __INT_LEAST64_MAX;
pub const UINT_FAST64_MAX = __UINT_LEAST64_MAX;
pub const INT32_MAX = INT32_C(__helpers.promoteIntLiteral(c_int, 2147483647, .decimal));
pub const INT32_MIN = -INT32_C(__helpers.promoteIntLiteral(c_int, 2147483647, .decimal)) - @as(c_int, 1);
pub const UINT32_MAX = UINT32_C(__helpers.promoteIntLiteral(c_int, 4294967295, .decimal));
pub const __INT_LEAST32_MIN = INT32_MIN;
pub const __INT_LEAST32_MAX = INT32_MAX;
pub const __UINT_LEAST32_MAX = UINT32_MAX;
pub const INT_LEAST32_MIN = __INT_LEAST32_MIN;
pub const INT_LEAST32_MAX = __INT_LEAST32_MAX;
pub const UINT_LEAST32_MAX = __UINT_LEAST32_MAX;
pub const INT_FAST32_MIN = __INT_LEAST32_MIN;
pub const INT_FAST32_MAX = __INT_LEAST32_MAX;
pub const UINT_FAST32_MAX = __UINT_LEAST32_MAX;
pub const INT16_MAX = INT16_C(@as(c_int, 32767));
pub const INT16_MIN = -INT16_C(@as(c_int, 32767)) - @as(c_int, 1);
pub const UINT16_MAX = UINT16_C(__helpers.promoteIntLiteral(c_int, 65535, .decimal));
pub const __INT_LEAST16_MIN = INT16_MIN;
pub const __INT_LEAST16_MAX = INT16_MAX;
pub const __UINT_LEAST16_MAX = UINT16_MAX;
pub const INT_LEAST16_MIN = __INT_LEAST16_MIN;
pub const INT_LEAST16_MAX = __INT_LEAST16_MAX;
pub const UINT_LEAST16_MAX = __UINT_LEAST16_MAX;
pub const INT_FAST16_MIN = __INT_LEAST16_MIN;
pub const INT_FAST16_MAX = __INT_LEAST16_MAX;
pub const UINT_FAST16_MAX = __UINT_LEAST16_MAX;
pub const INT8_MAX = INT8_C(@as(c_int, 127));
pub const INT8_MIN = -INT8_C(@as(c_int, 127)) - @as(c_int, 1);
pub const UINT8_MAX = UINT8_C(@as(c_int, 255));
pub const __INT_LEAST8_MIN = INT8_MIN;
pub const __INT_LEAST8_MAX = INT8_MAX;
pub const __UINT_LEAST8_MAX = UINT8_MAX;
pub const INT_LEAST8_MIN = __INT_LEAST8_MIN;
pub const INT_LEAST8_MAX = __INT_LEAST8_MAX;
pub const UINT_LEAST8_MAX = __UINT_LEAST8_MAX;
pub const INT_FAST8_MIN = __INT_LEAST8_MIN;
pub const INT_FAST8_MAX = __INT_LEAST8_MAX;
pub const UINT_FAST8_MAX = __UINT_LEAST8_MAX;
pub const __INTN_MIN = @compileError("unable to translate macro: undefined identifier `INT`");
// C:/Users/developer\tools\zig-0.17.0\zig-x86_64-windows-0.17.0\lib\include\stdint.h:764:10
pub const __INTN_MAX = @compileError("unable to translate macro: undefined identifier `INT`");
// C:/Users/developer\tools\zig-0.17.0\zig-x86_64-windows-0.17.0\lib\include\stdint.h:765:10
pub const __UINTN_MAX = @compileError("unable to translate macro: undefined identifier `UINT`");
// C:/Users/developer\tools\zig-0.17.0\zig-x86_64-windows-0.17.0\lib\include\stdint.h:766:9
pub const __INTN_C = @compileError("unable to translate macro: undefined identifier `INT`");
// C:/Users/developer\tools\zig-0.17.0\zig-x86_64-windows-0.17.0\lib\include\stdint.h:767:10
pub const __UINTN_C = @compileError("unable to translate macro: undefined identifier `UINT`");
// C:/Users/developer\tools\zig-0.17.0\zig-x86_64-windows-0.17.0\lib\include\stdint.h:768:9
pub const INTPTR_MIN = -__INTPTR_MAX__ - @as(c_int, 1);
pub const INTPTR_MAX = __INTPTR_MAX__;
pub const UINTPTR_MAX = __UINTPTR_MAX__;
pub const PTRDIFF_MIN = -__PTRDIFF_MAX__ - @as(c_int, 1);
pub const PTRDIFF_MAX = __PTRDIFF_MAX__;
pub const SIZE_MAX = __SIZE_MAX__;
pub const INTMAX_MIN = -__INTMAX_MAX__ - @as(c_int, 1);
pub const INTMAX_MAX = __INTMAX_MAX__;
pub const UINTMAX_MAX = __UINTMAX_MAX__;
pub const SIG_ATOMIC_MIN = __INTN_MIN(__SIG_ATOMIC_WIDTH__);
pub const SIG_ATOMIC_MAX = __INTN_MAX(__SIG_ATOMIC_WIDTH__);
pub const WINT_MIN = __UINTN_C(__WINT_WIDTH__, @as(c_int, 0));
pub const WINT_MAX = __UINTN_MAX(__WINT_WIDTH__);
pub const WCHAR_MAX = __WCHAR_MAX__;
pub const WCHAR_MIN = __UINTN_C(__WCHAR_WIDTH__, @as(c_int, 0));
pub inline fn INTMAX_C(v: anytype) @TypeOf(__INTMAX_C(v)) {
    _ = &v;
    return __INTMAX_C(v);
}
pub inline fn UINTMAX_C(v: anytype) @TypeOf(__UINTMAX_C(v)) {
    _ = &v;
    return __UINTMAX_C(v);
}
pub const CUDNN_VERSION_H_ = "";
pub const CUDNN_MAJOR = @as(c_int, 9);
pub const CUDNN_MINOR = @as(c_int, 24);
pub const CUDNN_PATCHLEVEL = @as(c_int, 0);
pub const CUDNN_VERSION = ((CUDNN_MAJOR * @as(c_int, 10000)) + (CUDNN_MINOR * @as(c_int, 100))) + CUDNN_PATCHLEVEL;
pub const CUDNN_MAX_SM_MAJOR_NUMBER = @as(c_int, 12);
pub const CUDNN_MAX_SM_MINOR_NUMBER = @as(c_int, 1);
pub const CUDNN_MAX_DEVICE_VERSION = (CUDNN_MAX_SM_MAJOR_NUMBER * @as(c_int, 100)) + (CUDNN_MAX_SM_MINOR_NUMBER * @as(c_int, 10));
pub const CUDNN_OPS_H_ = "";
pub const CUDNN_GRAPH_H_ = "";
pub const __CUDA_RUNTIME_API_H__ = "";
pub const CUDART_VERSION = @as(c_int, 13030);
pub const __CUDART_API_VERSION = CUDART_VERSION;
pub const __HOST_DEFINES_H__ = "";
pub const __no_return__ = @compileError("unable to translate macro: undefined identifier `noreturn`");
// C:\Development\prep\glm53-execution-20261015\cuda-headers\crt\host_defines.h:78:9
pub const __forceinline__ = @compileError("unable to translate macro: undefined identifier `always_inline`");
// C:\Development\prep\glm53-execution-20261015\cuda-headers\crt\host_defines.h:82:9
pub const __inline_hint__ = @compileError("unable to translate macro: undefined identifier `nv_inline_hint`");
// C:\Development\prep\glm53-execution-20261015\cuda-headers\crt\host_defines.h:84:9
pub const __align__ = @compileError("unable to translate macro: undefined identifier `aligned`");
// C:\Development\prep\glm53-execution-20261015\cuda-headers\crt\host_defines.h:86:9
pub const __maxnreg__ = @compileError("unable to translate macro: undefined identifier `maxnreg`");
// C:\Development\prep\glm53-execution-20261015\cuda-headers\crt\host_defines.h:88:9
pub const __local_maxnreg__ = @compileError("unable to translate macro: undefined identifier `local_maxnreg`");
// C:\Development\prep\glm53-execution-20261015\cuda-headers\crt\host_defines.h:90:9
pub const __thread__ = @compileError("unable to translate C expr: unexpected token '__thread'");
// C:\Development\prep\glm53-execution-20261015\cuda-headers\crt\host_defines.h:92:9
pub const __import__ = "";
pub const __export__ = "";
pub inline fn __location__(a: anytype) @TypeOf(__annotate__(a)) {
    _ = &a;
    return __annotate__(a);
}
pub const CUDARTAPI = "";
pub const CUDARTAPI_CDECL = "";
pub const __specialization_static = "";
pub inline fn __annotate__(a: anytype) void {
    _ = &a;
    return;
}
pub inline fn __builtin_align__(a: anytype) @TypeOf(__align__(a)) {
    _ = &a;
    return __align__(a);
}
pub const __grid_constant__ = @compileError("unable to translate macro: undefined identifier `grid_constant`");
// C:\Development\prep\glm53-execution-20261015\cuda-headers\crt\host_defines.h:206:9
pub const __host__ = @compileError("unable to translate macro: undefined identifier `host`");
// C:\Development\prep\glm53-execution-20261015\cuda-headers\crt\host_defines.h:211:9
pub const __device__ = @compileError("unable to translate macro: undefined identifier `device`");
// C:\Development\prep\glm53-execution-20261015\cuda-headers\crt\host_defines.h:215:9
pub const __global__ = @compileError("unable to translate macro: undefined identifier `global`");
// C:\Development\prep\glm53-execution-20261015\cuda-headers\crt\host_defines.h:219:9
pub const __tile_global__ = @compileError("unable to translate macro: undefined identifier `tile_global`");
// C:\Development\prep\glm53-execution-20261015\cuda-headers\crt\host_defines.h:223:9
pub const __tile__ = @compileError("unable to translate macro: undefined identifier `tile`");
// C:\Development\prep\glm53-execution-20261015\cuda-headers\crt\host_defines.h:227:9
pub const __tile_builtin__ = @compileError("unable to translate macro: undefined identifier `tile_builtin`");
// C:\Development\prep\glm53-execution-20261015\cuda-headers\crt\host_defines.h:235:9
pub const __shared__ = @compileError("unable to translate macro: undefined identifier `shared`");
// C:\Development\prep\glm53-execution-20261015\cuda-headers\crt\host_defines.h:239:9
pub const __constant__ = @compileError("unable to translate macro: undefined identifier `constant`");
// C:\Development\prep\glm53-execution-20261015\cuda-headers\crt\host_defines.h:243:9
pub const __managed__ = @compileError("unable to translate macro: undefined identifier `managed`");
// C:\Development\prep\glm53-execution-20261015\cuda-headers\crt\host_defines.h:247:9
pub const __nv_pure__ = @compileError("unable to translate macro: undefined identifier `nv_pure`");
// C:\Development\prep\glm53-execution-20261015\cuda-headers\crt\host_defines.h:251:9
pub const __device_builtin__ = "";
pub const __device_builtin_texture_type__ = "";
pub const __device_builtin_surface_type__ = "";
pub const __cudart_builtin__ = "";
pub const __cluster_dims__ = @compileError("unable to translate macro: undefined identifier `cluster_dims`");
// C:\Development\prep\glm53-execution-20261015\cuda-headers\crt\host_defines.h:276:9
pub const __block_size__ = @compileError("unable to translate macro: undefined identifier `block_size`");
// C:\Development\prep\glm53-execution-20261015\cuda-headers\crt\host_defines.h:287:9
pub const __CUDA_ARCH_HAS_FEATURE__ = @compileError("unable to translate macro: undefined identifier `__CUDA_ARCH_FEAT_`");
// C:\Development\prep\glm53-execution-20261015\cuda-headers\crt\host_defines.h:292:9
pub inline fn __CUDA_HAS_ARCH_FAMILY_SPECIFIC(N: anytype) @TypeOf(@as(c_int, 0)) {
    _ = &N;
    return @as(c_int, 0);
}
pub inline fn __CUDA_HAS_ARCH_SPECIFIC(N: anytype) @TypeOf(@as(c_int, 0)) {
    _ = &N;
    return @as(c_int, 0);
}
pub const __NV_TL__ = "";
pub const __NV_TL_BUILTIN__ = "";
pub const __CLZ_RETURN_TYPE = c_int;
pub const __CLZ_PARAMETER_TYPE = c_int;
pub const __CLZLL_PARAMETER_TYPE = c_longlong;
pub const __DEVICE_TYPES_H__ = "";
pub const __DRIVER_TYPES_H__ = "";
pub const __VECTOR_TYPES_H__ = "";
pub const __NV_SILENCE_HOST_DEPRECATION_BEGIN = @compileError("unable to translate macro: undefined identifier `_Pragma`");
// C:\Development\prep\glm53-execution-20261015\cuda-headers\vector_types.h:90:10
pub const __NV_SILENCE_HOST_DEPRECATION_END = @compileError("unable to translate macro: undefined identifier `_Pragma`");
// C:\Development\prep\glm53-execution-20261015\cuda-headers\vector_types.h:93:10
pub const __NV_SILENCE_DEPRECATION_BEGIN = __NV_SILENCE_HOST_DEPRECATION_BEGIN;
pub const __NV_SILENCE_DEPRECATION_END = __NV_SILENCE_HOST_DEPRECATION_END;
pub const _GCC_LIMITS_H_ = "";
pub const __CLANG_LIMITS_H = "";
pub const MB_LEN_MAX = @as(c_int, 1);
pub const LONG_LONG_MAX = __LONG_LONG_MAX__;
pub const LONG_LONG_MIN = -__LONG_LONG_MAX__ - @as(c_longlong, 1);
pub const ULONG_LONG_MAX = (__LONG_LONG_MAX__ * @as(c_ulonglong, 2)) + @as(c_ulonglong, 1);
pub const SCHAR_MAX = __SCHAR_MAX__;
pub const SHRT_MAX = __SHRT_MAX__;
pub const INT_MAX = __INT_MAX__;
pub const LONG_MAX = __LONG_MAX__;
pub const SCHAR_MIN = -__SCHAR_MAX__ - @as(c_int, 1);
pub const SHRT_MIN = -__SHRT_MAX__ - @as(c_int, 1);
pub const INT_MIN = -__INT_MAX__ - @as(c_int, 1);
pub const LONG_MIN = -__LONG_MAX__ - @as(c_long, 1);
pub const UCHAR_MAX = (__SCHAR_MAX__ * @as(c_int, 2)) + @as(c_int, 1);
pub const USHRT_MAX = (__SHRT_MAX__ * @as(c_int, 2)) + @as(c_int, 1);
pub const UINT_MAX = (__INT_MAX__ * @as(c_uint, 2)) + @as(c_uint, 1);
pub const ULONG_MAX = (__LONG_MAX__ * @as(c_ulong, 2)) + @as(c_ulong, 1);
pub const CHAR_BIT = __CHAR_BIT__;
pub const CHAR_MIN = SCHAR_MIN;
pub const CHAR_MAX = __SCHAR_MAX__;
pub const LLONG_MIN = -__LONG_LONG_MAX__ - @as(c_longlong, 1);
pub const LLONG_MAX = __LONG_LONG_MAX__;
pub const ULLONG_MAX = (__LONG_LONG_MAX__ * @as(c_ulonglong, 2)) + @as(c_ulonglong, 1);
pub const __STDC_VERSION_STDDEF_H__ = @as(c_long, 202311);
pub const NULL = __helpers.cast(?*anyopaque, @as(c_int, 0));
pub const offsetof = @compileError("unable to translate macro: undefined identifier `__builtin_offsetof`");
// C:/Users/developer\tools\zig-0.17.0\zig-x86_64-windows-0.17.0\lib\compiler\aro\include\stddef.h:18:9
pub const cudaHostAllocDefault = @as(c_int, 0x00);
pub const cudaHostAllocPortable = @as(c_int, 0x01);
pub const cudaHostAllocMapped = @as(c_int, 0x02);
pub const cudaHostAllocWriteCombined = @as(c_int, 0x04);
pub const cudaHostRegisterDefault = @as(c_int, 0x00);
pub const cudaHostRegisterPortable = @as(c_int, 0x01);
pub const cudaHostRegisterMapped = @as(c_int, 0x02);
pub const cudaHostRegisterIoMemory = @as(c_int, 0x04);
pub const cudaHostRegisterReadOnly = @as(c_int, 0x08);
pub const cudaPeerAccessDefault = @as(c_int, 0x00);
pub const cudaStreamDefault = @as(c_int, 0x00);
pub const cudaStreamNonBlocking = @as(c_int, 0x01);
pub const cudaStreamLegacy = __helpers.cast(cudaStream_t, @as(c_int, 0x1));
pub const cudaStreamPerThread = __helpers.cast(cudaStream_t, @as(c_int, 0x2));
pub const cudaEventDefault = @as(c_int, 0x00);
pub const cudaEventBlockingSync = @as(c_int, 0x01);
pub const cudaEventDisableTiming = @as(c_int, 0x02);
pub const cudaEventInterprocess = @as(c_int, 0x04);
pub const cudaEventRecordDefault = @as(c_int, 0x00);
pub const cudaEventRecordExternal = @as(c_int, 0x01);
pub const cudaEventWaitDefault = @as(c_int, 0x00);
pub const cudaEventWaitExternal = @as(c_int, 0x01);
pub const cudaDeviceScheduleAuto = @as(c_int, 0x00);
pub const cudaDeviceScheduleSpin = @as(c_int, 0x01);
pub const cudaDeviceScheduleYield = @as(c_int, 0x02);
pub const cudaDeviceScheduleBlockingSync = @as(c_int, 0x04);
pub const cudaDeviceBlockingSync = @as(c_int, 0x04);
pub const cudaDeviceScheduleMask = @as(c_int, 0x07);
pub const cudaDeviceMapHost = @as(c_int, 0x08);
pub const cudaDeviceLmemResizeToMax = @as(c_int, 0x10);
pub const cudaDeviceSyncMemops = @as(c_int, 0x80);
pub const cudaDeviceMask = @as(c_int, 0xff);
pub const cudaArrayDefault = @as(c_int, 0x00);
pub const cudaArrayLayered = @as(c_int, 0x01);
pub const cudaArraySurfaceLoadStore = @as(c_int, 0x02);
pub const cudaArrayCubemap = @as(c_int, 0x04);
pub const cudaArrayTextureGather = @as(c_int, 0x08);
pub const cudaArrayColorAttachment = @as(c_int, 0x20);
pub const cudaArraySparse = @as(c_int, 0x40);
pub const cudaArrayDeferredMapping = @as(c_int, 0x80);
pub const cudaIpcMemLazyEnablePeerAccess = @as(c_int, 0x01);
pub const cudaMemAttachGlobal = @as(c_int, 0x01);
pub const cudaMemAttachHost = @as(c_int, 0x02);
pub const cudaMemAttachSingle = @as(c_int, 0x04);
pub const cudaOccupancyDefault = @as(c_int, 0x00);
pub const cudaOccupancyDisableCachingOverride = @as(c_int, 0x01);
pub const cudaCpuDeviceId = __helpers.cast(c_int, -@as(c_int, 1));
pub const cudaInvalidDeviceId = __helpers.cast(c_int, -@as(c_int, 2));
pub const cudaInitDeviceFlagsAreValid = @as(c_int, 0x01);
pub const cudaArraySparsePropertiesSingleMipTail = @as(c_int, 0x1);
pub const CUDART_CB = @compileError("unable to translate C expr: unexpected token '__stdcall'");
// C:\Development\prep\glm53-execution-20261015\cuda-headers\driver_types.h:1436:9
pub const cudaMemPoolCreateUsageHwDecompress = @as(c_int, 0x2);
pub const CU_UUID_HAS_BEEN_DEFINED = "";
pub const CUDA_IPC_HANDLE_SIZE = @as(c_int, 64);
pub const cudaExternalMemoryDedicated = @as(c_int, 0x1);
pub const cudaExternalSemaphoreSignalSkipNvSciBufMemSync = @as(c_int, 0x01);
pub const cudaExternalSemaphoreWaitSkipNvSciBufMemSync = @as(c_int, 0x02);
pub const cudaNvSciSyncAttrSignal = @as(c_int, 0x1);
pub const cudaNvSciSyncAttrWait = @as(c_int, 0x2);
pub const cudaGraphKernelNodePortDefault = @as(c_int, 0);
pub const cudaGraphKernelNodePortProgrammatic = @as(c_int, 1);
pub const cudaGraphKernelNodePortLaunchCompletion = @as(c_int, 2);
pub const cudaStreamAttrID = cudaLaunchAttributeID;
pub const cudaStreamAttributeAccessPolicyWindow = cudaLaunchAttributeAccessPolicyWindow;
pub const cudaStreamAttributeSynchronizationPolicy = cudaLaunchAttributeSynchronizationPolicy;
pub const cudaStreamAttributeMemSyncDomainMap = cudaLaunchAttributeMemSyncDomainMap;
pub const cudaStreamAttributeMemSyncDomain = cudaLaunchAttributeMemSyncDomain;
pub const cudaStreamAttributePriority = cudaLaunchAttributePriority;
pub const cudaStreamAttrValue = cudaLaunchAttributeValue;
pub const cudaKernelNodeAttrID = cudaLaunchAttributeID;
pub const cudaKernelNodeAttributeAccessPolicyWindow = cudaLaunchAttributeAccessPolicyWindow;
pub const cudaKernelNodeAttributeCooperative = cudaLaunchAttributeCooperative;
pub const cudaKernelNodeAttributePriority = cudaLaunchAttributePriority;
pub const cudaKernelNodeAttributeClusterDimension = cudaLaunchAttributeClusterDimension;
pub const cudaKernelNodeAttributeClusterSchedulingPolicyPreference = cudaLaunchAttributeClusterSchedulingPolicyPreference;
pub const cudaKernelNodeAttributeMemSyncDomainMap = cudaLaunchAttributeMemSyncDomainMap;
pub const cudaKernelNodeAttributeMemSyncDomain = cudaLaunchAttributeMemSyncDomain;
pub const cudaKernelNodeAttributePreferredSharedMemoryCarveout = cudaLaunchAttributePreferredSharedMemoryCarveout;
pub const cudaKernelNodeAttributeDeviceUpdatableKernelNode = cudaLaunchAttributeDeviceUpdatableKernelNode;
pub const cudaKernelNodeAttributeNvlinkUtilCentricScheduling = cudaLaunchAttributeNvlinkUtilCentricScheduling;
pub const cudaKernelNodeAttrValue = cudaLaunchAttributeValue;
pub const __SURFACE_TYPES_H__ = "";
pub const cudaSurfaceType1D = @as(c_int, 0x01);
pub const cudaSurfaceType2D = @as(c_int, 0x02);
pub const cudaSurfaceType3D = @as(c_int, 0x03);
pub const cudaSurfaceTypeCubemap = @as(c_int, 0x0C);
pub const cudaSurfaceType1DLayered = @as(c_int, 0xF1);
pub const cudaSurfaceType2DLayered = @as(c_int, 0xF2);
pub const cudaSurfaceTypeCubemapLayered = @as(c_int, 0xFC);
pub const __TEXTURE_TYPES_H__ = "";
pub const cudaTextureType1D = @as(c_int, 0x01);
pub const cudaTextureType2D = @as(c_int, 0x02);
pub const cudaTextureType3D = @as(c_int, 0x03);
pub const cudaTextureTypeCubemap = @as(c_int, 0x0C);
pub const cudaTextureType1DLayered = @as(c_int, 0xF1);
pub const cudaTextureType2DLayered = @as(c_int, 0xF2);
pub const cudaTextureTypeCubemapLayered = @as(c_int, 0xFC);
pub const __CUDA_DEVICE_RUNTIME_API_H__ = "";
pub inline fn __CUDART_API_PTDS(api: anytype) @TypeOf(api) {
    _ = &api;
    return api;
}
pub inline fn __CUDART_API_PTSZ(api: anytype) @TypeOf(api) {
    _ = &api;
    return api;
}
pub const CUDART_DEVICE = __device__;
pub const __LIBRARY_TYPES_H__ = "";
pub const CUDNN_GRAPH_MAJOR = @as(c_int, 9);
pub const CUDNN_GRAPH_MINOR = @as(c_int, 24);
pub const CUDNN_GRAPH_PATCH = @as(c_int, 0);
pub const CUDNNWINAPI = @compileError("unable to translate C expr: unexpected token '__stdcall'");
// C:\Development\prep\glm53-native-20261012\deps\ngc-cudnn-headers\cudnn_graph.h:89:9
pub const CUDNN_DEPRECATED = "";
pub const CUDNN_DEPRECATED_ENUM = "";
pub inline fn CUDNN_STATUS_FULL_ERROR_CODE(category: anytype, specific_err: anytype) cudnnStatus_t {
    _ = &category;
    _ = &specific_err;
    return __helpers.cast(cudnnStatus_t, (@as(c_int, 0) + category) + specific_err);
}
pub inline fn CUDNN_STATUS_CATEGORY(full_error_code: anytype) @TypeOf(__helpers.div(full_error_code, @as(c_int, 1000)) * @as(c_int, 1000)) {
    _ = &full_error_code;
    return __helpers.div(full_error_code, @as(c_int, 1000)) * @as(c_int, 1000);
}
pub inline fn CUDNN_STATUS_SPECIFIC_ERROR(full_error_code: anytype) @TypeOf(__helpers.rem(full_error_code, @as(c_int, 1000))) {
    _ = &full_error_code;
    return __helpers.rem(full_error_code, @as(c_int, 1000));
}
pub const CUDNN_SEV_ERROR_EN = @as(c_uint, 1) << CUDNN_SEV_ERROR;
pub const CUDNN_SEV_WARNING_EN = @as(c_uint, 1) << CUDNN_SEV_WARNING;
pub const CUDNN_SEV_INFO_EN = @as(c_uint, 1) << CUDNN_SEV_INFO;
pub const CUDNN_DIM_MAX = @as(c_int, 8);
pub const CUDNN_OPS_MAJOR = @as(c_int, 9);
pub const CUDNN_OPS_MINOR = @as(c_int, 24);
pub const CUDNN_OPS_PATCH = @as(c_int, 0);
pub const CUDNN_LRN_MIN_N = @as(c_int, 1);
pub const CUDNN_LRN_MAX_N = @as(c_int, 16);
pub const CUDNN_LRN_MIN_K = @as(f64, 1e-5);
pub const CUDNN_LRN_MIN_BETA = @as(f64, 0.01);
pub const CUDNN_BN_MIN_EPSILON = @as(f64, 0.0);
pub const CUDNN_CNN_MAJOR = @as(c_int, 9);
pub const CUDNN_CNN_MINOR = @as(c_int, 24);
pub const CUDNN_CNN_PATCH = @as(c_int, 0);
pub const cudaRoundMode = enum_cudaRoundMode;
pub const cudaError = enum_cudaError;
pub const cudaChannelFormatKind = enum_cudaChannelFormatKind;
pub const CUdevResourceDesc_st = struct_CUdevResourceDesc_st;
pub const cudaExecutionContext_st = struct_cudaExecutionContext_st;
pub const cudaChannelFormatDesc = struct_cudaChannelFormatDesc;
pub const cudaArray = struct_cudaArray;
pub const cudaMipmappedArray = struct_cudaMipmappedArray;
pub const cudaArraySparseProperties = struct_cudaArraySparseProperties;
pub const cudaArrayMemoryRequirements = struct_cudaArrayMemoryRequirements;
pub const cudaMemoryType = enum_cudaMemoryType;
pub const cudaMemcpyKind = enum_cudaMemcpyKind;
pub const cudaPitchedPtr = struct_cudaPitchedPtr;
pub const cudaExtent = struct_cudaExtent;
pub const cudaPos = struct_cudaPos;
pub const cudaMemcpy3DParms = struct_cudaMemcpy3DParms;
pub const cudaMemcpyNodeParams = struct_cudaMemcpyNodeParams;
pub const cudaMemcpy3DPeerParms = struct_cudaMemcpy3DPeerParms;
pub const cudaMemsetParams = struct_cudaMemsetParams;
pub const cudaMemsetParamsV2 = struct_cudaMemsetParamsV2;
pub const cudaAccessProperty = enum_cudaAccessProperty;
pub const cudaAccessPolicyWindow = struct_cudaAccessPolicyWindow;
pub const cudaHostNodeParams = struct_cudaHostNodeParams;
pub const cudaHostNodeParamsV2 = struct_cudaHostNodeParamsV2;
pub const cudaStreamCaptureStatus = enum_cudaStreamCaptureStatus;
pub const cudaGraphRecaptureStatus = enum_cudaGraphRecaptureStatus;
pub const cudaStreamCaptureMode = enum_cudaStreamCaptureMode;
pub const cudaSynchronizationPolicy = enum_cudaSynchronizationPolicy;
pub const cudaClusterSchedulingPolicy = enum_cudaClusterSchedulingPolicy;
pub const cudaStreamUpdateCaptureDependenciesFlags = enum_cudaStreamUpdateCaptureDependenciesFlags;
pub const cudaUserObjectFlags = enum_cudaUserObjectFlags;
pub const cudaUserObjectRetainFlags = enum_cudaUserObjectRetainFlags;
pub const cudaHostTaskSyncMode = enum_cudaHostTaskSyncMode;
pub const cudaGraphicsResource = struct_cudaGraphicsResource;
pub const cudaGraphicsRegisterFlags = enum_cudaGraphicsRegisterFlags;
pub const cudaGraphicsMapFlags = enum_cudaGraphicsMapFlags;
pub const cudaGraphicsCubeFace = enum_cudaGraphicsCubeFace;
pub const cudaResourceType = enum_cudaResourceType;
pub const cudaResourceViewFormat = enum_cudaResourceViewFormat;
pub const cudaResourceDesc = struct_cudaResourceDesc;
pub const cudaResourceViewDesc = struct_cudaResourceViewDesc;
pub const cudaPointerAttributes = struct_cudaPointerAttributes;
pub const cudaFuncAttributes = struct_cudaFuncAttributes;
pub const cudaFuncAttribute = enum_cudaFuncAttribute;
pub const cudaFuncCache = enum_cudaFuncCache;
pub const cudaSharedMemConfig = enum_cudaSharedMemConfig;
pub const cudaSharedCarveout = enum_cudaSharedCarveout;
pub const cudaComputeMode = enum_cudaComputeMode;
pub const cudaLimit = enum_cudaLimit;
pub const cudaMemoryAdvise = enum_cudaMemoryAdvise;
pub const cudaMemRangeAttribute = enum_cudaMemRangeAttribute;
pub const cudaFlushGPUDirectRDMAWritesOptions = enum_cudaFlushGPUDirectRDMAWritesOptions;
pub const cudaGPUDirectRDMAWritesOrdering = enum_cudaGPUDirectRDMAWritesOrdering;
pub const cudaFlushGPUDirectRDMAWritesScope = enum_cudaFlushGPUDirectRDMAWritesScope;
pub const cudaFlushGPUDirectRDMAWritesTarget = enum_cudaFlushGPUDirectRDMAWritesTarget;
pub const cudaDeviceAttr = enum_cudaDeviceAttr;
pub const cudaMemPoolAttr = enum_cudaMemPoolAttr;
pub const cudaMemLocationType = enum_cudaMemLocationType;
pub const cudaMemLocation = struct_cudaMemLocation;
pub const cudaMemAccessFlags = enum_cudaMemAccessFlags;
pub const cudaMemAccessDesc = struct_cudaMemAccessDesc;
pub const cudaMemAllocationType = enum_cudaMemAllocationType;
pub const cudaMemAllocationHandleType = enum_cudaMemAllocationHandleType;
pub const cudaMemPoolProps = struct_cudaMemPoolProps;
pub const cudaMemPoolPtrExportData = struct_cudaMemPoolPtrExportData;
pub const cudaMemAllocNodeParams = struct_cudaMemAllocNodeParams;
pub const cudaMemAllocNodeParamsV2 = struct_cudaMemAllocNodeParamsV2;
pub const cudaMemFreeNodeParams = struct_cudaMemFreeNodeParams;
pub const cudaGraphMemAttributeType = enum_cudaGraphMemAttributeType;
pub const cudaMemcpyFlags = enum_cudaMemcpyFlags;
pub const cudaMemcpySrcAccessOrder = enum_cudaMemcpySrcAccessOrder;
pub const cudaMemcpyAttributes = struct_cudaMemcpyAttributes;
pub const cudaMemcpy3DOperandType = enum_cudaMemcpy3DOperandType;
pub const cudaOffset3D = struct_cudaOffset3D;
pub const cudaMemcpy3DOperand = struct_cudaMemcpy3DOperand;
pub const cudaMemcpy3DBatchOp = struct_cudaMemcpy3DBatchOp;
pub const cudaDeviceP2PAttr = enum_cudaDeviceP2PAttr;
pub const cudaAtomicOperation = enum_cudaAtomicOperation;
pub const cudaAtomicOperationCapability = enum_cudaAtomicOperationCapability;
pub const CUuuid_st = struct_CUuuid_st;
pub const cudaDeviceProp = struct_cudaDeviceProp;
pub const cudaIpcEventHandle_st = struct_cudaIpcEventHandle_st;
pub const cudaIpcMemHandle_st = struct_cudaIpcMemHandle_st;
pub const cudaMemFabricHandle_st = struct_cudaMemFabricHandle_st;
pub const cudaExternalMemoryHandleType = enum_cudaExternalMemoryHandleType;
pub const cudaExternalMemoryHandleDesc = struct_cudaExternalMemoryHandleDesc;
pub const cudaExternalMemoryBufferDesc = struct_cudaExternalMemoryBufferDesc;
pub const cudaExternalMemoryMipmappedArrayDesc = struct_cudaExternalMemoryMipmappedArrayDesc;
pub const cudaExternalSemaphoreHandleType = enum_cudaExternalSemaphoreHandleType;
pub const cudaExternalSemaphoreHandleDesc = struct_cudaExternalSemaphoreHandleDesc;
pub const cudaExternalSemaphoreSignalParams = struct_cudaExternalSemaphoreSignalParams;
pub const cudaExternalSemaphoreWaitParams = struct_cudaExternalSemaphoreWaitParams;
pub const cudaDevSmResourceGroup_flags = enum_cudaDevSmResourceGroup_flags;
pub const cudaDevSmResourceSplitByCount_flags = enum_cudaDevSmResourceSplitByCount_flags;
pub const cudaDevResourceType = enum_cudaDevResourceType;
pub const cudaDevSmResource = struct_cudaDevSmResource;
pub const cudaDevWorkqueueConfigScope = enum_cudaDevWorkqueueConfigScope;
pub const cudaDevWorkqueueConfigResource = struct_cudaDevWorkqueueConfigResource;
pub const cudaDevWorkqueueResource = struct_cudaDevWorkqueueResource;
pub const cudaDevSmResourceGroupParams_st = struct_cudaDevSmResourceGroupParams_st;
pub const cudaDevResource_st = struct_cudaDevResource_st;
pub const CUstream_st = struct_CUstream_st;
pub const CUevent_st = struct_CUevent_st;
pub const CUexternalMemory_st = struct_CUexternalMemory_st;
pub const CUexternalSemaphore_st = struct_CUexternalSemaphore_st;
pub const CUgraph_st = struct_CUgraph_st;
pub const CUgraphNode_st = struct_CUgraphNode_st;
pub const CUuserObject_st = struct_CUuserObject_st;
pub const CUfunc_st = struct_CUfunc_st;
pub const CUkern_st = struct_CUkern_st;
pub const cudaJitOption = enum_cudaJitOption;
pub const cudaLibraryOption = enum_cudaLibraryOption;
pub const cudalibraryHostUniversalFunctionAndDataTable = struct_cudalibraryHostUniversalFunctionAndDataTable;
pub const cudaJit_CacheMode = enum_cudaJit_CacheMode;
pub const cudaJit_Fallback = enum_cudaJit_Fallback;
pub const CUlib_st = struct_CUlib_st;
pub const CUmemPoolHandle_st = struct_CUmemPoolHandle_st;
pub const cudaCGScope = enum_cudaCGScope;
pub const cudaKernelNodeParams = struct_cudaKernelNodeParams;
pub const cudaKernelFunctionType = enum_cudaKernelFunctionType;
pub const cudaKernelNodeParamsV2 = struct_cudaKernelNodeParamsV2;
pub const cudaExternalSemaphoreSignalNodeParams = struct_cudaExternalSemaphoreSignalNodeParams;
pub const cudaExternalSemaphoreSignalNodeParamsV2 = struct_cudaExternalSemaphoreSignalNodeParamsV2;
pub const cudaExternalSemaphoreWaitNodeParams = struct_cudaExternalSemaphoreWaitNodeParams;
pub const cudaExternalSemaphoreWaitNodeParamsV2 = struct_cudaExternalSemaphoreWaitNodeParamsV2;
pub const cudaGraphConditionalHandleFlags = enum_cudaGraphConditionalHandleFlags;
pub const cudaGraphConditionalNodeType = enum_cudaGraphConditionalNodeType;
pub const cudaConditionalNodeParams = struct_cudaConditionalNodeParams;
pub const cudaGraphNodeType = enum_cudaGraphNodeType;
pub const cudaGraphChildGraphNodeOwnership = enum_cudaGraphChildGraphNodeOwnership;
pub const cudaChildGraphNodeParams = struct_cudaChildGraphNodeParams;
pub const cudaEventRecordNodeParams = struct_cudaEventRecordNodeParams;
pub const cudaEventWaitNodeParams = struct_cudaEventWaitNodeParams;
pub const cudaGraphNodeParams = struct_cudaGraphNodeParams;
pub const cudaGraphDependencyType_enum = enum_cudaGraphDependencyType_enum;
pub const cudaGraphEdgeData_st = struct_cudaGraphEdgeData_st;
pub const CUgraphExec_st = struct_CUgraphExec_st;
pub const cudaGraphExecUpdateResult = enum_cudaGraphExecUpdateResult;
pub const cudaGraphInstantiateParams_st = struct_cudaGraphInstantiateParams_st;
pub const cudaGraphExecUpdateResultInfo_st = struct_cudaGraphExecUpdateResultInfo_st;
pub const CUgraphDeviceUpdatableNode_st = struct_CUgraphDeviceUpdatableNode_st;
pub const cudaGraphKernelNodeField = enum_cudaGraphKernelNodeField;
pub const cudaGraphKernelNodeUpdate = struct_cudaGraphKernelNodeUpdate;
pub const cudaGetDriverEntryPointFlags = enum_cudaGetDriverEntryPointFlags;
pub const cudaDriverEntryPointQueryResult = enum_cudaDriverEntryPointQueryResult;
pub const cudaGraphDebugDotFlags = enum_cudaGraphDebugDotFlags;
pub const cudaGraphInstantiateFlags = enum_cudaGraphInstantiateFlags;
pub const cudaLaunchMemSyncDomainMap_st = struct_cudaLaunchMemSyncDomainMap_st;
pub const cudaLaunchAttribute_st = struct_cudaLaunchAttribute_st;
pub const cudaLaunchConfig_st = struct_cudaLaunchConfig_st;
pub const cudaDeviceNumaConfig = enum_cudaDeviceNumaConfig;
pub const cudaAsyncCallbackEntry = struct_cudaAsyncCallbackEntry;
pub const cudaAsyncNotificationType_enum = enum_cudaAsyncNotificationType_enum;
pub const cudaAsyncNotificationInfo = struct_cudaAsyncNotificationInfo;
pub const CUDAlogLevel_enum = enum_CUDAlogLevel_enum;
pub const CUlogsCallbackEntry_st = struct_CUlogsCallbackEntry_st;
pub const cudaFabricOpStatusSource = enum_cudaFabricOpStatusSource;
pub const cudaFabricOpStatusInfo = enum_cudaFabricOpStatusInfo;
pub const cudaSurfaceBoundaryMode = enum_cudaSurfaceBoundaryMode;
pub const cudaSurfaceFormatMode = enum_cudaSurfaceFormatMode;
pub const cudaTextureAddressMode = enum_cudaTextureAddressMode;
pub const cudaTextureFilterMode = enum_cudaTextureFilterMode;
pub const cudaTextureReadMode = enum_cudaTextureReadMode;
pub const cudaTextureDesc = struct_cudaTextureDesc;
pub const cudaGraphRecaptureCallbackData = struct_cudaGraphRecaptureCallbackData;
pub const cudnnContext = struct_cudnnContext;
pub const cudnnDebugStruct = struct_cudnnDebugStruct;
pub const cudnnFractionStruct = struct_cudnnFractionStruct;
pub const cudnnTensorStruct = struct_cudnnTensorStruct;
pub const cudnnPoolingStruct = struct_cudnnPoolingStruct;
pub const cudnnFilterStruct = struct_cudnnFilterStruct;
pub const cudnnLRNStruct = struct_cudnnLRNStruct;
pub const cudnnActivationStruct = struct_cudnnActivationStruct;
pub const cudnnSpatialTransformerStruct = struct_cudnnSpatialTransformerStruct;
pub const cudnnOpTensorStruct = struct_cudnnOpTensorStruct;
pub const cudnnReduceTensorStruct = struct_cudnnReduceTensorStruct;
pub const cudnnCTCLossStruct = struct_cudnnCTCLossStruct;
pub const cudnnTensorTransformStruct = struct_cudnnTensorTransformStruct;
pub const cudnnDropoutStruct = struct_cudnnDropoutStruct;
pub const cudnnConvolutionStruct = struct_cudnnConvolutionStruct;
pub const cudnnConvolutionFwdAlgoPerfStruct = struct_cudnnConvolutionFwdAlgoPerfStruct;
pub const cudnnConvolutionBwdDataAlgoPerfStruct = struct_cudnnConvolutionBwdDataAlgoPerfStruct;
pub const cudnnFusedOpsConstParamStruct = struct_cudnnFusedOpsConstParamStruct;
pub const cudnnFusedOpsVariantParamStruct = struct_cudnnFusedOpsVariantParamStruct;
pub const cudnnFusedOpsPlanStruct = struct_cudnnFusedOpsPlanStruct;
pub const cudnnConvolutionBwdFilterAlgoPerfStruct = struct_cudnnConvolutionBwdFilterAlgoPerfStruct;
