//! The ROCm backend's stand-ins for the CUDA driver, cuBLAS and cuBLASLt
//! calls in `gpu.rs`: the same names and shapes as cudarc's `sys` modules,
//! backed by HIP (libamdhip64) and hipBLAS, so `gpu.rs` is one code path for
//! both vendors. Enum values are HIP's (ROCm 7). TMA and the FP8 matmul have
//! no HIP counterpart here and report "not supported".
#![allow(
    unsafe_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

use std::ffi::{c_char, c_int, c_uint, c_void};

/// hipErrorNotSupported.
const NOT_SUPPORTED: c_int = 801;

/// hipBLASLt solutions tuned per matrix-multiply shape, by GPU architecture
/// (major * 10 + minor) and compute units: made with
/// scripts/tune_hipblaslt.sh on ROCm 7.14's hipBLASLt (solutions are
/// specific to a hipBLASLt build; others fall back to its own choices).
const TUNED_GEMMS: &[(u32, i32, &str, &str)] = &[(
    94,
    304,
    "hipblaslt-gfx942-304.txt",
    include_str!("../tuning/hipblaslt-gfx942-304.txt"),
)];

/// Points hipBLASLt at this GPU's tuned solutions (HIPBLASLT_TUNING_OVERRIDE_FILE,
/// read when hipBLASLt loads), unless that is already set or
/// POCKET_HIPBLASLT_TUNING=0. For decoding's 8 to 256 rows the tuned
/// solutions are up to 1.9 times as fast as hipBLASLt's heuristic picks.
pub fn use_tuned_gemms(arch: u32, cus: i32) {
    if std::env::var_os("HIPBLASLT_TUNING_OVERRIDE_FILE").is_some()
        || std::env::var_os("POCKET_HIPBLASLT_TUNING").is_some_and(|v| v == "0")
    {
        return;
    }
    let Some(&(_, _, name, table)) = TUNED_GEMMS.iter().find(|t| t.0 == arch && t.1 == cus) else {
        return;
    };
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    table.hash(&mut h);
    let path = std::env::temp_dir().join(format!(
        "pocket-engine-hipblaslt-{arch}-{cus}-{:016x}.txt",
        h.finish()
    ));
    if path.exists() || std::fs::write(&path, table).is_ok() {
        // SAFETY: called while creating the GPU, before hipBLASLt reads it.
        unsafe { std::env::set_var("HIPBLASLT_TUNING_OVERRIDE_FILE", &path) };
        eprintln!("hipBLASLt: tuned solutions from tuning/{name}");
    }
}

pub mod cu {
    use super::*;

    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct CUresult(pub c_int);
    impl CUresult {
        pub const CUDA_SUCCESS: Self = Self(0);
    }

    pub type CUdevice = c_int;
    pub type CUcontext = *mut c_void;
    pub type CUmodule = *mut c_void;
    pub type CUfunction = *mut c_void;
    pub type CUstream = *mut c_void;
    pub type CUevent = *mut c_void;
    pub type CUgraph = *mut c_void;
    pub type CUgraphExec = *mut c_void;
    pub type CUdeviceptr = u64;

    #[repr(transparent)]
    #[derive(Clone, Copy)]
    pub struct CUdevice_attribute(c_int);
    impl CUdevice_attribute {
        pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: Self = Self(23);
        pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: Self = Self(61);
        pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: Self = Self(63);
        pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: Self = Self(10002);
    }

    pub struct CUstream_flags;
    impl CUstream_flags {
        pub const CU_STREAM_NON_BLOCKING: u32 = 1;
    }
    pub struct CUevent_flags;
    impl CUevent_flags {
        pub const CU_EVENT_DISABLE_TIMING: u32 = 2;
    }
    #[repr(transparent)]
    #[derive(Clone, Copy)]
    pub struct CUstreamCaptureMode(c_int);
    impl CUstreamCaptureMode {
        pub const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: Self = Self(1);
    }
    #[repr(transparent)]
    #[derive(Clone, Copy)]
    pub struct CUfunction_attribute(c_int);
    impl CUfunction_attribute {
        pub const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: Self = Self(8);
    }

    #[repr(C, align(64))]
    pub struct CUtensorMap {
        pub opaque: [u64; 16],
    }
    pub struct CUtensorMapDataType;
    impl CUtensorMapDataType {
        pub const CU_TENSOR_MAP_DATA_TYPE_BFLOAT16: u32 = 0;
    }
    pub struct CUtensorMapInterleave;
    impl CUtensorMapInterleave {
        pub const CU_TENSOR_MAP_INTERLEAVE_NONE: u32 = 0;
    }
    pub struct CUtensorMapSwizzle;
    impl CUtensorMapSwizzle {
        pub const CU_TENSOR_MAP_SWIZZLE_128B: u32 = 0;
    }
    pub struct CUtensorMapL2promotion;
    impl CUtensorMapL2promotion {
        pub const CU_TENSOR_MAP_L2_PROMOTION_L2_256B: u32 = 0;
    }
    pub struct CUtensorMapFloatOOBfill;
    impl CUtensorMapFloatOOBfill {
        pub const CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE: u32 = 0;
    }

    #[link(name = "amdhip64")]
    unsafe extern "C" {
        fn hipInit(flags: c_uint) -> CUresult;
        fn hipDeviceGet(dev: *mut c_int, ordinal: c_int) -> CUresult;
        fn hipSetDevice(dev: c_int) -> CUresult;
        fn hipDeviceGetName(name: *mut c_char, len: c_int, dev: c_int) -> CUresult;
        fn hipDeviceGetAttribute(v: *mut c_int, attr: c_int, dev: c_int) -> CUresult;
        fn hipGetErrorString(e: CUresult) -> *const c_char;
        fn hipStreamCreateWithFlags(s: *mut CUstream, flags: c_uint) -> CUresult;
        fn hipStreamSynchronize(s: CUstream) -> CUresult;
        fn hipModuleLoadData(m: *mut CUmodule, image: *const c_void) -> CUresult;
        fn hipModuleGetFunction(f: *mut CUfunction, m: CUmodule, name: *const c_char) -> CUresult;
        fn hipModuleLaunchKernel(
            f: CUfunction,
            gx: c_uint,
            gy: c_uint,
            gz: c_uint,
            bx: c_uint,
            by: c_uint,
            bz: c_uint,
            smem: c_uint,
            s: CUstream,
            params: *mut *mut c_void,
            extra: *mut *mut c_void,
        ) -> CUresult;
        fn hipMalloc(p: *mut u64, bytes: usize) -> CUresult;
        fn hipFree(p: u64) -> CUresult;
        fn hipHostMalloc(p: *mut *mut c_void, bytes: usize, flags: c_uint) -> CUresult;
        fn hipHostFree(p: *mut c_void) -> CUresult;
        fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> CUresult;
        fn hipMemcpyHtoD(dst: u64, src: *const c_void, bytes: usize) -> CUresult;
        fn hipMemcpyHtoDAsync(dst: u64, src: *const c_void, bytes: usize, s: CUstream) -> CUresult;
        fn hipMemcpyDtoHAsync(dst: *mut c_void, src: u64, bytes: usize, s: CUstream) -> CUresult;
        fn hipMemcpyDtoDAsync(dst: u64, src: u64, bytes: usize, s: CUstream) -> CUresult;
        fn hipMemsetD8Async(dst: u64, v: u8, bytes: usize, s: CUstream) -> CUresult;
        fn hipEventCreateWithFlags(e: *mut CUevent, flags: c_uint) -> CUresult;
        fn hipEventRecord(e: CUevent, s: CUstream) -> CUresult;
        fn hipEventSynchronize(e: CUevent) -> CUresult;
        fn hipEventDestroy(e: CUevent) -> CUresult;
        fn hipEventElapsedTime(ms: *mut f32, a: CUevent, b: CUevent) -> CUresult;
        fn hipStreamWaitEvent(s: CUstream, e: CUevent, flags: c_uint) -> CUresult;
        fn hipStreamBeginCapture(s: CUstream, mode: CUstreamCaptureMode) -> CUresult;
        fn hipStreamEndCapture(s: CUstream, g: *mut CUgraph) -> CUresult;
        fn hipGraphInstantiateWithFlags(e: *mut CUgraphExec, g: CUgraph, flags: u64) -> CUresult;
        fn hipGraphLaunch(e: CUgraphExec, s: CUstream) -> CUresult;
        fn hipGraphExecDestroy(e: CUgraphExec) -> CUresult;
        fn hipGraphDestroy(g: CUgraph) -> CUresult;
    }

    pub unsafe fn cuInit(flags: c_uint) -> CUresult {
        unsafe { hipInit(flags) }
    }
    pub unsafe fn cuDeviceGet(dev: *mut CUdevice, ordinal: c_int) -> CUresult {
        unsafe { hipDeviceGet(dev, ordinal) }
    }
    /// HIP has no explicit contexts: selecting the device is enough.
    pub unsafe fn cuDevicePrimaryCtxRetain(ctx: *mut CUcontext, dev: CUdevice) -> CUresult {
        unsafe {
            *ctx = std::ptr::null_mut();
            hipSetDevice(dev)
        }
    }
    pub unsafe fn cuCtxSetCurrent(_ctx: CUcontext) -> CUresult {
        CUresult::CUDA_SUCCESS
    }
    pub unsafe fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult {
        unsafe { hipDeviceGetName(name, len, dev) }
    }
    pub unsafe fn cuDeviceGetAttribute(
        v: *mut c_int,
        attr: CUdevice_attribute,
        dev: CUdevice,
    ) -> CUresult {
        unsafe { hipDeviceGetAttribute(v, attr.0, dev) }
    }
    pub unsafe fn cuGetErrorString(e: CUresult, msg: *mut *const c_char) -> CUresult {
        unsafe { *msg = hipGetErrorString(e) };
        CUresult::CUDA_SUCCESS
    }
    pub unsafe fn cuStreamCreate(s: *mut CUstream, flags: c_uint) -> CUresult {
        unsafe { hipStreamCreateWithFlags(s, flags) }
    }
    pub unsafe fn cuStreamSynchronize(s: CUstream) -> CUresult {
        unsafe { hipStreamSynchronize(s) }
    }
    pub unsafe fn cuModuleLoadData(m: *mut CUmodule, image: *const c_void) -> CUresult {
        unsafe { hipModuleLoadData(m, image) }
    }
    pub unsafe fn cuModuleGetFunction(
        f: *mut CUfunction,
        m: CUmodule,
        name: *const c_char,
    ) -> CUresult {
        unsafe { hipModuleGetFunction(f, m, name) }
    }
    /// gfx942 gives every block up to 64 KB of LDS without opting in.
    pub unsafe fn cuFuncSetAttribute(
        _f: CUfunction,
        _attr: CUfunction_attribute,
        _v: c_int,
    ) -> CUresult {
        CUresult::CUDA_SUCCESS
    }
    pub unsafe fn cuLaunchKernel(
        f: CUfunction,
        gx: c_uint,
        gy: c_uint,
        gz: c_uint,
        bx: c_uint,
        by: c_uint,
        bz: c_uint,
        smem: c_uint,
        s: CUstream,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult {
        unsafe { hipModuleLaunchKernel(f, gx, gy, gz, bx, by, bz, smem, s, params, extra) }
    }
    pub unsafe fn cuMemAlloc_v2(p: *mut CUdeviceptr, bytes: usize) -> CUresult {
        unsafe { hipMalloc(p, bytes) }
    }
    pub unsafe fn cuMemFree_v2(p: CUdeviceptr) -> CUresult {
        unsafe { hipFree(p) }
    }
    pub unsafe fn cuMemHostAlloc(p: *mut *mut c_void, bytes: usize, flags: c_uint) -> CUresult {
        unsafe { hipHostMalloc(p, bytes, flags) }
    }
    pub unsafe fn cuMemFreeHost(p: *mut c_void) -> CUresult {
        unsafe { hipHostFree(p) }
    }
    pub unsafe fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult {
        unsafe { hipMemGetInfo(free, total) }
    }
    pub unsafe fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, bytes: usize) -> CUresult {
        unsafe { hipMemcpyHtoD(dst, src, bytes) }
    }
    pub unsafe fn cuMemcpyHtoDAsync_v2(
        dst: CUdeviceptr,
        src: *const c_void,
        bytes: usize,
        s: CUstream,
    ) -> CUresult {
        unsafe { hipMemcpyHtoDAsync(dst, src, bytes, s) }
    }
    pub unsafe fn cuMemcpyDtoHAsync_v2(
        dst: *mut c_void,
        src: CUdeviceptr,
        bytes: usize,
        s: CUstream,
    ) -> CUresult {
        unsafe { hipMemcpyDtoHAsync(dst, src, bytes, s) }
    }
    pub unsafe fn cuMemcpyDtoDAsync_v2(
        dst: CUdeviceptr,
        src: CUdeviceptr,
        bytes: usize,
        s: CUstream,
    ) -> CUresult {
        unsafe { hipMemcpyDtoDAsync(dst, src, bytes, s) }
    }
    pub unsafe fn cuMemsetD8Async(dst: CUdeviceptr, v: u8, bytes: usize, s: CUstream) -> CUresult {
        unsafe { hipMemsetD8Async(dst, v, bytes, s) }
    }
    pub unsafe fn cuEventCreate(e: *mut CUevent, flags: c_uint) -> CUresult {
        unsafe { hipEventCreateWithFlags(e, flags) }
    }
    pub unsafe fn cuEventRecord(e: CUevent, s: CUstream) -> CUresult {
        unsafe { hipEventRecord(e, s) }
    }
    pub unsafe fn cuEventSynchronize(e: CUevent) -> CUresult {
        unsafe { hipEventSynchronize(e) }
    }
    pub unsafe fn cuEventDestroy_v2(e: CUevent) -> CUresult {
        unsafe { hipEventDestroy(e) }
    }
    pub unsafe fn cuEventElapsedTime_v2(ms: *mut f32, a: CUevent, b: CUevent) -> CUresult {
        unsafe { hipEventElapsedTime(ms, a, b) }
    }
    pub unsafe fn cuStreamWaitEvent(s: CUstream, e: CUevent, flags: c_uint) -> CUresult {
        unsafe { hipStreamWaitEvent(s, e, flags) }
    }
    pub unsafe fn cuStreamBeginCapture_v2(s: CUstream, mode: CUstreamCaptureMode) -> CUresult {
        unsafe { hipStreamBeginCapture(s, mode) }
    }
    pub unsafe fn cuStreamEndCapture(s: CUstream, g: *mut CUgraph) -> CUresult {
        unsafe { hipStreamEndCapture(s, g) }
    }
    pub unsafe fn cuGraphInstantiateWithFlags(
        e: *mut CUgraphExec,
        g: CUgraph,
        flags: u64,
    ) -> CUresult {
        unsafe { hipGraphInstantiateWithFlags(e, g, flags) }
    }
    pub unsafe fn cuGraphLaunch(e: CUgraphExec, s: CUstream) -> CUresult {
        unsafe { hipGraphLaunch(e, s) }
    }
    pub unsafe fn cuGraphExecDestroy(e: CUgraphExec) -> CUresult {
        unsafe { hipGraphExecDestroy(e) }
    }
    pub unsafe fn cuGraphDestroy(g: CUgraph) -> CUresult {
        unsafe { hipGraphDestroy(g) }
    }
    /// TMA is Hopper's; the kernels that read tensor maps are not built for AMD.
    pub unsafe fn cuTensorMapEncodeTiled(
        _m: *mut CUtensorMap,
        _ty: u32,
        _rank: c_uint,
        _ptr: *mut c_void,
        _dims: *const u64,
        _strides: *const u64,
        _boxes: *const u32,
        _elem: *const u32,
        _interleave: u32,
        _swizzle: u32,
        _l2: u32,
        _oob: u32,
    ) -> CUresult {
        CUresult(NOT_SUPPORTED)
    }
}

pub mod cb {
    use super::*;

    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct cublasStatus_t(pub c_int);
    impl cublasStatus_t {
        pub const CUBLAS_STATUS_SUCCESS: Self = Self(0);
    }
    pub type cublasHandle_t = *mut c_void;

    #[repr(transparent)]
    #[derive(Clone, Copy)]
    pub struct cudaDataType_t(c_int);
    impl cudaDataType_t {
        pub const CUDA_R_32F: Self = Self(0);
        pub const CUDA_R_16BF: Self = Self(14);
    }
    #[repr(transparent)]
    #[derive(Clone, Copy)]
    pub struct cublasOperation_t(c_int);
    impl cublasOperation_t {
        pub const CUBLAS_OP_N: Self = Self(111);
        pub const CUBLAS_OP_T: Self = Self(112);
    }
    #[repr(transparent)]
    #[derive(Clone, Copy)]
    pub struct cublasComputeType_t(c_int);
    impl cublasComputeType_t {
        pub const CUBLAS_COMPUTE_32F: Self = Self(2);
    }
    #[repr(transparent)]
    #[derive(Clone, Copy)]
    pub struct cublasGemmAlgo_t(c_int);
    impl cublasGemmAlgo_t {
        pub const CUBLAS_GEMM_DEFAULT: Self = Self(160);
    }

    #[link(name = "hipblas")]
    unsafe extern "C" {
        fn hipblasCreate(h: *mut cublasHandle_t) -> cublasStatus_t;
        fn hipblasSetStream(h: cublasHandle_t, s: *mut c_void) -> cublasStatus_t;
        fn hipblasSetWorkspace(h: cublasHandle_t, p: *mut c_void, bytes: usize) -> cublasStatus_t;
        fn hipblasGemmEx(
            h: cublasHandle_t,
            ta: cublasOperation_t,
            tb: cublasOperation_t,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: *const c_void,
            a: *const c_void,
            at: cudaDataType_t,
            lda: c_int,
            b: *const c_void,
            bt: cudaDataType_t,
            ldb: c_int,
            beta: *const c_void,
            c: *mut c_void,
            ct: cudaDataType_t,
            ldc: c_int,
            compute: cublasComputeType_t,
            algo: cublasGemmAlgo_t,
        ) -> cublasStatus_t;
        fn hipblasGemmStridedBatchedEx(
            h: cublasHandle_t,
            ta: cublasOperation_t,
            tb: cublasOperation_t,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: *const c_void,
            a: *const c_void,
            at: cudaDataType_t,
            lda: c_int,
            sa: i64,
            b: *const c_void,
            bt: cudaDataType_t,
            ldb: c_int,
            sb: i64,
            beta: *const c_void,
            c: *mut c_void,
            ct: cudaDataType_t,
            ldc: c_int,
            sc: i64,
            batch: c_int,
            compute: cublasComputeType_t,
            algo: cublasGemmAlgo_t,
        ) -> cublasStatus_t;
    }

    /// hipBLAS runs on rocBLAS, whose own kernels pick poor tiles for the
    /// skinny products of decoding (an LM head at 64 rows took 871 us on an
    /// MI325X against hipBLASLt's 483): route them to hipBLASLt unless the
    /// environment says otherwise.
    pub unsafe fn cublasCreate_v2(h: *mut cublasHandle_t) -> cublasStatus_t {
        if std::env::var_os("ROCBLAS_USE_HIPBLASLT").is_none() {
            // SAFETY: called once while creating the GPU, before rocBLAS reads it.
            unsafe { std::env::set_var("ROCBLAS_USE_HIPBLASLT", "1") };
        }
        unsafe { hipblasCreate(h) }
    }
    pub unsafe fn cublasSetStream_v2(h: cublasHandle_t, s: *mut c_void) -> cublasStatus_t {
        unsafe { hipblasSetStream(h, s) }
    }
    pub unsafe fn cublasSetWorkspace_v2(
        h: cublasHandle_t,
        p: *mut c_void,
        bytes: usize,
    ) -> cublasStatus_t {
        unsafe { hipblasSetWorkspace(h, p, bytes) }
    }
    pub unsafe fn cublasGemmEx(
        h: cublasHandle_t,
        ta: cublasOperation_t,
        tb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        at: cudaDataType_t,
        lda: c_int,
        b: *const c_void,
        bt: cudaDataType_t,
        ldb: c_int,
        beta: *const c_void,
        c: *mut c_void,
        ct: cudaDataType_t,
        ldc: c_int,
        compute: cublasComputeType_t,
        algo: cublasGemmAlgo_t,
    ) -> cublasStatus_t {
        unsafe {
            hipblasGemmEx(
                h, ta, tb, m, n, k, alpha, a, at, lda, b, bt, ldb, beta, c, ct, ldc, compute, algo,
            )
        }
    }
    pub unsafe fn cublasGemmStridedBatchedEx(
        h: cublasHandle_t,
        ta: cublasOperation_t,
        tb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        at: cudaDataType_t,
        lda: c_int,
        sa: i64,
        b: *const c_void,
        bt: cudaDataType_t,
        ldb: c_int,
        sb: i64,
        beta: *const c_void,
        c: *mut c_void,
        ct: cudaDataType_t,
        ldc: c_int,
        sc: i64,
        batch: c_int,
        compute: cublasComputeType_t,
        algo: cublasGemmAlgo_t,
    ) -> cublasStatus_t {
        unsafe {
            hipblasGemmStridedBatchedEx(
                h, ta, tb, m, n, k, alpha, a, at, lda, sa, b, bt, ldb, sb, beta, c, ct, ldc, sc,
                batch, compute, algo,
            )
        }
    }
}

/// cuBLASLt runs only the FP8 matmul, which this backend does not offer
/// (`Gpu::fp8_mode` is None on AMD): a handle that refuses every plan.
pub mod lt {
    use super::*;

    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct cublasStatus_t(pub c_int);
    impl cublasStatus_t {
        pub const CUBLAS_STATUS_SUCCESS: Self = Self(0);
        pub const CUBLAS_STATUS_NOT_SUPPORTED: Self = Self(15);
    }
    pub type cublasLtHandle_t = *mut c_void;
    pub type cublasLtMatmulDesc_t = *mut c_void;
    pub type cublasLtMatrixLayout_t = *mut c_void;
    pub type cublasLtMatmulPreference_t = *mut c_void;
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct cublasLtMatmulAlgo_t {
        pub data: [u64; 8],
    }
    #[repr(C)]
    pub struct cublasLtMatmulHeuristicResult_t {
        pub algo: cublasLtMatmulAlgo_t,
    }
    #[repr(transparent)]
    #[derive(Clone, Copy)]
    pub struct cudaDataType_t(c_int);
    impl cudaDataType_t {
        pub const CUDA_R_32F: Self = Self(0);
        pub const CUDA_R_16BF: Self = Self(14);
        pub const CUDA_R_8F_E4M3: Self = Self(28);
    }
    #[repr(transparent)]
    #[derive(Clone, Copy)]
    pub struct cublasComputeType_t(c_int);
    impl cublasComputeType_t {
        pub const CUBLAS_COMPUTE_32F: Self = Self(2);
    }
    #[repr(transparent)]
    #[derive(Clone, Copy)]
    pub struct cublasLtMatmulDescAttributes_t(c_int);
    impl cublasLtMatmulDescAttributes_t {
        pub const CUBLASLT_MATMUL_DESC_TRANSA: Self = Self(3);
        pub const CUBLASLT_MATMUL_DESC_TRANSB: Self = Self(4);
        pub const CUBLASLT_MATMUL_DESC_A_SCALE_POINTER: Self = Self(17);
        pub const CUBLASLT_MATMUL_DESC_B_SCALE_POINTER: Self = Self(18);
        pub const CUBLASLT_MATMUL_DESC_A_SCALE_MODE: Self = Self(31);
        pub const CUBLASLT_MATMUL_DESC_B_SCALE_MODE: Self = Self(32);
    }
    #[repr(transparent)]
    #[derive(Clone, Copy)]
    pub struct cublasLtMatmulPreferenceAttributes_t(c_int);
    impl cublasLtMatmulPreferenceAttributes_t {
        pub const CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES: Self = Self(1);
    }

    const NO: cublasStatus_t = cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED;

    pub unsafe fn cublasLtCreate(h: *mut cublasLtHandle_t) -> cublasStatus_t {
        unsafe { *h = std::ptr::null_mut() };
        cublasStatus_t::CUBLAS_STATUS_SUCCESS
    }
    pub unsafe fn cublasLtMatmulDescCreate(
        _d: *mut cublasLtMatmulDesc_t,
        _c: cublasComputeType_t,
        _s: cudaDataType_t,
    ) -> cublasStatus_t {
        NO
    }
    pub unsafe fn cublasLtMatmulDescSetAttribute(
        _d: cublasLtMatmulDesc_t,
        _a: cublasLtMatmulDescAttributes_t,
        _v: *const c_void,
        _n: usize,
    ) -> cublasStatus_t {
        NO
    }
    pub unsafe fn cublasLtMatrixLayoutCreate(
        _l: *mut cublasLtMatrixLayout_t,
        _t: cudaDataType_t,
        _r: u64,
        _c: u64,
        _ld: i64,
    ) -> cublasStatus_t {
        NO
    }
    pub unsafe fn cublasLtMatmulPreferenceCreate(
        _p: *mut cublasLtMatmulPreference_t,
    ) -> cublasStatus_t {
        NO
    }
    pub unsafe fn cublasLtMatmulPreferenceSetAttribute(
        _p: cublasLtMatmulPreference_t,
        _a: cublasLtMatmulPreferenceAttributes_t,
        _v: *const c_void,
        _n: usize,
    ) -> cublasStatus_t {
        NO
    }
    pub unsafe fn cublasLtMatmulPreferenceDestroy(
        _p: cublasLtMatmulPreference_t,
    ) -> cublasStatus_t {
        NO
    }
    pub unsafe fn cublasLtMatmulAlgoGetHeuristic(
        _h: cublasLtHandle_t,
        _d: cublasLtMatmulDesc_t,
        _a: cublasLtMatrixLayout_t,
        _b: cublasLtMatrixLayout_t,
        _c: cublasLtMatrixLayout_t,
        _dd: cublasLtMatrixLayout_t,
        _p: cublasLtMatmulPreference_t,
        _n: c_int,
        _r: *mut cublasLtMatmulHeuristicResult_t,
        _found: *mut c_int,
    ) -> cublasStatus_t {
        NO
    }
    pub unsafe fn cublasLtMatmul(
        _h: cublasLtHandle_t,
        _d: cublasLtMatmulDesc_t,
        _alpha: *const c_void,
        _a: *const c_void,
        _al: cublasLtMatrixLayout_t,
        _b: *const c_void,
        _bl: cublasLtMatrixLayout_t,
        _beta: *const c_void,
        _c: *const c_void,
        _cl: cublasLtMatrixLayout_t,
        _dp: *mut c_void,
        _dl: cublasLtMatrixLayout_t,
        _algo: *const cublasLtMatmulAlgo_t,
        _ws: *mut c_void,
        _wsb: usize,
        _s: *mut c_void,
    ) -> cublasStatus_t {
        NO
    }
}
