//! The raw CUDA boundary: one context and stream, device buffers, kernel
//! launches, cuBLAS GEMMs and CUDA graphs. Everything here is driven from the
//! engine thread that created it.
#![allow(unsafe_code)]

#[cfg(feature = "rocm")]
use crate::hip::{cb, cu, lt};
use crate::{Error, Result};
#[cfg(not(feature = "rocm"))]
use cudarc::cublas::sys as cb;
#[cfg(not(feature = "rocm"))]
use cudarc::cublaslt::sys as lt;
#[cfg(not(feature = "rocm"))]
use cudarc::driver::sys as cu;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::{CStr, CString, c_void};

fn check(r: cu::CUresult, what: &'static str) -> Result<()> {
    if r == cu::CUresult::CUDA_SUCCESS {
        return Ok(());
    }
    let mut msg: *const std::ffi::c_char = std::ptr::null();
    // SAFETY: cuGetErrorString writes a pointer to a static string or leaves null.
    unsafe { cu::cuGetErrorString(r, &mut msg) };
    let text = if msg.is_null() {
        format!("{r:?}")
    } else {
        // SAFETY: non-null strings from the driver are static and NUL-terminated.
        unsafe { CStr::from_ptr(msg) }
            .to_string_lossy()
            .into_owned()
    };
    Err(Error::Cuda(what, text))
}

fn check_blas(r: cb::cublasStatus_t, what: &'static str) -> Result<()> {
    if r == cb::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(Error::Cuda(what, format!("{r:?}")))
    }
}

/// Device memory, freed on drop.
pub struct Buf {
    pub ptr: u64,
    pub bytes: usize,
}

impl Drop for Buf {
    fn drop(&mut self) {
        if self.ptr != 0 {
            // SAFETY: the pointer came from cuMemAlloc and is freed once.
            unsafe { cu::cuMemFree_v2(self.ptr) };
        }
    }
}

/// Page-locked host memory for staging transfers, freed on drop.
pub struct Pinned {
    ptr: *mut u8,
    pub bytes: usize,
}

impl Pinned {
    pub fn as_mut_slice<T: Copy>(&mut self) -> &mut [T] {
        // SAFETY: the allocation is `bytes` long and suitably aligned (page aligned).
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr.cast(), self.bytes / std::mem::size_of::<T>())
        }
    }
    pub fn as_slice<T: Copy>(&self) -> &[T] {
        // SAFETY: as above.
        unsafe {
            std::slice::from_raw_parts(self.ptr.cast(), self.bytes / std::mem::size_of::<T>())
        }
    }
}

// SAFETY: pinned memory is plain host memory; whichever thread drops it frees
// it with the context current (the engine's, or a Pinner's worker).
unsafe impl Send for Pinned {}

/// Allocates pinned memory from any thread, for work prepared off the engine
/// thread (see [`Gpu::pinner`]).
#[derive(Clone)]
pub struct Pinner(cu::CUcontext);

// SAFETY: a context handle may be made current on any thread.
unsafe impl Send for Pinner {}
unsafe impl Sync for Pinner {}

impl Pinner {
    pub fn pinned(&self, bytes: usize) -> Result<Pinned> {
        // SAFETY: the context outlives the engine's work; making it current
        // on this thread lets the allocation belong to it.
        check(unsafe { cu::cuCtxSetCurrent(self.0) }, "cuCtxSetCurrent")?;
        let mut ptr = std::ptr::null_mut();
        // SAFETY: out-parameter allocation; ownership moves into Pinned.
        check(
            unsafe { cu::cuMemHostAlloc(&mut ptr, bytes.max(8), 0) },
            "cuMemHostAlloc",
        )?;
        Ok(Pinned {
            ptr: ptr.cast(),
            bytes,
        })
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        // SAFETY: the pointer came from cuMemHostAlloc and is freed once.
        unsafe { cu::cuMemFreeHost(self.ptr.cast()) };
    }
}

/// Bytes to upload: a range of a file, read straight into pinned staging, or
/// host memory.
pub enum Piece<'a> {
    File {
        file: &'a std::fs::File,
        offset: u64,
        len: usize,
    },
    Mem(&'a [u8]),
}

impl Piece<'_> {
    pub fn len(&self) -> usize {
        match self {
            Piece::File { len, .. } => *len,
            Piece::Mem(m) => m.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Pinned buffers that loading cycles through: threads fill one from the
/// files while the copy engine drains the others.
struct Staging {
    bufs: Vec<(Pinned, cu::CUevent)>,
    next: usize,
    threads: usize,
    /// Device scratch the loader converts from, grown on demand.
    temp: [Option<Buf>; 2],
}

const STAGE_BYTES: usize = 64 << 20;
const STAGES: usize = 4;

impl Drop for Staging {
    fn drop(&mut self) {
        for (_, e) in &self.bufs {
            // SAFETY: each event was created once and is destroyed once, after the
            // owner synchronized the stream.
            unsafe { cu::cuEventDestroy_v2(*e) };
        }
    }
}

/// Fills `dst` from `file` at `offset`, splitting the read across threads.
fn read_parallel(file: &std::fs::File, offset: u64, dst: &mut [u8], threads: usize) -> Result<()> {
    use std::os::unix::fs::FileExt;
    let per = dst.len().div_ceil(threads).max(4 << 20);
    if per >= dst.len() {
        return Ok(file.read_exact_at(dst, offset)?);
    }
    std::thread::scope(|s| {
        let jobs: Vec<_> = dst
            .chunks_mut(per)
            .enumerate()
            .map(|(i, c)| s.spawn(move || file.read_exact_at(c, offset + (i * per) as u64)))
            .collect();
        jobs.into_iter()
            .try_for_each(|j| j.join().expect("reader thread panicked"))
    })?;
    Ok(())
}

#[derive(Clone, Copy)]
pub struct Kernel(cu::CUfunction);

/// A kernel argument, passed by value.
#[derive(Clone, Copy)]
pub enum Arg {
    P(u64),
    I(i32),
    F(f32),
    /// A TMA tensor map, passed by value (a `__grid_constant__ CUtensorMap`).
    Map(&'static TensorMap),
}

/// A TMA descriptor of a device tensor (see [`Gpu::tensor_map_3d`]).
pub struct TensorMap(cu::CUtensorMap);

/// A matrix operand of [`Gpu::gemm_batched`]: column-major with leading
/// dimension `ld`, `stride` elements between batch members, optionally
/// transposed, bf16 unless `f32`.
#[derive(Clone, Copy)]
pub struct Mat {
    pub ptr: u64,
    pub ld: usize,
    pub stride: usize,
    pub trans: bool,
    pub f32: bool,
}

pub struct Graph {
    graph: cu::CUgraph,
    exec: cu::CUgraphExec,
}

impl Drop for Graph {
    fn drop(&mut self) {
        // SAFETY: both handles came from capture/instantiation and are destroyed once.
        unsafe {
            cu::cuGraphExecDestroy(self.exec);
            cu::cuGraphDestroy(self.graph);
        }
    }
}

/// How FP8 matrices carry their scales, by what the device's cuBLASLt runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Mode {
    /// f32 scales per 128 x 128 weight tile and per 128 activations of a row
    /// (Hopper; Qwen's FP8 checkpoints use this layout).
    Block128,
    /// MXFP8: power-of-two (E8M0) scales per 32 values of a row, in
    /// cuBLASLt's 128 x 4 tiles (Blackwell).
    Mx32,
}

impl Fp8Mode {
    /// Values sharing one scale along K.
    pub fn group(self) -> usize {
        match self {
            Fp8Mode::Block128 => 128,
            Fp8Mode::Mx32 => 32,
        }
    }
    /// Bytes of activation scales for `rows` x `k`.
    pub fn act_scale_bytes(self, rows: usize, k: usize) -> usize {
        match self {
            Fp8Mode::Block128 => rows * k.div_ceil(128) * 4,
            Fp8Mode::Mx32 => mx_scale_bytes(rows, k),
        }
    }
    /// Bytes of weight scales for an `n` x `k` matrix.
    pub fn weight_scale_bytes(self, n: usize, k: usize) -> usize {
        match self {
            Fp8Mode::Block128 => n.div_ceil(128) * k.div_ceil(128) * 4,
            Fp8Mode::Mx32 => mx_scale_bytes(n, k),
        }
    }
}

/// MXFP8 scales of a `rows` x `k` matrix: 128-row by 4-block tiles of 512 bytes.
pub fn mx_scale_bytes(rows: usize, k: usize) -> usize {
    rows.div_ceil(128) * k.div_ceil(32).div_ceil(4) * 512
}

/// An FP8 matmul's operands: weights, activation scales, n, k, rows, ldy and
/// whether the output is f32.
type PlanKey = (u64, u64, usize, usize, usize, usize, bool);

struct LtPlan {
    desc: lt::cublasLtMatmulDesc_t,
    a: lt::cublasLtMatrixLayout_t,
    b: lt::cublasLtMatrixLayout_t,
    d: lt::cublasLtMatrixLayout_t,
    algo: lt::cublasLtMatmulAlgo_t,
    /// Algorithms checked against this plan, and whether they run it.
    #[cfg_attr(feature = "rocm", allow(dead_code))]
    fits: Vec<(crate::tuning::Algo, bool)>,
}

/// A matrix-multiply shape as tuning tables key it: y[rows, ldy] = x[rows,
/// k] . w[n, k]^T, its kind naming the input and output types ("bf16",
/// "bf16 f32", "fp8 block128", "fp8 mx32 f32", ...).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GemmShape {
    pub kind: String,
    pub n: usize,
    pub k: usize,
    pub ldy: usize,
}

/// The kind of a product, as [`GemmShape`] names it.
pub fn gemm_kind(fp8: Option<Fp8Mode>, f32_out: bool) -> String {
    let base = match fp8 {
        None => "bf16",
        Some(Fp8Mode::Block128) => "fp8 block128",
        Some(Fp8Mode::Mx32) => "fp8 mx32",
    };
    if f32_out {
        format!("{base} f32")
    } else {
        base.to_string()
    }
}

/// Where a model ran one shape (POCKET_RECORD_GEMMS=1): each layer's weights
/// and FP8 weight scales, and the inputs and outputs of its largest call,
/// which `pocket-engine tune` reuses to time algorithms on real buffers.
#[derive(Clone, Debug, Default)]
pub struct GemmUse {
    pub weights: Vec<(u64, u64)>,
    pub x: u64,
    pub x_scale: u64,
    pub y: u64,
    pub fp8: Option<Fp8Mode>,
    pub f32_out: bool,
    pub rows: BTreeSet<usize>,
}

/// GPU time by kernel, from events around each launch (POCKET_PROFILE=1).
/// Graph replays show as one entry, so profile with graphs off.
#[derive(Default)]
struct Profiler {
    /// POCKET_PROFILE=shapes: matmuls by shape.
    shapes: bool,
    names: HashMap<usize, String>,
    pending: Vec<(String, cu::CUevent, cu::CUevent)>,
    spare: Vec<cu::CUevent>,
    totals: HashMap<String, (u64, f64)>,
    capturing: bool,
}

pub struct Gpu {
    /// Streaming multiprocessors on the device.
    pub sms: usize,
    /// Compute capability, as major * 10 + minor.
    pub arch: u32,
    /// Shared memory per SM, in bytes.
    pub smem_per_sm: usize,
    ctx: cu::CUcontext,
    lt: lt::cublasLtHandle_t,
    lt_ws: Buf,
    lt_plans: RefCell<HashMap<PlanKey, LtPlan>>,
    /// Plans for tuned products, by shape, rows and scale pointers.
    #[cfg_attr(feature = "rocm", allow(dead_code))]
    lt_tuned: RefCell<HashMap<(GemmShape, usize, u64, u64), LtPlan>>,
    /// Settings measured for this GPU (see `tuning`).
    pub tuning: Option<crate::tuning::Table>,
    /// POCKET_RECORD_GEMMS=1: the products run so far.
    record: Option<RefCell<BTreeMap<GemmShape, GemmUse>>>,
    stream: cu::CUstream,
    /// A second stream for work that overlaps the main one between `fork`
    /// and `join`, and the events ordering the two.
    side: cu::CUstream,
    fork_ev: cu::CUevent,
    join_ev: cu::CUevent,
    blas: cb::cublasHandle_t,
    module: cu::CUmodule,
    _workspace: Buf,
    staging: RefCell<Option<Staging>>,
    prof: Option<RefCell<Profiler>>,
    pub name: String,
    /// AMD: matrix-vector kernels for products of 1 to 4 rows (skinny_bf16_r*).
    #[cfg(feature = "rocm")]
    skinny: [Kernel; 4],
}

const KERNEL_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/qwen35.fatbin"));

impl Gpu {
    pub fn new(ordinal: i32) -> Result<Self> {
        if KERNEL_IMAGE.is_empty() {
            return Err(Error::Config(
                "built without the `cuda` or `rocm` feature: no kernels".into(),
            ));
        }
        // SAFETY: plain driver calls with out-parameters; handles are owned by Self.
        unsafe {
            check(cu::cuInit(0), "cuInit")?;
            let mut dev = 0;
            check(cu::cuDeviceGet(&mut dev, ordinal), "cuDeviceGet")?;
            let mut ctx = std::ptr::null_mut();
            check(
                cu::cuDevicePrimaryCtxRetain(&mut ctx, dev),
                "cuDevicePrimaryCtxRetain",
            )?;
            check(cu::cuCtxSetCurrent(ctx), "cuCtxSetCurrent")?;
            let mut name = [0 as std::ffi::c_char; 256];
            check(
                cu::cuDeviceGetName(name.as_mut_ptr(), 256, dev),
                "cuDeviceGetName",
            )?;
            let name = CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned();
            let (mut major, mut minor) = (0, 0);
            check(
                cu::cuDeviceGetAttribute(
                    &mut major,
                    cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    dev,
                ),
                "cuDeviceGetAttribute",
            )?;
            check(
                cu::cuDeviceGetAttribute(
                    &mut minor,
                    cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    dev,
                ),
                "cuDeviceGetAttribute",
            )?;
            let mut sms = 0;
            check(
                cu::cuDeviceGetAttribute(
                    &mut sms,
                    cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                    dev,
                ),
                "cuDeviceGetAttribute",
            )?;
            let mut smem_per_sm = 0;
            check(
                cu::cuDeviceGetAttribute(
                    &mut smem_per_sm,
                    cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
                    dev,
                ),
                "cuDeviceGetAttribute",
            )?;
            let mut stream = std::ptr::null_mut();
            check(
                cu::cuStreamCreate(
                    &mut stream,
                    cu::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
                ),
                "cuStreamCreate",
            )?;
            let mut side = std::ptr::null_mut();
            check(
                cu::cuStreamCreate(&mut side, cu::CUstream_flags::CU_STREAM_NON_BLOCKING as u32),
                "cuStreamCreate",
            )?;
            let (mut fork_ev, mut join_ev) = (std::ptr::null_mut(), std::ptr::null_mut());
            for ev in [&mut fork_ev, &mut join_ev] {
                check(
                    cu::cuEventCreate(ev, cu::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32),
                    "cuEventCreate",
                )?;
            }
            let mut module = std::ptr::null_mut();
            // POCKET_KERNELS names a cubin or fatbin to load in place of the
            // built-in image, for timing kernel variants without a rebuild.
            let custom = std::env::var_os("POCKET_KERNELS")
                .map(std::fs::read)
                .transpose()
                .map_err(|e| Error::Config(format!("POCKET_KERNELS: {e}")))?;
            let bytes = custom.as_deref().unwrap_or(KERNEL_IMAGE);
            let image = {
                // The driver wants an aligned image.
                let mut v = vec![0u64; bytes.len().div_ceil(8)];
                std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), bytes.len())
                    .copy_from_slice(bytes);
                v
            };
            check(
                cu::cuModuleLoadData(&mut module, image.as_ptr().cast()),
                "cuModuleLoadData",
            )?;
            #[cfg(feature = "rocm")]
            let skinny = {
                let mut ks = [Kernel(std::ptr::null_mut()); 4];
                for (r, k) in ks.iter_mut().enumerate() {
                    let name =
                        CString::new(format!("skinny_bf16_r{}", r + 1)).expect("kernel name");
                    check(
                        cu::cuModuleGetFunction(&mut k.0, module, name.as_ptr()),
                        "cuModuleGetFunction",
                    )?;
                }
                ks
            };
            #[cfg(feature = "rocm")]
            crate::hip::use_tuned_gemms((major * 10 + minor) as u32, sms);
            let mut blas = std::ptr::null_mut();
            check_blas(cb::cublasCreate_v2(&mut blas), "cublasCreate")?;
            check_blas(
                cb::cublasSetStream_v2(blas, stream.cast()),
                "cublasSetStream",
            )?;
            // A fixed workspace keeps cuBLAS from allocating during graph capture.
            let ws_bytes = 64 << 20;
            let mut ws = 0;
            check(cu::cuMemAlloc_v2(&mut ws, ws_bytes), "cuMemAlloc")?;
            let workspace = Buf {
                ptr: ws,
                bytes: ws_bytes,
            };
            check_blas(
                cb::cublasSetWorkspace_v2(blas, ws as *mut c_void, ws_bytes),
                "cublasSetWorkspace",
            )?;
            let mut lth = std::ptr::null_mut();
            let r = lt::cublasLtCreate(&mut lth);
            if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(Error::Cuda("cublasLtCreate", format!("{r:?}")));
            }
            let lt_bytes = 64 << 20;
            let mut lw = 0;
            check(cu::cuMemAlloc_v2(&mut lw, lt_bytes), "cuMemAlloc")?;
            #[cfg(not(feature = "rocm"))]
            let tuning =
                crate::tuning::Table::for_gpu(&name, sms as usize, lt::cublasLtGetVersion());
            #[cfg(feature = "rocm")]
            let tuning = None;
            Ok(Self {
                sms: sms as usize,
                smem_per_sm: smem_per_sm as usize,
                ctx,
                arch: (major * 10 + minor) as u32,
                lt: lth,
                lt_ws: Buf {
                    ptr: lw,
                    bytes: lt_bytes,
                },
                lt_plans: RefCell::new(HashMap::new()),
                lt_tuned: RefCell::new(HashMap::new()),
                tuning,
                record: std::env::var_os("POCKET_RECORD_GEMMS")
                    .is_some_and(|v| v == "1")
                    .then(|| RefCell::new(BTreeMap::new())),
                stream,
                side,
                fork_ev,
                join_ev,
                blas,
                module,
                _workspace: workspace,
                staging: RefCell::new(None),
                prof: std::env::var("POCKET_PROFILE").ok().map(|v| {
                    RefCell::new(Profiler {
                        shapes: v == "shapes",
                        ..Profiler::default()
                    })
                }),
                name,
                #[cfg(feature = "rocm")]
                skinny,
            })
        }
    }

    pub fn kernel(&self, name: &str) -> Result<Kernel> {
        let cname = CString::new(name).expect("kernel name");
        let mut f = std::ptr::null_mut();
        // SAFETY: the module is loaded for the lifetime of self.
        check(
            unsafe { cu::cuModuleGetFunction(&mut f, self.module, cname.as_ptr()) },
            "cuModuleGetFunction",
        )?;
        if let Some(p) = &self.prof {
            p.borrow_mut().names.insert(f as usize, name.to_string());
        }
        Ok(Kernel(f))
    }

    /// Lets a kernel use `bytes` of dynamic shared memory (above 48 KB).
    /// A TMA descriptor of a bf16 tensor with dimensions `dims` (fastest first)
    /// and byte strides `strides` for dimensions 1 and 2, read in boxes of
    /// `boxes` elements into 128-byte-swizzled shared memory; reads past the
    /// dimensions return zeros. Leaked: it lives as long as the program.
    pub fn tensor_map_3d(
        &self,
        ptr: u64,
        dims: [u64; 3],
        strides: [u64; 2],
        boxes: [u32; 3],
    ) -> Result<&'static TensorMap> {
        let mut m = Box::new(TensorMap(cu::CUtensorMap { opaque: [0; 16] }));
        // SAFETY: the arrays hold rank (or rank - 1) entries; ptr is a live
        // device allocation covering dims and strides.
        check(
            unsafe {
                cu::cuTensorMapEncodeTiled(
                    &mut m.0,
                    cu::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                    3,
                    ptr as *mut c_void,
                    dims.as_ptr(),
                    strides.as_ptr(),
                    boxes.as_ptr(),
                    [1u32; 3].as_ptr(),
                    cu::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                    cu::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
                    cu::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                    cu::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
                )
            },
            "cuTensorMapEncodeTiled",
        )?;
        Ok(Box::leak(m))
    }

    pub fn set_smem(&self, k: Kernel, bytes: usize) -> Result<()> {
        // SAFETY: the function handle belongs to the loaded module.
        check(
            unsafe {
                cu::cuFuncSetAttribute(
                    k.0,
                    cu::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    bytes as i32,
                )
            },
            "cuFuncSetAttribute",
        )
    }

    pub fn alloc(&self, bytes: usize) -> Result<Buf> {
        let mut ptr = 0;
        // SAFETY: out-parameter allocation; ownership moves into Buf.
        check(
            unsafe { cu::cuMemAlloc_v2(&mut ptr, bytes.max(1)) },
            "cuMemAlloc",
        )?;
        Ok(Buf { ptr, bytes })
    }

    pub fn alloc_zeroed(&self, bytes: usize) -> Result<Buf> {
        let b = self.alloc(bytes)?;
        self.memset(b.ptr, bytes)?;
        Ok(b)
    }

    /// A handle that allocates pinned memory for this device from any thread.
    pub fn pinner(&self) -> Pinner {
        Pinner(self.ctx)
    }

    pub fn pinned(&self, bytes: usize) -> Result<Pinned> {
        let mut ptr = std::ptr::null_mut();
        // SAFETY: out-parameter allocation; ownership moves into Pinned.
        check(
            unsafe { cu::cuMemHostAlloc(&mut ptr, bytes.max(8), 0) },
            "cuMemHostAlloc",
        )?;
        Ok(Pinned {
            ptr: ptr.cast(),
            bytes,
        })
    }

    /// Starts loading: uploads go through pinned staging, ordered on the
    /// stream, until [`Gpu::end_uploads`]. Returns false, changing nothing,
    /// when uploads have already begun.
    pub fn begin_uploads(&self) -> Result<bool> {
        if self.staging.borrow().is_some() {
            return Ok(false);
        }
        let mut bufs = Vec::with_capacity(STAGES);
        for _ in 0..STAGES {
            let mut e = std::ptr::null_mut();
            // SAFETY: out-parameter; the event is owned by Staging.
            check(
                unsafe {
                    cu::cuEventCreate(&mut e, cu::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32)
                },
                "cuEventCreate",
            )?;
            bufs.push((self.pinned(STAGE_BYTES)?, e));
        }
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(8));
        *self.staging.borrow_mut() = Some(Staging {
            bufs,
            next: 0,
            threads,
            temp: [None, None],
        });
        Ok(true)
    }

    /// Waits for every staged upload and frees the staging buffers.
    pub fn end_uploads(&self) -> Result<()> {
        self.sync()?;
        *self.staging.borrow_mut() = None;
        Ok(())
    }

    /// Copies `piece` to device address `dst`, ordered on the stream. Needs
    /// [`Gpu::begin_uploads`]; the piece's bytes may be dropped on return.
    pub fn put(&self, dst: u64, piece: &Piece) -> Result<()> {
        let mut guard = self.staging.borrow_mut();
        let st = guard
            .as_mut()
            .ok_or_else(|| Error::Config("put outside begin_uploads".into()))?;
        let mut done = 0;
        while done < piece.len() {
            let n = (piece.len() - done).min(STAGE_BYTES);
            let (buf, ev) = &mut st.bufs[st.next];
            // SAFETY: the event was recorded after this buffer's last copy.
            check(unsafe { cu::cuEventSynchronize(*ev) }, "cuEventSynchronize")?;
            let dst_host = &mut buf.as_mut_slice::<u8>()[..n];
            match piece {
                Piece::File { file, offset, .. } => {
                    read_parallel(file, offset + done as u64, dst_host, st.threads)?
                }
                Piece::Mem(m) => dst_host.copy_from_slice(&m[done..done + n]),
            }
            // SAFETY: the pinned buffer stays untouched until the event fires.
            check(
                unsafe {
                    cu::cuMemcpyHtoDAsync_v2(dst + done as u64, buf.ptr.cast(), n, self.stream)
                },
                "cuMemcpyHtoDAsync",
            )?;
            // SAFETY: the event and stream are owned by self.
            check(
                unsafe { cu::cuEventRecord(*ev, self.stream) },
                "cuEventRecord",
            )?;
            st.next = (st.next + 1) % st.bufs.len();
            done += n;
        }
        Ok(())
    }

    /// A buffer holding the pieces back to back.
    pub fn upload_pieces(&self, pieces: &[Piece]) -> Result<Buf> {
        let b = self.alloc(pieces.iter().map(Piece::len).sum())?;
        let mut at = 0;
        for p in pieces {
            self.put(b.ptr + at as u64, p)?;
            at += p.len();
        }
        Ok(b)
    }

    /// Device scratch of at least `bytes` in one of two slots, valid until the
    /// next call for the same slot; stream-ordered like the uploads.
    pub fn temp(&self, slot: usize, bytes: usize) -> Result<u64> {
        let mut guard = self.staging.borrow_mut();
        let st = guard
            .as_mut()
            .ok_or_else(|| Error::Config("temp outside begin_uploads".into()))?;
        if st.temp[slot].as_ref().is_none_or(|b| b.bytes < bytes) {
            // Pending work may still read the old buffer.
            self.sync()?;
            st.temp[slot] = None;
            st.temp[slot] = Some(self.alloc(bytes)?);
        }
        Ok(st.temp[slot].as_ref().map_or(0, |b| b.ptr))
    }

    pub fn upload(&self, data: &[u8]) -> Result<Buf> {
        if self.staging.borrow().is_some() {
            return self.upload_pieces(&[Piece::Mem(data)]);
        }
        let b = self.alloc(data.len())?;
        // SAFETY: synchronous copy into an allocation of the same size.
        check(
            unsafe { cu::cuMemcpyHtoD_v2(b.ptr, data.as_ptr().cast(), data.len()) },
            "cuMemcpyHtoD",
        )?;
        Ok(b)
    }

    pub fn memset(&self, ptr: u64, bytes: usize) -> Result<()> {
        // SAFETY: the caller passes a live device range.
        check(
            unsafe { cu::cuMemsetD8Async(ptr, 0, bytes, self.stream) },
            "cuMemsetD8Async",
        )
    }

    /// Stream-ordered copy of `bytes` from pinned memory at `offset`; `src`
    /// must stay alive until the stream reaches it (callers synchronize each step).
    pub fn htod(&self, dst: u64, src: &Pinned, offset: usize, bytes: usize) -> Result<()> {
        assert!(offset + bytes <= src.bytes);
        // SAFETY: the range lies inside the pinned buffer, which outlives the copy.
        check(
            unsafe {
                cu::cuMemcpyHtoDAsync_v2(dst, src.ptr.add(offset).cast(), bytes, self.stream)
            },
            "cuMemcpyHtoDAsync",
        )
    }

    pub fn dtoh(&self, dst: &mut Pinned, src: u64, bytes: usize) -> Result<()> {
        assert!(bytes <= dst.bytes);
        // SAFETY: as above.
        check(
            unsafe { cu::cuMemcpyDtoHAsync_v2(dst.ptr.cast(), src, bytes, self.stream) },
            "cuMemcpyDtoHAsync",
        )
    }

    pub fn dtod(&self, dst: u64, src: u64, bytes: usize) -> Result<()> {
        // SAFETY: the caller passes live device ranges.
        check(
            unsafe { cu::cuMemcpyDtoDAsync_v2(dst, src, bytes, self.stream) },
            "cuMemcpyDtoDAsync",
        )
    }

    pub fn sync(&self) -> Result<()> {
        // SAFETY: the stream is owned by self.
        check(
            unsafe { cu::cuStreamSynchronize(self.stream) },
            "cuStreamSynchronize",
        )?;
        if let Some(p) = &self.prof {
            let mut p = p.borrow_mut();
            let pending = std::mem::take(&mut p.pending);
            for (name, a, b) in pending {
                let mut ms = 0f32;
                // SAFETY: both events were recorded before the sync above.
                unsafe { cu::cuEventElapsedTime_v2(&mut ms, a, b) };
                let e = p.totals.entry(name).or_default();
                e.0 += 1;
                e.1 += ms as f64;
                p.spare.push(a);
                p.spare.push(b);
            }
        }
        Ok(())
    }

    pub fn mem_info(&self) -> Result<(usize, usize)> {
        let (mut free, mut total) = (0, 0);
        // SAFETY: out-parameters.
        check(
            unsafe { cu::cuMemGetInfo_v2(&mut free, &mut total) },
            "cuMemGetInfo",
        )?;
        Ok((free, total))
    }

    pub fn launch(
        &self,
        k: Kernel,
        grid: (u32, u32, u32),
        block: u32,
        smem: u32,
        args: &[Arg],
    ) -> Result<()> {
        self.launch_in(self.stream, k, grid, block, smem, args)
    }

    /// Lets the side stream start work ordered after everything issued on
    /// the main stream so far.
    pub fn fork(&self) -> Result<()> {
        // On AMD the side stream's work runs on the main stream (see launch_side).
        if cfg!(feature = "rocm") {
            return Ok(());
        }
        // SAFETY: the event and streams are owned by self.
        unsafe {
            check(
                cu::cuEventRecord(self.fork_ev, self.stream),
                "cuEventRecord",
            )?;
            check(
                cu::cuStreamWaitEvent(self.side, self.fork_ev, 0),
                "cuStreamWaitEvent",
            )
        }
    }

    /// Orders later main-stream work after everything issued on the side stream.
    pub fn join(&self) -> Result<()> {
        if cfg!(feature = "rocm") {
            return Ok(());
        }
        // SAFETY: the event and streams are owned by self.
        unsafe {
            check(cu::cuEventRecord(self.join_ev, self.side), "cuEventRecord")?;
            check(
                cu::cuStreamWaitEvent(self.stream, self.join_ev, 0),
                "cuStreamWaitEvent",
            )
        }
    }

    /// `launch` on the side stream (between `fork` and `join`).
    pub fn launch_side(
        &self,
        k: Kernel,
        grid: (u32, u32, u32),
        block: u32,
        smem: u32,
        args: &[Arg],
    ) -> Result<()> {
        // On an MI325X (ROCm 7.14), LoRA's shrink overlapped with the layer's
        // matrix multiply through these events changed outputs (a zero
        // adapter moved 144 of 220 picks); in stream order it matches the
        // base model exactly, so AMD gives up the overlap.
        let stream = if cfg!(feature = "rocm") {
            self.stream
        } else {
            self.side
        };
        self.launch_in(stream, k, grid, block, smem, args)
    }

    fn launch_in(
        &self,
        stream: cu::CUstream,
        k: Kernel,
        grid: (u32, u32, u32),
        block: u32,
        smem: u32,
        args: &[Arg],
    ) -> Result<()> {
        let mut vals = [0u64; 32];
        let mut ptrs = [std::ptr::null_mut::<c_void>(); 32];
        for (i, a) in args.iter().enumerate() {
            // Little-endian: a 4-byte value sits at the start of its 8-byte slot.
            vals[i] = match *a {
                Arg::P(p) => p,
                Arg::I(v) => v as u32 as u64,
                Arg::F(v) => v.to_bits() as u64,
                Arg::Map(_) => 0,
            };
        }
        for (i, a) in args.iter().enumerate() {
            ptrs[i] = match *a {
                Arg::Map(m) => (&m.0 as *const cu::CUtensorMap).cast_mut().cast(),
                _ => (&mut vals[i] as *mut u64).cast(),
            };
        }
        let start = self.timed_start(stream);
        // SAFETY: argument count and types match the kernel signatures in
        // qwen35.cu; the driver copies the values at launch.
        let r = check(
            unsafe {
                cu::cuLaunchKernel(
                    k.0,
                    grid.0,
                    grid.1,
                    grid.2,
                    block,
                    1,
                    1,
                    smem,
                    stream,
                    ptrs.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        );
        if start.is_some() {
            let name = self
                .prof
                .as_ref()
                .and_then(|p| p.borrow().names.get(&(k.0 as usize)).cloned());
            self.timed_end(stream, start, name.as_deref().unwrap_or("?"));
        }
        r
    }

    /// y[rows, out] = x[rows, inp] . w[out, inp]^T, bf16 inputs, f32
    /// accumulation, bf16 or f32 output.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self,
        x: u64,
        w: u64,
        y: u64,
        rows: usize,
        inp: usize,
        out: usize,
        f32_out: bool,
    ) -> Result<()> {
        self.gemm_ld(x, w, y, rows, inp, out, out, f32_out)
    }

    /// As [`Gpu::gemm`], with output rows `ldy` elements apart.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_ld(
        &self,
        x: u64,
        w: u64,
        y: u64,
        rows: usize,
        inp: usize,
        out: usize,
        ldy: usize,
        f32_out: bool,
    ) -> Result<()> {
        let start = self.timed_start(self.stream);
        let r = self.gemm_ld_inner(x, w, y, rows, inp, out, ldy, f32_out);
        if start.is_some() && self.prof.as_ref().is_some_and(|p| p.borrow().shapes) {
            let name = format!(
                "cublas gemm {out}x{inp}, {}+ rows{}",
                1 << rows.ilog2(),
                if f32_out { ", f32" } else { "" }
            );
            self.timed_end(self.stream, start, &name);
        } else {
            self.timed_end(self.stream, start, "cublas gemm");
        }
        r
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_ld_inner(
        &self,
        x: u64,
        w: u64,
        y: u64,
        rows: usize,
        inp: usize,
        out: usize,
        ldy: usize,
        f32_out: bool,
    ) -> Result<()> {
        // On AMD, decoding's 1 to 4 rows stream the weights in skinny_bf16_r*.
        #[cfg(feature = "rocm")]
        if (1..=4).contains(&rows) && inp.is_multiple_of(8) {
            // Four waves of two rows each per block.
            return self.launch(
                self.skinny[rows - 1],
                (out.div_ceil(8) as u32, 1, 1),
                256,
                0,
                &[
                    Arg::P(w),
                    Arg::P(x),
                    Arg::P(y),
                    Arg::I(inp as i32),
                    Arg::I(out as i32),
                    Arg::I(ldy as i32),
                    Arg::I(f32_out as i32),
                ],
            );
        }
        self.note(None, out, inp, ldy, rows, f32_out, (w, 0), x, 0, y);
        #[cfg(not(feature = "rocm"))]
        if let Some(t) = &self.tuning {
            let shape = GemmShape {
                kind: gemm_kind(None, f32_out),
                n: out,
                k: inp,
                ldy,
            };
            if let Some(a) = t.gemm_algo(&shape.kind, out, inp, ldy, rows)
                && self.lt_run(&shape, rows, (w, 0), x, 0, y, &a)?
            {
                return Ok(());
            }
        }
        let (alpha, beta) = (1f32, 0f32);
        let c_type = if f32_out {
            cb::cudaDataType_t::CUDA_R_32F
        } else {
            cb::cudaDataType_t::CUDA_R_16BF
        };
        // Column-major view: y^T (out x rows) = W (out x inp) . x^T (inp x rows).
        // SAFETY: the caller passes device buffers of the stated shapes.
        check_blas(
            unsafe {
                cb::cublasGemmEx(
                    self.blas,
                    cb::cublasOperation_t::CUBLAS_OP_T,
                    cb::cublasOperation_t::CUBLAS_OP_N,
                    out as i32,
                    rows as i32,
                    inp as i32,
                    (&alpha as *const f32).cast(),
                    w as *const c_void,
                    cb::cudaDataType_t::CUDA_R_16BF,
                    inp as i32,
                    x as *const c_void,
                    cb::cudaDataType_t::CUDA_R_16BF,
                    inp as i32,
                    (&beta as *const f32).cast(),
                    y as *mut c_void,
                    c_type,
                    ldy as i32,
                    cb::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cb::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
            },
            "cublasGemmEx",
        )
    }

    /// Column-major strided batched GEMM, C = op(A) . op(B), with f32
    /// accumulation: C is m x n, op(A) m x k, op(B) k x n.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_batched(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: Mat,
        b: Mat,
        c: Mat,
        batch: usize,
    ) -> Result<()> {
        let start = self.timed_start(self.stream);
        let r = self.gemm_batched_inner(m, n, k, a, b, c, batch);
        self.timed_end(self.stream, start, "cublas gemm batched");
        r
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_batched_inner(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: Mat,
        b: Mat,
        c: Mat,
        batch: usize,
    ) -> Result<()> {
        let (alpha, beta) = (1f32, 0f32);
        let op = |t: bool| {
            if t {
                cb::cublasOperation_t::CUBLAS_OP_T
            } else {
                cb::cublasOperation_t::CUBLAS_OP_N
            }
        };
        let ty = |f: bool| {
            if f {
                cb::cudaDataType_t::CUDA_R_32F
            } else {
                cb::cudaDataType_t::CUDA_R_16BF
            }
        };
        // SAFETY: the caller passes device matrices of the stated shapes.
        check_blas(
            unsafe {
                cb::cublasGemmStridedBatchedEx(
                    self.blas,
                    op(a.trans),
                    op(b.trans),
                    m as i32,
                    n as i32,
                    k as i32,
                    (&alpha as *const f32).cast(),
                    a.ptr as *const c_void,
                    ty(a.f32),
                    a.ld as i32,
                    a.stride as i64,
                    b.ptr as *const c_void,
                    ty(b.f32),
                    b.ld as i32,
                    b.stride as i64,
                    (&beta as *const f32).cast(),
                    c.ptr as *mut c_void,
                    ty(c.f32),
                    c.ld as i32,
                    c.stride as i64,
                    batch as i32,
                    cb::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cb::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
            },
            "cublasGemmStridedBatchedEx",
        )
    }

    /// The FP8 layout this device's cuBLASLt runs: block scales on Hopper,
    /// MXFP8 on Blackwell, none before Hopper.
    pub fn fp8_mode(&self) -> Option<Fp8Mode> {
        match self.arch {
            90 => Some(Fp8Mode::Block128),
            100.. => Some(Fp8Mode::Mx32),
            _ => None,
        }
    }

    /// y[rows, ldy] (from column 0) = x[rows, k] . w[n, k]^T with E4M3 inputs
    /// scaled as `mode` describes, f32 accumulation, bf16 or f32 output.
    /// Plans (descriptor, layouts, algorithm) are cached per shape and
    /// operands, so replays and graph captures issue only the matmul.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn fp8_matmul(
        &self,
        mode: Fp8Mode,
        w: u64,
        w_scale: u64,
        x: u64,
        x_scale: u64,
        y: u64,
        n: usize,
        k: usize,
        rows: usize,
        ldy: usize,
        f32_out: bool,
    ) -> Result<()> {
        let start = self.timed_start(self.stream);
        let r = self.fp8_matmul_inner(mode, w, w_scale, x, x_scale, y, n, k, rows, ldy, f32_out);
        self.timed_end(self.stream, start, "cublaslt fp8");
        r
    }

    #[allow(clippy::too_many_arguments)]
    fn fp8_matmul_inner(
        &self,
        mode: Fp8Mode,
        w: u64,
        w_scale: u64,
        x: u64,
        x_scale: u64,
        y: u64,
        n: usize,
        k: usize,
        rows: usize,
        ldy: usize,
        f32_out: bool,
    ) -> Result<()> {
        self.note(
            Some(mode),
            n,
            k,
            ldy,
            rows,
            f32_out,
            (w, w_scale),
            x,
            x_scale,
            y,
        );
        let key = (w, x_scale, n, k, rows, ldy, f32_out);
        let mut plans = self.lt_plans.borrow_mut();
        let p = match plans.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let p = self.fp8_plan(mode, w_scale, x_scale, n, k, rows, ldy, f32_out)?;
                #[cfg(not(feature = "rocm"))]
                let p =
                    {
                        let mut p = p;
                        if let Some(a) = self.tuning.as_ref().and_then(|t| {
                            t.gemm_algo(&gemm_kind(Some(mode), f32_out), n, k, ldy, rows)
                        }) && self.fits(&mut p, &a)
                        {
                            p.algo = lt::cublasLtMatmulAlgo_t { data: a };
                        }
                        p
                    };
                e.insert(p)
            }
        };
        let (alpha, beta) = (1f32, 0f32);
        // SAFETY: the plan's operands match the buffers passed for its key.
        let r = unsafe {
            lt::cublasLtMatmul(
                self.lt,
                p.desc,
                (&alpha as *const f32).cast(),
                w as *const c_void,
                p.a,
                x as *const c_void,
                p.b,
                (&beta as *const f32).cast(),
                y as *const c_void,
                p.d,
                y as *mut c_void,
                p.d,
                &p.algo,
                self.lt_ws.ptr as *mut c_void,
                self.lt_ws.bytes,
                self.stream.cast(),
            )
        };
        if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(Error::Cuda("cublasLtMatmul", format!("{r:?}")));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn fp8_plan(
        &self,
        mode: Fp8Mode,
        w_scale: u64,
        x_scale: u64,
        n: usize,
        k: usize,
        rows: usize,
        ldy: usize,
        f32_out: bool,
    ) -> Result<LtPlan> {
        use lt::cublasLtMatmulDescAttributes_t as A;
        let ok = |r: lt::cublasStatus_t, what: &'static str| -> Result<()> {
            if r == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                Ok(())
            } else {
                Err(Error::Cuda(what, format!("{r:?}")))
            }
        };
        // cuBLASLt scale modes: SCALAR_32F = 0, VEC32_UE8M0 = 2, VEC128_32F = 4, BLK128x128_32F = 5.
        let (a_mode, b_mode): (u32, u32) = match mode {
            Fp8Mode::Block128 => (5, 4),
            Fp8Mode::Mx32 => (2, 2),
        };
        // SAFETY: descriptor and layout handles are created here and owned by the plan.
        unsafe {
            let mut desc = std::ptr::null_mut();
            ok(
                lt::cublasLtMatmulDescCreate(
                    &mut desc,
                    lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    lt::cudaDataType_t::CUDA_R_32F,
                ),
                "cublasLtMatmulDescCreate",
            )?;
            let set = |attr: A, v: *const c_void, size: usize| -> Result<()> {
                ok(
                    lt::cublasLtMatmulDescSetAttribute(desc, attr, v, size),
                    "cublasLtMatmulDescSetAttribute",
                )
            };
            // cublasOperation_t: N = 0, T = 1.
            let (ta, tb) = (1i32, 0i32);
            set(
                A::CUBLASLT_MATMUL_DESC_TRANSA,
                (&ta as *const i32).cast(),
                4,
            )?;
            set(
                A::CUBLASLT_MATMUL_DESC_TRANSB,
                (&tb as *const i32).cast(),
                4,
            )?;
            set(
                A::CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                (&a_mode as *const u32).cast(),
                4,
            )?;
            set(
                A::CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                (&b_mode as *const u32).cast(),
                4,
            )?;
            set(
                A::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                (&w_scale as *const u64).cast(),
                8,
            )?;
            set(
                A::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                (&x_scale as *const u64).cast(),
                8,
            )?;
            let layout = |ty: lt::cudaDataType_t,
                          r: usize,
                          c: usize,
                          ld: usize|
             -> Result<lt::cublasLtMatrixLayout_t> {
                let mut l = std::ptr::null_mut();
                ok(
                    lt::cublasLtMatrixLayoutCreate(&mut l, ty, r as u64, c as u64, ld as i64),
                    "cublasLtMatrixLayoutCreate",
                )?;
                Ok(l)
            };
            // Column-major views: W^T is k x n, x^T is k x rows, y^T is n x rows.
            let a = layout(lt::cudaDataType_t::CUDA_R_8F_E4M3, k, n, k)?;
            let b = layout(lt::cudaDataType_t::CUDA_R_8F_E4M3, k, rows, k)?;
            let d_ty = if f32_out {
                lt::cudaDataType_t::CUDA_R_32F
            } else {
                lt::cudaDataType_t::CUDA_R_16BF
            };
            let d = layout(d_ty, n, rows, ldy)?;
            let mut pref = std::ptr::null_mut();
            ok(
                lt::cublasLtMatmulPreferenceCreate(&mut pref),
                "cublasLtMatmulPreferenceCreate",
            )?;
            let ws = self.lt_ws.bytes;
            ok(
                lt::cublasLtMatmulPreferenceSetAttribute(
                    pref,
                    lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    (&ws as *const usize).cast(),
                    8,
                ),
                "cublasLtMatmulPreferenceSetAttribute",
            )?;
            let mut res: lt::cublasLtMatmulHeuristicResult_t = std::mem::zeroed();
            let mut found = 0;
            ok(
                lt::cublasLtMatmulAlgoGetHeuristic(
                    self.lt, desc, a, b, d, d, pref, 1, &mut res, &mut found,
                ),
                "cublasLtMatmulAlgoGetHeuristic",
            )?;
            lt::cublasLtMatmulPreferenceDestroy(pref);
            if found == 0 {
                return Err(Error::Cuda(
                    "cublasLtMatmulAlgoGetHeuristic",
                    format!("no FP8 algorithm for {mode:?} {n}x{k}, {rows} rows"),
                ));
            }
            Ok(LtPlan {
                desc,
                a,
                b,
                d,
                algo: res.algo,
                fits: Vec::new(),
            })
        }
    }

    /// Records a product for `pocket-engine tune` (POCKET_RECORD_GEMMS=1).
    #[allow(clippy::too_many_arguments)]
    fn note(
        &self,
        fp8: Option<Fp8Mode>,
        n: usize,
        k: usize,
        ldy: usize,
        rows: usize,
        f32_out: bool,
        w: (u64, u64),
        x: u64,
        x_scale: u64,
        y: u64,
    ) {
        let Some(r) = &self.record else { return };
        let mut r = r.borrow_mut();
        let u = r
            .entry(GemmShape {
                kind: gemm_kind(fp8, f32_out),
                n,
                k,
                ldy,
            })
            .or_default();
        if !u.weights.contains(&w) && u.weights.len() < 64 {
            u.weights.push(w);
        }
        if u.rows.last().is_none_or(|&m| rows >= m) {
            (u.x, u.x_scale, u.y) = (x, x_scale, y);
        }
        u.fp8 = fp8;
        u.f32_out = f32_out;
        u.rows.insert(rows);
    }

    /// The products recorded so far (POCKET_RECORD_GEMMS=1).
    pub fn recorded(&self) -> BTreeMap<GemmShape, GemmUse> {
        self.record
            .as_ref()
            .map(|r| r.borrow().clone())
            .unwrap_or_default()
    }

    /// Median time in microseconds of one call of `f(i)`, from `rounds`
    /// rounds of `reps` calls timed by events on the main stream.
    pub fn time_us(
        &self,
        reps: usize,
        rounds: usize,
        mut f: impl FnMut(usize) -> Result<()>,
    ) -> Result<f64> {
        let (mut a, mut b) = (std::ptr::null_mut(), std::ptr::null_mut());
        // SAFETY: events are created, used on this context's stream and destroyed here.
        unsafe {
            check(cu::cuEventCreate(&mut a, 0), "cuEventCreate")?;
            check(cu::cuEventCreate(&mut b, 0), "cuEventCreate")?;
        }
        let mut times = Vec::with_capacity(rounds);
        let mut run = || -> Result<()> {
            for _ in 0..rounds.max(1) {
                // SAFETY: as above.
                unsafe { check(cu::cuEventRecord(a, self.stream), "cuEventRecord")? };
                for i in 0..reps.max(1) {
                    f(i)?;
                }
                let mut ms = 0f32;
                // SAFETY: as above.
                unsafe {
                    check(cu::cuEventRecord(b, self.stream), "cuEventRecord")?;
                    check(cu::cuEventSynchronize(b), "cuEventSynchronize")?;
                    check(
                        cu::cuEventElapsedTime_v2(&mut ms, a, b),
                        "cuEventElapsedTime",
                    )?;
                }
                times.push(ms as f64 * 1e3 / reps.max(1) as f64);
            }
            Ok(())
        };
        let r = run();
        // SAFETY: as above.
        unsafe {
            cu::cuEventDestroy_v2(a);
            cu::cuEventDestroy_v2(b);
        }
        r?;
        times.sort_by(f64::total_cmp);
        Ok(times[times.len() / 2])
    }

    /// A cuBLASLt plan for a bf16 product: y^T (n x rows) = W^T' . x^T, as
    /// `gemm_ld` computes it.
    #[cfg(not(feature = "rocm"))]
    fn bf16_plan(
        &self,
        n: usize,
        k: usize,
        rows: usize,
        ldy: usize,
        f32_out: bool,
    ) -> Result<LtPlan> {
        use lt::cublasLtMatmulDescAttributes_t as A;
        let ok = |r: lt::cublasStatus_t, what: &'static str| -> Result<()> {
            if r == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                Ok(())
            } else {
                Err(Error::Cuda(what, format!("{r:?}")))
            }
        };
        // SAFETY: descriptor and layout handles are created here and owned by the plan.
        unsafe {
            let mut desc = std::ptr::null_mut();
            ok(
                lt::cublasLtMatmulDescCreate(
                    &mut desc,
                    lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    lt::cudaDataType_t::CUDA_R_32F,
                ),
                "cublasLtMatmulDescCreate",
            )?;
            let (ta, tb) = (1i32, 0i32);
            ok(
                lt::cublasLtMatmulDescSetAttribute(
                    desc,
                    A::CUBLASLT_MATMUL_DESC_TRANSA,
                    (&ta as *const i32).cast(),
                    4,
                ),
                "cublasLtMatmulDescSetAttribute",
            )?;
            ok(
                lt::cublasLtMatmulDescSetAttribute(
                    desc,
                    A::CUBLASLT_MATMUL_DESC_TRANSB,
                    (&tb as *const i32).cast(),
                    4,
                ),
                "cublasLtMatmulDescSetAttribute",
            )?;
            let layout = |ty: lt::cudaDataType_t, r: usize, c: usize, ld: usize| {
                let mut l = std::ptr::null_mut();
                ok(
                    lt::cublasLtMatrixLayoutCreate(&mut l, ty, r as u64, c as u64, ld as i64),
                    "cublasLtMatrixLayoutCreate",
                )
                .map(|_| l)
            };
            let bf = lt::cudaDataType_t::CUDA_R_16BF;
            let d_ty = if f32_out {
                lt::cudaDataType_t::CUDA_R_32F
            } else {
                bf
            };
            Ok(LtPlan {
                desc,
                a: layout(bf, k, n, k)?,
                b: layout(bf, k, rows, k)?,
                d: layout(d_ty, n, rows, ldy)?,
                algo: std::mem::zeroed(),
                fits: Vec::new(),
            })
        }
    }

    /// The plan for a product of a tuned shape at `rows` rows, made once.
    #[cfg(not(feature = "rocm"))]
    fn tuned_plan<'a>(
        &self,
        plans: &'a mut HashMap<(GemmShape, usize, u64, u64), LtPlan>,
        shape: &GemmShape,
        rows: usize,
        w_scale: u64,
        x_scale: u64,
    ) -> Result<&'a mut LtPlan> {
        let key = (shape.clone(), rows, w_scale, x_scale);
        if !plans.contains_key(&key) {
            let f32_out = shape.kind.ends_with(" f32");
            let p = if shape.kind.starts_with("fp8 block128") {
                self.fp8_plan(
                    Fp8Mode::Block128,
                    w_scale,
                    x_scale,
                    shape.n,
                    shape.k,
                    rows,
                    shape.ldy,
                    f32_out,
                )?
            } else if shape.kind.starts_with("fp8 mx32") {
                self.fp8_plan(
                    Fp8Mode::Mx32,
                    w_scale,
                    x_scale,
                    shape.n,
                    shape.k,
                    rows,
                    shape.ldy,
                    f32_out,
                )?
            } else {
                self.bf16_plan(shape.n, shape.k, rows, shape.ldy, f32_out)?
            };
            plans.insert(key.clone(), p);
        }
        Ok(plans.get_mut(&key).expect("inserted"))
    }

    /// Whether cuBLASLt runs `a` for this plan within the workspace.
    #[cfg(not(feature = "rocm"))]
    fn fits(&self, p: &mut LtPlan, a: &crate::tuning::Algo) -> bool {
        if let Some(&(_, ok)) = p.fits.iter().find(|f| f.0 == *a) {
            return ok;
        }
        let algo = lt::cublasLtMatmulAlgo_t { data: *a };
        // SAFETY: the plan's handles are live; the result is written in place.
        let ok = unsafe {
            let mut res: lt::cublasLtMatmulHeuristicResult_t = std::mem::zeroed();
            let r =
                lt::cublasLtMatmulAlgoCheck(self.lt, p.desc, p.a, p.b, p.d, p.d, &algo, &mut res);
            r == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                && res.state == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                && res.workspaceSize <= self.lt_ws.bytes
        };
        p.fits.push((*a, ok));
        ok
    }

    /// Runs a product with cuBLASLt algorithm `a`. Returns false, running
    /// nothing, when `a` cannot run it.
    #[cfg(not(feature = "rocm"))]
    #[allow(clippy::too_many_arguments)]
    pub fn lt_run(
        &self,
        shape: &GemmShape,
        rows: usize,
        (w, w_scale): (u64, u64),
        x: u64,
        x_scale: u64,
        y: u64,
        a: &crate::tuning::Algo,
    ) -> Result<bool> {
        let mut plans = self.lt_tuned.borrow_mut();
        let p = self.tuned_plan(&mut plans, shape, rows, w_scale, x_scale)?;
        if !self.fits(p, a) {
            return Ok(false);
        }
        let (alpha, beta) = (1f32, 0f32);
        let algo = lt::cublasLtMatmulAlgo_t { data: *a };
        // SAFETY: the plan matches the shape; the caller passes buffers of it.
        let r = unsafe {
            lt::cublasLtMatmul(
                self.lt,
                p.desc,
                (&alpha as *const f32).cast(),
                w as *const c_void,
                p.a,
                x as *const c_void,
                p.b,
                (&beta as *const f32).cast(),
                y as *const c_void,
                p.d,
                y as *mut c_void,
                p.d,
                &algo,
                self.lt_ws.ptr as *mut c_void,
                self.lt_ws.bytes,
                self.stream.cast(),
            )
        };
        if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(Error::Cuda("cublasLtMatmul", format!("{r:?}")));
        }
        Ok(true)
    }

    /// cuBLASLt's own candidates for a product at `rows` rows, best guess
    /// first: up to `count` that fit the workspace.
    #[cfg(not(feature = "rocm"))]
    pub fn lt_algos(
        &self,
        shape: &GemmShape,
        rows: usize,
        w_scale: u64,
        x_scale: u64,
        count: usize,
    ) -> Result<Vec<crate::tuning::Algo>> {
        let mut plans = self.lt_tuned.borrow_mut();
        let p = self.tuned_plan(&mut plans, shape, rows, w_scale, x_scale)?;
        // SAFETY: the preference is created and destroyed here; results are written in place.
        unsafe {
            let mut pref = std::ptr::null_mut();
            lt::cublasLtMatmulPreferenceCreate(&mut pref);
            let ws = self.lt_ws.bytes;
            lt::cublasLtMatmulPreferenceSetAttribute(
                pref,
                lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&ws as *const usize).cast(),
                8,
            );
            let mut res: Vec<lt::cublasLtMatmulHeuristicResult_t> = vec![std::mem::zeroed(); count];
            let mut found = 0;
            let r = lt::cublasLtMatmulAlgoGetHeuristic(
                self.lt,
                p.desc,
                p.a,
                p.b,
                p.d,
                p.d,
                pref,
                count as i32,
                res.as_mut_ptr(),
                &mut found,
            );
            lt::cublasLtMatmulPreferenceDestroy(pref);
            if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Ok(Vec::new());
            }
            Ok(res[..found as usize]
                .iter()
                .filter(|r| r.state == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS)
                .map(|r| r.algo.data)
                .collect())
        }
    }

    /// Starts timing `name` on `stream` when profiling; pass the result to `timed_end`.
    fn timed_start(&self, stream: cu::CUstream) -> Option<cu::CUevent> {
        let p = self.prof.as_ref()?;
        let mut p = p.borrow_mut();
        if p.capturing {
            return None;
        }
        let ev = p.spare.pop().unwrap_or_else(|| {
            let mut e = std::ptr::null_mut();
            // SAFETY: plain driver call with an out-parameter.
            unsafe { cu::cuEventCreate(&mut e, 0) };
            e
        });
        // SAFETY: the event and stream belong to this context.
        unsafe { cu::cuEventRecord(ev, stream) };
        Some(ev)
    }

    fn timed_end(&self, stream: cu::CUstream, start: Option<cu::CUevent>, name: &str) {
        let (Some(start), Some(p)) = (start, &self.prof) else {
            return;
        };
        let mut p = p.borrow_mut();
        let end = p.spare.pop().unwrap_or_else(|| {
            let mut e = std::ptr::null_mut();
            // SAFETY: plain driver call with an out-parameter.
            unsafe { cu::cuEventCreate(&mut e, 0) };
            e
        });
        // SAFETY: the event and stream belong to this context.
        unsafe { cu::cuEventRecord(end, stream) };
        p.pending.push((name.to_string(), start, end));
    }

    /// GPU time by kernel since the last report, most first, when profiling.
    pub fn profile_report(&self) -> Option<String> {
        let p = self.prof.as_ref()?;
        let mut p = p.borrow_mut();
        let total: f64 = p.totals.values().map(|v| v.1).sum();
        let mut v: Vec<_> = p.totals.drain().collect();
        v.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
        let mut out = format!("GPU time {total:.1} ms:");
        let rows = if p.shapes { 40 } else { 16 };
        for (name, (n, ms)) in v.iter().take(rows) {
            out += &format!(
                "\n  {:5.1}% {ms:9.2} ms {n:7}x  {name}",
                100.0 * ms / total.max(1e-9)
            );
        }
        Some(out)
    }

    pub fn begin_capture(&self) -> Result<()> {
        if let Some(p) = &self.prof {
            p.borrow_mut().capturing = true;
        }
        // SAFETY: the stream is owned by self.
        check(
            unsafe {
                cu::cuStreamBeginCapture_v2(
                    self.stream,
                    cu::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )
            },
            "cuStreamBeginCapture",
        )
    }

    pub fn end_capture(&self) -> Result<Graph> {
        if let Some(p) = &self.prof {
            p.borrow_mut().capturing = false;
        }
        let mut graph = std::ptr::null_mut();
        let mut exec = std::ptr::null_mut();
        // SAFETY: ends the capture begun on this stream; handles move into Graph.
        unsafe {
            check(
                cu::cuStreamEndCapture(self.stream, &mut graph),
                "cuStreamEndCapture",
            )?;
            check(
                cu::cuGraphInstantiateWithFlags(&mut exec, graph, 0),
                "cuGraphInstantiate",
            )?;
        }
        Ok(Graph { graph, exec })
    }

    pub fn replay(&self, g: &Graph) -> Result<()> {
        // SAFETY: the graph was instantiated in this context.
        check(
            unsafe { cu::cuGraphLaunch(g.exec, self.stream) },
            "cuGraphLaunch",
        )
    }
}
