//! The raw CUDA boundary: one context and stream, device buffers, kernel
//! launches, cuBLAS GEMMs and CUDA graphs. Everything here is driven from the
//! engine thread that created it.
#![allow(unsafe_code)]

use crate::{Error, Result};
use cudarc::cublas::sys as cb;
use cudarc::driver::sys as cu;
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

impl Drop for Pinned {
    fn drop(&mut self) {
        // SAFETY: the pointer came from cuMemHostAlloc and is freed once.
        unsafe { cu::cuMemFreeHost(self.ptr.cast()) };
    }
}

#[derive(Clone, Copy)]
pub struct Kernel(cu::CUfunction);

/// A kernel argument, passed by value.
#[derive(Clone, Copy)]
pub enum Arg {
    P(u64),
    I(i32),
    F(f32),
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

pub struct Gpu {
    stream: cu::CUstream,
    blas: cb::cublasHandle_t,
    module: cu::CUmodule,
    _workspace: Buf,
    pub name: String,
}

const KERNEL_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/qwen35.fatbin"));

impl Gpu {
    pub fn new(ordinal: i32) -> Result<Self> {
        if KERNEL_IMAGE.is_empty() {
            return Err(Error::Config(
                "built without the `cuda` feature: no kernels".into(),
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
            let mut stream = std::ptr::null_mut();
            check(
                cu::cuStreamCreate(
                    &mut stream,
                    cu::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
                ),
                "cuStreamCreate",
            )?;
            let mut module = std::ptr::null_mut();
            let image = {
                // The driver wants an aligned image.
                let mut v = vec![0u64; KERNEL_IMAGE.len().div_ceil(8)];
                std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), KERNEL_IMAGE.len())
                    .copy_from_slice(KERNEL_IMAGE);
                v
            };
            check(
                cu::cuModuleLoadData(&mut module, image.as_ptr().cast()),
                "cuModuleLoadData",
            )?;
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
            Ok(Self {
                stream,
                blas,
                module,
                _workspace: workspace,
                name,
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
        Ok(Kernel(f))
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

    pub fn upload(&self, data: &[u8]) -> Result<Buf> {
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
        )
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
        let mut vals = [0u64; 24];
        let mut ptrs = [std::ptr::null_mut::<c_void>(); 24];
        for (i, a) in args.iter().enumerate() {
            // Little-endian: a 4-byte value sits at the start of its 8-byte slot.
            vals[i] = match *a {
                Arg::P(p) => p,
                Arg::I(v) => v as u32 as u64,
                Arg::F(v) => v.to_bits() as u64,
            };
        }
        for i in 0..args.len() {
            ptrs[i] = (&mut vals[i] as *mut u64).cast();
        }
        // SAFETY: argument count and types match the kernel signatures in
        // qwen35.cu; the driver copies the values at launch.
        check(
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
                    self.stream,
                    ptrs.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
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
                    out as i32,
                    cb::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cb::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
            },
            "cublasGemmEx",
        )
    }

    pub fn begin_capture(&self) -> Result<()> {
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
