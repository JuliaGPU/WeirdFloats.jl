# WeirdFloats.jl

This package is for experimenting with interesting numerical formats available on some GPUs.

## Installation

To install this package:

```julia
import Pkg
Pkg.add(; url="https://github.com/JuliaGPU/WeirdFloats.jl")
```

This requires an AMD MI300 GPU or later generation and Julia v1.12 or later (to have LLVM 18 or later, to support that generation of GPU):

```julia-repl
julia> versioninfo()
Julia Version 1.12.0-beta4
Commit 600ac61d3d2 (2025-06-05 07:03 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 384 × AMD EPYC 9654 96-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, znver4)
  GC: Built with stock GC
Threads: 1 default, 1 interactive, 1 GC (on 384 virtual cores)

julia> AMDGPU.versioninfo()
[ Info: AMDGPU versioninfo
┌───────────┬──────────────────┬───────────┬────────────────────────────────────────────────────────────────────────────────────────┐
│ Available │ Name             │ Version   │ Path                                                                                   │
├───────────┼──────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│     +     │ LLD              │ -         │ /opt/rocm/llvm/bin/ld.lld                                                              │
│     +     │ Device Libraries │ -         │ /home/cceamgi/.julia/artifacts/b46ab46ef568406312e5f500efb677511199c2f9/amdgcn/bitcode │
│     +     │ HIP              │ 6.4.43482 │ /opt/rocm/lib/libamdhip64.so                                                           │
│     +     │ rocBLAS          │ 4.4.0     │ /opt/rocm/lib/librocblas.so                                                            │
│     +     │ rocSOLVER        │ 3.28.0    │ /opt/rocm/lib/librocsolver.so                                                          │
│     +     │ rocSPARSE        │ 3.4.0     │ /opt/rocm/lib/librocsparse.so                                                          │
│     +     │ rocRAND          │ 2.10.5    │ /opt/rocm/lib/librocrand.so                                                            │
│     +     │ rocFFT           │ 1.0.32    │ /opt/rocm/lib/librocfft.so                                                             │
│     +     │ MIOpen           │ 3.4.0     │ /opt/rocm/lib/libMIOpen.so                                                             │
└───────────┴──────────────────┴───────────┴────────────────────────────────────────────────────────────────────────────────────────┘

[ Info: AMDGPU devices
┌────┬─────────────────────┬────────────────────────┬───────────┬─────────────┬───────────────┐
│ Id │                Name │               GCN arch │ Wavefront │      Memory │ Shared Memory │
├────┼─────────────────────┼────────────────────────┼───────────┼─────────────┼───────────────┤
│  1 │ AMD Instinct MI300X │ gfx942:sramecc+:xnack- │        64 │ 191.984 GiB │    64.000 KiB │
│  2 │ AMD Instinct MI300X │ gfx942:sramecc+:xnack- │        64 │ 191.984 GiB │    64.000 KiB │
│  3 │ AMD Instinct MI300X │ gfx942:sramecc+:xnack- │        64 │ 191.984 GiB │    64.000 KiB │
│  4 │ AMD Instinct MI300X │ gfx942:sramecc+:xnack- │        64 │ 191.984 GiB │    64.000 KiB │
│  5 │ AMD Instinct MI300X │ gfx942:sramecc+:xnack- │        64 │ 191.984 GiB │    64.000 KiB │
│  6 │ AMD Instinct MI300X │ gfx942:sramecc+:xnack- │        64 │ 191.984 GiB │    64.000 KiB │
│  7 │ AMD Instinct MI300X │ gfx942:sramecc+:xnack- │        64 │ 191.984 GiB │    64.000 KiB │
│  8 │ AMD Instinct MI300X │ gfx942:sramecc+:xnack- │        64 │ 191.984 GiB │    64.000 KiB │
└────┴─────────────────────┴────────────────────────┴───────────┴─────────────┴───────────────┘
```

## Examples

### Downcasting to FP8 with stochasting rounding

We want to convert four `Float32` numbers to the 8-bit floating point format (FP8) used on AMD GPUs (or `DLFP8Types.Float8_E4M3FNUZ`, using the [`DLFP8Types.jl`](https://github.com/chengchingwen/DLFP8Types.jl) package)

```julia-repl
julia> using AMDGPU, WeirdFloats, DLFP8Types

julia> A = AMDGPU.rand(Float32, 4)
4-element ROCArray{Float32, 1, AMDGPU.Runtime.Mem.HIPBuffer}:
 0.3193292
 0.007127027
 0.77079546
 0.036067273

julia> B = AMDGPU.zeros(Int32, 1); C = AMDGPU.zeros(Int32, 1); D = AMDGPU.zeros(Int32, 1)
1-element ROCArray{Int32, 1, AMDGPU.Runtime.Mem.HIPBuffer}:
 0
```

With deterministic rounding we expect to have

```julia-repl
julia> Float8_E4M3FNUZ.(Array(A))
4-element Vector{Float8_E4M3FNUZ}:
 0.3125
 0.0068359375
 0.75
 0.03515625
```

Let's define a kernel which converts the elements of `A` to FP8 with [stochastic rounding](https://doi.org/10.1098/rsos.211631) by using `WeirdFloats.convert_sr`, and run the kernel multiple times, storing the result in the arrays `B`, `C`, and `D`:

```julia-repl
julia> @inbounds function kernel_sr(A, B)
           v = reinterpret(WeirdFloats.VFP8, Int32(0))
           v = WeirdFloats.convert_sr(v, A[1], rand(Int32), Int32(0))
           v = WeirdFloats.convert_sr(v, A[2], rand(Int32), Int32(1))
           v = WeirdFloats.convert_sr(v, A[3], rand(Int32), Int32(2))
           v = WeirdFloats.convert_sr(v, A[4], rand(Int32), Int32(3))
           B[1] = reinterpret(Int32, v)

           return nothing
       end
kernel_sr (generic function with 1 method)

julia> @device_code dir="fp8" @roc kernel_sr(A, B); B
1-element ROCArray{Int32, 1, AMDGPU.Runtime.Mem.HIPBuffer}:
 440141618

julia> @device_code dir="fp8" @roc kernel_sr(A, C); C
1-element ROCArray{Int32, 1, AMDGPU.Runtime.Mem.HIPBuffer}:
 423364658

julia> @device_code dir="fp8" @roc kernel_sr(A, D); D
1-element ROCArray{Int32, 1, AMDGPU.Runtime.Mem.HIPBuffer}:
 423364403
```

The four output FP8 numbers are packed in the one-element `Float32` arrays `B`, `C`, and `D`, we can unpack them to see their values:

```julia-repl
julia> for shift in (0, 8, 16, 24); println(reinterpret(Float8_E4M3FNUZ, (Array(B)[1] >> shift & 0xff) % UInt8)); end
0.3125
0.0068359375
0.75
0.0390625

julia> for shift in (0, 8, 16, 24); println(reinterpret(Float8_E4M3FNUZ, (Array(C)[1] >> shift & 0xff) % UInt8)); end
0.3125
0.0078125
0.75
0.03515625

julia> for shift in (0, 8, 16, 24); println(reinterpret(Float8_E4M3FNUZ, (Array(D)[1] >> shift & 0xff) % UInt8)); end
0.34375
0.0068359375
0.75
0.03515625
```

We can see that the elements of `A` have not been downcast deterministically to FP8 according to the expected result shown above, but in some cases they were rounded to the other adject number in FP8.
