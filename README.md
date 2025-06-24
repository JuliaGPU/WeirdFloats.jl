# WeirdFloats.jl

## Examples

### Downcasting to FP8 with stochasting rounding

```julia-repl
julia> using AMDGPU, WeirdFloats

julia> A = AMDGPU.rand(Float32, 4)
4-element ROCArray{Float32, 1, AMDGPU.Runtime.Mem.HIPBuffer}:
 0.14708285
 0.2024808
 0.42030886
 0.1691257

julia> B = AMDGPU.zeros(Int32, 1)
1-element ROCArray{Int32, 1, AMDGPU.Runtime.Mem.HIPBuffer}:
 0

julia> @inbounds function kernel_sr(A, B)
           v = reinterpret(WeirdFloats.VFP8, Int32(0))
           v = WeirdFloats.convert_sr(v, A[1], rand(Int32), Int32(0))
           v = WeirdFloats.convert_sr(v, A[2], Int32(1), Int32(1))
           v = WeirdFloats.convert_sr(v, A[3], Int32(2), Int32(2))
           v = WeirdFloats.convert_sr(v, A[4], rand(Int32), Int32(3))
           B[1] = reinterpret(Int32, v)

           return nothing
       end
kernel_sr (generic function with 1 method)

julia> @device_code dir="fp8" @roc kernel_sr(A, B); bitstring(AMDGPU.@allowscalar B[1])
"00101011001101010010110000101001"

julia> @device_code dir="fp8" @roc kernel_sr(A, B); bitstring(AMDGPU.@allowscalar B[1])
"00101011001101010010110000101001"

julia> @device_code dir="fp8" @roc kernel_sr(A, B); bitstring(AMDGPU.@allowscalar B[1])
"00101011001101010010110000101010"
```

Only the first and the last fp8 numbers packed in `B` are changing when calling the kernel multiple times.

```julia-repl
julia> @inbounds function kernel_sr(A, B)
           v = reinterpret(WeirdFloats.VFP8, Int32(0))
           v = WeirdFloats.convert_sr(v, A[1], Int32(0), Int32(0))
           v = WeirdFloats.convert_sr(v, A[2], Int32(1), Int32(1))
           v = WeirdFloats.convert_sr(v, A[3], Int32(2), Int32(2))
           v = WeirdFloats.convert_sr(v, A[4], Int32(3), Int32(3))
           B[1] = reinterpret(Int32, v)

           return nothing
       end
kernel_sr (generic function with 1 method)

julia> @device_code dir="fp8" @roc kernel_sr(A, B); bitstring(AMDGPU.@allowscalar B[1])
"00101010001101010010110000101001"

julia> @device_code dir="fp8" @roc kernel_sr(A, B); bitstring(AMDGPU.@allowscalar B[1])
"00101010001101010010110000101001"

julia> @device_code dir="fp8" @roc kernel_sr(A, B); bitstring(AMDGPU.@allowscalar B[1])
"00101010001101010010110000101001"
```

All seeds are fixed: the result of calling the kernel multiple times is always the same.

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

julia> @device_code dir="fp8" @roc kernel_sr(A, B); bitstring(AMDGPU.@allowscalar B[1])
"00101011001101010010110100101001"

julia> @device_code dir="fp8" @roc kernel_sr(A, B); bitstring(AMDGPU.@allowscalar B[1])
"00101011001101100010110100101001"

julia> @device_code dir="fp8" @roc kernel_sr(A, B); bitstring(AMDGPU.@allowscalar B[1])
"00101011001101100010110100101010"
```

Here all seeds are random, and all numbers change when calling the kernel multiple times.


This requires an AMD MI300 GPU and Julia v1.12 (to have LLVM 18, to support that generation of GPU):

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
