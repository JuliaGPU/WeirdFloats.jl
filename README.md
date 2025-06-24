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
