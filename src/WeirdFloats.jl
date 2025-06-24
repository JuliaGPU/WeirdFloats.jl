module WeirdFloats

# Packed 4xFP8 values
primitive type VFP8 32 end

# Packed 4xBF8 values
primitive type VBF8 32 end

# declare float @llvm.amdgcn.cvt.f32.bf8(i32, i32)
# declare float @llvm.amdgcn.cvt.f32.fp8(i32, i32)

function convert(::Type{Float32}, x::VBF8, sel::Int32)
    # Convert packed BF8 to FP32
    x = ccall("llvm.amdgcn.cvt.f32.bf8", llvmcall, Float32, (VBF8, Int32), x, sel)
    return x
end

function convert(::Type{Float32}, x::VFP8, sel::Int32)
    # Convert packed FP8 to FP32
    x = ccall("llvm.amdgcn.cvt.f32.fp8", llvmcall, Float32, (VFP8, Int32), x, sel)
    return x
end
    
# declare <2 x float> @llvm.amdgcn.cvt.pk.f32.bf8(i32, i1)
# declare <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32, i1)

function convert_pk(x::VBF8, sel::Bool)
    # Convert packed BF8 to packed FP32
    x = ccall("llvm.amdgcn.cvt.pk.f32.bf8", llvmcall, NTuple{2, Core.VecElement{Float32}}, (VBF8, Bool), x, sel)
    reinterpret(NTuple{2, Float32}, x)
end

function convert_pk(x::VFP8, sel::Bool)
    # Convert packed FP8 to packed FP32
    x = ccall("llvm.amdgcn.cvt.pk.f32.fp8", llvmcall, NTuple{2, Core.VecElement{Float32}}, (VFP8, Bool), x, sel)
    reinterpret(NTuple{2, Float32}, x)
end

# declare i32 @llvm.amdgcn.cvt.pk.bf8.f32(float, float, i32, i1)
# declare i32 @llvm.amdgcn.cvt.pk.fp8.f32(float, float, i32, i1)
function convert(old::VBF8, x::Float32, y::Float32, sel::Bool)
    ccall("llvm.amdgcn.cvt.pk.bf8.f32", llvmcall, VBF8, (Float32, Float32, VBF8, Bool), x, y, old, sel)
end

function convert(old::VFP8, x::Float32, y::Float32, sel::Bool)
    ccall("llvm.amdgcn.cvt.pk.fp8.f32", llvmcall, VFP8, (Float32, Float32, VFP8, Bool), x, y, old, sel)
end

# https://github.com/llvm/llvm-project/blob/9e33cb22f991fb25d606d89c0e5a13a3ebed52fe/llvm/include/llvm/IR/IntrinsicsAMDGPU.td#L3424
# declare i32 @llvm.amdgcn.cvt.sr.bf8.f32(float, i32, i32, i32)
# declare i32 @llvm.amdgcn.cvt.sr.fp8.f32(float, i32, i32, i32)
function convert_sr(old::VBF8, x::Float32, seed::Int32, sel::Int32)
    ccall("llvm.amdgcn.cvt.sr.bf8.f32", llvmcall, VBF8, (Float32, Int32, VBF8, Int32), x, seed, old, sel)
end

function convert_sr(old::VFP8, x::Float32, seed::Int32, sel::Int32)
    ccall("llvm.amdgcn.cvt.sr.fp8.f32", llvmcall, VFP8, (Float32, Int32, VFP8, Int32), x, seed, old, sel)
end


end # module WeirdFloats
