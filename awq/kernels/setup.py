import os
import shutil
import sys
from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

IS_ROCM = torch.version.hip is not None
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

if IS_ROCM:
    # On ROCm, torch hipifies the sources at build time and compiles them with
    # hipcc/amdclang. The NVIDIA-only nvcc flags (--use_fast_math, --threads,
    # --expt-*) do not apply; -ffast-math is the clang equivalent. The
    # -U__HIP_NO_HALF_* flags mirror CUDA's -U__CUDA_NO_HALF_* (implicit
    # half/bf16 conversions), and the force-included compat shim supplies the
    # bf16 typedefs hipify leaves untranslated.
    compat_header = os.path.join(THIS_DIR, "csrc", "awq_amd_compat.h")
    nvcc_args = [
        "-O3",
        "-std=c++20",
        "-DENABLE_BF16",
        "-ffast-math",
        "-U__HIP_NO_HALF_OPERATORS__",
        "-U__HIP_NO_HALF_CONVERSIONS__",
        "-U__HIP_NO_BFLOAT16_OPERATORS__",
        "-U__HIP_NO_BFLOAT16_CONVERSIONS__",
        "-U__HIP_NO_BFLOAT162_OPERATORS__",
        "-U__HIP_NO_BFLOAT162_CONVERSIONS__",
        "-include", compat_header,
    ]
else:
    nvcc_args = [
        "-O3",
        "-std=c++17",
        "-DENABLE_BF16",  # TODO
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--threads=8",
    ]

cxx_args = ["-g", "-O3", "-std=c++20", "-DENABLE_BF16"]
if not (sys.platform == "win32" and IS_ROCM):
    # -fopenmp/-lgomp are Linux GCC flags; on Windows+ROCm the pybind .cpp
    # is routed through hipcc (amdclang), which does not need them.
    cxx_args += ["-fopenmp", "-lgomp"]

extra_compile_args = {
    "cxx": cxx_args,
    "nvcc": nvcc_args,
}

sources = [
    "csrc/pybind.cpp",
    "csrc/quantization/gemm_cuda_gen.cu",
    "csrc/quantization/gemv_cuda.cu",
    "csrc/quantization_new/gemv/gemv_cuda.cu",
    "csrc/quantization_new/gemm/gemm_cuda.cu",
    "csrc/layernorm/layernorm.cu",
    "csrc/position_embedding/pos_encoding_kernels.cu",
    "csrc/attention/ft_attention.cpp",
    "csrc/attention/decoder_masked_multihead_attention.cu",
    "csrc/rope_new/fused_rope_with_pos.cu",
    "csrc/w8a8/w8a8_gemm_cuda.cu",
    "csrc/w8a8/quantization.cu",
    "csrc/w8a8/act.cu",
    "csrc/w8a8/layernorm.cu",
]

# The W8A8 GEMM calls hipBLASLt directly on ROCm.
libraries = ["hipblaslt"] if IS_ROCM else []

if IS_ROCM and sys.platform == "win32":
    # On Windows+ROCm, torch routes .cpp files through MSVC (cl.exe), which
    # cannot parse the GCC __attribute__ syntax in HIP runtime headers pulled
    # in through torch/extension.h. Copy each .cpp source to a *_winhip.cu
    # shim so the .cu extension routes it through hipcc (amdclang) instead.
    # The shim is transient; do NOT commit it (it is gitignored).
    new_sources = []
    for s in sources:
        if s.endswith(".cpp"):
            shim = s[:-4] + "_winhip.cu"
            shim_abs = os.path.join(THIS_DIR, shim)
            src_abs = os.path.join(THIS_DIR, s)
            if not os.path.exists(shim_abs) or os.path.getmtime(src_abs) > os.path.getmtime(shim_abs):
                shutil.copy2(src_abs, shim_abs)
            new_sources.append(shim)
        else:
            new_sources.append(s)
    sources = new_sources

# Use ninja on Windows+ROCm: the non-ninja MSVC compile path does not escape
# spaces in -I paths, splitting include dirs with spaces when forwarded to hipcc.
use_ninja = sys.platform == "win32" and IS_ROCM

setup(
    name="awq_inference_engine",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="awq_inference_engine",
            sources=sources,
            extra_compile_args=extra_compile_args,
            libraries=libraries,
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=use_ninja)},
    install_requires=["torch"],
)
