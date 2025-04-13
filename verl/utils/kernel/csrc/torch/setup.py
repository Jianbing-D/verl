from setuptools import setup, find_packages
from torch.utils import cpp_extension
import os
import glob
import itertools

def main():
    cppstd = 17

    arch_list = ["80", "90a"]
    archs = [f"-gencode arch=compute_{sm},code=sm_{sm}"
             for sm in arch_list]
    cuda_archs = list()
    for item in archs:
        cuda_archs.extend(item.split(" "))

    cxx_flags = ["-O3", f"-std=c++{cppstd}"]
    nvcc_flags = [
        "-O3", f"-std=c++{cppstd}",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-lineinfo",
        "--use_fast_math",
        "--Werror", "all-warnings",
        "-D_GLIBCXX_USE_CXX11_ABI=1"
    ]
    nvcc_flags.extend(cuda_archs)

    current_directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_directory, "./")
    suffix = ["cc", "cu"]
    sources = list(itertools.chain(*[glob.glob(os.path.join(path, f"*.{suf}"))
                   for suf in suffix]))

    project_path = os.path.abspath(os.path.join(current_directory, "../../"))
    library_path = os.path.abspath(os.path.join(project_path, "./build/library"))

    setup(
        name="linear_cross_entropy_package",
        ext_modules=[
            cpp_extension.CUDAExtension(
                name="linear_cross_entropy_extension",
                sources=sources,
                include_dirs=[path, project_path, os.path.abspath(os.path.join(project_path, "csrc/library/"))],
                library_dirs=[library_path],
                libraries=["lce"],
                extra_compile_args = {
                    "cxx": cxx_flags,
                    "nvcc": nvcc_flags
                },
                extra_link_args=[f"-Wl,-rpath,{library_path}"],
            )
        ],
        cmdclass={
            "build_ext": cpp_extension.BuildExtension
        },
    )

if __name__ == "__main__":
    main()
        
        