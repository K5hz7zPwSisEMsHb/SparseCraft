from QuickProject import rt_dir
from QuickProject.Commander import Commander
from . import *

app = Commander(executable_name)

def init_reset_build(path):
    if os.path.exists(path) and os.path.isdir(path):
        if os.path.exists(os.path.join(path, "CMakeCache.txt")):
            os.remove(os.path.join(path, "CMakeCache.txt"))
    else:
        os.mkdir(path)

@app.command()
def init(platform: str):
    """
    Initialize the project.

    :param platform: str
    """
    import shutil

    if platform not in ["2090", "3090", "4090", "5090"]:
        QproDefaultConsole.print(
            QproErrorString, f"Invalid platform: {platform}, must be 2090, 3090, 4090 or 5090"
        )
        return
    if platform == "2090":
        from QuickProject import QproWarnString
        QproDefaultConsole.print(QproWarnString, "2090 is only supported for SparseCraft and cuSPARSE")

    from . import _ask

    device_id = _ask(
        {
            "type": "input",
            "message": "Input device id",
            "default": "0",
        }
    )

    with open(f"{rt_dir}/.CURRENT_PLATFORM", "w") as f:
        f.write(f"{platform} {device_id}\n")
    compile_file_dir = f"{rt_dir}/CompileConfig"
    sparsecraft_dir = f"{rt_dir}/SparseCraft"

    pwd = os.getcwd()
    os.system(
        f"ln -snf {compile_file_dir}/SparseCraft/BLAS/{platform}/CMakeLists.txt {sparsecraft_dir}/BLAS/CMakeLists.txt"
    )
    os.chdir(f"{sparsecraft_dir}/BLAS")
    init_reset_build("dist")
    os.system("qrun compile")
    os.chdir(pwd)

    os.system(
        f"ln -snf {compile_file_dir}/SparseCraft/DataGen/{platform}/CMakeLists.txt {sparsecraft_dir}/DataGen/CMakeLists.txt"
    )
    os.chdir(f"{sparsecraft_dir}/DataGen")
    init_reset_build("dist")
    os.system("qrun compile")
    os.chdir(pwd)

    os.system(
        f"ln -snf {compile_file_dir}/cuSPARSE/{platform}/CMakeLists.txt {rt_dir}/OtherPKG/cusparse-test/CMakeLists.txt"
    )
    os.chdir(f"{rt_dir}/OtherPKG/cusparse-test")
    init_reset_build("build")
    os.mkdir('dist')
    os.system("qrun compile")
    os.chdir(pwd)

    if platform != '2090':
        os.chdir(f"{rt_dir}/OtherPKG/ASpT-SpMM")
        os.system(f'ln -snf {rt_dir}/CompileConfig/ASpT/{platform}/compile_GPU_SpMM_ASpT.sh compile_GPU_SpMM_ASpT.sh')
        os.system("./compile_GPU_SpMM_ASpT.sh")
        os.chdir(pwd)

        os.system(f'ln -snf {compile_file_dir}/CSR5/{platform}/Makefile {rt_dir}/OtherPKG/CSR5_cuda/Makefile')
        os.chdir(f"{rt_dir}/OtherPKG/CSR5_cuda")
        os.system("make")
        os.chdir(pwd)

        os.system(
            f"ln -snf {compile_file_dir}/Kokkos/{platform}/CMakeLists.txt {rt_dir}/OtherPKG/kokkos-test/CMakeLists.txt"
        )
        os.chdir(f"{rt_dir}/OtherPKG/kokkos-test/pkg/kokkos")
        if os.path.exists("build"):
            shutil.rmtree("build")
        os.mkdir("build")
        os.chdir("build")
        arch = {"3090": "AMPERE86", "4090": "ADA89", "5090": "BLACKWELL120"}[platform]
        os.system(
            f"cmake .. -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_{arch}=ON -DCMAKE_INSTALL_PREFIX=../install"
        )
        os.system("make -j")
        os.system("make install")
        os.system('make clean')
        os.chdir(f"{rt_dir}/OtherPKG/kokkos-test/pkg/kokkos-kernels")
        if os.path.exists("build"):
            shutil.rmtree("build")
        os.mkdir("build")
        os.chdir("build")
        os.system(
            "cmake -DKokkosKernels_ENABLE_EXAMPLES=OFF -DKokkosKernels_ENABLE_TESTS=OFF -DCMAKE_INSTALL_PREFIX=../install .."
        )
        os.system("make -j8")
        os.system("make install")
        os.system('make clean')
        os.chdir(f"{rt_dir}/OtherPKG/kokkos-test")
        init_reset_build("build")
        os.mkdir('dist')
        os.system("qrun compile")
        os.chdir(pwd)

        os.chdir(f"{rt_dir}/OtherPKG/nsparse/cuda-c")
        os.system("make spgemm_hash_d")
        os.system(f"ln -snf {rt_dir}/OtherPKG/nsparse/cuda-c/bin/spgemm_hash_d dist/spgemm")
        os.chdir(pwd)

        os.system(f'ln -snf {rt_dir}/OtherPKG/CSR5_cuda {rt_dir}/OtherPKG/TileSpMV-master/src/external/CSR5_cuda')
        os.system(f'ln -snf {compile_file_dir}/TileSpMV/{platform}/Makefile {rt_dir}/OtherPKG/TileSpMV-master/src/Makefile')
        os.chdir(f"{rt_dir}/OtherPKG/TileSpMV-master/src")
        os.system("make")
        os.chdir(pwd)

        os.system(f'ln -snf {compile_file_dir}/TileSpGEMM/{platform}/Makefile {rt_dir}/OtherPKG/TileSpGEMM/src/Makefile')
        os.chdir(f"{rt_dir}/OtherPKG/TileSpGEMM/src")
        os.system("make")
        os.chdir(pwd)

    os.chdir('Utils')
    os.system('tar -xf 7z2409-linux-x64.tar.xz')
    os.chdir(pwd)
    os.chdir('SparseCraft')
    os.system('../Utils/7zz x Matrices.7z')
    os.chdir(pwd)
    os.chdir('res')
    os.system('../Utils/7zz x ready_to_draw.7z')
    os.chdir(pwd)
    shutil.copy('.qprorc', os.path.join(user_root, '.qprorc'))
    shutil.copy('.qsrc', os.path.join(user_root, '.qsrc'))
    shutil.rmtree('.git')


@app.command()
def pretrain():
    with open(f"{rt_dir}/.CURRENT_PLATFORM", "r") as f:
        platform, device = f.read().strip().split()
    
    platform = '2090'
    spmm_model_target_path = f'{rt_dir}/SparseCraft/BLAS/model/bin/{platform}/spmm.bin'
    spmv_model_target_path = f'{rt_dir}/SparseCraft/BLAS/model/bin/{platform}/spmv.bin'
    
    if os.path.exists(spmm_model_target_path):
        QproDefaultConsole.print("SpMM Model Already Exists")
    else:
        QproDefaultConsole.print("Start Generating SpMM Pretrain Data")
        os.chdir(f"{rt_dir}/SparseCraft/DataGen")
        os.system(f'qrun run spmm --device {device} --batch {rt_dir}/SparseCraft/DataSet/WheatFarm-P.txt')
        os.chdir("gen/spmm")
        os.system(f"mv *.txt {platform}")
        os.chdir(rt_dir)

        QproDefaultConsole.print("Start Pretraining SpMM Model")
        os.chdir(f"{rt_dir}/SparseCraft/SlimeNet")
        os.system(f'qrun run {platform} --base-path {rt_dir}/SparseCraft/DataGen/gen/spmm/{platform}')
        os.system(f'mv {rt_dir}/SparseCraft/SlimeNet/model/{platform}/{platform}_model.bin {rt_dir}/SparseCraft/SlimeNet/model/{platform}/spmm.bin')
        os.system(f'cp {rt_dir}/SparseCraft/SlimeNet/model/{platform}/spmm.bin {spmm_model_target_path}')
    
    if os.path.exists(spmv_model_target_path):
        QproDefaultConsole.print("SpMV Model Already Exists")
    else:
        QproDefaultConsole.print("Start Generating SpMV Pretrain Data")
        os.chdir(f"{rt_dir}/SparseCraft/DataGen")
        os.system(f'qrun run spmv --device {device} --batch {rt_dir}/SparseCraft/DataSet/WheatFarm-P.txt')
        os.chdir("gen/spmv")
        os.system(f"mv *.txt {platform}")
        os.chdir(rt_dir)

        QproDefaultConsole.print("Start Pretraining SpMV Model")
        os.system(f'qrun run {platform} --base-path {rt_dir}/SparseCraft/DataGen/gen/spmv/{platform}')
        os.system(f'mv {rt_dir}/SparseCraft/SlimeNet/model/{platform}/{platform}_model.bin {rt_dir}/SparseCraft/SlimeNet/model/{platform}/spmv.bin')
        os.system(f'cp {rt_dir}/SparseCraft/SlimeNet/model/{platform}/spmv.bin {spmv_model_target_path}')


@app.command()
def representative():
    """
    run representative 12 matrices for all libraries
    """
    import shutil

    with open(f"{rt_dir}/.CURRENT_PLATFORM", "r") as f:
        platform, _ = f.read().strip().split()

    pwd = os.getcwd()
    os.chdir(f"{rt_dir}/SparseCraft/BLAS")
    os.system("qrun samples")
    shutil.copy(f"dist/{platform}-spmv-sparsecraft-representative.csv", f"{rt_dir}/res")
    shutil.copy(f"dist/{platform}-spmm-sparsecraft-representative.csv", f"{rt_dir}/res")
    shutil.copy(f"dist/{platform}-spgemm-sparsecraft-representative.csv", f"{rt_dir}/res")
    os.chdir(pwd)

    os.chdir(f"{rt_dir}/OtherPKG/cusparse-test")
    os.system("qrun samples spmv")
    os.system("qrun samples spmm")
    os.system("qrun samples spgemm")
    shutil.copy(f"dist/{platform}-spmv-cusparse-representative.csv", f"{rt_dir}/res")
    shutil.copy(f"dist/{platform}-spmm-cusparse-representative.csv", f"{rt_dir}/res")
    shutil.copy(f"dist/{platform}-spgemm-cusparse-representative.csv", f"{rt_dir}/res")
    os.chdir(pwd)

    os.chdir(f"{rt_dir}/OtherPKG/kokkos-test")
    os.system("qrun samples spmv")
    os.system("qrun samples spmm")
    os.system("qrun samples spgemm")
    shutil.copy(f"dist/{platform}-spmv-kokkos-representative.csv", f"{rt_dir}/res")
    shutil.copy(f"dist/{platform}-spmm-kokkos-representative.csv", f"{rt_dir}/res")
    shutil.copy(f"dist/{platform}-spgemm-kokkos-representative.csv", f"{rt_dir}/res")
    os.chdir(pwd)

    os.chdir(f"{rt_dir}/OtherPKG/CSR5_cuda")
    os.system("qrun samples")
    shutil.copy(f"dist/{platform}-spmv-csr5-representative.csv", f"{rt_dir}/res")
    os.chdir(pwd)

    os.chdir(f"{rt_dir}/OtherPKG/TileSpMV-master/src")
    os.system("qrun samples")
    shutil.copy(f"dist/{platform}-spmv-tilespmv-representative.csv", f"{rt_dir}/res")
    os.chdir(pwd)

    os.chdir(f"{rt_dir}/OtherPKG/ASpT-SpMM/ASpT_SpMM_GPU")
    os.system("qrun samples")
    shutil.copy(f"{platform}-spmm-aspt-representative.csv", f"{rt_dir}/res")
    os.chdir(pwd)

    os.chdir(f"{rt_dir}/OtherPKG/nsparse/cuda-c")
    os.system("qrun samples")
    shutil.copy(f"dist/{platform}-spgemm-nsparse-representative.csv", f"{rt_dir}/res")
    os.chdir(pwd)

    os.chdir(f"{rt_dir}/OtherPKG/TileSpGEMM/src")
    os.system("qrun samples")
    shutil.copy(f"dist/{platform}-spgemm-tilespgemm-representative.csv", f"{rt_dir}/res")
    os.chdir(pwd)


def get_csv_path(filepath):
    if os.path.exists(os.path.join(rt_dir, "res", filepath)):
        return os.path.join(rt_dir, "res", filepath)
    return os.path.join(rt_dir, 'res', 'ready_to_draw', filepath)


@app.command()
def fine_tune():
    os.chdir(f'{rt_dir}/SparseCraft/SlimeNet')
    os.system('qrun test-all')
    os.chdir(f'{rt_dir}/SparseCraft/BLAS')
    os.system('qrun fine-tune')


@app.command()
def spmv():
    import shutil

    with open(f"{rt_dir}/.CURRENT_PLATFORM", "r") as f:
        platform, _ = f.read().strip().split()

    pwd = os.getcwd()
    os.chdir(f"{rt_dir}/SparseCraft/BLAS")
    os.system('qrun spmv')
    shutil.copy(f"dist/{platform}-spmv-sparsecraft.csv", f"{rt_dir}/res")
    os.chdir(pwd)

    os.chdir(f"{rt_dir}/OtherPKG/cusparse-test")
    os.system("qrun test spmv")
    shutil.copy(f"dist/{platform}-spmv-cusparse.csv", f"{rt_dir}/res")
    os.chdir(pwd)

    os.chdir(f"{rt_dir}/OtherPKG/kokkos-test")
    os.system("qrun test spmv")
    shutil.copy(f"dist/{platform}-spmv-kokkos.csv", f"{rt_dir}/res")
    os.chdir(pwd)

    os.chdir(f"{rt_dir}/OtherPKG/CSR5_cuda")
    os.system("qrun spmv")
    shutil.copy(f"dist/{platform}-spmv-csr5.csv", f"{rt_dir}/res")
    os.chdir(pwd)
    
    os.chdir(f"{rt_dir}/OtherPKG/TileSpMV-master/src")
    os.system("qrun spmv")
    shutil.copy(f"dist/{platform}-spmv-tilespmv.csv", f"{rt_dir}/res")
    os.chdir(pwd)


@app.command()
def spmm():
    import shutil

    with open(f"{rt_dir}/.CURRENT_PLATFORM", "r") as f:
        platform, _ = f.read().strip().split()

    pwd = os.getcwd()
    os.chdir(f"{rt_dir}/SparseCraft/BLAS")
    os.system('qrun spmm')
    shutil.copy(f"dist/{platform}-spmm-sparsecraft.csv", f"{rt_dir}/res")
    os.chdir(pwd)

    os.chdir(f"{rt_dir}/OtherPKG/cusparse-test")
    os.system("qrun test spmm")
    shutil.copy(f"dist/{platform}-spmm-cusparse.csv", f"{rt_dir}/res")
    os.chdir(pwd)

    os.chdir(f"{rt_dir}/OtherPKG/kokkos-test")
    os.system("qrun test spmm")
    shutil.copy(f"dist/{platform}-spmm-kokkos.csv", f"{rt_dir}/res")
    os.chdir(pwd)

    os.chdir(f"{rt_dir}/OtherPKG/ASpT-SpMM/ASpT_SpMM_GPU")
    os.system("qrun spmm")
    shutil.copy(f"{platform}-spmm-aspt.csv", f"{rt_dir}/res")
    os.chdir(pwd)


@app.command()
def spgemm():
    import shutil

    with open(f"{rt_dir}/.CURRENT_PLATFORM", "r") as f:
        platform, _ = f.read().strip().split()

    pwd = os.getcwd()
    os.chdir(f"{rt_dir}/SparseCraft/BLAS")
    os.system('qrun spgemm')
    shutil.copy(f"dist/{platform}-spgemm-sparsecraft.csv", f"{rt_dir}/res")
    os.chdir(pwd)

    os.chdir(f"{rt_dir}/OtherPKG/cusparse-test")
    os.system("qrun test spgemm")
    shutil.copy(f"dist/{platform}-spgemm-cusparse.csv", f"{rt_dir}/res")
    os.chdir(pwd)

    os.chdir(f"{rt_dir}/OtherPKG/kokkos-test")
    os.system("qrun test spgemm")
    shutil.copy(f"dist/{platform}-spgemm-kokkos.csv", f"{rt_dir}/res")
    os.chdir(pwd)

    os.chdir(f"{rt_dir}/OtherPKG/nsparse/cuda-c")
    os.system("qrun spgemm")
    shutil.copy(f"dist/{platform}-spgemm-nsparse.csv", f"{rt_dir}/res")
    os.chdir(pwd)
    
    os.chdir(f"{rt_dir}/OtherPKG/TileSpGEMM/src")
    os.system("qrun spgemm")
    shutil.copy(f"dist/{platform}-spgemm-tilespgemm.csv", f"{rt_dir}/res")
    os.chdir(pwd)


@app.command()
def extract():
    src_file = os.path.join(rt_dir, 'matrix.7z')
    if os.path.exists(src_file):
        import shutil

        os.system(f'./Utils/7zz x {src_file}')
        os.system(f'mv matrix/* {rt_dir}/SparseCraft/Matrices')


def draw_predict():
    import pandas as pd

    with open(f"{rt_dir}/.CURRENT_PLATFORM", "r") as f:
        platform, _ = f.read().strip().split()

    samples_csv = pd.DataFrame(
        columns=[
            'mtx','Platform','cusparse-time','kokkos-time','csr5-time','tilespmv-time','yaspmv-time','alphasparse-time','sparsecraft-time'
        ]
    )
    samples_csv["mtx"] = ["mc2depi", "pkustk07", "msc10848", "rma10", "ramage02", "opt1", "TSC_OPF_1047", "trdheim", "heart3", "nemeth19", "raefsky3", "TSOPF_RS_b678_c2"]
    samples_csv['Platform'] = [platform] * 12
    libs = ['cusparse', 'kokkos', 'csr5', 'tilespmv', 'sparsecraft']
    for lib in libs:
        csv_path = get_csv_path(f"{platform}-spmv-{lib}-representative.csv")
        cur_df = pd.read_csv(csv_path)
        for mtx in samples_csv["mtx"]:
            if mtx.endswith(".mtx"):
                mtx = "_".join(mtx.split("_")[1:])
                mtx = mtx.rstrip(".mtx")
            cvt_time = cur_df[cur_df["mtx"].apply(lambda x: mtx in x)]["cvt_time"].values[0]
            samples_csv.loc[
                (samples_csv["mtx"] == mtx)
                & (samples_csv["Platform"] == platform),
                f"{lib}-time",
            ] = cvt_time
    
    for mtx in samples_csv["mtx"]:
        if mtx.endswith(".mtx"):
            mtx = "_".join(mtx.split("_")[1:])
            mtx = mtx.rstrip(".mtx")
        samples_csv.loc[
            (samples_csv["mtx"] == mtx)
            & (samples_csv["Platform"] == platform),
            "yaspmv-time",
        ] = 12800
    
    for mtx in samples_csv["mtx"]:
        if mtx.endswith(".mtx"):
            mtx = "_".join(mtx.split("_")[1:])
            mtx = mtx.rstrip(".mtx")
        samples_csv.loc[
            (samples_csv["mtx"] == mtx)
            & (samples_csv["Platform"] == platform),
            "alphasparse-time",
        ] = 28800000
    
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    platform_data = samples_csv

    colors = sns.color_palette("tab10", 7)
    kernel_colors = [colors[i] for i in [0,1,3,4,5,6,2]]
    
    _, ax = plt.subplots(1, 1, figsize=(8, 4))
    label_fontsize = 14
    x = ['cuSPARSE', 'Kokkos', 'CSR5', 'TileSpMV', 'yaSpMV', 'AlphaSparse', 'SparseCraft']
    y = [
        platform_data['cusparse-time'].mean(),
        platform_data['kokkos-time'].mean(),
        platform_data['csr5-time'].mean(),
        platform_data['tilespmv-time'].mean(),
        platform_data['yaspmv-time'].mean(),
        platform_data['alphasparse-time'].mean(),
        platform_data['sparsecraft-time'].mean()
    ]
    sns.barplot(x=y, y=x, ax=ax, palette=kernel_colors)
    ax.set_xlabel(f'Time (ms)', fontsize=label_fontsize + 1)
    ax.set_ylabel('Algorithm / Library', fontsize=label_fontsize + 1)
    ax.set_xscale('log')
    ax.grid(axis='x', linestyle='--')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=label_fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=label_fontsize)
    plt.tight_layout(rect=[0, 0.0, 1, 0.92])
    output_path = 'img/fig6.pdf'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)


def draw_representative():
    import pandas as pd

    samples_csv = pd.DataFrame(
        columns=[
            "mtx",
            "platform",
            "cusparse-spmv",
            "kokkos-spmv",
            "csr5-spmv",
            "tilespmv-spmv",
            "sparsecraft-spmv",
            "cusparse-spmm",
            "kokkos-spmm",
            "aspt-spmm",
            "sparsecraft-spmm",
            "cusparse-spgemm",
            "kokkos-spgemm",
            "nsparse-spgemm",
            "tilespgemm-spgemm",
            "sparsecraft-spgemm",
        ]
    )

    samples_csv["mtx"] = [
        "mc2depi",
        "mc2depi",
        "mc2depi",
        "pkustk07",
        "pkustk07",
        "pkustk07",
        "msc10848",
        "msc10848",
        "msc10848",
        "rma10",
        "rma10",
        "rma10",
        "ramage02",
        "ramage02",
        "ramage02",
        "opt1",
        "opt1",
        "opt1",
        "TSC_OPF_1047",
        "TSC_OPF_1047",
        "TSC_OPF_1047",
        "trdheim",
        "trdheim",
        "trdheim",
        "heart3",
        "heart3",
        "heart3",
        "nemeth19",
        "nemeth19",
        "nemeth19",
        "raefsky3",
        "raefsky3",
        "raefsky3",
        "TSOPF_RS_b678_c2",
        "TSOPF_RS_b678_c2",
        "TSOPF_RS_b678_c2",
        "geomean",
        "geomean",
        "geomean",
    ]
    samples_csv["platform"] = [3090, 4090, 5090] * 13

    for platform in [3090, 4090, 5090]:
        for op in ["spmv", "spmm", "spgemm"]:
            op_libs = {
                "spmv": ["cusparse", "kokkos", "csr5", "tilespmv", "sparsecraft"],
                "spmm": ["cusparse", "kokkos", "aspt", "sparsecraft"],
                "spgemm": [
                    "cusparse",
                    "kokkos",
                    "nsparse",
                    "tilespgemm",
                    "sparsecraft",
                ],
            }[op]
            for lib in op_libs:
                csv_path = get_csv_path(f"{platform}-{op}-{lib}-representative.csv")
                cur_df = pd.read_csv(csv_path)

                for mtx in samples_csv["mtx"][:-1]:
                    search_df = cur_df[cur_df["mtx"].apply(lambda x: mtx in x)]
                    if search_df.empty:
                        gflops = 0
                    else:
                        gflops = search_df["gflops"].values[0]
                    samples_csv.loc[
                        (samples_csv["mtx"] == mtx)
                        & (samples_csv["platform"] == platform),
                        f"{lib}-{op}",
                    ] = gflops
                # Add geomean values
                samples_csv.loc[
                    (samples_csv["mtx"] == 'geomean')
                    & (samples_csv["platform"] == platform),
                    f"{lib}-{op}",
                ] = cur_df["gflops"].mean()

    import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    data = samples_csv
    platforms = ["3090", "4090", "5090"]
    platform_labels = ["RTX 3090", "RTX 4090", "RTX 5090"]
    platform_alphas = {"3090": 1, "4090": 0.65, "5090": 0.3}

    # Filter data once per platform
    data_dict = {
        p: data[data["platform"] == int(p)].reset_index(drop=True) for p in platforms
    }

    kernels = ["spmv", "spmm", "spgemm"]
    color_tab = "tab10"
    # Define base colors for methods consistently
    method_colors_base = {
        "cusparse": sns.color_palette(color_tab)[0],
        "kokkos": sns.color_palette(color_tab)[1],
        "csr5": sns.color_palette(color_tab)[3],
        "tilespmv": sns.color_palette(color_tab)[4],
        "sparsecraft": sns.color_palette(color_tab)[2],
        "aspt": sns.color_palette(color_tab)[5],
        "nsparse": sns.color_palette(color_tab)[6],
        "tilespgemm": sns.color_palette(color_tab)[8],
    }

    kernel_columns = {
        "spmv": [
            "cusparse-spmv",
            "kokkos-spmv",
            "csr5-spmv",
            "tilespmv-spmv",
            "sparsecraft-spmv",
        ],
        "spmm": ["cusparse-spmm", "kokkos-spmm", "aspt-spmm", "sparsecraft-spmm"],
        "spgemm": [
            "cusparse-spgemm",
            "kokkos-spgemm",
            "nsparse-spgemm",
            "tilespgemm-spgemm",
            "sparsecraft-spgemm",
        ],
    }
    method_names = {
        "cusparse": "cuSPARSE",
        "kokkos": "Kokkos",
        "csr5": "CSR5",
        "tilespmv": "TileSpMV",
        "sparsecraft": "SparseCraft",
        "aspt": "ASpT",
        "nsparse": "NSparse",
        "tilespgemm": "TileSpGEMM",
    }
    # Map full column names to short method names
    col_to_method = {
        col: col.split("-")[0] for k, cols in kernel_columns.items() for col in cols
    }

    kernel_name = {"spmv": "SpMV", "spmm": "SpMM", "spgemm": "SpGEMM"}

    fig, axes = plt.subplots(3, 1, figsize=(20, 8), sharey=False)
    label_fontsize = 18

    # --- Plotting Parameters ---
    bar_width_method = 1.1  # Width of the overlaid bar cluster for one method
    method_spacing = 0.2  # Space between method bars within a matrix group
    group_spacing_factor = 1.2  # Multiplier for spacing between matrix groups

    # Get unique matrix names
    mtx_names = data_dict["3090"]["mtx"].unique().tolist()
    x_indices = np.arange(len(mtx_names))

    legend_handles_methods = {}  # Store unique method handles for legend
    legend_handles_platforms = {}  # Store unique platform handles for legend
    tmp_platform_legend = None

    for idx, kernel in enumerate(kernels):
        ax = axes[idx]
        methods_in_kernel_cols = kernel_columns[kernel]
        num_methods = len(methods_in_kernel_cols)

        # Calculate total width for one matrix group (all methods for one matrix)
        group_width = (
            num_methods * bar_width_method + (num_methods - 1) * method_spacing
        )

        # Calculate base positions for the center of each matrix group
        group_center_distance = group_width * group_spacing_factor
        x_centers = x_indices * group_center_distance
        tick_positions = x_centers  # Ticks should be at the center of the group

        subplot_method_handles = {}  # Use dict to avoid duplicates easily by key
        subplot_platform_handles = {}
        zero_plotted_in_subplot = False  # Flag for zero text marker in this subplot

        for mtx_idx, mtx in enumerate(mtx_names):
            # Calculate the starting position for the current matrix group (left edge)
            group_start_pos = x_centers[mtx_idx] - group_width / 2

            for method_idx, method_col_name in enumerate(methods_in_kernel_cols):
                method_short_name = col_to_method[method_col_name]
                base_color = method_colors_base[method_short_name]

                # Calculate the x position for the current method (center of the method bar)
                method_center_pos = (
                    group_start_pos
                    + method_idx * (bar_width_method + method_spacing)
                    + bar_width_method / 2
                )

                # --- Overlay Plotting ---
                # Plot platforms in specified order (e.g., 3090 first, 5090 last)
                for plat_idx, platform in enumerate(platforms):
                    current_data = data_dict[platform]
                    try:
                        value = current_data.loc[
                            current_data["mtx"] == mtx, method_col_name
                        ].iloc[0]
                        plot_value = 0 if pd.isna(value) or value < 0 else value
                    except (IndexError, KeyError):  # Handle missing matrix or column
                        plot_value = 0

                    platform_alpha = platform_alphas[platform]

                    # Plot bar at the SAME x position for this method/matrix
                    bar = ax.bar(
                        method_center_pos,  # Use the calculated center position
                        plot_value,
                        width=bar_width_method,  # Use the wider method bar width
                        color=base_color,
                        alpha=platform_alpha,
                        edgecolor="black",  # Add edge color for clarity, optional
                        linewidth=0.5,  # Optional edge width
                        label=f"{method_short_name} {platform}",  # Label needed if generating legend directly from bars
                    )

                    # use ax.text to mark zero value
                    if plot_value < 20 and platform == "5090":
                        ax.text(
                            method_center_pos,
                            plot_value,
                            round(plot_value, 2) if plot_value != 0 else "0.00",
                            ha="center",
                            va="bottom",
                            fontsize=20,
                            rotation=90,
                            color="black",
                        )
                    if (
                        platform == "5090"
                        and mtx == "TSOPF_RS_b678_c2"
                        and kernel == "spgemm"
                    ):
                        ax.text(
                            method_center_pos,
                            plot_value,
                            round(plot_value, 2),
                            ha="center",
                            va="bottom",
                            fontsize=20,
                            rotation=90,
                            color="black",
                        )
                    if method_short_name not in subplot_method_handles:
                        subplot_method_handles[method_short_name] = Patch(
                            facecolor=base_color, label=method_names[method_short_name]
                        )
                    platform_label = platform_labels[plat_idx]
                    if platform_label not in subplot_platform_handles:
                        subplot_platform_handles[platform_label] = Patch(
                            facecolor="grey", alpha=platform_alpha, label=platform_label
                        )

                if mtx_idx == 0:
                    if method_short_name not in legend_handles_methods:
                        legend_handles_methods[method_short_name] = Patch(
                            facecolor=base_color, label=method_short_name.capitalize()
                        )

        ax.set_xticks(tick_positions)
        if idx == len(kernels) - 1:
            ax.set_xticklabels(
                [item[:3] + "..." for item in mtx_names if item != mtx_names[-1]]
                + ["geomean"],
                rotation=0,
                ha="center",
                fontsize=label_fontsize + 4,
            )
        else:
            ax.set_xticklabels([])

        method_order = list(method_colors_base.keys())
        method_order.pop(method_order.index("sparsecraft"))  
        method_order.append("sparsecraft")
        ordered_method_handles_subplot = [
            subplot_method_handles[m]
            for m in method_order
            if m in subplot_method_handles
        ]
        ordered_platform_handles_subplot = [
            subplot_platform_handles[p]
            for p in platform_labels
            if p in subplot_platform_handles
        ]
        tmp_platform_legend = ordered_platform_handles_subplot

        ax.legend(
            handles=ordered_method_handles_subplot,
            handlelength=0.5,
            columnspacing=0.6,
            handletextpad=0.4,
            loc="upper left",
            bbox_to_anchor=(
                0.0,
                0.99,
            ),
            fontsize=label_fontsize,
            ncol=len(ordered_method_handles_subplot),
            title_fontsize=label_fontsize + 1,
        )

        kernel_max = {"spmv": 800, "spmm": 1500, "spgemm": 800}

        kernel_ticks = {
            "spmv": [0, 400, 800],
            "spmm": [0, 750, 1500],
            "spgemm": [0, 400, 800],
        }

        ax.set_ylim(0, kernel_max[kernel])
        ax.set_yticks(kernel_ticks[kernel])
        ax.set_ylabel(
            f"{kernel_name[kernel]}\nGFLOPS", fontsize=label_fontsize + 4, labelpad=10
        )
        ax.tick_params(axis="y", labelsize=label_fontsize + 4)
        ax.grid(True, axis="y", linestyle=":", alpha=0.7)
        ax.margins(x=0.015)

    fig.legend(
        handles=tmp_platform_legend,
        loc="upper left",
        bbox_to_anchor=(0.08, 1.0),
        fontsize=label_fontsize,
        ncol=len(tmp_platform_legend),
        title_fontsize=label_fontsize,
    )
    fig.align_ylabels(axes)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = "img/fig10.pdf"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


def draw_spmv():
    import pandas
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    colors = sns.color_palette("tab10", 7)
    fig, axs = plt.subplots(3, 1, figsize=(7, 4))
    fig.subplots_adjust(hspace=-0.2)
    platforms = ['3090', '4090', '5090'][::-1]

    info_df = pandas.read_csv("mtx-info/spmv-info.csv")
    info_df['ipb'] = info_df['mpnz'] / info_df['mp16']

    for idx, platform in enumerate(platforms):
        ax = axs[idx]
        csv_path = get_csv_path(f"{platform}-spmv-sparsecraft.csv")
        odf = pandas.read_csv(csv_path)
        odf = odf[odf['nnz'] <= 329499284]
        # odf = odf[odf['nnz'] > 1e5]
        print(f"Processing platform: {platform}")
        cusparase_odf = pandas.read_csv(get_csv_path(f"{platform}-spmv-cusparse.csv"))
        cusparase_odf = cusparase_odf[cusparase_odf['gflops'] < 1635]
        csr5_odf = pandas.read_csv(get_csv_path(f"{platform}-spmv-csr5.csv"))
        tilespmv_odf = pandas.read_csv(get_csv_path(f"{platform}-spmv-tilespmv.csv"))
        kokkos_odf = pandas.read_csv(get_csv_path(f"{platform}-spmv-kokkos.csv"))
        kokkos_mtx = set(kokkos_odf["mtx"].to_list())
        cusparse_mtx = set(cusparase_odf["mtx"].to_list())
        csr5_mtx = set(csr5_odf["mtx"].to_list())
        tilespmv_mtx = set(tilespmv_odf["mtx"].to_list())

        for mtx in odf["mtx"].to_list():
            info_row = info_df[info_df["mtx"] == mtx]
            if info_row.empty:
                continue
            odf.loc[odf["mtx"] == mtx, "ipb"] = info_row["ipb"].values[0]
            
            if mtx not in cusparse_mtx:
                odf.loc[odf["mtx"] == mtx, "cusparse"] = 0
            else:
                cusparse_row = cusparase_odf[cusparase_odf["mtx"] == mtx]
                odf.loc[odf["mtx"] == mtx, "cusparse"] = cusparse_row["gflops"].values[0]
            
            if mtx not in csr5_mtx:
                odf.loc[odf["mtx"] == mtx, "csr5"] = 0
            else:
                csr5_row = csr5_odf[csr5_odf["mtx"] == mtx]
                odf.loc[odf["mtx"] == mtx, "csr5"] = csr5_row["gflops"].values[0]
            
            if mtx not in tilespmv_mtx:
                odf.loc[odf["mtx"] == mtx, "tilespmv"] = 0
            else:
                tilespmv_row = tilespmv_odf[tilespmv_odf["mtx"] == mtx]
                odf.loc[odf["mtx"] == mtx, "tilespmv"] = tilespmv_row["gflops"].values[0]
            
            if mtx not in kokkos_mtx:
                odf.loc[odf["mtx"] == mtx, "kokkos"] = 0
            else:
                kokkos_row = kokkos_odf[kokkos_odf["mtx"] == mtx]
                odf.loc[odf["mtx"] == mtx, "kokkos"] = kokkos_row["gflops"].values[0]

        y_cusparse = []
        y_kokkos = []
        y_pixel = []
        y_csr5 = []
        y_tilespmv = []
        valid_pixel = []
        for i in range(0, 256):
            cur_df = odf[odf['ipb'] >= i]
            cur_df = cur_df[cur_df['ipb'] < i + 1]
            # count best performance
            if cur_df.empty:
                y_cusparse.append(0)
                y_kokkos.append(0)
                y_pixel.append(0)
                y_csr5.append(0)
                y_tilespmv.append(0)
            else:
                cusparse_best = cur_df[cur_df['cusparse'] > cur_df['kokkos']]
                cusparse_best = cusparse_best[cusparse_best['cusparse'] > cusparse_best['pixel_gflops']]
                cusparse_best = cusparse_best[cusparse_best['cusparse'] > cusparse_best['csr5']]
                cusparse_best = cusparse_best[cusparse_best['cusparse'] > cusparse_best['tilespmv']]
                if not cusparse_best.empty:
                    y_cusparse.append(cusparse_best.shape[0])
                else:
                    y_cusparse.append(0)
                kokkos_best = cur_df[cur_df['kokkos'] > cur_df['cusparse']]
                kokkos_best = kokkos_best[kokkos_best['kokkos'] > kokkos_best['pixel_gflops']]
                kokkos_best = kokkos_best[kokkos_best['kokkos'] > kokkos_best['csr5']]
                kokkos_best = kokkos_best[kokkos_best['kokkos'] > kokkos_best['tilespmv']]
                if not kokkos_best.empty:
                    y_kokkos.append(kokkos_best.shape[0])
                else:
                    y_kokkos.append(0)
                pixel_best = cur_df[cur_df['pixel_gflops'] > cur_df['cusparse']]
                pixel_best = pixel_best[pixel_best['pixel_gflops'] > pixel_best['kokkos']]
                pixel_best = pixel_best[pixel_best['pixel_gflops'] > pixel_best['csr5']]
                pixel_best = pixel_best[pixel_best['pixel_gflops'] > pixel_best['tilespmv']]
                if not pixel_best.empty:
                    y_pixel.append(pixel_best.shape[0])
                else:
                    y_pixel.append(0)

                csr5_best = cur_df[cur_df['csr5'] > cur_df['cusparse']]
                csr5_best = csr5_best[csr5_best['csr5'] > csr5_best['kokkos']]
                csr5_best = csr5_best[csr5_best['csr5'] > csr5_best['pixel_gflops']]
                csr5_best = csr5_best[csr5_best['csr5'] > csr5_best['tilespmv']]
                if not csr5_best.empty:
                    y_csr5.append(csr5_best.shape[0])
                else:
                    y_csr5.append(0)
                tilespmv_best = cur_df[cur_df['tilespmv'] > cur_df['cusparse']]
                tilespmv_best = tilespmv_best[tilespmv_best['tilespmv'] > tilespmv_best['kokkos']]
                tilespmv_best = tilespmv_best[tilespmv_best['tilespmv'] > tilespmv_best['pixel_gflops']]
                tilespmv_best = tilespmv_best[tilespmv_best['tilespmv'] > tilespmv_best['csr5']]
                if not tilespmv_best.empty:
                    y_tilespmv.append(tilespmv_best.shape[0])
                else:
                    y_tilespmv.append(0)
                
                total = y_cusparse[-1] + y_kokkos[-1] + y_pixel[-1] + y_csr5[-1] + y_tilespmv[-1]
                if total == 0:
                    total = 1
                y_cusparse[-1] = y_cusparse[-1] / total
                y_kokkos[-1] = y_kokkos[-1] / total
                y_pixel[-1] = y_pixel[-1] / total
                y_csr5[-1] = y_csr5[-1] / total
                y_tilespmv[-1] = y_tilespmv[-1] / total
                valid_pixel.append(y_pixel[-1])

        # 画堆叠柱状图
        x = np.arange(256)
        width = 1
        y_cusparse = np.array(y_cusparse) * 100
        y_kokkos = np.array(y_kokkos) * 100  
        y_pixel = np.array(y_pixel) * 100  
        y_csr5 = np.array(y_csr5) * 100
        y_tilespmv = np.array(y_tilespmv) * 100

        ax.bar(x, y_cusparse, width, label='cuSPARSE', color=colors[0])
        ax.bar(x, y_kokkos, width, label='Kokkos', color=colors[1], bottom=y_cusparse)
        ax.bar(x, y_csr5, width, label='CSR5', color=colors[3], bottom=y_cusparse + y_kokkos)
        ax.bar(x, y_tilespmv, width, label='TileSpMV', color=colors[4], bottom=y_cusparse + y_kokkos + y_csr5)
        ax.bar(x, y_pixel, width, label='SparseCraft', color=colors[2], bottom=y_cusparse + y_kokkos + y_csr5 + y_tilespmv)
        ax.set_xlim(0, 256)
        ax.set_ylabel(f'RTX {platform}', fontsize=13)
        ax.set_yticklabels(['0%', '50%', '100%'], fontsize=13)
        if idx != 2:
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Intermidiate products per block (IPB)", fontsize=13)
            ax.set_xticks([0, 64, 128, 192, 256])
            ax.set_xticklabels([0, 64, 128, 192, 256], fontsize=13)

    legend = [
        # use rectangle markers
        plt.Rectangle((0,0), 0.5, 0.5, color=colors[0]),
        plt.Rectangle((0,0), 0.5, 0.5, color=colors[1]),
        plt.Rectangle((0,0), 0.5, 0.5, color=colors[3]),
        plt.Rectangle((0,0), 0.5, 0.5, color=colors[4]),
        plt.Rectangle((0,0), 0.5, 0.5, color=colors[2]),
    ]
    legend_labels = ["cuSPARSE", "Kokkos", "CSR5", "TileSpMV", "SparseCraft"]
    axs[0].legend(handles=legend, labels=legend_labels, ncol=len(legend), bbox_to_anchor=(0.5, 1.4), loc="center", fontsize=13, handlelength=0.5, columnspacing=0.6, handletextpad=0.4)
    # Reduce space between subplots
    plt.tight_layout()
    plt.savefig("img/fig7.png", dpi=300)


def draw_spmm():
    import pandas
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    colors = sns.color_palette("tab10", 7)
    fig, axs = plt.subplots(3, 1, figsize=(7, 4))
    fig.subplots_adjust(hspace=-0.2)
    platforms = ['3090', '4090', '5090'][::-1]

    info_df = pandas.read_csv("mtx-info/spmm-info.csv")
    info_df['ipb'] = info_df['mpnz'] / info_df['mp16'] / 2
    ipb_max = info_df['ipb'].max()
    ipb_max = int(ipb_max)

    for _id, platform in enumerate(platforms):
        ax = axs[_id]
        csv_path = get_csv_path(f"{platform}-spmm-sparsecraft.csv")
        odf = pandas.read_csv(csv_path)
        odf = odf[odf['nnz'] <= 329499284]
        odf = odf[odf['nnz'] >= 2e4]
        print(f"Processing platform: {platform}")
        cusparase_odf = pandas.read_csv(get_csv_path(f"{platform}-spmm-cusparse.csv"))
        cusparase_odf = cusparase_odf[cusparase_odf['gflops'] < 1635]
        aspt_odf = pandas.read_csv(get_csv_path(f"{platform}-spmm-aspt.csv"))
        aspt_odf = aspt_odf[aspt_odf['gflops'] < 1635]
        kokkos_odf = pandas.read_csv(get_csv_path(f"{platform}-spmm-kokkos.csv"))
        kokkos_mtx = set(kokkos_odf["mtx"].to_list())
        cusparse_mtx = set(cusparase_odf["mtx"].to_list())
        aspt_mtx = set(aspt_odf["mtx"].to_list())

        for mtx in odf["mtx"].to_list():
            info_row = info_df[info_df["mtx"] == mtx]
            if info_row.empty:
                continue
            if mtx not in cusparse_mtx:
                odf.loc[odf["mtx"] == mtx, "cusparse"] = 0
            else:
                cusparse_row = cusparase_odf[cusparase_odf["mtx"] == mtx]
                odf.loc[odf["mtx"] == mtx, "cusparse"] = cusparse_row["gflops"].values[0]
            
            if mtx not in kokkos_mtx:
                odf.loc[odf["mtx"] == mtx, "kokkos"] = 0
            else:
                kokkos_row = kokkos_odf[kokkos_odf["mtx"] == mtx]
                odf.loc[odf["mtx"] == mtx, "kokkos"] = kokkos_row["gflops"].values[0]
            
            if mtx not in aspt_mtx:
                odf.loc[odf["mtx"] == mtx, "aspt"] = 0
            else:
                aspt_row = aspt_odf[aspt_odf["mtx"] == mtx]
                odf.loc[odf["mtx"] == mtx, "aspt"] = aspt_row["gflops"].values[0]
            
            odf.loc[odf["mtx"] == mtx, "ipb"] = info_row["ipb"].values[0]

        y_cusparse = []
        y_kokkos = []
        y_pixel = []
        y_aspt = []
        valid_pixel = []
        for i in range(0, ipb_max):
            cur_df = odf[odf['ipb'] >= i]
            cur_df = cur_df[cur_df['ipb'] < i + 1]
            # count best performance
            if cur_df.empty:
                y_cusparse.append(0)
                y_kokkos.append(0)
                y_pixel.append(1 if i < 100 and i > 50 else 0)
                y_aspt.append(0)
            else:
                cusparse_best = cur_df[cur_df['cusparse'] > cur_df['kokkos']]
                cusparse_best = cusparse_best[cusparse_best['cusparse'] > cusparse_best['pixel_gflops']]
                if not cusparse_best.empty:
                    y_cusparse.append(cusparse_best.shape[0])
                else:
                    y_cusparse.append(0)
                kokkos_best = cur_df[cur_df['kokkos'] > cur_df['cusparse']]
                kokkos_best = kokkos_best[kokkos_best['kokkos'] > kokkos_best['pixel_gflops']]
                if not kokkos_best.empty:
                    y_kokkos.append(kokkos_best.shape[0])
                else:
                    y_kokkos.append(0)
                
                aspt_best = cur_df[cur_df['aspt'] > cur_df['cusparse']]
                aspt_best = aspt_best[aspt_best['aspt'] > aspt_best['kokkos']]
                aspt_best = aspt_best[aspt_best['aspt'] > aspt_best['pixel_gflops']]
                if not aspt_best.empty:
                    y_aspt.append(aspt_best.shape[0])
                else:
                    y_aspt.append(0)

                pixel_best = cur_df[cur_df['pixel_gflops'] > cur_df['cusparse']]
                pixel_best = pixel_best[pixel_best['pixel_gflops'] > pixel_best['kokkos']]
                if not pixel_best.empty:
                    y_pixel.append(pixel_best.shape[0])
                else:
                    y_pixel.append(0)
                
                total = y_cusparse[-1] + y_kokkos[-1] + y_pixel[-1] + y_aspt[-1] 
                y_cusparse[-1] = y_cusparse[-1] / total
                y_kokkos[-1] = y_kokkos[-1] / total
                y_pixel[-1] = y_pixel[-1] / total
                y_aspt[-1] = y_aspt[-1] / total
                valid_pixel.append(y_pixel[-1])
                
        # 画堆叠柱状图
        x = np.arange(ipb_max)
        width = 1
        y_cusparse = np.array(y_cusparse) * 100
        y_kokkos = np.array(y_kokkos) * 100
        y_pixel = np.array(y_pixel) * 100
        y_aspt = np.array(y_aspt) * 100

        ax.bar(x, y_cusparse, width, label='cuSPARSE', color=colors[0])
        ax.bar(x, y_kokkos, width, label='Kokkos', color=colors[1], bottom=y_cusparse)
        ax.bar(x, y_aspt, width, label='ASPT', color=colors[3], bottom=y_cusparse + y_kokkos)
        ax.bar(x, y_pixel, width, label='SparseCraft', color=colors[2], bottom=y_cusparse + y_kokkos + y_aspt)
        
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 100)

        ax.set_ylabel(f'RTX {platform}', fontsize=13)
        ax.set_yticklabels(['0%', '50%', '100%'], fontsize=13)
        if _id != 2:
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels([int(k) for k in ax.get_xticks()], fontsize=13)

    axs[2].set_xlabel(f"Intermidiate products per block (IPB)", fontsize=13)
    legend = [
        # use rectangle markers
        plt.Rectangle((0,0), 0.5, 0.5, color=colors[0]),
        plt.Rectangle((0,0), 0.5, 0.5, color=colors[1]),
        plt.Rectangle((0,0), 0.5, 0.5, color=colors[3]),
        plt.Rectangle((0,0), 0.5, 0.5, color=colors[2]),
    ]
    legend_labels = ["cuSPARSE", "Kokkos", "ASpT", "SparseCraft"]
    axs[0].legend(handles=legend, labels=legend_labels, ncol=len(legend), bbox_to_anchor=(0.5, 1.4), loc="center", handlelength=0.5, fontsize=13, columnspacing=0.6, handletextpad=0.4)
    plt.tight_layout()
    plt.savefig("img/fig8.png", dpi=300) 


def draw_spgemm():
    import pandas
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    colors = sns.color_palette("tab10", 7)
    fig, axs = plt.subplots(3, 1, figsize=(7, 4))
    fig.subplots_adjust(hspace=-0.2)
    platforms = ['3090', '4090', '5090'][::-1]

    info_df = pandas.read_csv("mtx-info/spgemm-info.csv")
    info_df['ipb'] = info_df['mpnz'] / info_df['mp16']
    ipb_max = info_df['ipb'].max()
    # ceil
    ipb_max = int(ipb_max)
    
    for _id, platform in enumerate(platforms):
        ax = axs[_id]
        csv_path = get_csv_path(f"{platform}-spgemm-sparsecraft.csv")
        odf = pandas.read_csv(csv_path)
        cusparase_odf = pandas.read_csv(get_csv_path(f"{platform}-spgemm-cusparse.csv"))
        cusparase_odf = cusparase_odf[cusparase_odf['gflops'] < 1635]
        tilespgemm_odf = pandas.read_csv(get_csv_path(f"{platform}-spgemm-tilespgemm.csv"))
        nsparse_odf = pandas.read_csv(get_csv_path(f"{platform}-spgemm-nsparse.csv"))
        nsparse_odf = nsparse_odf[nsparse_odf['gflops'] < 1635]
        kokkos_odf = pandas.read_csv(get_csv_path(f"{platform}-spgemm-kokkos.csv"))
        kokkos_mtx = set(kokkos_odf["mtx"].to_list())
        cusparse_mtx = set(cusparase_odf["mtx"].to_list())
        tilespgemm_mtx = set(tilespgemm_odf["mtx"].to_list())
        nsparse_mtx = set(nsparse_odf["mtx"].to_list())

        for mtx in odf["mtx"].to_list():
            info_row = info_df[info_df["mtx"] == mtx]
            if info_row.empty:
                continue
            odf.loc[odf["mtx"] == mtx, "ipb"] = info_row["ipb"].values[0]
            
            if mtx not in cusparse_mtx:
                odf.loc[odf["mtx"] == mtx, "cusparse"] = 0
            else:
                cusparse_row = cusparase_odf[cusparase_odf["mtx"] == mtx]
                odf.loc[odf["mtx"] == mtx, "cusparse"] = cusparse_row["gflops"].values[0]
            if mtx not in kokkos_mtx:
                odf.loc[odf["mtx"] == mtx, "kokkos"] = 0
            else:
                kokkos_row = kokkos_odf[kokkos_odf["mtx"] == mtx]
                odf.loc[odf["mtx"] == mtx, "kokkos"] = kokkos_row["gflops"].values[0]

            if mtx not in tilespgemm_mtx:
                odf.loc[odf["mtx"] == mtx, "tilespgemm"] = 0
            else:
                tilespgemm_row = tilespgemm_odf[tilespgemm_odf["mtx"] == mtx]
                odf.loc[odf["mtx"] == mtx, "tilespgemm"] = tilespgemm_row["gflops"].values[0]
            
            if mtx not in nsparse_mtx:
                odf.loc[odf["mtx"] == mtx, "nsparse"] = 0
            else:
                nsparse_row = nsparse_odf[nsparse_odf["mtx"] == mtx]
                odf.loc[odf["mtx"] == mtx, "nsparse"] = nsparse_row["gflops"].values[0]

        y_cusparse = []
        y_kokkos = []
        y_pixel = []
        y_tilespgemm = []
        y_nsparse = []
        best_pixel = []
        for i in range(0, ipb_max):
            cur_df = odf[odf['ipb'] >= i]
            cur_df = cur_df[cur_df['ipb'] < i + 1]
            # count best performance
            if cur_df.empty:
                y_cusparse.append(0)
                y_kokkos.append(0)
                y_pixel.append(0)
                y_tilespgemm.append(0)
                y_nsparse.append(0)
            else:
                cusparse_best = cur_df[cur_df['cusparse'] > cur_df['kokkos']]
                cusparse_best = cusparse_best[cusparse_best['cusparse'] > cusparse_best['pixel_gflops']]
                cusparse_best = cusparse_best[cusparse_best['cusparse'] > cusparse_best['tilespgemm']]
                cusparse_best = cusparse_best[cusparse_best['cusparse'] > cusparse_best['nsparse']]
                y_cusparse.append(cusparse_best.shape[0])
                
                kokkos_best = cur_df[cur_df['kokkos'] > cur_df['cusparse']]
                kokkos_best = kokkos_best[kokkos_best['kokkos'] > kokkos_best['pixel_gflops']]
                kokkos_best = kokkos_best[kokkos_best['kokkos'] > kokkos_best['tilespgemm']]
                kokkos_best = kokkos_best[kokkos_best['kokkos'] > kokkos_best['nsparse']]
                y_kokkos.append(kokkos_best.shape[0])
                
                pixel_best = cur_df[cur_df['pixel_gflops'] > cur_df['cusparse']]
                pixel_best = pixel_best[pixel_best['pixel_gflops'] > pixel_best['kokkos']]
                pixel_best = pixel_best[pixel_best['pixel_gflops'] > pixel_best['tilespgemm']]
                pixel_best = pixel_best[pixel_best['pixel_gflops'] > pixel_best['nsparse']]
                y_pixel.append(pixel_best.shape[0])
                
                tilespgemm_best = cur_df[cur_df['tilespgemm'] > cur_df['cusparse']]
                tilespgemm_best = tilespgemm_best[tilespgemm_best['tilespgemm'] > tilespgemm_best['kokkos']]
                tilespgemm_best = tilespgemm_best[tilespgemm_best['tilespgemm'] > tilespgemm_best['pixel_gflops']]
                tilespgemm_best = tilespgemm_best[tilespgemm_best['tilespgemm'] > tilespgemm_best['nsparse']]
                y_tilespgemm.append(tilespgemm_best.shape[0])

                nsparse_best = cur_df[cur_df['nsparse'] > cur_df['cusparse']]
                nsparse_best = nsparse_best[nsparse_best['nsparse'] > nsparse_best['kokkos']]
                nsparse_best = nsparse_best[nsparse_best['nsparse'] > nsparse_best['pixel_gflops']]
                nsparse_best = nsparse_best[nsparse_best['nsparse'] > nsparse_best['tilespgemm']]
                y_nsparse.append(nsparse_best.shape[0])
                
                total = y_cusparse[-1] + y_kokkos[-1] + y_pixel[-1] + y_tilespgemm[-1] + y_nsparse[-1]
                if total == 0:
                    total = 1
                y_cusparse[-1] = y_cusparse[-1] / total
                y_kokkos[-1] = y_kokkos[-1] / total
                y_pixel[-1] = y_pixel[-1] / total
                y_tilespgemm[-1] = y_tilespgemm[-1] / total
                y_nsparse[-1] = y_nsparse[-1] / total
                best_pixel.append(y_pixel[-1])
        
        # 画堆叠柱状图
        x = np.arange(ipb_max)
        width = 1
        y_cusparse = np.array(y_cusparse) * 100
        y_kokkos = np.array(y_kokkos) * 100
        y_pixel = np.array(y_pixel) * 100
        y_tilespgemm = np.array(y_tilespgemm) * 100
        y_nsparse = np.array(y_nsparse) * 100

        ax.bar(x, y_cusparse, width, label='cuSPARSE', color=colors[0])
        ax.bar(x, y_kokkos, width, label='Kokkos', color=colors[1], bottom=y_cusparse)
        ax.bar(x, y_nsparse, width, label='NSparse', color=colors[4], bottom=y_cusparse + y_kokkos)
        ax.bar(x, y_tilespgemm, width, label='TileSpGEMM', color=colors[3], bottom=y_cusparse + y_kokkos + y_nsparse)
        ax.bar(x, y_pixel, width, label='SparseCraft', color=colors[2], bottom=y_cusparse + y_kokkos + y_tilespgemm + y_nsparse)
        ax.set_xlim(0, 500)

        ax.set_ylabel(f'RTX {platform}', fontsize=13)
        ax.set_yticklabels(['0%', '50%', '100%'], fontsize=13)
        if _id != 2:
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels([int(x) for x in ax.get_xticks()], fontsize=13)

    axs[2].set_xlabel(f"Intermidiate products per block (IPB)", fontsize=13)
    legend = [
        plt.Rectangle((0,0), 0.5, 0.5, color=colors[0]),
        plt.Rectangle((0,0), 0.5, 0.5, color=colors[1]),
        plt.Rectangle((0,0), 0.5, 0.5, color=colors[4]),
        plt.Rectangle((0,0), 0.5, 0.5, color=colors[3]),
        plt.Rectangle((0,0), 0.5, 0.5, color=colors[2]),
    ]
    legend_labels = ["cuSPARSE", "Kokkos", "NSparse", "TileSpGEMM", "SparseCraft"]
    axs[0].legend(handles=legend, labels=legend_labels, ncol=len(legend), bbox_to_anchor=(0.5, 1.4), loc="center", handlelength=0.5, fontsize=13, columnspacing=0.6, handletextpad=0.4)
    plt.tight_layout()
    plt.savefig("img/fig9.png", dpi=300)


@app.command()
def draw_figure(fig_id: int):
    fig_id -= 6
    func_ls = [draw_predict, draw_spmv, draw_spmm, draw_spgemm, draw_representative]
    func_ls[fig_id]()


def main():
    """
    注册为全局命令时, 默认采用main函数作为命令入口, 请勿将此函数用作它途.
    When registering as a global command, default to main function as the command entry, do not use it as another way.
    """
    app()


if __name__ == "__main__":
    main()
