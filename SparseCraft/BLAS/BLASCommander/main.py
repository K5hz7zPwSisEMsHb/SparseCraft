from QuickProject.Commander import Commander
from . import *


app = Commander(name)


@app.command()
def compile(
    debug: bool = False,
    check_result: bool = False,
    no_perf: bool = False,
):
    """
    compile
    """
    defines = []
    cur_func_fig = app.fig_table[1]

    for item in cur_func_fig["args"]:
        _name = item["name"]
        defines.append(f"-D{name}='{eval(_name)}'")
    for item in cur_func_fig["options"]:
        _name = item["name"].strip("-").replace("-", "_")
        if "args" not in item and eval(_name):
            defines.append(f"-D{_name.upper()}")
        elif "args" in item:
            defines.append(f"-D{_name}='{eval(_name)}'")
    defines = " ".join(defines)

    os.chdir("dist")
    print(f'cmake -DDEFINITIONS="{defines}" ..')
    st, ct = external_exec(f'cmake -DDEFINITIONS="{defines}" ..', without_output=True)
    if st:
        QproDefaultConsole.print(ct)
        return
    os.system("make -j")
    os.chdir("..")


def fmt_distribution(distribution: str):
    distribution = int(distribution, 16)
    distribution = bin(distribution)[2:]
    distribution = distribution[::-1][:256][::-1].zfill(256)
    return distribution


def find_and_select_mtx(mname):
    cwd = os.getcwd()
    os.chdir(mtx_dir_path)
    st, ct = external_exec(f"fd {mname}", without_output=True)
    mm = None
    if not st:
        ct = [i for i in ct.strip().split() if i.endswith(".mtx")]
        if len(ct) > 1:
            from . import _ask

            mm = _ask(
                {"type": "list", "message": "Select Matrix", "choices": ct, "default": ct[0]}
            )
        else:
            mm = ct[0]
    os.chdir(cwd)
    return mm


@app.command()
def mrun(mname: str, op: str, repeat: int = 1000, device: int = 0, right_n: int = 8):
    """
    run BLAS through matrix name

    :param mname: matrix name
    """
    mm = find_and_select_mtx(mname)
    if mm:
        external_exec(
            f"./dist/{name} {mtx_dir_path}/{mm} {repeat} {right_n} {device} {op}",
            __expose=True,
        )


@app.command()
def spmv():
    """
    run all spmv
    """
    import pandas

    with open('../../.CURRENT_PLATFORM', 'r') as f:
        info = f.read().strip()
        current_platform, device = info.split()

    csv_path = f"dist/{current_platform}-spmv-sparsecraft.csv"

    ls = [i for i in os.listdir(mtx_dir_path) if i.endswith(".mtx") and not i.startswith('_')]
    if os.path.exists(csv_path):
        odf = pandas.read_csv(csv_path)
        have_mtx = set(odf["mtx"].to_list())
        ls = [i for i in ls if i not in have_mtx]
    else:
        odf = pandas.DataFrame(
            columns=[
                "index",
                "mtx",
                "m",
                "n",
                "nnz",
                "cvt_time",
                "pixel_time",
                "pixel_gflops",
            ]
        )
    
    app.real_call('switch_model', 'spmm', current_platform)
    from .framework import start_framework
    start_framework("BLAS", ls, odf, csv_path, 1000, device, 'spmv')


@app.command()
def spmm(right_n: int = 8, device: int = 0):
    """
    run all spmm
    """
    import pandas

    with open('../../.CURRENT_PLATFORM', 'r') as f:
        info = f.read().strip()
        current_platform, device = info.split()

    csv_path = f"dist/{current_platform}-spmm-sparsecraft.csv"
    ls = pandas.read_csv('dist/odf/spmm-256.csv')["mtx"].to_list()
    if os.path.exists(csv_path):
        odf = pandas.read_csv(csv_path)
        have_mtx = set(odf["mtx"].to_list())
        ls = [i for i in ls if i not in have_mtx and 'mawi' not in i and 'kron_g' not in i]
    else:
        odf = pandas.DataFrame(
            columns=[
                "index",
                "mtx",
                "m",
                "n",
                "nnz",
                "right-n",
                "pixel_gflops",
            ]
        )

    app.real_call('switch_model', 'spmm', current_platform)
    from .framework import start_framework
    start_framework("BLAS", ls, odf, csv_path, 1000, right_n, device, "spmm")


@app.command()
def spgemm(device: int = 0):
    """
    📚 spgemm
    """
    import pandas

    with open('/root/.CURRENT_PLATFORM', 'r') as f:
        info = f.read().strip()
        current_platform, device = info.split()

    csv_path = f"dist/{current_platform}-spgemm-sparsecraft.csv"
    ls = pandas.read_csv('dist/spgemm-task.csv')['mtx'].to_list()
    mtx_ls = [i for i in os.listdir(mtx_dir_path) if i.endswith(".mtx") and not i.startswith('_')]
    ls = [i for i in ls if i in mtx_ls]

    if os.path.exists(csv_path):
        odf = pandas.read_csv(csv_path)
        have_mtx = set(odf["mtx"].to_list())
        ls = [i for i in ls if i not in have_mtx and 'mawi' not in i and 'kron_g' not in i]
        print(len(ls))
    else:
        odf = pandas.DataFrame(
            columns=[
                "index",
                "mtx",
                "pixel_gflops",
            ]
        )
    
    app.real_call('switch_model', 'spmm', current_platform)
    from .framework import start_framework
    start_framework("BLAS", ls, odf, csv_path, 100, device, 'spgemm')


@app.command()
def samples():
    """
    run samples
    """
    with open('../../.CURRENT_PLATFORM', 'r') as f:
        info = f.read().strip()
        current_platform, device = info.split()
    mtx = ['2430_mc2depi.mtx', '2453_pkustk07.mtx', '2305_msc10848.mtx', '2449_rma10.mtx', '2479_ramage02.mtx', '2412_opt1.mtx', '2420_TSC_OPF_1047.mtx', '2413_trdheim.mtx', '2148_heart3.mtx', '2200_nemeth19.mtx', '2340_raefsky3.mtx', '2634_TSOPF_RS_b678_c2.mtx']
    repeat = {
        'spmv': 1000,
        'spmm': 1000,
        'spgemm': 100
    }

    from QuickProject import QproDefaultStatus
    
    QproDefaultStatus.start()
    for op in ['spmv', 'spmm', 'spgemm']:
        app.real_call('switch_model', {
            'spmv': 'spmv',
            'spmm': 'spmm',
            'spgemm': 'spmm'
        }[op], current_platform)
        perf_ls = []
        right_n = '8' if op == 'spmm' else ''
        for mname in mtx:
            QproDefaultStatus(f"SparseCraft: {op}, {mname}")
            st, ct = external_exec(f"./dist/{name} ../Matrices/{mname} {repeat[op]} {right_n} {device} {op}", without_output=True)
            if st:
                print(ct)
                return
            perf = ct.strip().split('\n')[-1]
            perf_ls.append(perf)
        with open(f'dist/{current_platform}-{op}-sparsecraft-representative.csv', 'w') as f:
            if op in ['spmm', 'spgemm']:
                f.write('mtx,gflops\n')
                for i, mname in enumerate(mtx):
                    f.write(f'{mname},{perf_ls[i].split(",")[-1]}\n')
            else:
                f.write('mtx,cvt_time,gflops\n')
                for i, mname in enumerate(mtx):
                    _ls = perf_ls[i].split(",")
                    f.write(f'{mname},{_ls[-3]},{_ls[-1]}\n')
    QproDefaultStatus.stop()


@app.command()
def switch_model(op: str, platform: str = "4090"):
    """
    🔄 切换模型
    """
    from QuickProject import rt_dir

    os.chdir(rt_dir)
    # find all .bin files
    os.system(f"ln -snf \"{os.path.join(rt_dir, 'model/bin', platform, f'{op}.bin')}\" {os.path.join(rt_dir, 'model/slime_net.bin')}")


def main():
    """
    * 注册为全局命令时, 默认采用main函数作为命令入口, 请勿将此函数用作它途.
    * When registering as a global command, default to main function as the command entry, do not use it as another way.
    """
    app()


if __name__ == "__main__":
    main()
