from QuickProject.Commander import Commander
from . import *

app = Commander(executable_name)


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
                {"type": "list", "message": "选择矩阵", "choices": ct, "default": ct[0]}
            )
        else:
            mm = ct[0]
    os.chdir(cwd)
    return mm


@app.custom_complete("platform")
def compile():
    return ["4090", "3090"]


@app.command()
def compile(platform: str = "4090"):
    """
    support platform: 4090, 3090
    """
    os.chdir("build")
    os.system(f"cmake -DPLATFORM={platform} ..")
    os.system("make -j")
    os.chdir("..")


@app.custom_complete("opt")
def mrun():
    return ["spmv", "spmm", "spgemm"]


@app.command()
def mrun(device_id: int, mname: str, opt: str):
    """
    run the model
    """
    right_n = 8
    mm = find_and_select_mtx(mname)
    if mm:
        if opt.startswith('spmm'):
            external_exec(f"./build/cusparse_test {mtx_dir_path}/{mm} {right_n} {device_id} --{opt}")
        else:
            external_exec(f"./build/cusparse_test {mtx_dir_path}/{mm} {device_id} --{opt}")


@app.custom_complete("opt")
def test():
    return ["spmv", "spmm", "spgemm"]


@app.command()
def test(opt: str):
    """
    run spmv
    """
    import pandas as pd

    with open('../../.CURRENT_PLATFORM', 'r') as f:
        info = f.read().strip()
        current_platform, device_id = info.split()

    csv_path = f"dist/{current_platform}-{opt}-cusparse.csv"
    ls = None
    if opt == 'spgemm':
        ls = pd.read_csv('../../SparseCraft/BLAS/dist/spgemm-task.csv')['mtx'].tolist()
        mtx_ls = [i for i in os.listdir(mtx_dir_path) if i.endswith(".mtx") and not i.startswith('_')]
        ls = [i for i in ls if i in mtx_ls]
    else:
        ls = [i for i in os.listdir(mtx_dir_path) if i.endswith(".mtx") and not i.startswith('_')]
    
    if os.path.exists(csv_path):
        odf = pd.read_csv(csv_path)
        have_mtx = set(odf["mtx"].to_list())
        ls = [i for i in ls if i not in have_mtx]
    else:
        odf = pd.DataFrame(columns=["index", "mtx", "cvt_time", "gflops"])

    from .framework import start_framework
    
    if opt == 'spmm':
        start_framework("cusparse_test", ls, odf, csv_path, 8, device_id, f"--{opt}")
    else:
        start_framework("cusparse_test", ls, odf, csv_path, device_id, f"--{opt}")


@app.command()
def samples(opt: str):
    import pandas as pd

    with open('../../.CURRENT_PLATFORM', 'r') as f:
        info = f.read().strip()
        current_platform, device_id = info.split()

    csv_path = f"dist/{current_platform}-{opt}-cusparse-representative.csv"
    ls = ['2430_mc2depi.mtx', '2453_pkustk07.mtx', '2305_msc10848.mtx', '2449_rma10.mtx', '2479_ramage02.mtx', '2412_opt1.mtx', '2420_TSC_OPF_1047.mtx', '2413_trdheim.mtx', '2148_heart3.mtx', '2200_nemeth19.mtx', '2340_raefsky3.mtx', '2634_TSOPF_RS_b678_c2.mtx']
    
    right_n = 8
    if os.path.exists(csv_path):
        odf = pd.read_csv(csv_path)
        have_mtx = set(odf["mtx"].to_list())
        ls = [i for i in ls if i not in have_mtx]
    else:
        odf = pd.DataFrame(columns=["index", "mtx", "cvt_time", "gflops"])

    from .framework import start_framework

    if opt == 'spmm':
        start_framework("cusparse_test", ls, odf, csv_path, right_n, device_id, f"--{opt}")
    else:
        start_framework("cusparse_test", ls, odf, csv_path, device_id, f"--{opt}")


def main():
    """
    When registering as a global command, default to main function as the command entry, do not use it as another way.
    """
    app()


if __name__ == "__main__":
    main()
