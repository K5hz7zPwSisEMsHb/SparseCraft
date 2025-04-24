from QuickProject.Commander import Commander
from . import *

app = Commander(executable_name)


@app.command()
def spmv():
    import pandas

    with open('../../../.CURRENT_PLATFORM', 'r') as f:
        info = f.read().strip()
        current_platform, device = info.split()

    csv_path = f"dist/{current_platform}-spmv-tilespmv.csv"
    ls = [i for i in os.listdir(mtx_dir_path) if i.endswith(".mtx") and not i.startswith('_')]
    if os.path.exists(csv_path):
        odf = pandas.read_csv(csv_path)
        have_mtx = set(odf["mtx"].to_list())
        ls = [i for i in ls if i not in have_mtx]
        print(len(ls))
    else:
        odf = pandas.DataFrame(
            columns=[
                "index",
                "mtx",
                'cvt_time',
                "gflops",
            ]
        )
    
    from .framework import start_framework

    start_framework("spmv", ls, odf, csv_path, '-d', device)


@app.command()
def samples():
    import pandas

    with open('../../../.CURRENT_PLATFORM', 'r') as f:
        info = f.read().strip()
        current_platform, device = info.split()

    csv_path = f"dist/{current_platform}-spmv-tilespmv-representative.csv"
    ls = ['2430_mc2depi.mtx', '2453_pkustk07.mtx', '2305_msc10848.mtx', '2449_rma10.mtx', '2479_ramage02.mtx', '2412_opt1.mtx', '2420_TSC_OPF_1047.mtx', '2413_trdheim.mtx', '2148_heart3.mtx', '2200_nemeth19.mtx', '2340_raefsky3.mtx', '2634_TSOPF_RS_b678_c2.mtx']

    if os.path.exists(csv_path):
        odf = pandas.read_csv(csv_path)
        have_mtx = set(odf["mtx"].to_list())
        ls = [i for i in ls if i not in have_mtx]
        print(len(ls))
    else:
        odf = pandas.DataFrame(
            columns=[
                "index",
                "mtx",
                'cvt_time',
                "gflops",
            ]
        )
    
    from .framework import start_framework

    start_framework("spmv", ls, odf, csv_path, '-d', device)


def main():
    """
    注册为全局命令时, 默认采用main函数作为命令入口, 请勿将此函数用作它途.
    When registering as a global command, default to main function as the command entry, do not use it as another way.
    """
    app()


if __name__ == "__main__":
    main()
