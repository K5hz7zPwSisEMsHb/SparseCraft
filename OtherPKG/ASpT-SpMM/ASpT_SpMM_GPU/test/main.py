from QuickProject.Commander import Commander
from . import *

app = Commander(executable_name)


@app.command()
def spmm():
    """
    📚 spmv
    """
    import pandas

    with open('/root/.CURRENT_PLATFORM', 'r') as f:
        info = f.read().strip()
        current_platform, device = info.split()

    csv_path = f"{current_platform}-spmm-aspt.csv"
    ls = [i for i in os.listdir(mtx_dir_path) if i.endswith('.mtx')]

    for mtx in ls:
        print(f'./dspmm_32 {mtx_dir_path}/{mtx} 8')
        os.system(f'export CUDA_VISIBLE_DEVICES={device} && ./dspmm_32 {mtx_dir_path}/{mtx} 8')
    
    df = pandas.DataFrame(columns=["mtx", 'gflops'])
    with open('ASpT-results-double-32.csv', 'r') as f:
        for _id, line in enumerate(f.readlines()):
            df.loc[df.shape[0]] = [ls[_id], line.strip().split(',')[-1]]

    df.to_csv(csv_path)
    os.remove('ASpT-results-double-32.csv')


@app.command()
def samples():
    """
    📚 spmv
    """
    import pandas

    with open('/root/.CURRENT_PLATFORM', 'r') as f:
        info = f.read().strip()
        current_platform, device = info.split()

    csv_path = f"{current_platform}-spmm-aspt-representative.csv"
    ls = ['2430_mc2depi.mtx', '2453_pkustk07.mtx', '2305_msc10848.mtx', '2449_rma10.mtx', '2479_ramage02.mtx', '2412_opt1.mtx', '2420_TSC_OPF_1047.mtx', '2413_trdheim.mtx', '2148_heart3.mtx', '2200_nemeth19.mtx', '2340_raefsky3.mtx', '2634_TSOPF_RS_b678_c2.mtx']
    
    # set CUDA_VIS

    for mtx in ls:
        print(f'./dspmm_32 {mtx_dir_path}/{mtx} 8')
        os.system(f'export CUDA_VISIBLE_DEVICES={device} && ./dspmm_32 {mtx_dir_path}/{mtx} 8')
    
    df = pandas.DataFrame(columns=["mtx", 'gflops'])
    with open('ASpT-results-double-32.csv', 'r') as f:
        for _id, line in enumerate(f.readlines()):
            df.loc[df.shape[0]] = [ls[_id], line.strip().split(',')[-1]]

    df.to_csv(csv_path)
    os.remove('ASpT-results-double-32.csv')

def main():
    """
    注册为全局命令时, 默认采用main函数作为命令入口, 请勿将此函数用作它途.
    When registering as a global command, default to main function as the command entry, do not use it as another way.
    """
    app()


if __name__ == "__main__":
    main()
