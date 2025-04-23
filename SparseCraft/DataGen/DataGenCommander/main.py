from QuickProject.Commander import Commander
from . import *
import glob
import os

app = Commander(name)


@app.command()
def compile(debug: bool = False):
    """
    🔨 编译
    """
    # * 在此函数的参数中添加你需要的define, bool类型参数将自动转换为-D <参数名>，其他类型参数将自动转换为-D <参数名>=<参数值>。因此推荐为bool类型参数设置默认值为False。
    # ! 不支持list类型参数
    defines = []
    cur_func_fig = app.fig_table[1]

    for item in cur_func_fig["args"]:
        _name = item["name"]
        defines.append(f"-D{name}='{eval(_name)}'")
    for item in cur_func_fig["options"]:
        _name = item["name"].strip("-")
        if "args" not in item and eval(_name):
            defines.append(f"-D{_name.upper()}")
        elif "args" in item:
            defines.append(f"-D{_name}='{eval(_name)}'")
    defines = " ".join(defines)

    os.chdir("dist")
    os.system(f'cmake -DDEFINITIONS="{defines}" ..')
    os.system("make -j")


@app.command()
def run(op: str, device: int = 0, batch: str = 'dist/distributions.txt'):
    """
    🏃 运行
    """
    external_exec(f"./dist/{name} {batch} {device} {op}", __expose=True)



@app.command()
def run_all_spmv(device: int = 0):
    """
    Quick run all mtx file
    """
    from QuickProject import rt_dir
    mtx_files = glob.glob(f"{rt_dir}/dist/*.mtx")
    if not mtx_files:
        print("No mtx files")
        return
    
    # 记录失败的矩阵
    failed_matrices = []
    mtx_files = ['2544_pdb1HYS.mtx', '2592_consph.mtx', '2534_cant.mtx', '2663_pwtk.mtx', '2449_rma10.mtx', '2409_conf5_4-8x8-05.mtx', '2621_shipsec1.mtx']
    for mtx_file in mtx_files[1:]:
        mtx_file = os.path.join(rt_dir, "dist", mtx_file)
        base_name = os.path.splitext(os.path.basename(mtx_file))[0]
        size = base_name.split('_')[0]
        
        # 创建输出目录
        output_dir = os.path.join(rt_dir, "gen/spmv", f"{size}_{base_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n处理文件: {base_name}")
        try:
            # 使用相对路径执行命令，保持在原目录
            st, ct = external_exec(f"./dist/{name}{device} {mtx_file} {device} spmv", __expose=True)
            
            # 从spmv文件夹获取txt文件
            spmv_dir = f"{rt_dir}/gen/spmv"
            spmv_txt_files = glob.glob(os.path.join(spmv_dir, "*.txt"))

            external_exec(f"python3 {rt_dir}/gen/spmv/classify_matrices.py", __expose=True)
            
            if spmv_txt_files:  # 如果有生成的文件，就复制
                for txt_file in spmv_txt_files:
                    src_path = txt_file
                    dst_path = os.path.join(output_dir, os.path.basename(txt_file))
                    import shutil
                    shutil.copy2(src_path, dst_path)
                print(f"Copied results to: {output_dir}")
            
            if st != 0:
                failed_matrices.append(base_name)
                print(f"警告: {base_name} 处理返回值为 {st}: {ct}")
        except Exception as e:
            failed_matrices.append(base_name)
            print(f"错误: 处理 {base_name} 时发生异常: {str(e)}")

    
    # 输出处理结果摘要
    print("\n========== 处理结果摘要 ==========")
    print(f"总矩阵数: {len(mtx_files)}")
    print(f"成功数量: {len(mtx_files) - len(failed_matrices)}")
    print(f"失败数量: {len(failed_matrices)}")
    
    if failed_matrices:
        print("\n失败的矩阵:")
        for matrix in failed_matrices:
            print(f"- {matrix}")


@app.command()
def run_all_spmm(device: int = 0):
    """
    🚀 快速运行所有mtx文件
    """
    # 获取所有mtx文件
    mtx_files = glob.glob("/home/lhc/CV-Tile/DataGen/dist/*.mtx")
    if not mtx_files:
        print("没有找到mtx文件")
        return
    
    # 记录失败的矩阵
    failed_matrices = []
    mtx_files = ['2592_consph.mtx', '2534_cant.mtx', '2663_pwtk.mtx', '2409_conf5_4-8x8-05.mtx', '2621_shipsec1.mtx', '2544_pdb1HYS.mtx', '2449_rma10.mtx']
    
    for mtx_file in mtx_files[1:]:
        mtx_file = os.path.join("/home/lhc/CV-Tile/DataGen/dist", mtx_file)
        base_name = os.path.splitext(os.path.basename(mtx_file))[0]
        size = base_name.split('_')[0]
        
        # 创建输出目录
        output_dir = os.path.join("/home/lhc/CV-Tile/DataGen/gen/spmm", f"{size}_{base_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n处理文件: {base_name}")
        try:
            # 使用相对路径执行命令，保持在原目录
            st, ct = external_exec(f"./dist/{name} {mtx_file} {device} spmm", __expose=True)
            
            # 从spmm文件夹获取txt文件
            spmm_dir = "/home/lhc/CV-Tile/DataGen/gen/spmm"
            spmm_txt_files = glob.glob(os.path.join(spmm_dir, "*.txt"))

            # external_exec(f"python3 /home/lhc/CV-Tile/DataGen/gen/spmm/classify_matrices.py", __expose=True)
            
            if spmm_txt_files:  # 如果有生成的文件，就复制
                for txt_file in spmm_txt_files:
                    src_path = txt_file
                    dst_path = os.path.join(output_dir, os.path.basename(txt_file))
                    import shutil
                    shutil.copy2(src_path, dst_path)
                print(f"已复制结果文件到: {output_dir}")
            
            if st != 0:
                failed_matrices.append(base_name)
                print(f"警告: {base_name} 处理返回值为 {st}: {ct}")
        except Exception as e:
            failed_matrices.append(base_name)
            print(f"错误: 处理 {base_name} 时发生异常: {str(e)}")

    
    # 输出处理结果摘要
    print("\n========== 处理结果摘要 ==========")
    print(f"总矩阵数: {len(mtx_files)}")
    print(f"成功数量: {len(mtx_files) - len(failed_matrices)}")
    print(f"失败数量: {len(failed_matrices)}")
    
    if failed_matrices:
        print("\n失败的矩阵:")
        for matrix in failed_matrices:
            print(f"- {matrix}")


@app.command()
def run_all_spgemm(device: int = 0):
    """
    🚀 快速运行所有mtx文件
    """
    # 获取所有mtx文件
    failed_matrices = []
    mtx_files = ['2592_consph.mtx', '2534_cant.mtx', '2663_pwtk.mtx', '2409_conf5_4-8x8-05.mtx', '2621_shipsec1.mtx', '2544_pdb1HYS.mtx', '2449_rma10.mtx']
    
    for mtx_file in mtx_files:
        # mtx_file = os.path.join("/home/lhc/CV-Tile/Trace/TraceGenSpGEMM/trace", mtx_file)
        mtx_file = os.path.join("/home/lhc/CV-Tile/DataGen/dist", mtx_file)
        base_name = os.path.splitext(os.path.basename(mtx_file))[0]
        size = base_name.split('_')[0]
        
        # 创建输出目录
        output_dir = os.path.join("/home/lhc/CV-Tile/DataGen/gen/spgemm", f"{size}_{base_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n处理文件: {base_name}")
        try:
            # 使用相对路径执行命令，保持在原目录
            st, ct = external_exec(f"./dist/{name} {mtx_file} {device} spgemmB", __expose=True)
            
            # 从spmm文件夹获取txt文件
            spgemm_dir = "/home/lhc/CV-Tile/DataGen/gen/spgemm"
            spgemm_txt_files = glob.glob(os.path.join(spgemm_dir, "*.txt"))
            
            if spgemm_txt_files:  # 如果有生成的文件，就复制
                for txt_file in spgemm_txt_files:
                    src_path = txt_file
                    dst_path = os.path.join(output_dir, os.path.basename(txt_file))
                    import shutil
                    shutil.copy2(src_path, dst_path)
                    os.remove(txt_file)
                print(f"已复制结果文件到: {output_dir}")
            
            if st != 0:
                failed_matrices.append(base_name)
                print(f"警告: {base_name} 处理返回值为 {st}: {ct}")
        except Exception as e:
            failed_matrices.append(base_name)
            print(f"错误: 处理 {base_name} 时发生异常: {str(e)}")

    
    # 输出处理结果摘要
    print("\n========== 处理结果摘要 ==========")
    print(f"总矩阵数: {len(mtx_files)}")
    print(f"成功数量: {len(mtx_files) - len(failed_matrices)}")
    print(f"失败数量: {len(failed_matrices)}")
    
    if failed_matrices:
        print("\n失败的矩阵:")
        for matrix in failed_matrices:
            print(f"- {matrix}")


@app.command()
def fmt_data(old_p: float, new_p: float):
    fmt_id = {
        'COO': 0,
        'CSR': 1,
        'ELL': 2,
        'HYB': 3,
        'DRW': 4,
        'DCL': 5,
        'DNS': 6,
    }
    # parse fmt_id by filepath
    import os
    dirs = os.listdir("./build/backup")
    for i in dirs:
        if '2240_2240_scircuit' in i:
            continue
        for fmt in fmt_id:
            filepath = f"./build/backup/{i}/{fmt}.txt"
            with open(filepath, 'r') as f:
                items = len(f.readlines())
            external_exec(f"./dist/{name} {filepath} {fmt_id[fmt]} {items} format-data", __expose=True)


def main():
    """
    * 注册为全局命令时, 默认采用main函数作为命令入口, 请勿将此函数用作它途.
    * When registering as a global command, default to main function as the command entry, do not use it as another way.
    """
    app()


if __name__ == "__main__":
    main()
