import tempfile
import re
import json
import subprocess
from datetime import datetime
from QuickProject.Commander import Commander
from . import *
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from rich.table import Table


app = Commander(name)


@app.command()
def compile(debug: bool = False):
    """
    🔨 Compile
    """
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
    if os.path.exists("CMakeCache.txt"):
        os.remove("CMakeCache.txt")
    if os.path.exists("CMakeFiles"):
        import shutil
        shutil.rmtree("CMakeFiles")
        
    st, ct = external_exec(f'cmake -DDEFINITIONS="{defines}" ..', without_output=True)
    if st:
        QproDefaultConsole.print(ct)
        return
    st, ct = external_exec("make -j", without_output=True)
    if st:
        QproDefaultConsole.print(ct)
        return
    os.chdir("..")


@app.command()
def compile_test_only(debug: bool = False):
    """
    🔨 Compile Test Program Only
    """
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
    if os.path.exists("CMakeCache.txt"):
        os.remove("CMakeCache.txt")
    if os.path.exists("CMakeFiles"):
        import shutil
        shutil.rmtree("CMakeFiles")
        
    st, ct = external_exec(f'cmake -DDEFINITIONS="{defines} -DTEST_ONLY=1" ..', without_output=True)
    if st:
        QproDefaultConsole.print(ct)
        return
    
    st, ct = external_exec("make -j test_only", without_output=True)
    if st:
        QproDefaultConsole.print(ct)
        return
    os.chdir("..")
    QproDefaultConsole.print("Test program compilation completed")


@app.command()
def run(platform: str):
    """
    🏃 Run Training

    :param platform: GPU Platform (3090/4090/5090)
    """
    if platform not in ['3090', '4090', '5090']:
        QproDefaultConsole.print(f"Error: Unsupported platform {platform}, please use 3090/4090/5090")
        return
    
    base_path = f"data/{platform}"
    train_path = f"{base_path}/train"
    test_path = f"{base_path}/test"
    model_path = f"model/{platform}_more/model.bin"
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    if not os.path.exists(train_path):
        QproDefaultConsole.print(f"Error: Training data path does not exist: {train_path}")
        return
    
    if not os.path.exists(test_path):
        QproDefaultConsole.print(f"Error: Test data path does not exist: {test_path}")
        return
    
    QproDefaultConsole.print(f"Starting training for {platform} platform")
    QproDefaultConsole.print(f"Training set: {train_path}")
    QproDefaultConsole.print(f"Test set: {test_path}")
    QproDefaultConsole.print(f"Model output: {model_path}")
    
    cmd = f"./dist/{name} {train_path} {model_path} {test_path}"
    QproDefaultConsole.print(f"Executing command: {cmd}")
    external_exec(cmd, __expose=True)


@app.command()
def test(platform: str, model_path: str):
    """
    🔍 Test Model

    :param platform: Platform (3090/4090/5090)
    :param model_path: Model file path
    """
    if platform not in ['3090', '4090', '5090']:
        QproDefaultConsole.print(f"Error: Unsupported platform {platform}, please use 3090/4090/5090")
        return
    
    test_data = f"data/{platform}/test"
    
    if not os.path.exists(model_path):
        QproDefaultConsole.print(f"Error: Model file does not exist: {model_path}")
        return
    
    if not os.path.exists(test_data):
        QproDefaultConsole.print(f"Error: Test dataset does not exist: {test_data}")
        return
    
    QproDefaultConsole.print(f"Starting test for {platform} platform model")
    QproDefaultConsole.print(f"Model file: {model_path}")
    QproDefaultConsole.print(f"Test set: {test_data}")
    
    cmd = f"./dist/test_only {model_path} {test_data}"
    
    QproDefaultConsole.print(f"Executing command: {cmd}")
    st, ct = external_exec(cmd, __expose=True)
    if st:
        QproDefaultConsole.print("Test failed")
        return
    QproDefaultConsole.print("Test completed")


def _run_single_test(test_args):
    test_platform, model_path, test_data = test_args
    cmd = f"./dist/test_only {model_path} {test_data}"
    st, ct = external_exec(cmd, __expose=True)
    
    # If test succeeded and has output
    if not st and ct:
        # Extract accuracy
        match = re.search(r"accuracy: ([\d.]+)%", ct)
        if match:
            accuracy = match.group(1)
            model_name = os.path.basename(model_path)
            source_platform = model_name.split('_')[0]
            model_type = "fine-tuned" if "fine_model" in model_name else "native"
            
            # Build explanation text
            if model_type == "fine-tuned":
                QproDefaultConsole.print(
                    f"[Explanation] This is the test result of {source_platform} platform model fine-tuned for {test_platform} platform, tested on {test_platform}'s test set"
                )
            else:
                QproDefaultConsole.print(
                    f"[Explanation] This is the test result of {source_platform} platform's native model tested on {test_platform}'s test set"
                )
    
    return st, ct

@app.command()
def test_all():
    """
    🔍 Test All Models on All Platforms (Parallel Execution)
    """
    platforms = ['3090', '4090', '5090']
    phase1_tasks = []
    phase2_tasks = []
    
    # Phase 1: Native models on all platforms (9 tests)
    QproDefaultConsole.print("\nPhase 1: Testing native models across all platforms")
    for model_platform in platforms:
        model_dir = f"model/{model_platform}"
        native_model = f"{model_dir}/{model_platform}_model.bin"
        if os.path.exists(native_model):
            for test_platform in platforms:
                test_data = f"data/{test_platform}/test"
                if os.path.exists(test_data):
                    phase1_tasks.append((test_platform, native_model, test_data))
    
    # Execute Phase 1
    QproDefaultConsole.print(f"\nExecuting Phase 1: {len(phase1_tasks)} cross-platform tests")
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        phase1_results = list(executor.map(_run_single_test, phase1_tasks))
    
    # 显示 Phase 1 的 3x3 矩阵结果
    phase1_table = Table(title="Phase 1 - Native Models Cross-platform Performance")
    phase1_table.add_column("Model↓ Test→", style="bold")
    for platform in platforms:
        phase1_table.add_column(platform, style="cyan")
    
    # 构建并填充 Phase 1 矩阵
    phase1_results_matrix = {model: {test: "N/A" for test in platforms} for model in platforms}
    for i, (test_platform, model_path, _) in enumerate(phase1_tasks):
        status, output = phase1_results[i]
        model_platform = os.path.basename(model_path).split('_')[0]
        
        accuracy = "N/A"
        if not status and output:
            match = re.search(r"accuracy: ([\d.]+)%", output)
            if match:
                accuracy = f"{match.group(1)}%"
        phase1_results_matrix[model_platform][test_platform] = accuracy
    
    # 输出 Phase 1 矩阵
    for model_platform in platforms:
        row = [model_platform]
        for test_platform in platforms:
            row.append(phase1_results_matrix[model_platform][test_platform])
        phase1_table.add_row(*row)
    
    QproDefaultConsole.print("\nPhase 1 Summary:")
    QproDefaultConsole.print(phase1_table)
    
    # Phase 2: Testing fine-tuned models on their target platforms
    QproDefaultConsole.print("\nPhase 2: Testing fine-tuned models on target platforms")
    for platform in platforms:
        test_data = f"data/{platform}/test"
        if not os.path.exists(test_data):
            continue
        
        # 测试该平台文件夹下的所有模型（原生模型和两个微调模型）
        model_dir = f"model/{platform}"
        if os.path.exists(model_dir):
            # 测试原生模型
            native_model = f"{model_dir}/{platform}_model.bin"
            if os.path.exists(native_model):
                phase2_tasks.append((platform, native_model, test_data))
            
            # 测试该文件夹下的两个微调模型
            for other_platform in platforms:
                if other_platform != platform:
                    fine_model = f"{model_dir}/{other_platform}_fine_model.bin"
                    if os.path.exists(fine_model):
                        phase2_tasks.append((platform, fine_model, test_data))
    
    # Execute Phase 2
    QproDefaultConsole.print(f"\nExecuting Phase 2: {len(phase2_tasks)} fine-tuning tests")
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        phase2_results = list(executor.map(_run_single_test, phase2_tasks))
    
    # 创建最终总结表格
    final_phase1_table = Table(title="Final Summary - Phase 1: Native Models Cross-platform Performance")
    final_phase1_table.add_column("Model↓ Test→", style="bold")
    for platform in platforms:
        final_phase1_table.add_column(platform, style="cyan")
    
    final_phase2_table = Table(title="Final Summary - Phase 2: Fine-tuned Models Performance")
    final_phase2_table.add_column("Model↓ Test→", style="bold")
    for platform in platforms:
        final_phase2_table.add_column(platform, style="cyan")
    
    # 复用之前的 Phase 1 结果
    for model_platform in platforms:
        row = [model_platform]
        for test_platform in platforms:
            row.append(phase1_results_matrix[model_platform][test_platform])
        final_phase1_table.add_row(*row)
    
    # 填充 Phase 2 表格
    phase2_results_matrix = {model: {test: "N/A" for test in platforms} for model in platforms}
    for i, (test_platform, model_path, _) in enumerate(phase2_tasks):
        status, output = phase2_results[i]
        model_name = os.path.basename(model_path)
        source_platform = model_name.split('_')[0]
        
        accuracy = "N/A"
        if not status and output:
            match = re.search(r"accuracy: ([\d.]+)%", output)
            if match:
                accuracy = f"{match.group(1)}%"
        if "fine_model" in model_name:
            accuracy = f"[blue]{accuracy}[/blue]"
        phase2_results_matrix[source_platform][test_platform] = accuracy
    
    # 输出 Phase 2 矩阵
    for source_platform in platforms:
        row = [source_platform]
        for test_platform in platforms:
            row.append(phase2_results_matrix[source_platform][test_platform])
        final_phase2_table.add_row(*row)
    
    # 显示最终总结
    QproDefaultConsole.print("\nFinal Summary:")
    QproDefaultConsole.print(final_phase1_table)
    QproDefaultConsole.print("\n")
    QproDefaultConsole.print(final_phase2_table)
    
    # 添加图例说明
    QproDefaultConsole.print("\n[bold]Legend:[/bold]")
    QproDefaultConsole.print("• Normal text: Native model performance")
    QproDefaultConsole.print("• [blue]Blue text[/blue]: Fine-tuned model performance")
    QproDefaultConsole.print("• Rows: Source model platform")
    QproDefaultConsole.print("• Columns: Test platform")


def main():
    app()


if __name__ == "__main__":
    main()
