import os

def classify_matrices():
    # 获取目录下所有格式文件
    input_dir = '.'
    format_files = {}
    
    # 从目录中读取所有txt文件作为格式文件
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.txt') and not file_name.endswith('_best.txt'):
            format_name = file_name[:-4]  # 移除.txt后缀
            format_files[format_name] = []
    
    # 读取所有格式文件中的数据
    format_data = {}
    for format_name in format_files.keys():
        file_path = os.path.join(input_dir, f'{format_name}.txt')
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                format_data[format_name] = {}
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        matrix = parts[0]
                        speed = float(parts[2])
                        format_data[format_name][matrix] = (speed, line)
    
    # 对每个矩阵找到最快的格式
    all_matrices = set()
    for data in format_data.values():
        all_matrices.update(data.keys())
    
    for matrix in all_matrices:
        best_format = None
        best_speed = -1
        best_line = None
        
        # 在所有格式中找到最快的
        for format_name, data in format_data.items():
            if matrix in data:
                speed, line = data[matrix]
                if speed > best_speed:
                    best_speed = speed
                    best_format = format_name
                    best_line = line
        
        if best_format:
            format_files[best_format].append(best_line)
    
    # 将矩阵写入对应的最佳格式文件
    for format_name, matrices in format_files.items():
        if matrices:
            output_file = os.path.join(input_dir, f'{format_name}.txt')
            with open(output_file, 'w') as f:
                f.writelines(matrices)
            print(f'已写入 {len(matrices)} 个矩阵到 {format_name}_best.txt')

if __name__ == '__main__':
    classify_matrices()
