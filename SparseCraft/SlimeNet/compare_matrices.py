import os
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm  # 添加进度条库

def read_matrix_file(file_path):
    """使用生成器读取矩阵文件，减少内存使用"""
    with open(file_path, 'r') as f:
        for line in f:
            # 只取空格前的字符串（第一列），并确保只取前256位
            matrix_str = line.split()[0][:256]
            yield np.array(list(matrix_str), dtype=np.int8)  # 使用int8节省内存

def write_matrix_file(file_path, matrices):
    """批量写入文件"""
    with open(file_path, 'w', buffering=8192*1024) as f:  # 增加缓冲区大小
        f.writelines(''.join(map(str, matrix)) + '\n' for matrix in matrices)

def redistribute_csr_matrices(base_dir):
    folders = {
        "3090": os.path.join(base_dir, "train3090"),
        "4090": os.path.join(base_dir, "train4090"),
        "5090": os.path.join(base_dir, "train5090")
    }
    
    # 使用字典存储矩阵哈希值和对应的格式信息
    matrix_formats = defaultdict(dict)
    print("读取所有格式的矩阵...")
    
    # 读取所有平台的所有格式文件
    all_formats = ['CSR.txt', 'COO.txt', 'DCL.txt', 'DNS.txt', 'DRW.txt', 'ELL.txt', 'HYB.txt']
    for platform, folder in folders.items():
        print(f"\n处理 {platform} 平台的文件...")
        for fmt in all_formats:
            file_path = os.path.join(folder, fmt)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    total_lines = sum(1 for _ in f)
                
                for matrix in tqdm(read_matrix_file(file_path), 
                                 total=total_lines, 
                                 desc=f"读取 {fmt}"):
                    matrix_hash = hash(matrix.tobytes())
                    if matrix_hash not in matrix_formats:
                        matrix_formats[matrix_hash] = {'matrix': matrix}
                    if platform not in matrix_formats[matrix_hash]:
                        matrix_formats[matrix_hash][platform] = []
                    matrix_formats[matrix_hash][platform].append(fmt)
    
    # 找出在三个平台上都是CSR格式的矩阵
    common_csr_matrices = []
    for matrix_hash, info in matrix_formats.items():
        if all(platform in info and info[platform] == ['CSR.txt'] 
               for platform in folders.keys()):
            common_csr_matrices.append(info['matrix'])
    
    print(f"\n找到 {len(common_csr_matrices)} 个在三个平台上都是CSR格式的矩阵")
    
    # 随机选择35405个矩阵删除
    if len(common_csr_matrices) >= 25405:
        matrices_to_remove = np.array(random.sample(common_csr_matrices, 25405))
        matrices_to_remove_set = set(map(lambda x: hash(x.tobytes()), matrices_to_remove))
        
        # 更新CSR文件
        print("\n更新CSR文件...")
        for platform, folder in folders.items():
            csr_file = os.path.join(folder, 'CSR.txt')
            print(f"处理 {platform} 的CSR文件...")
            current = [matrix for matrix in tqdm(read_matrix_file(csr_file),
                                               desc=f"过滤 {platform} 矩阵")
                      if hash(matrix.tobytes()) not in matrices_to_remove_set]
            write_matrix_file(csr_file, current)
            print(f"完成更新 {platform} 的CSR文件，删除了 35405 个矩阵")
        
        print("\n所有平台处理完成！")
    else:
        print(f"\n错误：找到的共同CSR矩阵数量（{len(common_csr_matrices)}）小于需要删除的数量（35405）")

if __name__ == "__main__":
    base_dir = "/home/rhythmlian/slime_s/data"
    redistribute_csr_matrices(base_dir)