import numpy as np

def generate_high_density_matrix():
    # 生成一个随机矩阵，1的概率为0.95（确保大于90%的非零元含量）
    matrix = np.random.choice([0, 1], size=256, p=[0.05, 0.95])
    return ''.join(map(str, matrix))

# 生成6000个矩阵
with open('/home/lhc/CV-Tile/DataGen/dist/distributions_dns.txt', 'w') as f:
    # 首先写入矩阵数量
    f.write('6000\n')
    
    # 生成每个矩阵并写入文件
    for _ in range(6000):
        matrix = generate_high_density_matrix()
        f.write(matrix + '\n')