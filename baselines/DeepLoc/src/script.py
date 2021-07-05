import os
import sys
from itertools import product


project = sys.argv[1]
start = 0
end = 1000
if len(sys.argv) == 4:
    start = int(sys.argv[2])
    end = int(sys.argv[3])
assert project in ('birt', 'eclipse', 'jdt', 'swt', 'tomcat')
num_neg = 200

project_log_dir = f'/home/LAB/qibh/Documents/BugLocalizationResearch/DreamLoc_for_github/DeepLoc/data/{project}/log/neg_{num_neg}'
batch_size = [32, 64, 128]
learning_rate = [0.01, 0.001]
n_kernel = [50, 100]
epoch = [100]
early_stop = 30

param = list(product(batch_size, learning_rate, n_kernel, epoch))[start:end]
for idx, (bz, lr, nk, ep) in enumerate(param):
    cmd = f'python -u deep_loc.py --project {project} --lr {lr} --batch_size {bz} '\
          f'--epochs {ep} --n_kernels {nk} --num_neg {num_neg} --early_stop {early_stop} > {project_log_dir}/{bz}_{lr}_{ep}_{nk}.log'
    print(f'{idx + 1}/{len(param)}:')
    print(cmd)
    os.system(cmd)
