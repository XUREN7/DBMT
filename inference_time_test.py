import numpy as np
import torch
from torch.backends import cudnn
import tqdm
cudnn.benchmark = True
from models.DBMT import DBMT


model = DBMT().to('cuda')
repetitions = 300
dummy_input = torch.rand(1, 3, 256, 256).to('cuda')
print('warm up ...\n')
with torch.no_grad():
    for _ in range(100):
        _ = model(dummy_input)

# synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
torch.cuda.synchronize()

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
timings = np.zeros((repetitions, 1))

print('testing ...\n')
with torch.no_grad():
    for rep in tqdm.tqdm(range(repetitions)):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        torch.cuda.synchronize() # 等待GPU任务完成
        curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒32
        timings[rep] = curr_time

avg = timings.sum()/repetitions
print('\navg={}\n'.format(avg))