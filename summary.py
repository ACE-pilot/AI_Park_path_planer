from parl.utils import summary
import numpy as np

for i in range(10):
    x = np.random.random(1000)
    summary.add_histogram('distribution centers', x + i, i)

"""tensorboard --logdir=./train_log"""