import jittor as jt
import numpy as np
from jittor import nn, Module, init

class Model(Module):
    def __init__(self):
        self.layer1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(10, 1)
        import pdb; pdb.set_trace()
        
    def execute (self,x) :
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

np.random.seed(0)
jt.set_seed(3)
n = 1000
batch_size = 50

def get_data(n):
    for i in range(n):
        x = np.random.rand(batch_size, 1)
        y = x*x
        yield jt.float32(x), jt.float32(y)

model = Model()
learning_rate = 0.1
optim = nn.SGD(model.parameters(), learning_rate)

for i,(x,y) in enumerate(get_data(n)):
    pred_y = model(x)
    loss = jt.sqr(pred_y - y)
    loss_mean = loss.mean()
    optim.step(loss_mean)
    print(f"step {i}, loss = {loss_mean.numpy().sum()}")

assert loss_mean.numpy() < 0.005