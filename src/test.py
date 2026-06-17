from models.generator import TSCNet
import torch
from torchsummary import summary
print("load model")

model = TSCNet(num_channel=128, num_features=400 // 2 + 1).cuda()

input = torch.rand([2,201,201]).cuda()
print("get output")
# real,imag = model(input)
print("summary")
summary(model,(2,201,201),batch_size=4)


