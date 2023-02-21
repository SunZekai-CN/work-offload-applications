import torch
print("finish load pytorch")
torch.cuda._tls.is_initializing = True
device=torch.device("cuda:0")
img = torch.ones(512).to(device)
for _ in range(10):
    img_t = torch.div(img,2.0)
    print("inference once")