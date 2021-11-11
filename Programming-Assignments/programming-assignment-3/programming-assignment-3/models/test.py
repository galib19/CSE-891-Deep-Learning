import torch

a = torch.rand(3, 2,1)
b = torch.rand(2,2, 3)
print('a:',a)
print('b:',b)
# c = a.expand_as(b)
# print('c:',c)

print(a.shape)
# print(torch.unsqueeze(a, 0))
# print(torch.unsqueeze(a, 2))
print(a.transpose(1,2).shape)