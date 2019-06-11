import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from mobilenetv3 import mobilenetv3


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


testdir = '../val_img'

testset = torchvision.datasets.ImageFolder(testdir, transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=32)

'''
print(testset.class_to_idx)
print(testset.classes)
print(testset.imgs)

dataiter = iter(test_loader)
images, labels = dataiter.next()
print(type(labels), labels, labels.size())
imshow(torchvision.utils.make_grid(images))
'''

net = mobilenetv3(mode='small')
state_dict = torch.load('mobilenetv3_small_67.218.pth.tar')
net.load_state_dict(state_dict)
net.cuda()
net.eval()

correct_num = 0
total_num = 0
for data in test_loader:
    images, labels = data
    images = images.cuda()
    labels = labels.cuda()
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total_num += labels.size(0)
    correct_num += (predicted == labels).sum()
    print('processed: ', total_num)

print('##############')
print('Accuracy: ', correct_num.cpu().numpy()/total_num)
print('##############')
