import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from mobilenetv3 import mobilenetv3


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_random_sample(testdir ='../val_img'):
    dirs = os.listdir(testdir)
    dirs.sort()
    label = np.random.randint(1000)
    dir_name = os.path.join(testdir,dirs[label])
    img_names = os.listdir(dir_name)
    img_name = img_names[np.random.randint(len(img_names))]
    img = Image.open(os.path.join(dir_name,img_name))
    print(dir_name,img_name,label)
    return img, label


if __name__ == '__main__':
    net = mobilenetv3(mode='small')
    state_dict = torch.load('mobilenetv3_small_67.218.pth.tar')
    net.load_state_dict(state_dict)
    net.cuda()
    net.eval()

    data_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    image, label = get_random_sample()
    image = data_transform(image)
    image = image.reshape([1,3,224,224]).cuda()

    outputs = net(Variable(image))
    _, predicted = torch.max(outputs.data, 1)

    print((predicted[0].cpu() == label).numpy().astype(np.bool))