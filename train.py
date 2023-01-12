import json
import time

import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import torch.optim as optim
from model import AlexNet
from torch.utils.data import DataLoader  # 不引用这句就是查不到DataLoader是吧离谱
import torch
from torchvision import transforms, datasets, utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),  # crop 裁剪
                                 transforms.RandomHorizontalFlip(),  # 水平翻转
                                 # 随机反转后对于网络是全新的训练样例，增大训练集数量，防止过拟合
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])
}
train_dataset = datasets.ImageFolder(root="flower_data/train", transform=data_transform["train"])
train_num = len(train_dataset)  # 3306
# ---------------------------------------------------
# {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
flower_list = train_dataset.class_to_idx  # 设断点通过调试可以查看class_to_idx

# {0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)  # 通过调试查看，太方便了卧槽
with open('class_indices.json', 'w') as json_file:  # indices 目录
    json_file.write(json_str)
# ---------------------------------------------------

batch_size = 32
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root="flower_data/val", transform=data_transform["val"])
val_num = len(validate_dataset)  # 364
validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)
# ----验证集图像查看 将validate_loader中 batch_size=4, shuffle=True
# test_data_iter = iter(validate_loader)
# test_image, test_label = test_data_iter.next()
#
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#     print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
# imshow(utils.make_grid(test_image))

net = AlexNet(num_classes=5, init_weights=True)

net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0002)

save_path = './AlexNet.pth'
best_acc = 0.0
for epoch in range(10):
    net.train()  # Sets the module in training mode.我们希望只在训练过程中随机失活神经元
    running_loss = 0.0
    t1 = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    print(time.perf_counter() - t1)

    # evaluation
    net.eval()
    acc = 0.0
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))  # output[batch,5]
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item() #预测正确的样本个数
        accurate_val = acc / val_num
        if accurate_val > best_acc:
            best_acc = accurate_val
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / step, accurate_val))
print("Finished")
