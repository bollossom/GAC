import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision


class GradCAM:
    def __init__(self, model, target_layer, size=(224, 224), num_cls=1000, mean=None, std=None):
        self.model = model
        self.model.eval()
        getattr(self.model, target_layer).register_forward_hook(self.__forward_hook)
        getattr(self.model, target_layer).register_full_backward_hook(self.__backward_hook)
        self.size = size
        self.num_cls = num_cls
        self.mean, self.std = mean, std
        self.ann = {}
        
    def forward(self, path, show=True, write=False,snn=True):
        # 读取图片
        origin_img = cv2.imread(path)
#         origin_img = cv2.resize(origin_img, (224,224))
#         cv2.imwrite("input.jpg", origin_img)
        origin_size = (origin_img.shape[1], origin_img.shape[0])  # [H, W, C]
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((origin_img.shape[1], origin_img.shape[0])),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])
        img = transform(origin_img[:, :, ::-1]).unsqueeze(0)

        # 输入模型以获取特征图和梯度
        output = self.model(img)
        
        self.model.zero_grad()
        loss,index = torch.max(output,dim=1)  

        loss.backward()
        # 计算cam图片
        
        if snn==True:
            cam = np.zeros(self.fmaps.shape[2:], dtype=np.float32)
        else:
            cam = np.zeros(self.fmaps.shape[1:], dtype=np.float32)
            
        if snn==True:
            alpha = np.mean(self.grads, axis=(2, 3)) #6,512,7,7_T
            alpha = alpha.mean(0)
            self.fmaps = self.fmaps.mean(0)
        else:
            alpha = np.mean(self.grads, axis=(1, 2))

        
        for k, ak in enumerate(alpha):
#             print(k) #0-511
#             print(alpha.shape) #512
#             print(ak) # 一个数
#             print(cam.shape) # 7x7
#             print(self.fmaps[k].shape) # 7x7
            
            cam += ak * self.fmaps[k]  # linear combination

            
            
        cam[cam < 0] = 0
        cam = cv2.resize(np.array(cam), origin_size)
        cam /= np.max(cam)
#         print(cam.shape,output.shape)
        # 把cam图变成热力图，再与原图相加
        cam= cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cam = np.float32(cam) + np.float32(origin_img)
        # 两幅图的色彩相加后范围要回归到（0，1）之间,再乘以255
        cam = np.uint8(255 * cam / np.max(cam))
        if write:
            if snn==True:
                cv2.imwrite("camcam_pandans1_GAC_MS34.jpg", cam)
                
            else:
                cv2.imwrite("origin_img.jpg", origin_img)
                cv2.imwrite("camcam_pandans1_ANN.jpg", cam)
        if show:
            # 要显示RGB的图片，如果是BGR的 热力图是反过来的
            plt.imshow(cam[:, :, ::-1])
            plt.show()

    def __backward_hook(self, module, grad_in, grad_out):
        self.grads = np.array(grad_out[0].detach().squeeze())
    def __forward_hook(self, module, input, output):
        self.fmaps = np.array(output.detach().squeeze())

# 调用函数
from models.MS_ResNet import resnet34
net = resnet34()
state_dict = torch.load('.pth', map_location=torch.device('cuda'))
if 'module' in list(state_dict.keys())[0]:
    state_dict = {k[7:]: v for k, v in state_dict.items()} # remove the 'module.' prefix

net.load_state_dict(state_dict)


grad_cam = GradCAM(net, 'conv5_x', (224, 224), 1000, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

grad_cam.forward('/dataset/ImageNet2012/val/n02510455/ILSVRC2012_val_00026307.JPEG', show=True, write=True,snn=True) 


            



