"""
unet.py
Benjamin Smith

Sourced:
1. U-Net paper : https://arxiv.org/pdf/1505.04597.pdf
2. https://github.com/usuyama/pytorch-unet 
3. https://github.com/milesial/Pytorch-UNet/tree/master/unet 

"""


import torch.nn as nn
import torch
import torch.nn.functional as F


class UNet(nn.Module):

    def __init__(self, n_classes):
        super(UNet, self).__init__()

        self.cp0 = self.convrelu(3,16,3,1)
        self.cp1 = self.convrelu(16,32,3,1)
        self.cp2 = self.convrelu(32,64,3,1)    
        self.cp3 = self.convrelu(64,128,3,1)        
        self.cp4 = self.convrelu(128,256,3,1)
        self.cp5 = self.convrelu(256,512,3,1)

        # bottom layer
        self.b1 = nn.Sequential(
                nn.Conv2d(512,1024,kernel_size=3,padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024,512,kernel_size=3,padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
        )
        # ----

    
        self.up1 = self.upconvrelu(512+512,512,3,1)
        self.up2 = self.upconvrelu(256+512,256,3,1)
        self.up3 = self.upconvrelu(128+256,128,3,1)
        self.up4 = self.upconvrelu(128+64,64,3,1)
        self.up5 = self.upconvrelu(64+32,32,3,1)
        self.up6 = self.upconvrelu(32+16,16,3,1)

        self.conv_orig0 = self.convrelu(3,64,3,1)
        self.conv_orig1 = self.convrelu(64,128,3,1)
        self.conv_orig2 = self.upconvrelu(128+16, 64, 3, 1)


        self.conv1x1 = nn.Sequential(
                        nn.Conv2d(16,n_classes,kernel_size=1,padding=0),
                        #nn.Conv2d(n_classes,n_classes,kernel_size=1,padding=1),
                        #nn.BatchNorm2d(n_classes),
                        #nn.ReLU(inplace=True)
        )

    
    def convrelu(self, in_channels, out_channels, kernel, padding):
        return nn.Sequential(
                    nn.Conv2d(in_channels,out_channels,kernel_size=kernel,padding=padding),
                    nn.Conv2d(out_channels,out_channels,kernel_size=kernel,padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                    nn.MaxPool2d(2, stride=2)
                    )

    # FIXED: remove function calls and place in-situ
    def upsample(self, layer, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        layer = self.pad(layer, x)
        x = torch.cat((x, layer),1) # Memory Explosion TODO: https://stackoverflow.com/questions/54645349/torch-cat-memory-explode 
        return layer, x

    def upconvrelu(self, in_channels, out_channels, kernel, padding):
        return nn.Sequential(
                    nn.Conv2d(in_channels,out_channels,kernel_size=kernel,padding=padding),
                    nn.Conv2d(out_channels,out_channels,kernel_size=kernel,padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                    )

    #https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    def pad(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        return x1
        
    def forward(self, input):

        #x_orig = [0,]

        #x_orig[0] = self.conv_orig0(input)
        #x_orig[0] = self.conv_orig1(x_orig[0])

        # Using lists to act as pointers
        layer_00 = [0,]
        layer_01 = [0,]
        layer_0 = [0,]
        layer_1 = [0,]
        layer_2 = [0,]
        layer_3 = [0,]
        layer_4 = [0,]
        x = [0,]
        
        layer_00[0] = self.cp0(input)
        layer_01[0] = self.cp1(layer_00[0])
        layer_0[0] = self.cp2(layer_01[0])
        layer_1[0] = self.cp3(layer_0[0])
        layer_2[0] = self.cp4(layer_1[0])
        layer_3[0] = self.cp5(layer_2[0])
        
        layer_4[0] = self.b1(layer_3[0]) # Bottom layer

        x[0] = F.interpolate(layer_4[0], scale_factor=2, mode='bilinear', align_corners=True)
        #layer_3[0] = self.pad(layer_3[0], x[0])
        x[0] = self.pad(x[0],layer_3[0])
        x[0] = torch.cat((x[0],layer_3[0]),1)
        x[0] = self.up1(x[0])

        x[0] = F.interpolate(x[0], scale_factor=2, mode='bilinear', align_corners=True)
        #layer_2[0] = self.pad(layer_2[0], x[0])
        x[0] = self.pad(x[0],layer_2[0])
        x[0] = torch.cat((x[0], layer_2[0]),1)
        x[0] = self.up2(x[0])

        x[0] = F.interpolate(x[0], scale_factor=2, mode='bilinear', align_corners=True)
        #layer_1[0] = self.pad(layer_1[0], x[0])
        x[0] = self.pad(x[0],layer_1[0])
        x[0] = torch.cat((x[0], layer_1[0]),1)
        x[0] = self.up3(x[0])

        x[0] = F.interpolate(x[0], scale_factor=2, mode='bilinear', align_corners=True)
        #layer_0[0] = self.pad(layer_0[0], x[0])
        x[0] = self.pad(x[0],layer_0[0])
        x[0] = torch.cat((x[0], layer_0[0]),1)
        x[0] = self.up4(x[0])

        x[0] = F.interpolate(x[0], scale_factor=2, mode='bilinear', align_corners=True)
        #layer_01[0] = self.pad(layer_01[0], x[0])
        x[0] = self.pad(x[0],layer_01[0])
        x[0] = torch.cat((x[0], layer_01[0]),1)
        x[0] = self.up5(x[0])
        
        x[0] = F.interpolate(x[0], scale_factor=2, mode='bilinear', align_corners=True)
        #layer_00[0] = self.pad(layer_00[0], x[0])
        x[0] = self.pad(x[0],layer_00[0])
        x[0] = torch.cat((x[0], layer_00[0]),1)
        x[0] = self.up6(x[0])

        x[0] = F.interpolate(x[0], scale_factor=2, mode='bilinear', align_corners=True)
        #x_orig[0] = self.pad(x_orig[0], x[0])
        #x[0] = torch.cat((x[0], x_orig[0]),1)
        #x[0] = self.conv_orig2(x[0])

        out = self.conv1x1(x[0])
        #out = F.log_softmax(out, dim=1)
        return out

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g,p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(1)
    model = model.to(device)
    
    from torchsummary import summary
    summary(model, input_size=(3,768,768))
