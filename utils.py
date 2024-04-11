import time
import torch
import numpy as np
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()

    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}

def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item())

# initialize weights
# in this example using zero-centered Normal dist with std 0.02

def init_weights(net, gain=0.02):
    def init_func(model):
        classname = model.__class__.__name__
        if hasattr(model, 'weight') and 'Conv' in classname:
            nn.init.normal_(model.weight.data, mean=0.0, std=gain)

            if hasattr(model, 'bias') and model.bias is not None:
                nn.init.constant_(model.bias.data, 0.0)

        elif 'BatchNorm2d' in classname:
            nn.init.normal_(model.weight.data, 1., gain)
            nn.init.constant_(model.bias.data, 0.)

    net.apply(init_func)
    print(f"model initialized with norm initialization")
    return net


def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model

def visualize(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.color
    gray = model.gray

    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(gray[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        fake_i = transforms.ToPILImage()(fake_color[i].cpu())
        fake_i = np.array(fake_i) / 255
        ax.imshow(fake_i)
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        real_i = transforms.ToPILImage()(real_color[i].cpu())
        real_i = np.array(real_i)
        ax.imshow(real_i)
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")

def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

def resume(model, filename):
    model.load_state_dict(torch.load(filename))