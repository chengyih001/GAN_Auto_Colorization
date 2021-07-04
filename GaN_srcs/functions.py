from average_meter import AverageMeter
from skimage.color import lab2rgb
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch
import numpy as np

def train_model_cycleGAN(model, train_dl, val_dl, epochs, display_every=20):
    data = next(iter(val_dl))
    a1 = []
    b1 = []
    c1 = []
    for e in range(epochs):
        print("epoch: ", e)
        loss_meter_dict= create_loss_meters_2() # function returing a dictionary of objects to 
        i = 0                                  # log the losses of the complete network
        for data in tqdm(train_dl):
            model.setup_input(data) 
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1
    
        if e % display_every == 0:
            print(f"\nEpoch {e+1}/{epochs}")
            print(f"Iteration {i}/{len(train_dl)}")
            log_results(loss_meter_dict) # function to print out the losses
            visualize_xxxx(model, data, save=False) # function displaying the model's outputs

        if (e != 0):
            a1.append(loss_meter_dict["loss_D_b"].avg)
            b1.append(loss_meter_dict["loss_D_a"].avg)
            c1.append(loss_meter_dict["loss_G"].avg)


    return a1, b1, c1

def train_model(model, train_dl, val_dl, epochs, display_every=20):
    data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    a1 = []
    b1 = []
    c1 = []
    d1 = []
    e1 = []
    f1 = []
    for e in range(epochs):
        print("epoch: ", e)
        loss_meter_dict= create_loss_meters() # function returing a dictionary of objects to 
        i = 0                                  # log the losses of the complete network
        for data in tqdm(train_dl):
            model.setup_input(data) 
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1
    
        if e % display_every == 0:
            print(f"\nEpoch {e+1}/{epochs}")
            print(f"Iteration {i}/{len(train_dl)}")
            log_results(loss_meter_dict) # function to print out the losses
            visualize_xxx(model, data, save=False) # function displaying the model's outputs
        
        if (e != 0):
            a1.append(loss_meter_dict["loss_D_fake"].avg)
            b1.append(loss_meter_dict["loss_D_real"].avg)
            c1.append(loss_meter_dict["loss_D"].avg)
            d1.append(loss_meter_dict["loss_G_GAN"].avg)
            e1.append(loss_meter_dict["loss_G_L1"].avg)
            f1.append(loss_meter_dict["loss_G"].avg)

    return a1, b1, c1, d1, e1, f1

def test_model(model,test_dl):
    for data in tqdm(test_dl):
        model.setup_input(data)
        loss_meter_dict = create_loss_meters()
        #update_losses(model, loss_meter_dict, count=data['L'].size(0))
        log_results(loss_meter_dict)
        visualize_test(model,data,save=False)



def test_model_cycleGAN(model,test_dl):
    for data in tqdm(test_dl):
        model.setup_input(data)
        loss_meter_dict = create_loss_meters_2()
        #update_losses(model, loss_meter_dict, count=data['L'].size(0))
        log_results(loss_meter_dict)
        visualize_test_2(model,data,save=False)

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

def create_loss_meters_2():
    loss_D_b = AverageMeter()
    loss_D_a = AverageMeter()
    loss_G = AverageMeter()
    
    return {'loss_D_b': loss_D_b,
            'loss_D_a': loss_D_a,
            'loss_G': loss_G}

def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)
    
def visualize(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")

def visualize_test(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    #model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(3):
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 3, i + 1 + 3)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3,3, i + 1 + 6)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_test_{time.time()}.png")

def visualize_test_2(model, data, save=True):
    model.net_G_a2b.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    #model.net_G.train()
    fake_color = model.fake_B.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(3):
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 3, i + 1 + 3)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3,3, i + 1 + 6)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_test_{time.time()}.png")

def visualize_xxx(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(4):
        ax = plt.subplot(3, 4, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 4, i + 1 + 4)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3,4, i + 1 + 8)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")

def visualize_xxxx(model, data, save=True):
    model.net_G_a2b.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G_a2b.train()
    fake_color = model.fake_B.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i+5][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i+5])
        ax.axis("off")
        ax = plt.subplot(3,5, i + 1 + 10)
        ax.imshow(real_imgs[i+5])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")
        
def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")