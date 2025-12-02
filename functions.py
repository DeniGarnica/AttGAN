# Preparamos la data

import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


# Dataset para CelebAMask-HQ

# Este dataset:
# - Carga solamente los atributos seleccionados 
# - Lee el archivo CelebAMask-HQ-attribute-anno.txt
# - Convierte atributos de {-1, 1} a {0, 1} para compatibilidad con BCE
# - Divide el dataset en train/val/test siguiendo el esquema del repo AttGAN
# - Preprocesa imágenes al tamaño requerido por el modelo (128x128)

class CelebAMaskHQSimple(data.Dataset):
    def __init__(self, data_path, attr_path, image_size, mode, selected_attrs):
        super().__init__()
        self.data_path = data_path

        # Cargamos la línea que contiene los nombres de atributos
        # El índice +1 es porque la primera columna es el nombre de la imagen
        lines = open(attr_path, 'r', encoding='utf-8').readlines()
        header = lines[1].split()
        attr_indices = [header.index(att) + 1 for att in selected_attrs]

        # Leemos cada fila: filename + valores de atributos
        entries = lines[2:]
        image_names = []
        labels = []

        for line in entries:
            parts = line.split()
            image_names.append(parts[0])  
            vals = [int(parts[i]) for i in attr_indices]  # Solo atributos seleccionados
            labels.append(vals)

        self.images = np.array(image_names)
        self.labels = np.array(labels)

        # División fija usada comúnmente en CelebA-HQ: 28000/500/1500
        if mode == "train":
            self.images = self.images[:28000]
            self.labels = self.labels[:28000]
        elif mode == "valid":
            self.images = self.images[28000:28500]
            self.labels = self.labels[28000:28500]
        else:  # test
            self.images = self.images[28500:]
            self.labels = self.labels[28500:]

        # Transformaciones : resize, tensorización y normalización simétrica
        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, idx):
        # Cargamos la imagen y aplicamos transformaciones.
        img_path = os.path.join(self.data_path, self.images[idx])
        img = self.tf(Image.open(img_path).convert('RGB'))

        # Convertimos atributos de {-1, 1} a {0, 1} porque AttGAN usa BCE con logits
        att = torch.tensor((self.labels[idx] + 1) // 2).float()
        return img, att

    def __len__(self):
        return len(self.images)

# Normalización y activaciones
# -------------------------------
# Estos helpers construyen dinámicamente las capas de normalización y activación
# según los parámetros definidos en el modelo (batchnorm, instancenorm, lrelu, etc.)

def add_normalization_1d(layers, fn, n_out):
    # Aplica normalización 1D a capas lineales.
    # La opción 'instancenorm' requiere insertar reshape (Unsqueeze/Squeeze)
    if fn == 'none':
        pass
    elif fn == 'batchnorm':
        layers.append(nn.BatchNorm1d(n_out))
    elif fn == 'instancenorm':
        # InstanceNorm1d opera sobre (N, C, L); se ajusta la forma temporalmente
        layers.append(Unsqueeze(-1))
        layers.append(nn.InstanceNorm1d(n_out, affine=True))
        layers.append(Squeeze(-1))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers

def add_normalization_2d(layers, fn, n_out):
    # Normalización 2D para capas convolucionales (feature maps)
    if fn == 'none':
        pass
    elif fn == 'batchnorm':
        layers.append(nn.BatchNorm2d(n_out))
    elif fn == 'instancenorm':
        layers.append(nn.InstanceNorm2d(n_out, affine=True))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers

def add_activation(layers, fn):
    # Inserta la función de activación especificada
    if fn == 'none':
        pass
    elif fn == 'relu':
        layers.append(nn.ReLU())
    elif fn == 'lrelu':
        layers.append(nn.LeakyReLU())
    elif fn == 'sigmoid':
        layers.append(nn.Sigmoid())
    elif fn == 'tanh':
        layers.append(nn.Tanh())
    else:
        raise Exception('Unsupported activation function: ' + str(fn))
    return layers


# Utilidades de reshape
# -------------------------------
# Se usan exclusivamente para permitir InstanceNorm1d en vectores lineales
# Evitan código repetido en los bloques

class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
         # Elimina dimensión 'dim'; necesario tras InstanceNorm1d
        self.dim = dim
    
    def forward(self, x):
        return x.squeeze(self.dim)

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        # Añade dimensión 'dim'; preparación para InstanceNorm1d
        return x.unsqueeze(self.dim)

# Bloques modulares
# -------------------------------
# Estos bloques son la base estructural de AttGAN:
# - LinearBlock se usa en el clasificador del discriminador
# - Conv2dBlock compone el encoder y el discriminador
# - ConvTranspose2dBlock compone el decoder del generador

class LinearBlock(nn.Module):
    def __init__(self, n_in, n_out, norm_fn='none', acti_fn='none'):
        super(LinearBlock, self).__init__()
        # Capa lineal con sesgo solo si no hay normalización
        layers = [nn.Linear(n_in, n_out, bias=(norm_fn=='none'))]
        layers = add_normalization_1d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class Conv2dBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0, 
                 norm_fn=None, acti_fn=None):
        super(Conv2dBlock, self).__init__()
        # Convolución básica para extraer características
        layers = [nn.Conv2d(n_in, n_out, kernel_size, stride=stride, padding=padding, bias=(norm_fn=='none'))]
        layers = add_normalization_2d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class ConvTranspose2dBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0, 
                 norm_fn=False, acti_fn=None):
        super(ConvTranspose2dBlock, self).__init__()
        # Convolución transpuesta para upsampling (decoder)
        layers = [nn.ConvTranspose2d(n_in, n_out, kernel_size, stride=stride, padding=padding, bias=(norm_fn=='none'))]
        layers = add_normalization_2d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    

import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

def save_samples(attgan, fixed_imgs, fixed_atts, step, args):
    """
    fixed_imgs: tensor shape [B,3,H,W]  (al menos B=2)
    fixed_atts: tensor shape [B,n_attrs]
    """
    attgan.eval()

    # Queremos solo 4 filas
    num_rows = 4
    imgs_to_show = []
    labels = []

    for row in range(num_rows):
        img0 = fixed_imgs[row:row+1]     # mantener dimensión batch=1
        att0 = fixed_atts[row:row+1]

        # Imagen original
        imgs_to_show.append(img0)
        labels.append("ORIGINAL")

        # Flips de atributos
        for i, attr_name in enumerate(args.attrs):
            att_mod = att0.clone()
            old_val = int(att0[0, i].item())
            new_val = 1 - old_val

            att_mod[:, i] = new_val
            att_mod_scaled = (att_mod * 2 - 1) * args.test_int

            gen = attgan.G(img0, att_mod_scaled)
            imgs_to_show.append(gen)
            labels.append(f"{attr_name}: {old_val} -> {new_val}")

    # Convertimos a PIL + texto
    pil_imgs = []
    for img, label in zip(imgs_to_show, labels):
        img = img[0]                          # quitar batch
        img = (img * 0.5 + 0.5).clamp(0, 1)   # [-1,1] → [0,1]
        img_pil = TF.to_pil_image(img.cpu())

        # Añadimos barra superior con texto
        canvas = Image.new("RGB", (img_pil.width, img_pil.height + 32), (255, 255, 255))
        canvas.paste(img_pil, (0, 32))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), label, fill=(0, 0, 0))
        pil_imgs.append(canvas)

    # Crear grid (4 filas)
    cols = len(args.attrs) + 1     # original + flips
    rows = 4
    w, h = pil_imgs[0].size

    grid = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))

    for idx, img in enumerate(pil_imgs):
        r = idx // cols
        c = idx % cols
        grid.paste(img, (c * w, r * h))

    # Guardar
    save_path = f"{args.output_dir}/samples/sample_{step}.jpg"
    grid.save(save_path)
    print(f"Muestra guardada en {save_path}")

    attgan.train()
