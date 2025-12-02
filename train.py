import os
import torch
import torch.utils.data as data
import torchvision.utils as vutils

from AttGan import AttGAN
from functions import *
from functions import CelebAMaskHQSimple 
from Arguments import Args


# Configuración del experimento

# class Args:
#     # Atributos a aprender
#     attrs = ['Smiling', 'Bald', 'Heavy_Makeup', 'Eyeglasses']

#     # Rutas
#     data_path = "data/CelebAMask-HQ/CelebA-HQ-img"
#     attr_path = "data/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt"

#     # Tamaño y modelo
#     img_size = 128
#     shortcut_layers = 1
#     inject_layers = 0
#     enc_dim = 64
#     dec_dim = 64
#     dis_dim = 64
#     dis_fc_dim = 1024
#     enc_layers = 5
#     dec_layers = 5
#     dis_layers = 5
#     enc_norm = "batchnorm"
#     dec_norm = "batchnorm"
#     dis_norm = "instancenorm"
#     dis_fc_norm = "none"
#     enc_acti = "lrelu"
#     dec_acti = "relu"
#     dis_acti = "lrelu"
#     dis_fc_acti = "relu"

#     # Pérdidas
#     lambda_1 = 100.0
#     lambda_2 = 10.0
#     lambda_3 = 1.0
#     lambda_gp = 10.0
#     mode = "wgan"

#     # Entrenamiento
#     epochs = 200
#     batch_size = 16
#     num_workers = 4
#     lr = 0.0002
#     beta1 = 0.5
#     beta2 = 0.999
#     n_d = 5   # actualizaciones D por cada actualización G

#     # Inferencia
#     thres_int = 0.5
#     test_int = 1.0

#     # Guardado
#     save_interval = 2000
#     sample_interval = 2000
#     output_dir = "output"

#     gpu = True


args = Args()
args.n_attrs = len(args.attrs)
args.betas = (args.beta1, args.beta2)

os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(f"{args.output_dir}/samples", exist_ok=True)
os.makedirs(f"{args.output_dir}/checkpoints", exist_ok=True)


# ===========
# Dataset
# ===========

train_dataset = CelebAMaskHQSimple(
    data_path=args.data_path,
    attr_path=args.attr_path,
    image_size=args.img_size,
    mode="train",
    selected_attrs=args.attrs
)

valid_dataset = CelebAMaskHQSimple(
    data_path=args.data_path,
    attr_path=args.attr_path,
    image_size=args.img_size,
    mode="valid",
    selected_attrs=args.attrs
)

train_loader = data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=args.num_workers
)

valid_loader = data.DataLoader(
    valid_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False
)

print("Training images:", len(train_dataset))


# Inicializar AttGAN

attgan = AttGAN(args)
device = torch.device("cuda" if args.gpu else "cpu")
attgan.G.to(device)
attgan.D.to(device)

fixed_img, fixed_att = next(iter(valid_loader))
fixed_img = fixed_img.to(device)
fixed_att = fixed_att.to(device)

import csv
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

# Loop de entrenamiento
# Inicializar TensorBoard 
writer = SummaryWriter(f"{args.output_dir}/tensorboard")

# Inicializar CSV con header
csv_file = open(f"{args.output_dir}/training_log.csv", "w", newline="")
csv_logger = csv.writer(csv_file)
csv_logger.writerow(["step", "d_loss", "g_loss", "df_loss", "gc_loss", "gr_loss"])

step = 0
for epoch in range(args.epochs):
    for img_a, att_a in train_loader:
        img_a = img_a.to(device)
        att_a = att_a.to(device)

        # Atributos destino b = permutación aleatoria
        idx = torch.randperm(len(att_a))
        att_b = att_a[idx]

        # Escalar atributos [-thres_int, thres_int]
        att_a_ = (att_a * 2 - 1) * args.thres_int
        att_b_ = (att_b * 2 - 1) * args.thres_int

        
        # Entrenar Discriminador
        if (step + 1) % (args.n_d + 1) != 0:
            errD = attgan.trainD(img_a, att_a, att_a_, att_b, att_b_)

            # === Guardar scalars de D ===
            writer.add_scalar("Loss/D_loss", errD['d_loss'], step)
            writer.add_scalar("Loss/DF_loss", errD['df_loss'], step)
            writer.add_scalar("Loss/GP", errD['df_gp'], step)
            writer.add_scalar("Loss/DC_loss", errD['dc_loss'], step)

        else:
            # Entrenar Generador
            errG = attgan.trainG(img_a, att_a, att_a_, att_b, att_b_)

            # Guardar scalars de Generador
            writer.add_scalar("Loss/G_loss", errG['g_loss'], step)
            writer.add_scalar("Loss/GF_loss", errG['gf_loss'], step)
            writer.add_scalar("Loss/GC_loss", errG['gc_loss'], step)
            writer.add_scalar("Loss/GR_loss", errG['gr_loss'], step)

            print(f"Epoch {epoch} | Step {step} | D_loss: {errD['d_loss']:.4f} | G_loss: {errG['g_loss']:.4f}")

        # Guardar checkpoint
        if (step + 1) % args.save_interval == 0:
            torch.save(attgan.G.state_dict(),
                       f"{args.output_dir}/checkpoints/G_step_{step}.pth")
            print("Checkpoint guardado.")

        # Guardar muestras
        if (step + 1) % args.sample_interval == 0:
            save_samples(attgan, fixed_img, fixed_att, step, args)


        step += 1
