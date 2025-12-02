class Args:
    # Atributos a aprender
    attrs = ['Smiling', 'Bald', 'Heavy_Makeup', 'Eyeglasses']

    # Rutas
    data_path = "data/CelebAMask-HQ/CelebA-HQ-img"
    attr_path = "data/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt"

    # Tamaño y modelo
    img_size = 128
    shortcut_layers = 1
    inject_layers = 0
    enc_dim = 64
    dec_dim = 64
    dis_dim = 64
    dis_fc_dim = 1024
    enc_layers = 5
    dec_layers = 5
    dis_layers = 5
    enc_norm = "batchnorm"
    dec_norm = "batchnorm"
    dis_norm = "instancenorm"
    dis_fc_norm = "none"
    enc_acti = "lrelu"
    dec_acti = "relu"
    dis_acti = "lrelu"
    dis_fc_acti = "relu"

    # Pérdidas
    lambda_1 = 100.0
    lambda_2 = 10.0
    lambda_3 = 1.0
    lambda_gp = 10.0
    mode = "wgan"

    # Entrenamiento
    epochs = 200
    batch_size = 16
    num_workers = 4
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    n_d = 5   # actualizaciones D por cada actualización G

    # Inferencia
    thres_int = 0.5
    test_int = 1.0

    # Guardado
    save_interval = 2000
    sample_interval = 2000
    output_dir = "output"

    gpu = True
