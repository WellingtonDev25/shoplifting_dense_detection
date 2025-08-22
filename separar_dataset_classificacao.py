import os
import shutil
import random

# Defina o caminho do dataset original
dataset_path = "recortes_processados"
output_path = "dataset"  # Novo diret�rio organizado

# Defina a propor��o de divis�o
train_ratio = 0.75
test_ratio = 0.15
val_ratio = 0.10

# Verifique se o diret�rio de sa�da existe, se n�o, cria
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Criar pastas de sa�da
for split in ["train", "test", "val"]:
    os.makedirs(os.path.join(output_path, split), exist_ok=True)

# Percorrer cada classe no dataset original
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)

    # Ignorar arquivos que n�o sejam diret�rios (pastas de classes)
    if not os.path.isdir(class_path):
        continue

    # Criar as pastas da classe dentro de cada split (train, test, val)
    for split in ["train", "test", "val"]:
        os.makedirs(os.path.join(output_path, split, class_name), exist_ok=True)

    # Listar todas as imagens da classe e embaralhar
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))]
    random.shuffle(images)

    # Determinar quantidades para cada divis�o
    total_images = len(images)
    train_count = int(total_images * train_ratio)
    test_count = int(total_images * test_ratio)
    val_count = total_images - train_count - test_count  # Garante que soma 100%

    # Fun��o para copiar arquivos e tratar exce��es
    def copy_files(image_list, split):
        for img in image_list:
            try:
                src = os.path.join(class_path, img)
                dst = os.path.join(output_path, split, class_name, img)
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"Erro ao copiar {img}: {e}")

    # Copiar imagens para as pastas correspondentes
    copy_files(images[:train_count], "train")
    copy_files(images[train_count:train_count + test_count], "test")
    copy_files(images[train_count + test_count:], "val")

print("? Dataset copiado e organizado com sucesso! ??")
