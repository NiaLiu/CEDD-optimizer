import os
import csv
from torchvision.datasets import CIFAR10
from torchvision import transforms
from PIL import Image
import numpy as np
import datetime

def main(root='./data', out_dir='./compressed_cifar10'):
    # Load CIFAR-10
    train_ds = CIFAR10(root=root, train=True, download=True, transform=None)
    test_ds  = CIFAR10(root=root, train=False, download=True, transform=None)

    # Prepare output dirs
    for split in ['train', 'test']:
        for cls in train_ds.classes:
            os.makedirs(os.path.join(out_dir, split, cls), exist_ok=True)

    # Helper to save images and labels
    def save_split(ds, split):
        csv_path = os.path.join(out_dir, f'{split}_labels.csv')
        with open(csv_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename', 'label'])
            for idx, (img, label) in enumerate(ds):
                cls_name = ds.classes[label]
                fn = f'{idx:05d}.png'
                img_path = os.path.join(out_dir, split, cls_name, fn)
                # Ensure PIL Image
                if not isinstance(img, Image.Image):
                    img = transforms.ToPILImage()(img)
                img.save(img_path, format='PNG')
                writer.writerow([os.path.join(split, cls_name, fn), cls_name])
        print(f'Wrote {len(ds)} images and labels to {split}_labels.csv')

    save_split(train_ds, 'train')
    save_split(test_ds, 'test')

def load_split(out_dir, split):
    csv_path = os.path.join(out_dir, f'{split}_labels.csv')
    images = []
    labels = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img_path = os.path.join(out_dir, row['filename'])
            img = Image.open(img_path).convert('RGB')
            arr = np.array(img, dtype=np.uint8)
            images.append(arr)
            labels.append(row['label'])
    images = np.stack(images)       # shape: (N, 32, 32, 3)
    labels = np.array(labels)      # shape: (N,)
    print(f'Loaded {len(labels)} samples from {split}')
    return images, labels

def main_1(out_dir='./compressed_cifar10', save_path='cifar10_reconstructed.npz'):
    train_images, train_labels = load_split(out_dir, 'train')
    test_images, test_labels   = load_split(out_dir, 'test')

    np.savez(
        save_path,
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels
    )
    print(f'Saved reconstructed arrays to {save_path}')

def main_2(root='./data', out_dir='./compressed_cifar10_webp', quality=80):
    # Download/load CIFAR-10
    train_ds = CIFAR10(root=root, train=True,  download=True, transform=None)
    test_ds  = CIFAR10(root=root, train=False, download=True, transform=None)

    # Make class subdirectories
    for split in ['train', 'test']:
        for cls in train_ds.classes:
            os.makedirs(os.path.join(out_dir, split, cls), exist_ok=True)

    def save_split(ds, split):
        csv_path = os.path.join(out_dir, f'{split}_labels.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'label'])
            for idx, (img, label) in enumerate(ds):
                cls_name = ds.classes[label]
                fn = f'{idx:05d}.webp'
                img_path = os.path.join(out_dir, split, cls_name, fn)

                # Ensure PIL Image
                if not isinstance(img, Image.Image):
                    img = transforms.ToPILImage()(img)

                # Save as WebP with given quality
                img.save(img_path, format='WEBP', quality=quality)
                writer.writerow([os.path.join(split, cls_name, fn), cls_name])
        print(f'  • Wrote {len(ds)} images to `{split}/` and labels to `{split}_labels.csv`')

    print(f'Compressing CIFAR-10 to WebP (quality={quality})...')
    save_split(train_ds, 'train')
    save_split(test_ds,  'test')
    print('Done.')
def main_3(out_dir='./compressed_cifar10_webp', save_path='cifar10_webp_reconstructed.npz'):
    train_imgs, train_lbls = load_split(out_dir, 'train')
    test_imgs,  test_lbls  = load_split(out_dir, 'test')

    np.savez(
        save_path,
        train_images=train_imgs,
        train_labels=train_lbls,
        test_images=test_imgs,
        test_labels=test_lbls
    )
    print(f'Saved reconstructed arrays to `{save_path}`')

def load_as_array(root, train=True):
    ds = CIFAR10(root=root, train=train, download=True, transform=None)
    imgs = []
    labels = []
    to_np = transforms.ToTensor()  # produces float tensor in [0,1]
    for img, label in ds:
        # get as uint8 array
        arr = np.array(img, dtype=np.uint8)  # shape (32,32,3)
        imgs.append(arr)
        labels.append(label)
    return np.stack(imgs, axis=0), np.array(labels, dtype=np.int64)
def main_4(root='./data', out_path='cifar10_int8q.npz'):
    # load train/test
    X_train, y_train = load_as_array(root, train=True)
    X_test,  y_test  = load_as_array(root, train=False)

    # quantize: uint8 [0,255] -> int8 by subtracting 128
    X_train_q = (X_train.astype(np.int16) - 128).astype(np.int8)
    X_test_q  = (X_test.astype(np.int16)  - 128).astype(np.int8)

    # save compressed
    np.savez_compressed(
        out_path,
        train_images = X_train_q,
        train_labels = y_train,
        test_images  = X_test_q,
        test_labels  = y_test
    )
    print(f'Compressed CIFAR-10 saved to: {out_path}')
    print(f'  • train_images:  {X_train_q.shape}, dtype={X_train_q.dtype}')
    print(f'  • test_images:   {X_test_q.shape},  dtype={X_test_q.dtype}')

def main_5(in_path='cifar10_int8q.npz', out_path='cifar10_reconstructed.npz'):
    # load compressed
    data = np.load(in_path)
    X_train_q, y_train = data['train_images'], data['train_labels']
    X_test_q,  y_test  = data['test_images'],  data['test_labels']

    # de-quantize: int8 [-128,127] -> uint8 [0,255]
    X_train = (X_train_q.astype(np.int16) + 128).astype(np.uint8)
    X_test  = (X_test_q.astype(np.int16)  + 128).astype(np.uint8)

    # save reconstructed
    np.savez(
        out_path,
        train_images = X_train,
        train_labels = y_train,
        test_images  = X_test,
        test_labels  = y_test
    )
    print(f'Reconstructed CIFAR-10 saved to: {out_path}')
    print(f'  • train_images:  {X_train.shape}, dtype={X_train.dtype}')
    print(f'  • test_images:   {X_test.shape},  dtype={X_test.dtype}')

if __name__ == '__main__':
    print(datetime.datetime.now())
    # main()
    main_2(quality=80)
    print(datetime.datetime.now())
    # main_1()
    main_3
    print(datetime.datetime.now())