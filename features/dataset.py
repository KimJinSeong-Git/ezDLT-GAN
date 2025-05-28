import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class TrainDataset(Dataset):
    def __init__(self, root_dataset, image_size=(224, 224)):
        self.image_paths = []
        self.labels = []

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        class_names = sorted([
            d for d in os.listdir(root_dataset)
            if os.path.isdir(os.path.join(root_dataset, d))
        ])
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        for class_name in class_names:
            class_dir = os.path.join(root_dataset, class_name)
            label = self.class_to_idx[class_name]
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]
        return image, label
    
class CelebADataset(Dataset):
    def __init__(self, root_dir, image_size=(224, 224), partition_file=None, phase='train'):
        self.image_dir = os.path.join(root_dir, "img_align_celeba")
        self.attr_path = os.path.join(root_dir, "list_attr_celeba.txt")
        self.phase = phase
        self.image_paths = []
        self.labels = []

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Load attributes
        self._load_attributes(partition_file)

    def _load_attributes(self, partition_file):
        with open(self.attr_path, 'r') as f:
            lines = f.readlines()

        attr_names = lines[1].strip().split()
        attr_data = lines[2:]  # Skip header

        if partition_file:  # If split info is provided
            partition_path = os.path.join(os.path.dirname(self.attr_path), partition_file)
            with open(partition_path, 'r') as pf:
                partition_dict = {
                    line.split()[0]: int(line.split()[1]) for line in pf.readlines()
                }
        else:
            partition_dict = None

        for line in attr_data:
            tokens = line.strip().split()
            filename = tokens[0]
            attrs = torch.tensor([1 if int(x) == 1 else 0 for x in tokens[1:]], dtype=torch.float)

            # Use phase filter if provided
            if partition_dict:
                phase_code = {'train': 0, 'val': 1, 'test': 2}[self.phase]
                if partition_dict[filename] != phase_code:
                    continue

            self.image_paths.append(os.path.join(self.image_dir, filename))
            self.labels.append(attrs)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]
        return image, label
