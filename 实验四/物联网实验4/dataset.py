import os
import os.path
from typing import Any, Callable, Optional, Tuple
from PIL import Image
from torch.utils.data import Dataset


class MyImageFolder(Dataset):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        # 加载所有图片路径和标签
        self.classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for target_class in self.classes:
            class_index = self.class_to_idx[target_class]
            class_dir = os.path.join(root, target_class)

            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif')):
                    path = os.path.join(class_dir, filename)
                    self.samples.append((path, class_index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]

        # 打开图片（兼容所有格式）
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        # 应用 transform
        if self.transform is not None:
            img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target