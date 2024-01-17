import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

# sy here: newly added
import os
import torch
from torchvision.datasets.vision import VisionDataset
import matplotlib.pyplot as plt
import hashlib
import time
import json
import threading
from io import StringIO
from multiprocessing import current_process 
import atexit

log_buffers = {}

def save_logs():
    process_id = os.getpid()
    if process_id in log_buffers:
        try:
            filepath = f'log_0_process_{process_id}.json'  # define node_num accordingly
            with open(filepath, 'a') as file:
                file.write(log_buffers[process_id].getvalue())
            print(f"Logs for process {process_id} successfully saved to {filepath}")
            log_buffers[process_id] = StringIO()
        except Exception as e:
            print(f"Error writing logs for process {process_id} to file: {e}")

def _log_time(in_out, index, path, timestamp):
    process_id = os.getpid()
    if process_id not in log_buffers:
        log_buffers[process_id] = StringIO()

    ssd_num = _get_ssd_number(path)  
    log_entry = {'in_out': in_out, 'ssd': ssd_num, 'index': index, 'time': timestamp}
    log_buffers[process_id].write(json.dumps(log_entry) + "\n")


def _get_ssd_number(path):
    ssd_part = path.split('/')[2]
    ssd_number = int(ssd_part.replace('ssd', ''))  # extracting number
    return ssd_number


class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        extract: bool = False,
    ) -> None:

        super(CIFAR10, self).__init__(root, transform=transform, target_transform=target_transform)
        self.process_id = current_process().name   
        self.node_num = 0 # somehow figure out node_num
        
        if download:
            self.download()
            
        self.train = train
        
        self._load_meta()
        
        extracted_path = os.path.join(self.root, 'train' if self.train else 'test')
        if extract and not os.path.exists(extracted_path):
            self.extract_images()
        self.samples = self._make_dataset()

    def worker_init_fn(worker_id):
        process_id = os.getpid()
        log_buffers[process_id] = StringIO()
        print(f"Initializing worker {worker_id} with process id {process_id}")

        def cleanup():
            print(f"Cleaning up worker {worker_id} with process id {process_id}")
            save_logs()
        atexit.register(cleanup)


    def extract_images(self):
        data_list = self.train_list if self.train else self.test_list
        for file_name, _ in data_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                data, labels = entry['data'], entry['labels']
                data = data.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))

            for i, (img_data, label) in enumerate(zip(data, labels)):
                # modulo for ssd subdird name
                subdirectory_name = f'ssd{i % 3}'
                ssd_dir = os.path.join(self.root, subdirectory_name)
                os.makedirs(ssd_dir, exist_ok=True)

                # train/test
                dataset_type = 'train' if self.train else 'test'
                dataset_dir = os.path.join(ssd_dir, dataset_type)
                os.makedirs(dataset_dir, exist_ok=True)

                # classes
                class_dir = os.path.join(dataset_dir, str(label))
                os.makedirs(class_dir, exist_ok=True)

                img_path = os.path.join(class_dir, f'{i}.png')
                img = Image.fromarray(img_data, 'RGB')
                img.save(img_path, format='PNG')

                loaded_img = Image.open(img_path)
                if not np.array_equal(np.array(img), np.array(loaded_img)):
                    print(f"Discrepancy detected in saving/loading image at {img_path}")

    def _make_dataset(self):
        def is_image_file(filename):
            return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.bmp'])

        samples = []
        self.targets = [] 

        # ssd direc
        for ssd_subdir in sorted(os.listdir(self.root)):
            ssd_dir = os.path.join(self.root, ssd_subdir)
            if not os.path.isdir(ssd_dir):
                continue

            # train/test
            data_dir = os.path.join(ssd_dir, 'train' if self.train else 'test')
            if not os.path.isdir(data_dir):
                continue

            # class dirs
            for label in range(len(self.classes)):
                class_dir = os.path.join(data_dir, str(label))
                if not os.path.isdir(class_dir):
                    continue

                # datafiles
                for fname in sorted(os.listdir(class_dir)):
                    if is_image_file(fname):
                        path = os.path.join(class_dir, fname)
                        samples.append((path, label))  # path + label(class)
                        self.targets.append(label)

        return samples

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        path, target = self.samples[index]

        try:
            start_time = time.perf_counter()
            _log_time('in', index, path, start_time)
            with open(path, 'rb') as f:
                img = Image.open(f)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                if self.transform is not None:
                    img = self.transform(img)

                if self.target_transform is not None:
                    target = self.target_transform(target)

            end_time = time.perf_counter()
            _log_time('out', index, path, end_time)
#            if index == len(self) - 1:
#                self.save_logs()
            
        except Exception as e:
            print(f"Error in __getitem__ for index {index}: {e}")
            raise e

        return img, target

            
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
     
    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)


    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

