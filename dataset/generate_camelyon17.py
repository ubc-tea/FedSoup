import numpy as np
import os
import random
import torchvision.transforms as transforms
import torch.utils.data as data
from utils.dataset_utils import split_data, save_file
from os import path
from scipy.io import loadmat
from PIL import Image
from torch.utils.data import DataLoader

import torch
from torch.utils.data import Dataset
import pandas as pd
import glob
from PIL import Image
import os
import ujson


class Camelyon17Dataset(data.Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        super(Camelyon17Dataset, self).__init__()
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        if img.shape[0] != 1:
            # transpose to Image type, so that the transform function can be used
            img = Image.fromarray(np.uint8(np.asarray(img.transpose((1, 2, 0)))))

        elif img.shape[0] == 1:
            im = np.uint8(np.asarray(img))
            # turn the raw image into 3 channels
            im = np.vstack([im, im, im]).transpose((1, 2, 0))
            img = Image.fromarray(im)

        # do transform with PIL
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return self.data.shape[0]


class RawCamelyon17Dataset(Dataset):
    def __init__(self, root, train, node, transform):
        self.root = root
        self.train = train
        self.node = node
        self.transform = transform

        self.img_paths = []
        self.img_labels = []
        self.patient_ids = []

        dir_match = self.root + "patches/" + "*node_" + str(self.node) + "/"
        total_dirpath = glob.glob(dir_match)
        if self.node == 3:
            test_patient_num = int(len(total_dirpath) * 0.7)
        else:
            test_patient_num = int(len(total_dirpath) * 0.5)
        if self.train:
            sampling_dirpath = random.sample(
                total_dirpath, len(total_dirpath) - test_patient_num
            )
        else:
            sampling_dirpath = random.sample(total_dirpath, test_patient_num)
        pid_match = []
        for s_dir in sampling_dirpath:
            dirname = s_dir.split("/")[-2]
            s_dir_pid = int(dirname.split("_")[1])
            pid_match.append(s_dir_pid)

        label_file_path = self.root + "metadata.csv"
        img_labels_pd = pd.read_csv(label_file_path)
        node_pd = img_labels_pd.query("node==@self.node")

        for idx, rows in node_pd.iterrows():
            patient = rows["patient"]
            if int(patient) in pid_match:
                label = rows["tumor"]
                x_coord = rows["x_coord"]
                y_coord = rows["y_coord"]
                self.img_labels.append(label)
                self.patient_ids.append(patient)
                img_filename = (
                    "patch_patient_"
                    + str(patient).zfill(3)
                    + "_node_"
                    + str(self.node)
                    + "_x_"
                    + str(x_coord)
                    + "_y_"
                    + str(y_coord)
                    + ".png"
                )
                full_img_path = (
                    self.root
                    + "patches/"
                    + "patient_"
                    + str(patient).zfill(3)
                    + "_node_"
                    + str(self.node)
                    + "/"
                    + img_filename
                )
                self.img_paths.append(full_img_path)

        one_index_list = [i for i, x in enumerate(self.img_labels) if x == 1]
        zero_index_list = [i for i, x in enumerate(self.img_labels) if x == 0]

        random.seed(44)
        random.shuffle(one_index_list)
        random.shuffle(zero_index_list)
        # generate tiny camelyon17
        # if self.train:
        #     one_index_list = one_index_list[:320]
        #     zero_index_list = zero_index_list[:320]
        # else:
        #     one_index_list = one_index_list[:140]
        #     zero_index_list = zero_index_list[:140]
        # total: 46k
        if self.train:
            one_index_list = one_index_list[:3200]
            zero_index_list = zero_index_list[:3200]
        else:
            one_index_list = one_index_list[:1400]
            zero_index_list = zero_index_list[:1400]

        new_img_paths = []
        new_img_labels = []
        new_patient_ids = []
        new_img_paths.extend([self.img_paths[i] for i in one_index_list])
        new_img_paths.extend([self.img_paths[i] for i in zero_index_list])

        new_img_labels.extend([self.img_labels[i] for i in one_index_list])
        new_img_labels.extend([self.img_labels[i] for i in zero_index_list])

        new_patient_ids.extend([self.patient_ids[i] for i in one_index_list])
        new_patient_ids.extend([self.patient_ids[i] for i in zero_index_list])

        self.img_paths = new_img_paths
        self.img_labels = new_img_labels
        self.patient_ids = new_patient_ids

        # dir_match = self.root + "patches/" + "*node_" + str(self.node) + '/'
        # total_dirpath = glob.glob(dir_match)
        # total_filepath = []
        # for dir in total_dirpath:
        #     if self.tiny:
        #         all_img_path = glob.glob(dir + "/*.png")
        #         load_img_path = random.sample(all_img_path, int(len(all_img_path) * 0.1))
        #         total_filepath.extend(load_img_path)
        #     else:
        #         total_filepath.extend(glob.glob(dir + "/*.png"))
        # for filepath in total_filepath:
        #     self.img_paths.append(filepath)
        #     filename = filepath.split('/')[-1]
        #     patient_id = filename.split('_')[2]
        #     node = filename.split('_')[4]
        #     x_coord = filename.split('_')[6]
        #     y_coord = filename.split('_')[8]
        #     label = img_labels_pd.query("patient==@patient_id and node==@node and x_coord==@x_coord and y_coord==@y_coord")["tumor"]
        #     self.img_labels.append(label)
        #     self.patient_ids.append(int(patient_id))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        label = self.img_labels[idx]
        patient_id = self.patient_ids[idx]
        image = Image.open(image_path)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, patient_id


def rawcamelyon17_dataset_read(base_path, node_id):
    # define the transform function
    transform = transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    node_train_dataset = RawCamelyon17Dataset(
        root=base_path, train=True, node=node_id, transform=transform
    )
    node_test_dataset = RawCamelyon17Dataset(
        root=base_path, train=False, node=node_id, transform=transform
    )
    node_train_loader = DataLoader(
        dataset=node_train_dataset, batch_size=len(node_train_dataset), shuffle=False
    )
    node_test_loader = DataLoader(
        dataset=node_test_dataset, batch_size=len(node_test_dataset), shuffle=False
    )

    return node_train_loader, node_test_loader


# Allocate data to usersz``
def generate_camelyon17(dir_path, class_balanced=False, num_balanced=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    root = data_path

    y = []
    for node_id in range(5):
        X_train, X_test = [], []
        print("Node: ", node_id)
        node_train_loader, node_test_loader = rawcamelyon17_dataset_read(root, node_id)

        statistic = [[] for _ in range(5)]

        for _, tt in enumerate(node_train_loader):
            node_train_data, node_train_label, node_train_pid = tt
        for _, tt in enumerate(node_test_loader):
            node_test_data, node_test_label, node_test_pid = tt

        X_train.extend(node_train_data.cpu().detach().numpy())
        X_test.extend(node_test_data.cpu().detach().numpy())

        node_y = []
        y_train = node_train_label.cpu().detach().numpy()
        y_test = node_test_label.cpu().detach().numpy()

        num_samples = {"train": [], "test": []}
        train_data = {"x": X_train, "y": y_train}
        num_samples["train"].append(len(y_train))
        test_data = {"x": X_test, "y": y_test}
        num_samples["test"].append(len(y_test))

        with open(train_path + str(node_id) + ".npz", "wb") as f:
            np.savez_compressed(f, data=train_data)
        with open(test_path + str(node_id) + ".npz", "wb") as f:
            np.savez_compressed(f, data=test_data)

        node_y.extend(y_train)
        node_y.extend(y_test)
        y.append(np.array(node_y))

        del X_train, X_test, train_data, test_data

    labelss = []
    for yy in y:
        labelss.append(len(set(yy)))
    num_clients = len(y)
    print(f"Number of labels: {labelss}")
    print(f"Number of clients: {num_clients}")

    statistic = [[] for _ in range(num_clients)]
    for client in range(num_clients):
        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))
    config = {
        "num_clients": num_clients,
        "num_classes": max(labelss),
        "non_iid": None,
        "balance": None,
        "partition": None,
        "Size of samples for labels in clients": statistic,
        "alpha": 0.1,
        "batch_size": 10,
    }
    with open(config_path, "w") as f:
        ujson.dump(config, f)


random.seed(43)
np.random.seed(43)
ROOT_DIR = "/localscratch/chenmh.61223108.0/tmp_dataset/"
# SAVE_ROOT_DIR = (
#     "/home/chenmh/projects/rrg-timsbc/chenmh/cmh_proj/dataset/tiny_camelyon17/"
# )
SAVE_ROOT_DIR = (
    "/home/chenmh/projects/rrg-timsbc/chenmh/cmh_proj/dataset/camelyon17_46k/"
)
data_path = ROOT_DIR
dir_path = SAVE_ROOT_DIR
# dir_path = "/home/chenmh/projects/rrg-timsbc/chenmh/cmh_proj/dataset/tiny_camelyon17_balanced_class/"
# dir_path = "/home/chenmh/projects/rrg-timsbc/chenmh/cmh_proj/dataset/tiny_camelyon17_balanced_class_num/"
# dir_path = "/home/chenmh/projects/rrg-timsbc/chenmh/cmh_proj/dataset/camelyon17_balanced_class/"
# dir_path = "/home/chenmh/projects/rrg-timsbc/chenmh/cmh_proj/dataset/camelyon17_balanced_class_num/"


if __name__ == "__main__":
    generate_camelyon17(dir_path, class_balanced=True, num_balanced=False)
