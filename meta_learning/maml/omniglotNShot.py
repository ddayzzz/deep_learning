from dataset.omniglot.omniglot import Omniglot
import torchvision.transforms as transforms
from PIL import Image
import os.path
import numpy as np


class OmniglotNShot:

    def __init__(self, root, batchsz, n_way, k_shot, k_query, imgsz):
        """
        设置 Omniglot 的数据集的格式
        :param root: 保存有 omniglot 数据集的路径 (e.g. dataset/omniglot)
        :param batchsz: batch size
        :param n_way: 几个类别
        :param k_shot: 一个类别有多训练多少个样本? (元学习中的测试集为 support set/meta training set)
        :param k_qry: 一个列表测试集有多少个样本? (元学习中的测试集为 query set/meta test set)
        :param imgsz: 图像大小
        """

        self.resize = imgsz
        target_file = os.path.join(root, f'omniglot_imgsz[{imgsz}].npy')
        if not os.path.isfile(target_file):
            # if root/data.npy does not exist, just download it
            self.x = Omniglot(root, download=True,
                              transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
                                                            lambda x: x.resize((imgsz, imgsz)),
                                                            lambda x: np.reshape(x, (imgsz, imgsz, 1)),
                                                            lambda x: np.transpose(x, [2, 0, 1]),
                                                            lambda x: x / 255.])
                              )

            temp = dict()  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
            # 这里按照类排序
            for (img, label) in self.x:
                if label in temp.keys():
                    temp[label].append(img)
                else:
                    temp[label] = [img]

            self.x = []
            for label, imgs in temp.items():  # labels info deserted , each label contains 20imgs
                self.x.append(np.array(imgs))

            # as different class may have different number of imgs
            # 每个类别只有20张图像
            self.x = np.array(self.x).astype(np.float)  # [[20 imgs],..., 1623 classes in total]
            # each character contains 20 imgs
            print('data shape:', self.x.shape)  # [1623, 20, 84, 84, 1]
            temp = []  # Free memory
            # save all dataset into npy file.
            np.save(os.path.join(root, target_file), self.x)
            print('write into omniglot.npy.')
        else:
            # if data.npy exists, just load it.
            self.x = np.load(target_file)
            print(f'load from {target_file}.')

        # [1623, 20, 84, 84, 1]
        # 这里直接将1623个类别的前1200类作为训练, 后423作为测试
        self.x_train, self.x_test = self.x[:1200], self.x[1200:]
        # TODO 这里没有 normalization, 很奇怪
        # self.normalization()

        self.batchsz = batchsz
        self.n_cls = self.x.shape[0]  # 1623
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        assert (k_shot + k_query) <= 20

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}  # original data cached
        print("DB: train", self.x_train.shape, "test", self.x_test.shape)
        # 建立数据的缓存: [(x_spt, y_spt, x_query, y_query), ], 元组的数量和 episode 相同. x_spt 的样本数量(第二维) N_WAY * N_SHOT, 第一维为 BATCH_SIZE
        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"]),  # current epoch data cached
                               "test": self.load_data_cache(self.datasets["test"])}

    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sdt of 1
        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        # print("before norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)

    def load_data_cache(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  # one batch means one set

                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                # 选择哪些类?
                selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)

                for j, cur_class in enumerate(selected_cls):
                    # 按照当前的类别选择样本
                    # TODO 每个类有20个样本
                    selected_img = np.random.choice(20, self.k_shot + self.k_query, False)

                    # 分割成: meta-training and meta-test
                    x_spt.append(data_pack[cur_class][selected_img[:self.k_shot]])
                    x_qry.append(data_pack[cur_class][selected_img[self.k_shot:]])
                    # 设置标签. 这里的 x_spt 和 x_query 是同一个类, 设置为 index
                    y_spt.append([j for _ in range(self.k_shot)])
                    y_qry.append([j for _ in range(self.k_query)])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, 1, self.resize, self.resize)[perm]  # [N, C, H, W], 且打乱顺序
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, 1, self.resize, self.resize)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)

            # [b, setsz, 1, 84, 84]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, 1, self.resize, self.resize)
            y_spts = np.array(y_spts).astype(np.int64).reshape(self.batchsz, setsz)
            # [b, qrysz, 1, 84, 84]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, 1, self.resize, self.resize)
            y_qrys = np.array(y_qrys).astype(np.int64).reshape(self.batchsz, querysz)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch


if __name__ == '__main__':
    import os
    import time
    import torch
    path = os.sep.join((os.path.dirname(__file__), 'dataset', 'omniglot'))

    db = OmniglotNShot(path, batchsz=20, n_way=5, k_shot=5, k_query=15, imgsz=64)

    for i in range(1):
        x_spt, y_spt, x_qry, y_qry = db.next('train')
        test_x_spt, test_y_spt, _, _ = db.next('test')
        # [b, setsz, h, w, c] => [b, setsz, c, w, h] => [b, setsz, 3c, w, h]
        x_spt = torch.from_numpy(x_spt)
        x_qry = torch.from_numpy(x_qry)
        y_spt = torch.from_numpy(y_spt)
        y_qry = torch.from_numpy(y_qry)
        batchsz, setsz, c, h, w = x_spt.size()

