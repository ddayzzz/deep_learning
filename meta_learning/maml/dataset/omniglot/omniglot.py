import torch.utils.data as data
import os
import os.path
import errno


class Omniglot(data.Dataset):
    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    '''
    The items are (filename,category). The index of all the categories can be found in self.idx_classes
    Args:
    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    - download: need to download the dataset
    '''

    def __init__(self, root, transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')
        # 图像的信息
        self.all_items = find_classes(os.path.join(self.root, self.processed_folder))
        # Dict: 图像的ID(由字符类别+对应其中的编号组成)->对应index
        self.idx_classes = index_classes(self.all_items)

    def __getitem__(self, index):

        filename = self.all_items[index][0]
        # 完整的文件路径
        img = str.join(os.sep, [self.all_items[index][2], filename])
        # 类型名称: 字符集名称/字符的类型名
        target = self.idx_classes[self.all_items[index][1]]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.all_items)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, "images_evaluation")) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, "images_background"))

    def download(self):
        """
        下载文件
        :return:
        """
        from six.moves import urllib
        import zipfile

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            file_processed = os.path.join(self.root, self.processed_folder)
            print("== Unzip from " + file_path + " to " + file_processed)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(file_processed)
            zip_ref.close()
        print("Download finished.")


def find_classes(root_dir):
    retour = []
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            # 遍历的文件名
            if f.endswith("png"):
                r = root.split(os.sep)
                lr = len(r)
                # 保存: (文件名, 字符集名称/字符的类型名(这里用character[xx]来表示), 文件存在的路径前缀)
                retour.append((f, r[lr - 2] + os.sep + r[lr - 1], root))
    print("== Found %d items " % len(retour))
    return retour


def index_classes(items):
    idx = {}
    for i in items:
        if i[1] not in idx:
            idx[i[1]] = len(idx)
    print("== Found %d classes" % len(idx))
    return idx


if __name__ == '__main__':
    train = Omniglot(root='.')
    g = 5