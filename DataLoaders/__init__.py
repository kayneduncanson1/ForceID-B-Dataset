from torch.utils.data import Dataset, DataLoader
from Utils import set_seed

seed = 42
g = set_seed(seed)


# This class relies on pre-loaded data and labels objects. It returns a sample and label from these objects at a
# specified index:
class SingleDatasetLoaderLabels(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):

        sample = self.data[index]
        label = self.labels[index]

        return sample, label

    def __len__(self):
        return len(self.labels)


# This class is used when applying the Siamese variant of a neural network that takes two separate inputs. It is as per
# SingleDatasetLoaderLabels, except that the input dataset is a tuple containing two items. In this implementation, each
# item is the data object for a specific stance side:
class DualDatasetLoaderLabels(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):

        sample1 = self.data[0][index]
        sample2 = self.data[1][index]
        label = self.labels[index]

        return sample1, sample2, label

    def __len__(self):
        return len(self.labels)


# As per SingleDatasetLoaderLabels but returns a sample *without* its label. This can be used when there is a set of
# data and associated labels, and the data is loaded in batches *without shuffling*. This is the case for validation
# and test sets in this implementation (see init_data_loaders_no_labs_va_te function further down):
class SingleDatasetLoaderNoLabels(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):

        sample = self.data[index]

        return sample

    def __len__(self):

        # Note that the Dataset must be a tensor to have the size attribute:
        return self.data.size(0)


# As per DualDatasetLoaderLabels, but returns input samples *without* their labels:
class DualDatasetLoaderNoLabels(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):

        sample1 = self.data[0][index]
        sample2 = self.data[1][index]

        return sample1, sample2

    # Note that the items within the Dataset tuple must be tensors to have the size attribute:
    def __len__(self):
        return self.data[0].size(0)


# This function initialises data loaders for training, validation and test sets. It does not input or output labels for
# validation and test sets because they are not shuffled (there is no need to trace the labels through data loading):
def init_data_loaders_no_labs_va_te(data_loader_class_tr,
                                    data_loader_class_eval,
                                    data_tr,
                                    labs_tr,
                                    data_va,
                                    data_te,
                                    batch_size,
                                    sampler_tr):

    loader_tr = DataLoader(data_loader_class_tr(data_tr, labs_tr),
                           batch_size=batch_size,
                           sampler=sampler_tr,
                           num_workers=0,
                           generator=g)
    loader_va = DataLoader(data_loader_class_eval(data_va),
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=0,
                           generator=g)
    loader_te = DataLoader(data_loader_class_eval(data_te),
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=0,
                           generator=g)

    return loader_tr, loader_va, loader_te
