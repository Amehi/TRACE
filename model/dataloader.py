import torch
import torch.utils.data as data_utils
import random

class TRACEDataloader:
    def __init__(self, dataset) -> None:
        dataset = dataset.load_dataset()
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.smap = dataset['smap']
        self.smap_l = dataset['smap_l']
        self.item_count = len(self.smap)
        self.item_count_l = len(self.smap_l)
        self.CLOZE_MASK_TOKEN = self.item_count + 1
        self.batch_size = 128

        trajs = self.test[0]
        gds = self.test[1]
        times = self.test[2]

        self.traj_gd_time = [(l1, l2, l3) for l1, l2, l3 in zip(trajs, gds, times)]


    def get_pytorch_dataloaders(self):
        train_loader = self.get_train_loader()
        val_loader = self.get_val_loader()
        test_loader = self.get_test_loaders()
        return train_loader, val_loader, test_loader

    def get_train_loader(self):
        dataset = self.get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def get_train_dataset(self):
        dataset = TrainDataset(self.train, self.CLOZE_MASK_TOKEN, self.item_count, self.smap_l, self.smap)
        return dataset
    
    def get_val_loader(self):
        return self.get_eval_loader(mode='val')

    def get_test_loaders(self):
        batch_size = self.batch_size
        dataset = TestDataset(self.traj_gd_time, self.CLOZE_MASK_TOKEN, self.smap_l, self.smap)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader
    def get_eval_loader(self, mode):
        batch_size = self.batch_size
        dataset = self.get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader
    
    def get_eval_dataset(self, mode):
        answers = self.val if mode == 'val' else self.test
        dataset = EvalDataset(answers, self.CLOZE_MASK_TOKEN, self.smap_l, self.smap)
        return dataset

class TrainDataset(data_utils.Dataset):
    def __init__(self, train, mask_token, num_items,smap_l,smap):
        self.trajs = train[0]
        self.gds = train[1]
        self.times = train[2]
        self.mask_token = mask_token
        self.num_items = num_items
        self.max_len = 100
        self.rng = random.Random(2024)
        self.mask_prob = 0.2
        self.smap = list(smap.values())
        self.smap_l = list(smap_l.values())
    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, index):
        traj = self.trajs[index] + [self.mask_token]
        gd = self.gds[index]
        time = self.times[index]+[0]
        candidates = self.smap_l
        labels = [0] * len(self.smap_l)
        for idx in range(len(candidates)):
            if candidates[idx] in gd:
                labels[idx] = 1

        
        traj = traj[-self.max_len:]
        time = time[-self.max_len:]

        mask_len = self.max_len - len(traj)

        traj = [0] * mask_len + traj
        time = [0] * mask_len + time

        return torch.LongTensor(traj), torch.LongTensor(candidates), torch.LongTensor(labels), torch.LongTensor(time)

class EvalDataset(data_utils.Dataset):
    def __init__(self, eval, mask_token, smap_l, smap):
        self.trajs = eval[0]
        self.gds = eval[1]
        self.times = eval[2]
        self.max_len = 100
        self.mask_token = mask_token
        self.smap_l = smap_l
        self.smap = smap

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, index):
        candidates = list(self.smap_l.values())

        traj = self.trajs[index] + [self.mask_token]
        gd = self.gds[index]
        time = self.times[index] + [0]

        labels = [0] * len(self.smap_l)
        for idx in range(len(candidates)):
            if candidates[idx] in gd:
                labels[idx] = 1
        
        traj = traj[-self.max_len:]
        time = time[-self.max_len:]

        mask_len_traj = self.max_len - len(traj)

        traj = [0] * mask_len_traj + traj
        time = [0] * mask_len_traj + time 


        return torch.LongTensor(traj), torch.LongTensor(candidates), torch.LongTensor(labels), torch.LongTensor(time)
    
class TestDataset(data_utils.Dataset):
    def __init__(self, eval, mask_token, smap_l, smap):
        self.test = eval
        self.max_len = 100
        self.mask_token = mask_token
        self.smap_l = smap_l
        self.smap = smap

    def __len__(self):
        return len(self.test)

    def __getitem__(self, index):
        item = self.test[index]
        traj = item[0]
        gd = item[1]
        time = item[2]

        candidates = list(self.smap_l.values())

        traj = traj + [self.mask_token]
        time = time + [0]

        labels = [0] * len(self.smap_l)
        for idx in range(len(candidates)):
            if candidates[idx] in gd:
                labels[idx] = 1
        
        traj = traj[-self.max_len:]
        time = time[-self.max_len:]
        mask_len_traj = self.max_len - len(traj)
        # mask_len_gd = 100 - len(gd)
        traj = [0] * mask_len_traj + traj
        time = [0] * mask_len_traj + time

        return torch.LongTensor(traj), torch.LongTensor(candidates), torch.LongTensor(labels), torch.LongTensor(time)
