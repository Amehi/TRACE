from pathlib import Path
import pickle
import json
import random

class myEHR:
    def __init__(self, dataset_name) -> None:
        self.dataset = dataset_name
    def load_dataset(self):
        self.preprocess()
        dataset_path = self.get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def read_and_augmentation(self):
        
        with open(f'Data/{self.dataset}/trajs.pkl', 'rb') as f:
            trajs = pickle.load(f)
        with open(f'Data/{self.dataset}/gds.pkl', 'rb') as f:
            gds = pickle.load(f)
        with open(f'Data/{self.dataset}/action_dict.json', 'r') as f:
            action_dict = json.load(f)
        with open(f'Data/{self.dataset}/action_dict_l.json', 'r') as f:
            action_dict_l = json.load(f)
        assert len(trajs) == len(gds)
        with open (f'Data/{self.dataset}/time.pkl', 'rb') as f:
            times = pickle.load(f)
        assert len(trajs) == len(times)

        return trajs,gds,action_dict,action_dict_l,times
    
    def preprocess(self):
        dataset_path = self.get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        trajs,gds,action_dict,action_dict_l,times = self.read_and_augmentation()

        #shuffle
        random.seed(2024)
        n = len(trajs)
        indices = list(range(n))
        random.shuffle(indices)
        trajs = [trajs[i] for i in indices]
        gds = [gds[i] for i in indices]
        times = [times[i] for i in indices]

        #split
        trajs_train = trajs[ : int(0.75 * n)]
        gds_train = gds[: int(0.75 * n)]
        times_train = times[: int(0.75 * n)]
        trajs_val = trajs[int(0.75 * n) : int(0.85 * n)]
        gds_val = gds[int(0.75 * n) : int(0.85 * n)]
        times_val = times[int(0.75 * n) : int(0.85 * n)]
        trajs_test = trajs[int(0.85 * n) : ]
        gds_test = gds[int(0.85 * n) : ]
        times_test = times[int(0.85 * n) : ]

        #store
        dataset = {
            'train':(trajs_train, gds_train, times_train),
            'val' : (trajs_val, gds_val, times_val),
            'test' : (trajs_test, gds_test, times_test),
            'smap' : action_dict,
            'smap_l' : action_dict_l,
        }
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def get_rawdata_root_path(self):
        return Path('Data')
    
    def get_preprocessed_root_path(self):
        root = self.get_rawdata_root_path()
        return root.joinpath('preprocessed')
    
    def get_preprocessed_dataset_path(self):
        folder = self.get_preprocessed_root_path()
        return folder.joinpath(f'dataset_{self.dataset}.pkl')

        