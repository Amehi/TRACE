from model.dataset import myEHR
from model.dataloader import TRACEDataloader
from model.model import TRACEModel
from model.trainer import TRACETrainer
from datetime import date
import argparse
import os
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def create_experiment_export_folder(dataset_name):
    experiment_dir = f'/home/liangyuyang/myEHR-ML-Model/Experiment/{dataset_name}'
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    experiment_path = os.path.join(experiment_dir, ('Experiment' + "_" + str(date.today())))
    idx = _get_experiment_index(experiment_path)
    experiment_path = experiment_path + "_" + str(idx)
    os.mkdir(experiment_path)
    print('Folder created: ' + os.path.abspath(experiment_path))
    return experiment_path

def _get_experiment_index(experiment_path):
    idx = 0
    while os.path.exists(experiment_path + "_" + str(idx)):
        idx += 1
    return idx

def train(args):
    export_root = create_experiment_export_folder(args.dataset)
    dataset = myEHR(args.dataset)
    dataloader = TRACEDataloader(dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()
    model = TRACEModel(dataloader.item_count, dataloader.item_count_l, args).to(device)
    trainer = TRACETrainer(model, train_loader = train, val_loader=val, test_loader=test, export_root=export_root,device=device)
    trainer.train()
    trainer.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--dataset', type=str, default='mimic4')
    args = parser.parse_args()
    train(args)