import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import AverageMeterSet
from logger import *
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, recall_score
import numpy as np
import torch.nn.functional as F

STATE_DICT_KEY = 'model_state_dict'
OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'


class TRACETrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, export_root, device):
        self.device = device
        self.model = model.to(self.device)
        self.is_parallel = False
        if self.is_parallel:
            self.model = nn.DataParallel(self.model)

        # Adam param
        self.lr = 0.001
        self.weight_decay = 0
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader


        self.optimizer = self._create_optimizer()

        self.num_epochs = 25
        self.metric_ks = [5,10,20]
        self.best_metric = "AUC_PR"

        self.export_root = export_root
        self.writer, self.train_loggers, self.val_loggers = self._create_loggers()
        self.logger_service = LoggerService(self.train_loggers, self.val_loggers)
        self.train_batch_size = 128
        self.log_period_as_iter = 12800
        
        self.ce = nn.CrossEntropyLoss()


    def train(self):
        accum_iter = 0
        # self.validate(0, accum_iter)
        for epoch in range(self.num_epochs):
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            self.validate(epoch, accum_iter)
        self.logger_service.complete({
            'state_dict': (self._create_state_dict()),
        })
        self.writer.close()

    def train_one_epoch(self, epoch, accum_iter):
        self.model.train()
        # if self.args.enable_lr_schedule:
        #     self.lr_scheduler.step()
        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.train_loader)

        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch_size = batch[0].size(0)
            batch = [x.to(self.device) for x in batch]


            self.optimizer.zero_grad()
            loss_ce, loss_mask = self.calculate_loss(batch)
            loss = loss_ce + 0.01 *loss_mask
            loss.backward()

            self.optimizer.step()

            average_meter_set.update('loss_ce', loss_ce.item())
            average_meter_set.update('loss_mask', loss_mask)
            tqdm_dataloader.set_description(
                'Epoch {}, loss_ce {:.3f} loss_mask {:.3f}'.format(epoch+1, average_meter_set['loss_ce'].avg, average_meter_set['loss_mask'].avg))

            accum_iter += batch_size

            if self._needs_to_log(accum_iter):
                tqdm_dataloader.set_description('Logging to Tensorboard')
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch+1,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                # self.log_extra_train_info(log_data)
                self.logger_service.log_train(log_data)

        return accum_iter

    def validate(self, epoch, accum_iter):
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.mlc_eval(batch)
                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['AUC_PR'] +\
                                        ['AUC_ROC']
                description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch+1,
                'accum_iter': accum_iter,
            }
            log_data.update(average_meter_set.averages())
            self.logger_service.log_val(log_data)

    def multilabel_classification_task_eval(self, logits, targets, threshold=0.5):
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        preds_np = preds.cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten()
        probs_np = probs.cpu().numpy().flatten()

        #acc, recall, F1
        f1_micro = f1_score(targets_np, preds_np, average='micro')
        f1_macro = f1_score(targets_np, preds_np, average='macro')
        acc = accuracy_score(targets_np, preds_np)
        recall_micro = recall_score(targets_np, preds_np, average='micro')
        recall_macro = recall_score(targets_np, preds_np, average='macro')
        #AUC-PR AUC-ROC
        auc_pr = average_precision_score(targets_np, probs_np)
        auc_roc = roc_auc_score(targets_np, probs_np)
        avg_lab_num = np.sum(preds_np) / preds.shape[0]

        return f1_micro, f1_macro, acc, recall_micro, recall_macro, auc_pr, auc_roc, avg_lab_num
    
    def mlc_eval(self, batch):
        metric = {}
        seqs, candidates, labels, times = batch
        scores = self.model(seqs, times)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C
        f1_micro, f1_macro, acc, recall_micro, recall_macro, auc_pr, auc_roc, avg_lab_num = self.multilabel_classification_task_eval(scores, labels)
        metric = self.recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        metric['ACC'] = acc
        metric['Recall_micro'] = recall_micro
        metric['Recall_macro'] = recall_macro
        metric['F1_micro'] = f1_micro
        metric['F1_macro'] = f1_macro
        metric['AUC_PR'] = auc_pr
        metric['AUC_ROC'] = auc_roc
        metric['AVG_#_SIGN'] = avg_lab_num
        return metric
    
    def test(self):
        print('Test best model with test set!')

        best_model = torch.load(os.path.join(self.export_root, 'models', 'best_acc_model.pth'))
        print(best_model.get('epoch'))
        best_model = best_model.get('model_state_dict')
        new_state_dict = {}
        
        for key, value in best_model.items():
            new_key = key
            if self.is_parallel:
                if not key.startswith('module.'):
                    new_key = 'module.' + key
            new_state_dict[new_key] = value
        
        
        self.model.load_state_dict(new_state_dict)
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                bc_metrics = self.mlc_eval(batch)
                for k, v in bc_metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['AUC_PR'] +\
                                    ['AUC_ROC']
                description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            average_metrics = average_meter_set.averages()
            with open(os.path.join(self.export_root, 'logs', 'test_metrics' + '.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)
            print(average_metrics)

    def _create_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _create_loggers(self):
        root = Path(self.export_root)
        writer = SummaryWriter(root.joinpath('logs'))
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricGraphPrinter(writer, key='epoch', graph_name='Epoch', group_name='Train'),
            MetricGraphPrinter(writer, key='loss', graph_name='Loss', group_name='Train'),
        ]

        val_loggers = []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))
        val_loggers.append(RecentModelLogger(model_checkpoint))
        val_loggers.append(BestModelLogger(model_checkpoint, metric_key=self.best_metric))
        return writer, train_loggers, val_loggers

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }

    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.train_batch_size and accum_iter != 0
    

    def recalls_and_ndcgs_for_ks(self, scores, labels, ks):
        metrics = {}
        scores = scores
        labels = labels
        answer_count = labels.sum(1)

        labels_float = labels.float()
        rank = (-scores).argsort(dim=1)
        cut = rank
        for k in sorted(ks, reverse=True):
            cut = cut[:, :k]
            hits = labels_float.gather(1, cut)
            metrics['Recall@%d' % k] = (hits.sum(1) / labels.sum(1).float()).mean().cpu().item()
            metrics['Precision@%d' % k] = (hits.sum(1) / k).mean().cpu().item()
            position = torch.tensor([2 for _ in range(k)])
            weights = 1 / torch.log2(position.float())
            dcg = (hits * weights.to(hits.device)).sum(1)
            idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
            ndcg = (dcg / idcg).mean()
            metrics['NDCG@%d' % k] = ndcg.cpu().item()

        return metrics
    
    def calculate_loss(self, batch):
        seqs, candidates, labels, times = batch
        logits = self.model(seqs, times)  # B x T x V

        logits = logits[:, -1, :]  # (B*T) x V
        logits = logits.gather(1, candidates)
        loss_ce = self.ce(logits.view(-1), labels.view(-1).float())
        loss_mask = self.calculate_attention_sparsity_loss()
        return loss_ce, loss_mask

    def calculate_attention_sparsity_loss(self):
        loss = 0
        if self.is_parallel:
            Zs = [block.attention.attention.mask() for block in self.model.module.trace.transformer_blocks]
        else:
            Zs = [block.attention.attention.mask() for block in self.model.trace.transformer_blocks]
        for Z in Zs:
            loss += torch.sum(torch.abs(Z)) - 0.5 * torch.mean(Z)
        return loss