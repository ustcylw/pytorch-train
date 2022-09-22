import os
import tempfile
import warnings
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Union
from torch.cuda.amp import autocast, GradScaler

import time
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torchinfo import summary as TISummary
import numpy as np
from datetime import datetime
from tqdm import trange
from tqdm import tqdm, tqdm_notebook
from tqdm.contrib.logging import logging_redirect_tqdm
import logging
from logging import handlers

# from PyUtils.PLModuleInterface import PLMInterface, GraphCallback
from PyUtils.logs.print import *

import os
import argparse
import torch
from torch.nn import SyncBatchNorm
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torchvision import models
import torchvision.transforms as transforms
from torch.optim import lr_scheduler

import torch.utils.tensorboard.writer as TBWriter
from prettytable import PrettyTable

from PyUtils.utils.meter import AverageMeter
from PyUtils.pytorch.callback import LogCallback, TrainCallback, GraphCallback, Callback
from PyUtils.pytorch.module import TrainModule, Trainer

from src.nets.yolov1.net import vgg16, vgg16_bn
from src.nets.yolov1.mobilev2 import get_mobilenet_v2
from src.nets.yolov1.resnet_yolo import resnet50, resnet18
from src.loss.yolov1.yoloLoss import yoloLoss
from src.dataset.yolov1.dataset import yoloDataset
# from src.utils.visualize import Visualizer
from src.config.yolov1.config_v1 import CONFIGS



class LRSchedulerCB(Callback):
    def on_train_epoch_end(self, local_rank, epoch, module: "TrainModule", trainer: "Trainer"):
        is_change = False
        # epoch -= trainer.current_epoch
        # if epoch > 5 and epoch < 20:
        #     learning_rate=0.0001
        #     is_change = True
        # if epoch >= 20 and epoch < 40:
        #     learning_rate=0.00001
        #     is_change = True
        # # if epoch == 80:
        # #     learning_rate=0.000001
        # #     is_change = True
        # if is_change:
        #     for param_group in module.optimizer.param_groups:
        #         param_group['lr'] = learning_rate
        #     print(f'[************]  Learning Rate for this {epoch=}  {learning_rate=}')
        module.lr_scheduler.step()
        print(f'[************]  Learning Rate for this {epoch=}  {module.optimizer.param_groups[0]["lr"]=}')


class SaveModelCB(Callback):
    def on_train_epoch_end(self, local_rank, epoch, module: "TrainModule", trainer: "Trainer"):
        if local_rank == 0 and epoch % module.configs.TRAIN_SAVE_MODEL_INTERVAL == 0:
            torch.save(module.model.module.state_dict(), os.path.join(f'{module.configs.CHECKPOINT_DIR}', f'{epoch}.pth'))


class YoloV1TrainModule(TrainModule):

    def __init__(self, configs=None, pretrained=None):
        super(YoloV1TrainModule, self).__init__(pretrained)
        self.configs = configs
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.train_dataset = None
        self.train_sampler = None
        self.train_loader = None
        self.val_dataset = None
        self.val_sampler = None
        self.val_loader = None
        self.test_dataset = None
        self.test_sampler = None
        self.test_loader = None
        self.pretrained = pretrained

    def create_model(self, local_rank):
        # create local model
        if self.configs.NET_STRUCETURE == 'resnet':
            phi = 's'
            backbone = 'cspdarknet'
            pretrained = False
            self.model = resnet50()

            resnet = models.resnet50(pretrained=True)
            new_state_dict = resnet.state_dict()
            dd = self.model.state_dict()
            for k in new_state_dict.keys():
                print(k)
                if k in dd.keys() and not k.startswith('fc'):
                    print('yes')
                    dd[k] = new_state_dict[k]
            self.model.load_state_dict(dd)
        else:
            self.model = get_mobilenet_v2(pretrained=True)

        if self.pretrained and os.path.exists(self.pretrained) and local_rank == 0:
            sllog << f'[------------][{local_rank}]  loading pre-trained model[{self.pretrained}] ...'
            # self.model.load_state_dict(torch.load(pretrained).module.state_dict())
            self.model.load_state_dict(torch.load(self.pretrained))
            sllog << f'[------------][{local_rank}]  load pre-trained model complete.'
            
        self.model.to(local_rank)
        dist.barrier()

    def create_loss(self):
        self.criterion = yoloLoss(7,2,5,0.5)

    def create_optim(self, model):
        params=[]
        params_dict = dict(self.model.named_parameters())
        for key,value in params_dict.items():
            if key.startswith('features'):
                params += [{'params':[value],'lr':CONFIGS.TRAIN_LR_INIT*1}]
            else:
                params += [{'params':[value],'lr':CONFIGS.TRAIN_LR_INIT}]
        # self.optimizer = torch.optim.SGD(
        #     params, 
        #     lr=CONFIGS.TRAIN_LR_INIT, 
        #     momentum=0.9, 
        #     weight_decay=CONFIGS.TRAIN_WEIGHT_DECAY
        # )
        self.optimizer = torch.optim.Adam(
            params,
            lr=self.configs.TRAIN_LR_INIT,
            weight_decay=self.configs.TRAIN_WEIGHT_DECAY
        )

        # self.lr_scheduler = ReduceLROnPlateau(
        #     self.optimizer, 
        #     factor=0.6, 
        #     patience=2, 
        #     verbose=True, 
        #     mode="min", 
        #     threshold=1e-3, 
        #     min_lr=1e-8, 
        #     eps=1e-8
        # )
        self.lr_scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=20, 
            eta_min=1e-9
        )
        # self.lr_scheduler = None

    def create_data_loader(self):
        
        
        # sampler = DistributedSampler(dataset) if is_distributed else None
        # >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        # ...                     sampler=sampler)
        # >>> for epoch in range(start_epoch, n_epochs):
        # ...     if is_distributed:
        # ...         sampler.set_epoch(epoch)
        # ...     train(loader)
        
        
        
        self.train_dataset = yoloDataset(
            root=self.configs.DATASET_DIRS[0],
            list_file=self.configs.DATA_FILES,
            train=True,
            transform = [transforms.ToTensor()]
        )
        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.configs.TRAIN_BATCH_SIZE,
            num_workers=self.configs.TRAIN_NUMBER_WORKERS,
            # shuffle=True,
            sampler=self.train_sampler,
            collate_fn=self.train_dataset.yolo_dataset_collate
        )
        
        self.val_dataset = yoloDataset(
            root=self.configs.DATASET_DIRS[0],
            list_file=self.configs.TEST_DATA_FILES,
            train=False,
            transform = [transforms.ToTensor()]
        )
        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.configs.TRAIN_BATCH_SIZE,
            num_workers=self.configs.TRAIN_NUMBER_WORKERS,
            # shuffle=False,
            # sampler=self.val_sampler,
            collate_fn=self.val_dataset.yolo_dataset_collate
        )

        self.test_dataset = yoloDataset(
            root=self.configs.DATASET_DIRS[0],
            list_file=self.configs.TEST_DATA_FILES,
            train=False,
            transform = [transforms.ToTensor()]
        )
        self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.configs.TEST_BATCH_SIZE,
            # num_workers=self.configs.TRAIN_NUMBER_WORKERS,
            shuffle=False,
            # sampler=self.test_sampler,
            collate_fn=self.test_dataset.yolo_dataset_collate
        )

    def train_step(self, batch_idx, batch, local_rank):
        imgs, target = batch
        # imgs.cuda(f'cuda:{local_rank}')
        # target.cuda(f'cuda:{local_rank}')
        imgs = imgs.to(self.model.device)
        target = target.to(self.model.device)

        pred = self.model(imgs)
        loss = self.criterion(pred, target)

        return {'loss': loss}

    def train_step_end(self, step_output: Union[None, torch.Tensor, dict], batch_idx=None, batch=None, local_rank=None):
        ...

    def train_epoch_end(self):
        ...
        
    @torch.no_grad()
    def eval_step(self, batch_idx, batch, local_rank):
        imgs, target = batch
        imgs = imgs.to(self.model.device)
        target = target.to(self.model.device)

        pred = self.model(imgs)
        loss = self.criterion(pred,target)

        return {'val_loss': loss}

    def eval_step_end(self, step_output: Union[None, torch.Tensor, dict], batch_idx=None, batch=None, local_rank=None):
        ...

    @torch.no_grad()
    def test_step(self, batch_idx, batch, local_rank):
        # images, targets, y_trues = batch[0], batch[1], batch[2]

        # outputs = self.model(images.to(local_rank))
        # loss_all  = 0
        # losses = [0, 0, 0]
        # for l, output in enumerate(outputs):
        #     loss_item = self.criterion(
        #         l,
        #         output,
        #         targets,
        #         y_trues[l],
        #         batch
        #     )
        #     loss_all  += loss_item
        #     losses[l] += loss_item.item()
        # return {'loss': loss_all, 'loss1':losses[0], 'loss2':losses[1], 'loss3':losses[2]}
        return {'test_loss': 0}

    def test_step_end(self, step_output: Union[None, torch.Tensor, dict], batch_idx=None, batch=None, local_rank=None):
        ...

    @torch.no_grad()
    def predict(self, images, device_id):
        ...

    def predict_end(self, predicts, device_id):
        ...

    def set_callbacks(self):
        graph_cb = TrainCallback(
            interval=30, 
            log_dir=self.configs.LOGS_DIR, 
            dummy_input=np.zeros(shape=(2, 3, 446, 446))
        )
        log_cbs = LogCallback(
            train_meters={
                'loss': AverageMeter(name='loss', fmt=':4f')
            }, 
            val_meters={
                'val_loss': AverageMeter(name='loss', fmt=':4f')
            }, 
            test_meters={
                'test_loss': AverageMeter(name='loss', fmt=':4f')
            }, 
            log_dir=self.configs.LOGS_DIR, log_surfix='default',
            interval=30
        )
        lr_cb = LRSchedulerCB()
        save_model = SaveModelCB()
        return [graph_cb, log_cbs, lr_cb, save_model]



if __name__=="__main__":
    train_module = YoloV1TrainModule(configs=CONFIGS, pretrained=os.path.join(CONFIGS.TRAIN_PRETRAINED_DIR, CONFIGS.TRAIN_PRETRAINED_MODEL_NAME))
    
    trainer = Trainer(
        train_module=train_module, 
        configs=CONFIGS,
        mode='train', 
        accelarate=1, 
        precision=True,
        grad_average=False,
        sync_bn=True,
        limit_train_iters=1.0,
        limit_val_iters=1.0,
        limit_test_iters=1.0
    )
    trainer.fit()
