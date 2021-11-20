
import torch
from torch import nn
from optimizer_helper import get_optim_and_scheduler
from itertools import cycle
import numpy as np


#### Implement Step2

def _do_epoch(args,feature_extractor,rot_cls,obj_cls,source_loader,target_loader_train,target_loader_eval,optimizer,device):

    criterion = nn.CrossEntropyLoss()
    feature_extractor.train()
    obj_cls.train()
    rot_cls.train()

    target_loader_train = cycle(target_loader_train)

    for it, (data_source, class_l_source, _, _) in enumerate(source_loader):

        (data_target, _, data_target_rot, rot_l_target) = next(target_loader_train)

        data_source, class_l_source  = data_source.to(device), class_l_source.to(device)
        data_target, data_target_rot, rot_l_target  = data_target.to(device), data_target_rot.to(device), rot_l_target.to(device)

        optimizer.zero_grad()


        class_loss = ....
        rot_loss = ....

        loss = class_loss + args.weight_RotTask_step2*rot_loss

        loss.backward()

        optimizer.step()

        _, cls_pred = ...
        _, rot_pred = ...

    acc_cls = ...
    acc_rot = ...

    print("Class Loss %.4f, Class Accuracy %.4f,Rot Loss %.4f, Rot Accuracy %.4f" % (class_loss.item(), acc_cls, rot_loss.item(), acc_rot))


    #### Implement the final evaluation step, computing OS*, UNK and HOS
    feature_extractor.eval()
    obj_cls.eval()
    rot_cls.eval()

    with torch.no_grad():
        for it, (data, class_l,_,_) in enumerate(target_loader_eval):



def step2(args,feature_extractor,rot_cls,obj_cls,source_loader,target_loader_train,target_loader_eval,device):
    optimizer, scheduler = get_optim_and_scheduler(feature_extractor,rot_cls,obj_cls, args.epochs_step2, args.learning_rate, args.train_all)


    for epoch in range(args.epochs_step2):
        _do_epoch(args,feature_extractor,rot_cls,obj_cls,source_loader,target_loader_train,target_loader_eval,optimizer,device)
        scheduler.step()