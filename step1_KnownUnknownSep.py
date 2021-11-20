
import torch
from torch import nn
from optimizer_helper import get_optim_and_scheduler

#### Implement Step1

def _do_epoch(args,feature_extractor,rot_cls,obj_cls,source_loader,optimizer,device):

    criterion = nn.CrossEntropyLoss()
    feature_extractor.train()
    obj_cls.train()
    rot_cls.train()


    for it, (data, class_l, data_rot, rot_l) in enumerate(source_loader):
        data, class_l, data_rot, rot_l  = data.to(device), class_l.to(device), data_rot.to(device), rot_l.to(device)
        optimizer.zero_grad()


        class_loss = ...
        rot_loss = ...

        loss = class_loss + args.weight_RotTask_step1*rot_loss

        loss.backward()

        optimizer.step()

        _, cls_pred = ...
        _, rot_pred = ...


    acc_cls = ...
    acc_rot = ...

    return class_loss, acc_cls, rot_loss, acc_rot


def step1(args,feature_extractor,rot_cls,obj_cls,source_loader,device):
    optimizer, scheduler = get_optim_and_scheduler(feature_extractor,rot_cls,obj_cls, args.epochs_step1, args.learning_rate, args.train_all)


    for epoch in range(args.epochs_step1):
        print('Epoch: ',epoch)
        class_loss, acc_cls, rot_loss, acc_rot = _do_epoch(args,feature_extractor,rot_cls,obj_cls,source_loader,optimizer,device)
        print("Class Loss %.4f, Class Accuracy %.4f,Rot Loss %.4f, Rot Accuracy %.4f" % (class_loss.item(),acc_cls,rot_loss.item(), acc_rot))
        scheduler.step()