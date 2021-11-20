from torch import optim


def get_optim_and_scheduler(feature_extractor,rot_cls,obj_cls, epochs, lr, train_all):

    if train_all:
        params = list(feature_extractor.parameters()) + list(rot_cls.parameters()) + list(obj_cls.parameters())
    else:
        params = list(rot_cls.parameters()) + list(obj_cls.parameters())

    optimizer = optim.SGD(params, weight_decay=.0005, momentum=.9, lr=lr)
    step_size = int(epochs * .8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)


    print("Step size: %d" % step_size)
    return optimizer, scheduler