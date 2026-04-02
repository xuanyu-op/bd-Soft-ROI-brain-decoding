import torch
import torch.nn.functional as F
import math


def charbonnier_loss(x, y, eps=1e-3):
    return torch.mean(torch.sqrt((x - y) ** 2 + eps ** 2))


def get_loss_fn(loss_name):
    """
    Return the corresponding loss function based on the given loss name. Supports mse, l1, huber, quantile, and charb.
    """
    loss_fns = {
        'mse': F.mse_loss,
        'l1': F.l1_loss,
        'huber': F.smooth_l1_loss,
        'quantile': lambda x, y: torch.quantile(torch.abs(x - y), 0.9),
        'charb': charbonnier_loss
    }
    return loss_fns[loss_name]


def build_optimizer(model, args, accelerator):
    """
    Build the AdamW optimizer.
    Parameters are split into groups: regular parameters use weight decay, while bias and LayerNorm-related parameters do not use weight decay.
    The maximum learning rate is also scaled by the number of processes.
    """
    max_lr = args.max_lr * accelerator.num_processes

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    opt_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 1e-2
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)
    return optimizer, max_lr


def compute_steps(subjects, num_train_samples, batch_size, world_size, num_epochs):
    """
    Compute the number of batches per subject, the total number of steps per epoch, and the total number of training steps based on the subject list, sample counts, batch size, number of processes, and number of epochs.
    """
    global_batch_size = batch_size * world_size

    batches_per_subject = {
        subj: max(1, math.ceil(num_train_samples[subj] / global_batch_size))
        for subj in subjects
    }

    steps_per_epoch = sum(batches_per_subject.values())
    total_steps = int(num_epochs * steps_per_epoch)

    return batches_per_subject, steps_per_epoch, total_steps


def build_scheduler(optimizer, args, total_steps, max_lr):
    """
    Build the learning rate scheduler according to the configuration. Supports linear and cycle scheduling strategies. Returns None if no valid scheduler type is matched.
    """
    if args.lr_scheduler_type == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=total_steps,
            last_epoch=-1
        )
    elif args.lr_scheduler_type == 'cycle':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            final_div_factor=1000,
            last_epoch=-1,
            pct_start=2 / args.num_epochs
        )
    else:
        lr_scheduler = None

    return lr_scheduler