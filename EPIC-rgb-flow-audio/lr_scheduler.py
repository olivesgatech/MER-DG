import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineSchedule(_LRScheduler):
    """
    Implements a learning rate schedule with linear warmup followed by cosine decay.
    Linearly increases learning rate from 0 to max_lr over `warmup_steps` training steps.
    Decreases learning rate from max_lr to 0 over remaining `total_steps - warmup_steps` steps following a cosine curve.
    
    This is commonly used in VICReg and self-supervised learning methods with LARS optimizer.
    """
    
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, 
                 last_epoch=-1):
        """
        Args:
            optimizer: Wrapped optimizer.
            warmup_steps: Number of steps for the warmup phase.
            total_steps: Total number of training steps.
            min_lr: Minimum learning rate at the end of scheduling. Default: 0.
            last_epoch: The index of the last epoch. Default: -1.
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super(WarmupCosineSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            lr_scale = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [base_lr * lr_scale for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = float(self.last_epoch - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay
                    for base_lr in self.base_lrs]


class LinearWarmupCosineDecayScheduler:
    """
    A simplified wrapper class to adjust learning rate with warmup and cosine decay.
    Can be used directly in the training loop without _LRScheduler inheritance.
    """
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=0.0, 
                 steps_per_epoch=None, start_lr=None):
        """
        Args:
            optimizer: The optimizer to adjust LR for
            warmup_epochs: Number of epochs for warmup
            total_epochs: Total number of epochs for training
            min_lr: Minimum LR at end of schedule (default: 0.0)
            steps_per_epoch: Steps per epoch (if None, will do epoch-wise adjustment)
            start_lr: Optional starting learning rate (if None, will use optimizer's LR)
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.steps_per_epoch = steps_per_epoch
        
        # Get base learning rates from optimizer
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            self.base_lrs.append(param_group['lr'])
        
        if start_lr is not None:
            self.start_lr = start_lr
        else:
            self.start_lr = self.base_lrs[0]
            
    def step(self, epoch, step=None):
        """
        Update learning rate based on current epoch and step
        
        Args:
            epoch: Current epoch number (0-indexed)
            step: Current step within the epoch (optional)
        """
        if step is not None and self.steps_per_epoch is not None:
            # Convert to overall step if step-level precision is desired
            overall_step = epoch * self.steps_per_epoch + step
            warmup_steps = self.warmup_epochs * self.steps_per_epoch
            total_steps = self.total_epochs * self.steps_per_epoch
            
            if overall_step < warmup_steps:
                # Linear warmup phase
                lr_scale = float(overall_step) / float(max(1, warmup_steps))
                lr = [base_lr * lr_scale for base_lr in self.base_lrs]
            else:
                # Cosine decay phase
                progress = float(overall_step - warmup_steps) / float(
                    max(1, total_steps - warmup_steps)
                )
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = [self.min_lr + (base_lr - self.min_lr) * cosine_decay
                      for base_lr in self.base_lrs]
        else:
            # Epoch-level adjustment
            if epoch < self.warmup_epochs:
                # Linear warmup phase
                lr_scale = float(epoch) / float(max(1, self.warmup_epochs))
                lr = [base_lr * lr_scale for base_lr in self.base_lrs]
            else:
                # Cosine decay phase
                progress = float(epoch - self.warmup_epochs) / float(
                    max(1, self.total_epochs - self.warmup_epochs)
                )
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = [self.min_lr + (base_lr - self.min_lr) * cosine_decay
                      for base_lr in self.base_lrs]
        
        # Update optimizer learning rates
        for param_group, new_lr in zip(self.optimizer.param_groups, lr):
            param_group['lr'] = new_lr
        
        return lr[0]  # Return current learning rate
