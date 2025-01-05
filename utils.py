import ramps
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

# def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=[21, 65, 85]):
#     if epoch <= 9:
#         de_in = 2 - ((epoch - 1) / 4)
#         decay = decay_rate ** (de_in)
#     elif epoch < decay_epoch[0]:
#         decay = 1
#     elif epoch < decay_epoch[1]:
#         decay = decay_rate ** (1)
#     elif epoch < decay_epoch[2]:
#         decay = decay_rate ** (2)
#     else:
#         decay = decay_rate ** (3)
# 
#     # decay = decay_rate ** (epoch // decay_epoch)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = decay * init_lr
#         lr = param_group['lr']
#     return lr
def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=[21, 65, 85]):
    if epoch < decay_epoch[0]:
        decay = 1
    elif epoch < decay_epoch[1]:
        decay = decay_rate ** (1)
    elif epoch < decay_epoch[2]:
        decay = decay_rate ** (2)
    else:
        decay = decay_rate ** (3)

    # decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * init_lr
        lr = param_group['lr']
    return lr
class consistency_weight(object):
    def __init__(self, final_w, iters_per_epoch, rampup_starts=0, rampup_ends=7, ramp_type='sigmoid_rampup'):
        self.final_w = final_w   #30
        self.iters_per_epoch = iters_per_epoch   #2000
        self.rampup_starts = rampup_starts * iters_per_epoch  #0
        self.rampup_ends = rampup_ends * iters_per_epoch   #18*2000
        self.rampup_length = (self.rampup_ends - self.rampup_starts)  #18*2000
        self.rampup_func = getattr(ramps, ramp_type)
        self.current_rampup = 0

    def __call__(self, epoch, curr_iter):
        cur_total_iter = self.iters_per_epoch * epoch + curr_iter
        if cur_total_iter < self.rampup_starts:
            return 0
        self.current_rampup = self.rampup_func(cur_total_iter - self.rampup_starts, self.rampup_length) #[0,e-5]
        return self.final_w * self.current_rampup   #[0.2-30]