import os, shutil
import torch
import torch.nn.functional as F
import sys
import cv2
import numpy as np
from torch import nn
from datetime import datetime
from Encoder import Mnet
from itertools import chain, cycle
from datetime import datetime
import torch.optim as optim
from tqdm import tqdm
import random
from masked import maske_aug
from Diceloss import dice_loss
from data import get_loader, test_dataset
from utils import clip_gradient, adjust_lr, consistency_weight
from Decoder import Main_Decoder, Aux_decoders, Decoder
from torch.utils.data import DataLoader
from torch.nn import functional as F
from loss_F import iou_loss, structure_loss, pr_loss, Sm_loss1
# from smooth_loss import get_saliency_smoothness
import torch.backends.cudnn as cudnn
from options import opt
import torchvision.transforms as transforms
cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
BCE = torch.nn.BCEWithLogitsLoss()
def cal_ual(seg_logits, seg_gts):
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    sigmoid_x = seg_logits.sigmoid()
    loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
    return loss_map.mean()


def sup_loss(score_body, score_edge, score_1, score, label, label_bodys, label_edges):
    body_loss = structure_loss(score_body, label_bodys) +  pr_loss(score_body, label_bodys)
    edge_loss = structure_loss(score_edge, label_edges) + Sm_loss1(score_edge, label_edges)
    sal_loss1 = structure_loss(score_1, label) +  pr_loss(score_1, label) +  Sm_loss1(score_1, label)
    sal_loss = structure_loss(score, label) +  pr_loss(score, label) + Sm_loss1(score, label)
    ual_loss = cal_ual(score, label)
    return sal_loss + sal_loss1 + 0.8 * edge_loss +  0.8 * body_loss + ual_loss


# train function
def train(train_loader, Encoder, Main_D, Aux_decoders, optimizer, epoch, save_path):
    global step, best_avg_loss, image_unsup_root, best_avg_epoch
    Encoder.train()
    Main_D.train()
    Aux_decoders.train()
    Encoder.zero_grad()
    Main_D.zero_grad()
    Aux_decoders.zero_grad()
    total_step = len(train_loader_unsup)
    loss_all = 0
    epoch_step = 0

    try:
        # for i, (images, gts, t) in enumerate(train_loader, start=1):   image, t, gt, body, edge
        for batch_idx in range(len(train_loader_unsup)):
            (sup_images, sup_t, sup_gts, sup_bodys, sup_edges), (unsup_images, unsup_t, unsup_gts, _, _) = next(train_loader)
            sup_rgb = sup_images.cuda()
            sup_t = sup_t.cuda()
            sup_gts = sup_gts.cuda()
            N, c, h, w = sup_gts.size()
            sup_bodys = sup_bodys.cuda()
            sup_edges = sup_edges.cuda()
            unsup_rgb = unsup_images.cuda()
            unsup_t = unsup_t.cuda()
            unsup_gts = unsup_gts.cuda()
            optimizer.zero_grad()

            unsup_rgb_ori = unsup_rgb
            unsup_t_ori = unsup_t
            random_number = random.randint(1, 12)
            if random_number >= 10:
                sup_rgb = maske_aug(sup_rgb)
                sup_t = maske_aug(sup_t)
                unsup_rgb = maske_aug(unsup_rgb)
                unsup_t = maske_aug(unsup_t)
            elif random_number % 2 ==0:
                sup_rgb = maske_aug(sup_rgb)
                unsup_rgb = maske_aug(unsup_rgb)
            elif random_number != 1:
                sup_t = maske_aug(sup_t)
                unsup_t = maske_aug(unsup_t)
            # i_iter = (epoch-10) * len(train_loader_unsup) + batch_idx

            if epoch <= opt.start_weakly_epoch:
############### compute supervised loss
                sup_body, sup_edge, sup_all = Encoder(sup_rgb, sup_t)
                score_body, score_edge, score_1, score = Main_D(sup_body, sup_edge, sup_all)
                loss_sup = sup_loss(score_body, score_edge, score_1, score, sup_gts, sup_bodys, sup_edges)

############### compute unsupervised loss
                unsup_body, unsup_edge, unsup_all = Encoder(unsup_rgb, unsup_t)
                un_score_body, un_score_edge, un_score_1, un_score = Main_D(unsup_body, unsup_edge, unsup_all)
                # x_unsup = unsup_all
                outputs_ul = [aux_decoder(unsup_all, un_score.detach()) for aux_decoder in Aux_decoders]
                targets = un_score.detach()
                targets_1 = un_score_1.detach()
    
                loss_unsup = sum([(dice_loss(u[0], targets)) for u in outputs_ul]) + 0.5 * sum([(dice_loss(u[0], targets_1)) for u in outputs_ul])
                loss_unsup = (loss_unsup / len(outputs_ul))
                       
                weight_u = cons_w_unsup(epoch=epoch, curr_iter=batch_idx)
                loss_unsup = loss_unsup * weight_u

                loss_weaklysup = loss_unsup * 0
                total_loss = loss_unsup + loss_sup + loss_weaklysup
    
                sal_loss = total_loss.mean()
                loss = sal_loss
                loss.backward()
                clip_gradient(optimizer, opt.clip)
                optimizer.step()

            else:
############### compute supervised loss
                sup_body, sup_edge, sup_all = Encoder(sup_rgb, sup_t)
                score_body, score_edge, score_1, score = Main_D(sup_body, sup_edge, sup_all)
                loss_sup = sup_loss(score_body, score_edge, score_1, score, sup_gts, sup_bodys, sup_edges)

############### compute weakly-supervised loss
                unsup_body, unsup_edge, unsup_all = Encoder(unsup_rgb, unsup_t)
                un_score_body, un_score_edge, un_score_1, un_score = Main_D(unsup_body, unsup_edge, unsup_all)
                loss_weaklysup = structure_loss(un_score, unsup_gts) + 0.8 * structure_loss(un_score_1, unsup_gts)

                weight_weakly = cons_w_weakly(epoch=(epoch - opt.start_weakly_epoch), curr_iter=batch_idx)
                weight_u = weight_weakly
                loss_weaklysup = weight_weakly * loss_weaklysup

############### compute unsupervised loss
                # x_unsup = unsup_all
                outputs_ul = [aux_decoder(unsup_all, un_score.detach()) for aux_decoder in Aux_decoders]
                targets = un_score.detach()
                loss_unsup = sum([(dice_loss(u[0], targets)) for u in outputs_ul]) + sum([(dice_loss(u[0], unsup_gts)) for u in outputs_ul])

                loss_unsup = (loss_unsup / len(outputs_ul))
                loss_unsup = loss_unsup  * 2

                total_loss = loss_unsup + loss_sup + loss_weaklysup

                sal_loss = total_loss.mean()
                loss = sal_loss
                loss.backward()
                clip_gradient(optimizer, opt.clip)
                optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data
            if batch_idx % 20 == 0 or batch_idx == total_step or batch_idx == 1:
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}, un_w:{:.4f}||sal_loss:{:4f}||loss_sup:{:4f}||loss_weaklysup:{:4f}||loss_unsup:{:4f}'.format(
                        datetime.now(), epoch, opt.epoch, batch_idx, total_step,
                        optimizer.state_dict()['param_groups'][0]['lr'], weight_u, loss.data, loss_sup.data,
                        loss_weaklysup.data, loss_unsup.data))

            del sup_images, sup_t, sup_gts, unsup_images, unsup_t, unsup_gts, sup_bodys, sup_edges
            del total_loss, loss_unsup
        loss_all /= epoch_step
        print('Epoch [{:03d}/{:03d}], ||epoch_avg_loss:{:4f}  ... Best_AVG_loss:{} ... Best_AVG_epoch:{}'.format(epoch,
                                                                                                                 opt.epoch,
                                                                                                                 loss_all.data,
                                                                                                                 best_avg_loss,
                                                                                                                 best_avg_epoch))
        if epoch <= opt.start_weakly_epoch + opt.weakly_warm_epochs:
            best_avg_loss = loss_all
        else:
            if best_avg_loss >= loss_all:
                best_avg_loss = loss_all
                best_avg_epoch = epoch
                torch.save(Encoder.state_dict(), save_path + 'Best_AVG_E_epoch.pth')
                torch.save(Main_D.state_dict(), save_path + 'Best_AVG_D_epoch.pth')
                print('Best AVG loss:{}, Best AVG epoch:{}'.format(best_avg_loss, epoch))
        if epoch == opt.epoch -1:
            torch.save(Encoder.state_dict(), save_path + 'E_epoch_{}.pth'.format(epoch))
            torch.save(Main_D.state_dict(), save_path + 'D_epoch_{}.pth'.format(epoch))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # torch.save(model.state_dict(), save_path + 'epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


def Pseudo_label_gen(epoch, Encoder, Main_D, image_unsup_root, root_path):
    Encoder.eval()
    Main_D.eval()
    out_path = os.path.join(image_unsup_root, 'GT/')
    with torch.no_grad():
        print('Generate Pseudo label data...')
        test_loader = test_dataset(image_unsup_root, root_path, opt.trainsize)
        for i in tqdm(range(test_loader.size)):
            image, gt, t, name = test_loader.load_data()
            label = transforms.ToTensor()(gt)
            label = label.unsqueeze(0).cuda()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            t = t.cuda()
            unsup_body, unsup_edge, unsup_all = Encoder(image, t)
            un_score_body, un_score_edge, un_score_1, un_score = Main_D(unsup_body, unsup_edge, unsup_all)
            res = F.interpolate(un_score, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            unsup_pre = (res - res.min()) / (res.max() - res.min() + 1e-8)
            unsup_pre[unsup_pre >= 0.7] = 1
            unsup_pre[unsup_pre != 1] = 0
            cv2.imwrite(out_path + name, unsup_pre * 255)

def val(test_loader, Encoder, Main_D, epoch, save_path):
    global best_mae, best_mae_epoch
    Encoder.eval()
    Main_D.eval()
    with torch.no_grad():
        mae_sum=0
        for i in range(test_loader.size):
            image, gt, t, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            t = t.cuda()
            sup_body, sup_edge, sup_all = Encoder(image, t)
            score_body, score_edge, score_1, score = Main_D(sup_body, sup_edge, sup_all)
            res = F.interpolate(score, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res-gt))*1.0/(gt.shape[0]*gt.shape[1])

        mae = mae_sum/test_loader.size
        print('Epoch: {} -MAE: {}  ...  bestMAE: {} ...  best MAE Epoch: {}'.format(epoch,mae,best_mae,best_mae_epoch))
        if epoch==1:
            best_mae = mae
        else:
            if mae<best_mae:
                best_mae   = mae
                best_mae_epoch = epoch
                torch.save(Encoder.state_dict(), save_path + 'Best_MAE_E_epoch.pth')
                torch.save(Main_D.state_dict(), save_path + 'Best_MAE_D_epoch.pth')
                print('best MAE epoch:{}'.format(epoch))

if __name__ == '__main__':
    random.seed(118)
    np.random.seed(118)
    torch.manual_seed(118)
    torch.cuda.manual_seed(118)
    torch.cuda.manual_seed_all(118)

    image_sup_root = opt.sup_gt_root
    image_unsup_root = opt.unsup_gt_root
    root_path = opt.root_path
    test_gt_root = opt.test_gt_root
    test_root_path = opt.test_root_path
    
    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    Encoder = nn.DataParallel(Mnet()).cuda()
    Main_D = nn.DataParallel(Main_Decoder()).cuda()
    Aux_decoders = Aux_decoders().cuda()

    num_gpus = torch.cuda.device_count()
    print(f"========>num_gpus:{num_gpus}==========")

    num_parms = 0
    for p in Encoder.parameters():
        num_parms += p.numel()
    for p_ in Main_D.parameters():
        num_parms += p_.numel()
    for p__ in Aux_decoders.parameters():
        num_parms += p__.numel()
    print("Total Parameters (For Reference): {}".format(num_parms))

    params_E = Encoder.parameters()
    params_D = Main_D.parameters()
    params_aux = Aux_decoders.parameters()

    parameters = chain(params_E, params_D, params_aux)
    optimizer = torch.optim.Adam(parameters, opt.lr)

    test_loader = test_dataset(test_gt_root, test_root_path, opt.trainsize)
    print('load data...')
    train_loader_sup = get_loader(image_sup_root, root_path, batchsize=opt.batchsize, trainsize=opt.trainsize)
    train_loader_unsup = get_loader(image_unsup_root, root_path, batchsize=opt.batchsize, trainsize=opt.trainsize)
    cons_w_unsup = consistency_weight(final_w = opt.un_warm_value, iters_per_epoch=len(train_loader_unsup), rampup_ends= opt.un_warm_epochs)
    cons_w_weakly = consistency_weight(final_w=opt.weakly_warm_value, iters_per_epoch=len(train_loader_unsup), rampup_ends=opt.weakly_warm_epochs)

    step = 0
    best_mae = 1
    best_mae_epoch = 0
    best_avg_loss = 1
    best_avg_epoch = 0

    print("Start train...")
    for epoch in range(1, opt.epoch):
        if epoch >= opt.start_weakly_epoch + 1:
            print('load data...')
            train_loader_sup = get_loader(image_sup_root, root_path, batchsize=opt.batchsize, trainsize=opt.trainsize)
            train_loader_unsup = get_loader(image_unsup_root, root_path, batchsize=opt.batchsize, trainsize=opt.trainsize)
        train_loader = iter(zip(cycle(train_loader_sup), train_loader_unsup))
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch_list)
        train(train_loader, Encoder, Main_D, Aux_decoders, optimizer, epoch, save_path)
        if epoch % 2 ==0:
            val(test_loader, Encoder, Main_D, epoch, save_path)
        if epoch >= opt.start_weakly_epoch and epoch % 2 == 0 and epoch < opt.epoch-1:
            Pseudo_label_gen(epoch, Encoder, Main_D, image_unsup_root, root_path)
