import os
import cv2
from tqdm import tqdm

import sod_metrics as M

FM = M.Fmeasure()
WFM = M.WeightedFmeasure()
SM = M.Smeasure()
EM = M.Emeasure()
MAE = M.MAE()

mask_root = '/home/wj/SOD/RGB-T datasets/'
pred_root = './output/'

pred_datasets = os.listdir(pred_root)
for pred_dataset in pred_datasets:
    pred_path = pred_root + '/' + str(pred_dataset) + '/'
    if pred_dataset[:6] == 'vt5000':
        mask_path = mask_root + '/vt5000/' + '/GT/'
    elif pred_dataset[:6] == 'vt1000':
        mask_path = mask_root + '/vt1000/' + '/GT/'
    else:
        mask_path = mask_root + '/vt821/' + '/GT/'

    mask_name_list = sorted(os.listdir(mask_path))
    for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
        mask_path = os.path.join(mask_path, mask_name)
        pred_path = os.path.join(pred_path, mask_name[:-4] + '.png')
        print(mask_path)
        print(pred_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        MAE.step(pred=pred, gt=mask)

    fm = FM.get_results()['fm']
    wfm = WFM.get_results()['wfm']
    sm = SM.get_results()['sm']
    em = EM.get_results()['em']
    mae = MAE.get_results()['mae']
    print(pred_dataset)
    print(
        'Smeasure:', sm.round(4), '; ',
        'wFmeasure:', wfm.round(4), '; ',
        'MAE:', mae.round(4), '; ',
        'adpEm:', em['adp'].round(4), '; ',
        'meanEm:', '-' if em['curve'] is None else em['curve'].mean().round(4), '; ',
        'maxEm:', '-' if em['curve'] is None else em['curve'].max().round(4), '; ',
        'adpFm:', fm['adp'].round(4), '; ',
        'meanFm:', fm['curve'].mean().round(4), '; ',
        'maxFm:', fm['curve'].max().round(4),
        sep=''
    )

    with open("../result.txt", "a+") as f:
        print(
            'Smeasure:', sm.round(4), '; ',
            'wFmeasure:', wfm.round(4), '; ',
            'MAE:', mae.round(4), '; ',
            'adpEm:', em['adp'].round(4), '; ',
            'meanEm:', '-' if em['curve'] is None else em['curve'].mean().round(4), '; ',
            'maxEm:', '-' if em['curve'] is None else em['curve'].max().round(4), '; ',
            'adpFm:', fm['adp'].round(4), '; ',
            'meanFm:', fm['curve'].mean().round(4), '; ',
            'maxFm:', fm['curve'].max().round(4), '; ',
            file=f
        )
        # print('Smeasure:', sm.round(3), '; ',
        #       'meanEm:', '-' if em['curve'] is None else em['curve'].mean().round(3), '; ',
        #       'wFmeasure:', wfm.round(3), '; ',
        #       'MAE:', mae.round(3), '; ',
        #       file=f
        #       )
