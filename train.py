import os, gc, sys, logging
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from utils.tools import get_parser_with_args_from_json, get_random_sample_indices
from utils.helper import seed_torch, get_pseudo_label, built_model
from dataset.dataset import DynamicLabelDataset




args = get_parser_with_args_from_json('configs/configs.json')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_torch(args.seed)




site_list = [
    #'site_usa_ia',
    #'site_usa_il',
    #'site_bz',
    'site_ag',
    #'site_hl',
]

year_list =[
    '2022',
    #'2023',
    #'2024'
]



for SITE in site_list:
    for YEAR in year_list:
        print(f'==================={SITE}+{YEAR}==================')
        
        # =============================== import data ===============================
        data_x = np.load(f'E:/PEACE-net_data/0_data/{SITE}/{YEAR}/x_sr_s.npy')
        data_y = np.load(f'E:/PEACE-net_data/0_data/{SITE}/{YEAR}/label.npy')
        print('总样本个数：', data_x.shape[0])
        
        pseudo_label = get_pseudo_label(data_x, data_y, SITE)
        #pseudo_label = data_y.copy()
        
        mod_save_dir = f'./model_save/PEACE_Net/{SITE}_{YEAR}/'
        os.makedirs(mod_save_dir, exist_ok=True)
        
        # =============================== dataset ===============================
        n_samples = int(5000*5000*0.005)
        train_size = 0.5
        
        idx = get_random_sample_indices(pseudo_label, n_samples = n_samples)
        x_train, x_eval = data_x[idx][:int(n_samples*2*train_size)], data_x[idx][int(n_samples*2*train_size):]
        y_train, y_eval = data_y[idx][:int(n_samples*2*train_size)], data_y[idx][int(n_samples*2*train_size):]
        pseudo_label_train, pseudo_label_eval = pseudo_label[idx][:int(n_samples*2*train_size)], pseudo_label[idx][int(n_samples*2*train_size):]
        
        print('训练集个数：', x_train.shape[0])
        print('验证集个数：', x_eval.shape[0])

        print(x_train.shape)

        del data_x, data_y, pseudo_label

        


        dataset_train = DynamicLabelDataset(x_train, pseudo_label_train, y_train)
        loader_train =  DataLoader(dataset_train, 
                            batch_size=args.batch_size, 
                            shuffle=False, drop_last=False,
                            #num_workers= 2, 
                            )

        dataset_val = DynamicLabelDataset(x_eval, pseudo_label_eval, y_eval)
        loader_val =  DataLoader(dataset_val,
                            batch_size=args.val_batch_size, 
                            shuffle=False, drop_last=False,
                            #num_workers= 2, 
                            )
        
        
        
        # =============================== model set up ===============================
        model = built_model('PEACE_Net', args)
        model.to(DEVICE)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,                          
            weight_decay=args.weight_decay
        )

        lr_scheduler = StepLR(optimizer, 
                              step_size=args.UPDATE_STEP*2, 
                              gamma=0.9
                              )
        
        wight_contrastiv = 1.0
        wight_classification = 1.0
        

        
        # =============================== train ===============================
        from tqdm import tqdm
        import matplotlib.pyplot as plt
        
        from utils.loss import sim_loss_with_margin, binary_soft_ce_loss
        
        loss_list_train = []
        loss_list_val = []
        acc_p_list = []
        acc_t_list = []
        pseudo_label_acc_list = []
        best_acc = 0.0

        T = 0.8

        for epoch in tqdm(range(args.EPOCHS), desc="Training"):
            epoch_id = epoch + 1

            # =================== train ===================
            model.train()
            loss_total_train = 0.0
            co_loss_total_train = 0.0
            cl_loss_total_train = 0.0

            for input_x, y_pseudo, y_true in loader_train:
                input_x = input_x.to(DEVICE, non_blocking=True)
                y_pseudo = y_pseudo.to(DEVICE, non_blocking=True)
                y_true = y_true.to(DEVICE, non_blocking=True)

                contrastive_out, class_out, _, _ = model(input_x)

                co_loss = sim_loss_with_margin(contrastive_out, y_pseudo)
                cl_loss = binary_soft_ce_loss(class_out, y_pseudo)
                loss = wight_contrastiv * co_loss + wight_classification * cl_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                co_loss_total_train += co_loss.item()
                cl_loss_total_train += cl_loss.item()
                loss_total_train += loss.item()

            avg_train_loss = loss_total_train / len(loader_train)
            avg_train_co = co_loss_total_train / len(loader_train)
            avg_train_cl = cl_loss_total_train / len(loader_train)
            loss_list_train.append(avg_train_loss)

            lr_scheduler.step()

            tqdm.write('========================================================')
            tqdm.write('training')
            tqdm.write(f"-Epoch {epoch_id}/{args.EPOCHS}, ***Loss***: {avg_train_loss:.4f}")
            tqdm.write(f"-co_loss: {avg_train_co:.4f}, -cl_loss: {avg_train_cl:.4f}")
            tqdm.write("-----------------------------------------")

            # =================== 更新 soft label ===================
            all_new_probs = []
            all_old_label = []
            all_y_t = []

            model.eval()
            with torch.no_grad():
                for input_x, y_pseudo, y_true in loader_train:
                    input_x = input_x.to(DEVICE, non_blocking=True)
                    y_pseudo = y_pseudo.to(DEVICE, non_blocking=True)
                    y_true = y_true.to(DEVICE, non_blocking=True)

                    _, logits, _, _ = model(input_x)
                    probs = F.softmax(logits / T, dim=1)  # (B, C)

                    all_new_probs.append(probs)
                    all_old_label.append(y_pseudo)
                    all_y_t.append(y_true)

            all_new_probs = torch.cat(all_new_probs, dim=0)
            all_old_label = torch.cat(all_old_label, dim=0)
            all_y_t = torch.cat(all_y_t, dim=0)

            pseudo_label_acc = np.mean(
                torch.argmax(all_old_label, dim=1).cpu().numpy()
                == all_y_t.cpu().numpy()
            )
            pseudo_label_acc_list.append(pseudo_label_acc)
            tqdm.write(f"伪标签准确率: {pseudo_label_acc:.4f}")

            # print("更新前label:", all_old_label[-1, :])
            # print("待更新label:", all_new_probs[-1, :])
            # print("true:", all_y_t[-1])
            if args.use_soft_prob_update:
                if epoch_id >= args.warmup_epochs and epoch_id % args.UPDATE_STEP == 0:
                    alpha = 0.5 + 0.4 * (epoch_id / args.EPOCHS)
                    dataset_train.update_soft_labels(all_old_label, all_new_probs, alpha=alpha)
                    # tqdm.write(f"更新 soft labels, alpha = {alpha:.4f}")

            # =================== validation ===================
            y_hard_val = []
            label_true = []
            label_pseudo = []

            loss_total_val = 0.0
            co_loss_total_val = 0.0
            cl_loss_total_val = 0.0

            model.eval()
            with torch.no_grad():
                for input_x, y_pseudo, y_true in loader_val:
                    input_x = input_x.to(DEVICE, non_blocking=True)
                    y_pseudo = y_pseudo.to(DEVICE, non_blocking=True)
                    y_true = y_true.to(DEVICE, non_blocking=True)

                    contrastive_out, class_out, _, _ = model(input_x)

                    co_loss = sim_loss_with_margin(contrastive_out, y_pseudo)
                    cl_loss = binary_soft_ce_loss(class_out, y_pseudo)
                    loss = wight_contrastiv * co_loss + wight_classification * cl_loss

                    co_loss_total_val += co_loss.item()
                    cl_loss_total_val += cl_loss.item()
                    loss_total_val += loss.item()

                    y_out = torch.argmax(class_out, dim=1)
                    y_hard_val.append(y_out.cpu())
                    label_pseudo.append(y_pseudo.cpu())
                    label_true.append(y_true.cpu())

            avg_val_loss = loss_total_val / len(loader_val)
            avg_val_co = co_loss_total_val / len(loader_val)
            avg_val_cl = cl_loss_total_val / len(loader_val)
            loss_list_val.append(avg_val_loss)

            y_hard_val = torch.cat(y_hard_val, dim=0).numpy()
            label_pseudo = torch.argmax(torch.cat(label_pseudo, dim=0), dim=1).numpy()
            label_true = torch.cat(label_true, dim=0).numpy()

            acc_p = np.mean(y_hard_val == label_pseudo)
            acc_t = np.mean(y_hard_val == label_true)

            acc_p_list.append(acc_p)
            acc_t_list.append(acc_t)

            tqdm.write('validation')
            tqdm.write(f"-Epoch {epoch_id}/{args.EPOCHS}, ***Loss***: {avg_val_loss:.4f}")
            tqdm.write(f"accuracy_with_pseudo: {acc_p:.4f}")
            tqdm.write(f"accuracy_with_true  : {acc_t:.4f}")

            # =================== save model ===================
            if acc_t >= best_acc:
                best_acc = acc_t
                save_path = os.path.join(mod_save_dir, f'peace-model-{best_acc:.4f}.pth')
                torch.save(model.state_dict(), save_path)
                tqdm.write(f'---模型已保存，准确率提升至 {best_acc:.4f}---')
            else:
                tqdm.write(f'---模型未保存，当前准确率 {acc_t:.4f}---')

            # 清空缓存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # =================== 保存 acc 曲线 ===================
        np.savez(f'acc_list_{SITE}_{YEAR}.npz',
                acc_p=acc_p_list,
                acc_t=acc_t_list,
                pseudo_acc=pseudo_label_acc_list)

        plt.figure(figsize=(10, 5))
        plt.title(f'Loss Curve {SITE}_{YEAR}')
        plt.scatter(range(len(loss_list_train)), loss_list_train, color='blue', label='Loss_train')
        plt.scatter(range(len(loss_list_val)), loss_list_val,color='red', label='Loss_val')
        plt.ylim(0, 1)
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.tight_layout()
        fig_save_dir = f'./fig/loss/PEACE_Net'
        os.makedirs(fig_save_dir, exist_ok=True)
        plt.savefig(os.path.join(fig_save_dir, f'peace-net_{SITE}_{YEAR}.png'), dpi=300)
        # plt.show()
        
        
        print(f'finish,best acc:{best_acc:.4f}')