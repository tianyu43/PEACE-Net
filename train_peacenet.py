import os
import gc
import argparse as ag

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from utils.tools import get_parser_with_args_from_json, get_random_sample_indices, print_metrics
from utils.helper import seed_torch, get_pseudo_label_with_tw, built_model
from dataset.dataset import DynamicLabelDataset


def parse_args():
    parser = ag.ArgumentParser(description="PEACE-Net training (multi-site, multi-year)")
    parser.add_argument(
        "--config",
        default="configs/configs.json",
        help="Path to config json (default: configs/configs.json)",
    )
    parser.add_argument(
        "--data_root",
        default="F:/PEACE_Net_data",
        help="Dataset root path containing SITE/YEAR folders",
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        default=["site_usa_ia"],
        help="List of site names, e.g. site_usa_ia site_usa_il",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2022", "2023"],
        help="List of years, e.g. 2020 2021 2022",
    )
    parser.add_argument(
        "--save_root",
        default="./model_save/PEACE_Net",
        help="Root directory for saving checkpoints",
    )
    parser.add_argument(
        "--loss_root",
        default="./fig/loss/PEACE_Net",
        help="Root directory for saving loss curves",
    )
    parser.add_argument(
        "--share_labels",
        action="store_true",
        help="Store soft labels in shared memory for multi-worker loading",
    )
    return parser.parse_args()


def main():
    cli_args = parse_args()
    args = get_parser_with_args_from_json(cli_args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_torch(args.seed)

    for site in cli_args.sites:
        for year in cli_args.years:
            print(f"==================={site}+{year}==================")

            data_x = np.load(os.path.join(cli_args.data_root, site, year, "x_sr_s.npy"))
            data_y = np.load(os.path.join(cli_args.data_root, site, year, "label.npy"))

            pseudo_label = get_pseudo_label_with_tw(
                data_x,
                data_y,
                start_t_gwcci=12,
                start_t_gwcci_2=14,
                start_t_smci=14,
                m="max",
            )

            save_dir = os.path.join(cli_args.save_root, f"{site}_{year}")
            os.makedirs(save_dir, exist_ok=True)

            n_samples = int(5000 * 5000 * 0.005 * 2)
            train_size = 0.5

            idx = get_random_sample_indices(
                pseudo_label, n_samples=n_samples, seed=args.seed
            )

            split_idx = int(n_samples * train_size)
            x_train, x_eval = data_x[idx][:split_idx], data_x[idx][split_idx:]
            y_train, y_eval = data_y[idx][:split_idx], data_y[idx][split_idx:]
            pseudo_train = pseudo_label[idx][:split_idx]
            pseudo_eval = pseudo_label[idx][split_idx:]

            print_metrics(y_train, pseudo_train)
            print("train_size:", x_train.shape[0])
            print("val_size:", x_eval.shape[0])

            del data_x, data_y, pseudo_label

            pin_memory = torch.cuda.is_available()
            share_labels = cli_args.share_labels or (
                args.use_soft_prob_update and args.num_workers > 0
            )
            dataset_train = DynamicLabelDataset(
                x_train, pseudo_train, y_train, share_memory=share_labels
            )
            loader_train = DataLoader(
                dataset_train,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                persistent_workers=args.num_workers > 0,
            )

            dataset_val = DynamicLabelDataset(
                x_eval, pseudo_eval, y_eval, share_memory=share_labels
            )
            loader_val = DataLoader(
                dataset_val,
                batch_size=args.val_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.val_num_workers,
                pin_memory=pin_memory,
                persistent_workers=args.val_num_workers > 0,
            )

            model = built_model("PEACE_Net", args).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
            lr_scheduler = StepLR(
                optimizer, step_size=args.UPDATE_STEP * 2, gamma=0.9
            )

            weight_contrastive = args.weight_contrastive
            weight_classification = args.weight_classification
            temperature = args.temperature

            from utils.loss import sim_loss_with_margin, binary_soft_ce_loss

            loss_list_train = []
            loss_list_val = []
            acc_p_list = []
            acc_t_list = []
            pseudo_label_acc_list = []
            best_acc = 0.0

            for epoch in range(args.EPOCHS):
                epoch_id = epoch + 1
                model.train()
                loss_total_train = 0.0
                co_loss_total_train = 0.0
                cl_loss_total_train = 0.0

                for input_x, y_pseudo, y_true in loader_train:
                    input_x = input_x.to(device, non_blocking=True)
                    y_pseudo = y_pseudo.to(device, non_blocking=True)
                    y_true = y_true.to(device, non_blocking=True)

                    contrastive_out, class_out, _, _ = model(input_x)
                    co_loss = sim_loss_with_margin(contrastive_out, y_pseudo)
                    cl_loss = binary_soft_ce_loss(class_out, y_pseudo)
                    loss = weight_contrastive * co_loss + weight_classification * cl_loss

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                    co_loss_total_train += co_loss.item()
                    cl_loss_total_train += cl_loss.item()
                    loss_total_train += loss.item()

                loss_list_train.append(loss_total_train / len(loader_train))
                lr_scheduler.step()

                # update soft labels
                all_new_probs = []
                all_old_label = []
                all_y_t = []
                model.eval()
                with torch.no_grad():
                    for input_x, y_pseudo, y_true in loader_train:
                        input_x = input_x.to(device, non_blocking=True)
                        y_pseudo = y_pseudo.to(device, non_blocking=True)
                        y_true = y_true.to(device, non_blocking=True)

                        _, logits, _, _ = model(input_x)
                        probs = F.softmax(logits / temperature, dim=1)
                        all_new_probs.append(probs)
                        all_old_label.append(y_pseudo)
                        all_y_t.append(y_true)

                all_new_probs = torch.cat(all_new_probs, dim=0).float().cpu()
                all_old_label = torch.cat(all_old_label, dim=0).float().cpu()
                all_y_t = torch.cat(all_y_t, dim=0).cpu()

                pseudo_label_acc = np.mean(
                    torch.argmax(all_old_label, dim=1).numpy() == all_y_t.numpy()
                )
                pseudo_label_acc_list.append(pseudo_label_acc)
                changed_ratio = np.mean(
                    torch.argmax(all_old_label, dim=1).numpy()
                    != torch.argmax(all_new_probs, dim=1).numpy()
                )
                delta_mean = torch.mean(torch.abs(all_new_probs - all_old_label)).item()
                print(
                    f"epoch {epoch_id}: pseudo-label-acc={pseudo_label_acc:.4f} "
                    f"changed_ratio={changed_ratio:.4f} delta_mean={delta_mean:.6f}"
                )

                if args.use_soft_prob_update:
                    if epoch_id >= args.warmup_epochs and epoch_id % args.UPDATE_STEP == 0:
                        alpha = 0.5 + 0.4 * (epoch_id / args.EPOCHS)
                        dataset_train.update_soft_labels(
                            all_old_label, all_new_probs, alpha=alpha
                        )

                # validation
                y_hard_val = []
                label_true = []
                label_pseudo = []
                loss_total_val = 0.0
                co_loss_total_val = 0.0
                cl_loss_total_val = 0.0

                model.eval()
                with torch.no_grad():
                    for input_x, y_pseudo, y_true in loader_val:
                        input_x = input_x.to(device, non_blocking=True)
                        y_pseudo = y_pseudo.to(device, non_blocking=True)
                        y_true = y_true.to(device, non_blocking=True)

                        contrastive_out, class_out, _, _ = model(input_x)
                        co_loss = sim_loss_with_margin(contrastive_out, y_pseudo)
                        cl_loss = binary_soft_ce_loss(class_out, y_pseudo)
                        loss = (
                            weight_contrastive * co_loss
                            + weight_classification * cl_loss
                        )

                        co_loss_total_val += co_loss.item()
                        cl_loss_total_val += cl_loss.item()
                        loss_total_val += loss.item()

                        y_out = torch.argmax(class_out, dim=1)
                        y_hard_val.append(y_out.cpu())
                        label_pseudo.append(y_pseudo.cpu())
                        label_true.append(y_true.cpu())

                loss_list_val.append(loss_total_val / len(loader_val))
                y_hard_val = torch.cat(y_hard_val, dim=0).numpy()
                label_pseudo = torch.argmax(torch.cat(label_pseudo, dim=0), dim=1).numpy()
                label_true = torch.cat(label_true, dim=0).numpy()

                acc_p = np.mean(y_hard_val == label_pseudo)
                acc_t = np.mean(y_hard_val == label_true)
                acc_p_list.append(acc_p)
                acc_t_list.append(acc_t)

                print(
                    f"epoch {epoch_id}: val_loss={loss_list_val[-1]:.4f} "
                    f"acc_p={acc_p:.4f} acc_t={acc_t:.4f}"
                )

                if acc_t >= best_acc:
                    best_acc = acc_t
                    save_path = os.path.join(save_dir, f"peace-model-{best_acc:.4f}.pth")
                    save_path_best = os.path.join(save_dir, "peace-model-best.pth")
                    torch.save(model.state_dict(), save_path)
                    torch.save(model.state_dict(), save_path_best)
                    print(f"model saved: best_acc={best_acc:.4f}")

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # save acc curves
            np.savez(
                f"acc_list_{site}_{year}.npz",
                acc_p=acc_p_list,
                acc_t=acc_t_list,
                pseudo_acc=pseudo_label_acc_list,
            )

            try:
                import matplotlib.pyplot as plt

                os.makedirs(cli_args.loss_root, exist_ok=True)
                plt.figure(figsize=(10, 5))
                plt.title(f"Loss Curve {site}_{year}")
                plt.scatter(range(len(loss_list_train)), loss_list_train, color="blue", label="Loss_train")
                plt.scatter(range(len(loss_list_val)), loss_list_val, color="red", label="Loss_val")
                plt.ylim(0, 1)
                plt.legend()
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.tight_layout()
                plt.savefig(os.path.join(cli_args.loss_root, f"peace-net_{site}_{year}.png"), dpi=300)
            except Exception as exc:
                print(f"skip loss plot: {exc}")

            print(f"*******finish,best acc:{best_acc:.4f}*******")


if __name__ == "__main__":
    main()
