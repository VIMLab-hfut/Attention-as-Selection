
import argparse
import os
import random
import time
from multiprocessing import freeze_support

import numpy as np
import torch
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from data_provider.data_factory import data_provider
from models import AAS
from utils.logger import get_logger
from utils.tools import EarlyStopping, adjust_learning_rate, load_content, vali


os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


parser = argparse.ArgumentParser(description="BALM-TSF")

# basic
parser.add_argument("--task_name", type=str, default="long_term_forecast")
parser.add_argument("--is_training", type=int, default=1)
parser.add_argument("--model_id", type=str, default="test")
parser.add_argument("--model_comment", type=str, default="none")
parser.add_argument("--model", type=str, default="BALM")
parser.add_argument("--seed", type=int, default=2021)

# data
parser.add_argument("--data", type=str, default="ETTm1")
parser.add_argument("--root_path", type=str, default="./dataset")
parser.add_argument("--data_path", type=str, default="ETTm1.csv")
parser.add_argument("--features", type=str, default="M")
parser.add_argument("--target", type=str, default="MT_320")
parser.add_argument("--loader", type=str, default="modal")
parser.add_argument("--freq", type=str, default="h")
parser.add_argument("--checkpoints", type=str, default="./checkpoints/")

# forecasting
parser.add_argument("--seq_len", type=int, default=96)
parser.add_argument("--label_len", type=int, default=48)
parser.add_argument("--pred_len", type=int, default=96)
parser.add_argument("--seasonal_patterns", type=str, default="Monthly")

# model
parser.add_argument("--enc_in", type=int, default=7)
parser.add_argument("--dec_in", type=int, default=7)
parser.add_argument("--c_out", type=int, default=7)
parser.add_argument("--d_model", type=int, default=16)
parser.add_argument("--n_heads", type=int, default=8)
parser.add_argument("--e_layers", type=int, default=2)
parser.add_argument("--d_layers", type=int, default=1)
parser.add_argument("--d_ff", type=int, default=32)
parser.add_argument("--moving_avg", type=int, default=25)
parser.add_argument("--factor", type=int, default=1)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--embed", type=str, default="fixed")
parser.add_argument("--activation", type=str, default="gelu")
parser.add_argument("--output_attention", action="store_true")
parser.add_argument("--patch_len", type=int, default=16)
parser.add_argument("--stride", type=int, default=8)
parser.add_argument("--prompt_domain", type=int, default=0)
parser.add_argument("--llm_model", type=str, default="GPT2")
parser.add_argument("--llm_dim", type=int, default=768)
parser.add_argument("--mask_rate", type=float, default=0)

# optimization
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--itr", type=int, default=1)
parser.add_argument("--train_epochs", type=int, default=10)
parser.add_argument("--align_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--eval_batch_size", type=int, default=8)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--des", type=str, default="test")
parser.add_argument("--loss", type=str, default="MSE")
parser.add_argument("--lradj", type=str, default="type1")
parser.add_argument("--pct_start", type=float, default=0.2)
parser.add_argument("--use_amp", action="store_true")
parser.add_argument("--llm_layers", type=int, default=6)
parser.add_argument("--percent", type=int, default=100)
parser.add_argument("--device", type=str, default="cuda:0")
# wandb
parser.add_argument("--version_num", default="BALM", type=str)
parser.add_argument("--run_name", default="test", type=str)
parser.add_argument("--wandb_flag", type=int, default=1)
parser.add_argument("--wd_project", default="BALM_test", type=str)



def main():
    args = parser.parse_args()
    wandb.login()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        mixed_precision="bf16", 
    )

    for ii in range(args.itr):
        setting = (f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_"
                   f"ft{args.features}_sl{args.seq_len}_ll{args.label_len}_"
                   f"pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_"
                   f"el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_"
                   f"fc{args.factor}_eb{args.embed}_{args.des}_{ii}_{args.llm_model}")

        run_name = (f"v{args.version_num}_r{args.run_name}_nh{args.n_heads}_"
                    f"dn{args.data_path}_pl{args.pred_len}_mask{args.mask_rate}")

        wandb_group_name = setting
        wandb_run_name   = f"{run_name}_seed{args.seed}_it{ii}"

        if args.wandb_flag and accelerator.is_local_main_process:
            run = wandb.init(project=args.wd_project,
                             name=wandb_run_name,
                             group=wandb_group_name,
                             config=args)
        else:
            run = wandb.init(mode="disabled")

        train_data, train_loader = data_provider(args, "train")
        vali_data,   vali_loader   = data_provider(args, "val")
        test_data,   test_loader   = data_provider(args, "test")

        model = AAS.Model(args).float()
        path = os.path.join(args.checkpoints, setting + "-" + args.model_comment)
        # args.content = load_content(args)
        if not os.path.exists(path) and accelerator.is_local_main_process:
            os.makedirs(path, exist_ok=True)
        logger = get_logger(path, __name__, f"record_s{args.seed}.log")
        logger.info(args)
        args.logger = logger

        time_now       = time.time()
        trainning_time = 0
        train_steps    = len(train_loader)
        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

        trained_params = [p for p in model.parameters() if p.requires_grad]
        model_optim    = optim.Adam(trained_params, lr=args.learning_rate)

        if args.lradj == "COS":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                model_optim, T_max=20, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(
                optimizer=model_optim,
                steps_per_epoch=train_steps,
                pct_start=args.pct_start,
                epochs=args.train_epochs,
                max_lr=args.learning_rate)

        criterion  = nn.MSELoss()
        mae_metric = nn.L1Loss()

        train_loader, vali_loader, test_loader, model, model_optim, scheduler = \
            accelerator.prepare(train_loader, vali_loader, test_loader,
                                model, model_optim, scheduler)
        
        
        if accelerator.is_local_main_process:
            model.cross_t2s.selector.save_analysis = True
            model.cross_t2s.selector.analysis_dir = \
                f"./analysis_cache/{setting}/"
            model.cross_t2s.selector.step_counter = 0
        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []
            model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,embeddings) in \
                    tqdm(enumerate(train_loader), total=len(train_loader), disable=not accelerator.is_local_main_process):

                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float()
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float()
                batch_y_mark = batch_y_mark.float()

                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :])
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1)

                # ========== forward ==========
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        if args.output_attention:
                            outputs = model(
                                batch_x, batch_x_mark,
                                dec_inp, batch_y_mark,
                                embeddings,
                                batch_y=batch_y,
                                epoch=epoch,
                                mode="train"
                            )[0]
                        else:
                            outputs, alignment_loss = model(
                                batch_x, batch_x_mark,
                                dec_inp, batch_y_mark,
                                embeddings,
                                batch_y=batch_y,
                                epoch=epoch,
                                mode="train"
                            )
                else:
                    if args.output_attention:
                        outputs = model(
                            batch_x, batch_x_mark,
                            dec_inp, batch_y_mark,
                            embeddings=embeddings,
                            batch_y=batch_y,
                            epoch=epoch,
                            mode="train"
                        )[0]
                    else:
                        outputs, alignment_loss = model(
                            batch_x, batch_x_mark,
                            dec_inp, batch_y_mark,
                            embeddings=embeddings,
                            batch_y=batch_y,
                            epoch=epoch,
                            mode="train"
                        )

                f_dim = -1 if args.features == "MS" else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]
                # print(criterion(outputs, batch_y))
                # print(alignment_loss)
                loss = criterion(outputs, batch_y) + alignment_loss
                train_loss.append(loss.item())

                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    accelerator.backward(loss)
                    model_optim.step()

                if args.lradj == "TST":
                    adjust_learning_rate(accelerator, model_optim, scheduler,
                                         epoch + 1, args, printout=False)
                    scheduler.step()


            one_epoch_time = time.time() - epoch_time
            trainning_time += one_epoch_time

            train_loss = np.average(train_loss)
            vali_loss, vali_mae, vali_mape, _, _ = vali(
                args, accelerator, model, vali_data, vali_loader,
                criterion, mae_metric, epoch, mode="vali")
            test_loss, test_mae, test_mape, _, _ = vali(
                args, accelerator, model, test_data, test_loader,
                criterion, mae_metric, epoch, mode="test")

            accelerator.print(
                f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} "
                f"Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f} MAE: {test_mae:.7f}")

            if accelerator.is_local_main_process:
                wandb.log({"Train Loss": train_loss,
                           "Vali Loss": vali_loss,
                           "Test MSE": test_loss,
                           "Test MAE": test_mae,
                           "Test MAPE": test_mape,
                           "epoch": epoch})

            early_stopping(vali_loss, model, path)
            if early_stopping.early_stop:
                accelerator.print("Early stopping")
                break

            if args.lradj != "TST":
                if args.lradj == "COS":
                    scheduler.step()
                else:
                    adjust_learning_rate(accelerator, model_optim, scheduler,
                                         epoch + 1, args, printout=True)


        if accelerator.is_local_main_process:
            run.finish()

    accelerator.wait_for_everyone()



if __name__ == "__main__":
    freeze_support()
    main()