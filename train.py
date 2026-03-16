import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from model import UNet
from datapipe import TimeSeriesDataset
from bsp_torch import spec_loss


def load_config(config_path: str = "config.yaml") -> dict:
    """Load training configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def batch_loss(model, batch, rollout_steps: int):
    pred = model(batch[:, 0])
    loss = torch.mean((pred - batch[:, 1]) ** 2)
    loss += spec_loss(pred, batch[:, 1])
    for i in range(2, rollout_steps):
        # Pushforward trick
        x_in = pred.detach().clone()
        pred = model(x_in)
        w = 0.9 ** i
        loss += w * torch.mean((pred - batch[:, i]) ** 2)
        loss += w * spec_loss(pred, batch[:, i])
    return 10.0 * loss


def main():
    print(torch.__version__)
    print("Num GPUS : ", torch.cuda.device_count())

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--load", action="store_true", help="Load previous checkpoint if it exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    chkpts_path_outputs = Path(cfg["chkpts_path_outputs"])
    chkpts_path_outputs.mkdir(parents=True, exist_ok=True)
    net_name = cfg["net_name"]
    batch_size = cfg["batch_size"]
    batch_time = cfg["batch_time"]
    learning_rate = cfg["learning_rate"]
    epochs = cfg["epochs"]
    train_size = cfg.get("train_size", 320)
    print(net_name + " Load: ", args.load)

    cfg_device = cfg.get("device", "cpu")
    if cfg_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(cfg_device)

    with open(cfg["data_path"], "rb") as f:
        data = np.swapaxes(np.load(f), 1, -1)[:, :3]
    print("Shape:", data.shape, " Max: ", np.max(data), " Min: ", np.min(data))

    if train_size <= batch_time:
        raise ValueError("train_size must be greater than batch_time for windowed training.")
    if data.shape[0] - train_size <= batch_time:
        raise ValueError("Test split must contain more than batch_time steps.")

    train_dataset = TimeSeriesDataset(data[:train_size], seq_len=batch_time)
    test_dataset = TimeSeriesDataset(data[train_size:], seq_len=batch_time)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    del data

    x_ch = cfg.get("n_channels", 3)
    torch.set_printoptions(precision=8)

    model = UNet(n_channels=x_ch).to(device)
    ckpt_path = chkpts_path_outputs / f"{net_name}_chkpt.pt"
    if args.load:
        if ckpt_path.exists():
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Loaded checkpoint: {ckpt_path}")
        else:
            print(f"Checkpoint not found, training from scratch: {ckpt_path}")

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    print("Params : ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, batch, rollout_steps=batch_time)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        model.eval()
        test_running = 0.0
        with torch.no_grad():
            for test_batch in test_loader:
                test_batch = test_batch.to(device)
                test_running += batch_loss(model, test_batch, rollout_steps=batch_time).item()
        test_loss = test_running / len(test_loader)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch: {ep}/{epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, LR: {current_lr:.3e}")

        if test_loss < best_loss:
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), ckpt_path)
            best_loss = test_loss
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
