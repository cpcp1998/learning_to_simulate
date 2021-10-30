import argparse

from tqdm import tqdm
import torch
import torch_geometric as pyg

import dataset
import model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--noise", type=float, default=6.7e-4)
    args = parser.parse_args()

    train_dataset = dataset.OneStepDataset(args.data_path, "train", noise_std=args.noise)
    valid_dataset = dataset.OneStepDataset(args.data_path, "valid", noise_std=0.0)
    train_loader = pyg.loader.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    valid_loader = pyg.loader.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    simulator = model.LearnedSimulator()
    simulator = simulator.cuda()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(simulator.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1 / 5e6))

    for epoch in range(args.epoch):
        simulator.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        total_loss = 0
        batch_count = 0
        for data in progress_bar:
            optimizer.zero_grad()
            data = data.cuda()
            pred = simulator(data)
            loss = loss_fn(pred, data.y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix({"loss": loss.item(), "avg_loss": total_loss / batch_count, "lr": optimizer.param_groups[0]["lr"]})

        simulator.eval()
        total_loss = 0
        batch_count = 0
        with torch.no_grad():
            for data in tqdm(valid_loader):
                data = data.cuda()
                pred = simulator(data)
                loss = loss_fn(pred, data.y)
                total_loss += loss.item()
                batch_count += 1
        print(f"Eval loss: {total_loss / batch_count}")

if __name__ == "__main__":
    main()
