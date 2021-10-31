import os
import argparse

from tqdm import tqdm
import torch
import torch_geometric as pyg

import dataset
import model
import rollout
import visualize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--noise", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=100000)
    parser.add_argument("--vis-interval", type=int, default=100000)
    parser.add_argument("--save-interval", type=int, default=100000)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    train_dataset = dataset.OneStepDataset(args.data_path, "train", noise_std=args.noise)
    valid_dataset = dataset.OneStepDataset(args.data_path, "valid", noise_std=args.noise)
    train_loader = pyg.loader.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    valid_loader = pyg.loader.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    rollout_dataset = dataset.RolloutDataset(args.data_path, "valid")

    simulator = model.LearnedSimulator()
    simulator = simulator.cuda()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(simulator.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1 / 5e6))

    total_batch = 0
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

            total_batch += 1
            if args.eval_interval and total_batch % args.eval_interval == 0:
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
                simulator.train()

            if args.vis_interval and total_batch % args.vis_interval == 0:
                simulator.eval()
                rollout_data = rollout_dataset[0]
                rollout_out = rollout.rollout(simulator, rollout_data, rollout_dataset.metadata, args.noise)
                rollout_out = rollout_out.permute(1, 0, 2)
                anim = visualize.visualize_pair(rollout_data["particle_type"], rollout_out, rollout_data["position"], rollout_dataset.metadata)
                anim.save(os.path.join(args.output_path, f"rollout_{total_batch}.gif"), writer="ffmpeg", fps=60)
                simulator.train()

            if args.save_interval and total_batch % args.save_interval == 0:
                torch.save(
                    {
                        "model": simulator.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "args": vars(args),
                    },
                    os.path.join(args.output_path, f"checkpoint_{total_batch}.pt")
                )


if __name__ == "__main__":
    main()
