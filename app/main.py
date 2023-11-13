import time
from pathlib import Path

import pandas as pd
import torch

# import f1 score
from sklearn.metrics import f1_score
from torch import nn

from app.dataset import device, get_dataloader
from app.model import RNNModel

CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR.parent / "data"

BATCH_SIZE = 128
LEARNING_RATE = 1e-3


def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    total_acc, total_count, total_f1_score = 0, 0, 0
    log_interval = 500
    start_time = time.time()
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch_num, (label, text, text_lens) in enumerate(dataloader):
        optimizer.zero_grad()
        # Compute prediction and loss
        predicted_label = model(text, text_lens)
        loss = loss_fn(predicted_label, label)

        # Backpropagation
        loss.backward()
        # clip the gradient norm to 1.0 to prevent exploding gradients. A common
        # problem with RNNs and LSTMs is the "exploding gradient" problem. This
        # is where the gradient for a particular parameter gets larger and larger
        # as the number of layers increases. This can result in the gradient
        # becoming so large that the weights overflow (i.e. become NaN) and the
        # model fails to train.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        predictions = (predicted_label > 0.5).float()

        total_acc += (predictions == label).sum().item()
        total_count += label.size(0)
        total_f1_score += f1_score(
            label.detach().cpu(), predictions.detach().cpu(), average="macro"
        )

        if batch_num % log_interval == 0 and batch_num > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}"
                "| f1_score {:8.3f}"
                "| ms/batch {:5.2f}".format(
                    epoch,
                    batch_num,
                    len(dataloader),
                    total_acc / total_count,
                    total_f1_score / log_interval,
                    elapsed * 1000 / log_interval,
                )
            )
            total_acc, total_count, total_f1_score = 0, 0, 0
            start_time = time.time()


def evaluate(dataloader, model, criterion):
    model.eval()
    total_acc, total_count, total_f1_score = 0, 0, 0
    total_loss = 0

    with torch.no_grad():
        for _, (label, text, offsets) in enumerate(dataloader):
            predicted_label = (model(text, offsets) > 0.5).float()
            total_loss += criterion(predicted_label, label)

            total_acc += (predicted_label == label).sum().item()
            total_f1_score += f1_score(
                label.detach().cpu(), predicted_label.detach().cpu(), average="macro"
            )
            total_count += label.size(0)

    print(
        "Evaluation - loss: {:.6f}  "
        "accuracy: {:.3f}  f1_score: {:.3f}\n".format(
            total_loss / len(dataloader),
            total_acc / total_count,
            total_f1_score / len(dataloader),
        )
    )
    return total_acc / total_count, total_f1_score / len(dataloader)


if __name__ == "__main__":
    # Load csv dataset
    train_data = pd.read_csv(DATA_DIR / "train_set.csv")
    train_data = train_data[train_data["question_text"].str.len() > 5]
    test_data = pd.read_csv(DATA_DIR / "test_set.csv")
    test_data = test_data[test_data["question_text"].str.len() > 5]

    # Turn csv to dataloader
    train_dataloader = get_dataloader(train_data, batch_size=BATCH_SIZE)
    test_dataloader = get_dataloader(test_data, batch_size=BATCH_SIZE)

    # Instantiate model
    model = RNNModel(
        vocab_size=len(train_dataloader.dataset.vocab),
        embedding_dim=8,
        hidden_dim=8,
        output_dim=1,
        n_layers=2,
        bidirectional=True,
        dropout=0,
    )

    # Send model to device
    model = model.to(device)

    # Initialize the loss function
    pos_weight = torch.tensor([7.0])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    epochs = 50
    # Measure time
    tic = time.time()
    for t in range(epochs):
        epoch_start_time = time.time()
        print(f"Epoch {t}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, t)
        # eval
        accu_val, f1_val = evaluate(test_dataloader, model, loss_fn)
        print("-" * 59)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s"
            "| valid accuracy {:8.3f} "
            "| valid f1 score {:8.3f}".format(
                t,
                time.time() - epoch_start_time,
                accu_val,
                f1_val,
            )
        )
    print("-" * 59)
    toc = time.time()
    print("Done!", f"Training time: {toc - tic:>.3f} seconds")
    # Train model
    train_loop(train_dataloader, test_dataloader, model)
