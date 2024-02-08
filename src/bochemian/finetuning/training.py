import torch
import wandb
from bochemian.data.dataset import weight_func


def beta_scheduler(epoch, max_epochs, min_epochs=5, start_beta=0.1, end_beta=1.0):
    """
    Linearly increases beta from start_beta to end_beta over max_epochs.

    Args:
        epoch (int): Current epoch.
        max_epochs (int): Maximum number of epochs for the scheduler.
        start_beta (float): Starting value of beta.
        end_beta (float): Final value of beta.

    Returns:
        float: The beta value for the current epoch.
    """

    if epoch > max_epochs:
        return end_beta
    beta_range = end_beta - start_beta
    beta = start_beta + (beta_range * (epoch / max_epochs))
    if epoch < min_epochs:
        beta = start_beta
    return beta


def train_finetuning_model(
    model,
    train_loader,
    emb_criterion,
    mse_criterion,
    optimizer,
    num_epochs,
    beta,
):
    # Set the model in training mode
    model.train()
    wandb.log({"beta": beta})
    for epoch in range(num_epochs):
        running_loss = 0.0

        # Iterate through the training data
        for inputs, targets, weights in train_loader:

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            embeddings = model.fc1(model.adapter(inputs))

            # Compute loss
            norm_factor = targets.max()

            mse_loss = mse_criterion(
                outputs / norm_factor, targets / norm_factor
            ) / len(inputs)
            logratio_loss = (
                emb_criterion(embeddings, targets / norm_factor) if emb_criterion else 0
            )

            # Weight the loss by function evaluations (yields)
            # weights = 5 * weight_func(targets)
            weighted_prediction_loss = torch.mean(mse_loss)
            loss = 10 * weighted_prediction_loss + logratio_loss

            # Backpropagation and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

        # Print the average loss for this epoch
        avg_loss = running_loss / len(train_loader)
        wandb.log({"finetuning_loss": avg_loss})
        # print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    print("Finetuning finished")


def evaluate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            outputs = model(inputs)
            loss = torch.mean(criterion(outputs, targets))
            running_loss += loss.item()

    avg_loss = running_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
