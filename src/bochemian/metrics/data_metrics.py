import torch
import wandb


def calculate_data_metrics(data_module):
    target_stat_max = torch.max(data_module.y)
    target_stat_mean = torch.mean(data_module.y)
    target_stat_std = torch.std(data_module.y)
    target_stat_var = torch.var(data_module.y)
    input_stat_feature_dimension = data_module.x.shape[-1]
    input_stat_n_points = data_module.x.shape[0]

    target_q75 = torch.quantile(data_module.y, 0.75)
    target_q90 = torch.quantile(data_module.y, 0.9)
    target_q95 = torch.quantile(data_module.y, 0.95)
    target_q99 = torch.quantile(data_module.y, 0.99)

    top_3_values, _ = torch.topk(data_module.y, 3, dim=0)
    top_5_values, _ = torch.topk(data_module.y, 5, dim=0)
    top_10_values, _ = torch.topk(data_module.y, 10, dim=0)

    top_1 = torch.max(data_module.y)
    top_3 = top_3_values[-1]
    top_5 = top_5_values[-1]
    top_10 = top_10_values[-1]

    return {
        "target_stat_max": target_stat_max,
        "target_stat_mean": target_stat_mean,
        "target_stat_std": target_stat_std,
        "target_stat_var": target_stat_var,
        "input_stat_feature_dimension": input_stat_feature_dimension,
        "input_stat_n_points": input_stat_n_points,
        "target_q75": target_q75,
        "target_q90": target_q90,
        "target_q95": target_q95,
        "target_q99": target_q99,
        "top_1": top_1,
        "top_3": top_3,
        "top_5": top_5,
        "top_10": top_10,
    }


def log_data_metrics(data_metrics, train_y, epoch=0):
    log_top_n_counts(data_metrics, train_y, epoch)
    log_quantile_counts(data_metrics, train_y, epoch)


def log_data_stats(data_metrics):
    for key, value in data_metrics.items():
        if "stat" in key:
            wandb.summary[key] = value.item() if torch.is_tensor(value) else value


def log_top_n_counts(data_metrics, train_y, epoch=0):
    for n in [1, 3, 5, 10]:
        threshold = data_metrics[f"top_{n}"]
        count = (train_y >= threshold).sum().item()
        wandb.log({f"top_{n}_count": count, "epoch": epoch})


def log_quantile_counts(data_metrics, train_y, epoch=0):
    for q in [0.75, 0.9, 0.95, 0.99]:
        threshold = data_metrics[f"target_q{int(q * 100)}"]
        count = (train_y >= threshold).sum().item()
        wandb.log({f"quantile_{int(q * 100)}_count": count, "epoch": epoch})
