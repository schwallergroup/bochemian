import torch


def update_embeddings(model, original_data, concatenation=True):
    """
    Update embeddings based on the model's adapter and a scaling factor alpha.

    Args:
        model: The model with an adapter method for fine-tuning.
        original_data: The original embeddings.

    Returns:
        The updated embeddings.
    """
    # Compute the fine-tuned embeddings
    adapter_embeddings = model.adapter(original_data)
    assert not torch.isnan(
        adapter_embeddings
    ).any(), "Adapter embeddings contain NaN values."

    finetuned_embeddings = model.fc1(adapter_embeddings)
    assert not torch.isnan(
        finetuned_embeddings
    ).any(), "Finetuned embeddings contain NaN values."

    concatenated_embeddings = torch.cat((original_data, finetuned_embeddings), dim=1)

    return concatenated_embeddings if concatenation else finetuned_embeddings
