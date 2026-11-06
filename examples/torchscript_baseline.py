"""Adapter for an exported classifier; match the winning solution's preprocessing exactly."""
from pathlib import Path


def create_scorer(config, class_names, device):
    import torch
    from torchvision import transforms

    output_classes = config["output_classes"]
    if len(output_classes) != len(class_names) or set(output_classes) != set(class_names):
        raise ValueError("baseline.output_classes must match the experiment classes")
    order = [output_classes.index(name) for name in class_names]
    model = torch.jit.load(str(Path(config["checkpoint"])), map_location=device).eval()
    preprocess = transforms.Compose([
        transforms.Resize(config["resize"]),
        transforms.CenterCrop(config["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize(config["mean"], config["std"]),
    ])

    @torch.inference_mode()
    def score(images):
        batch = torch.stack([preprocess(image.convert("RGB")) for image in images]).to(device)
        return model(batch)[:, order].float().cpu().numpy()

    return score
