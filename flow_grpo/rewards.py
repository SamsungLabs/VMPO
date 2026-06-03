from PIL import Image
import io
import numpy as np
import torch
from collections import defaultdict

def ocr_score(device):
    from flow_grpo.ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def multi_score(device, score_dict):
    score_functions = {
        "ocr": ocr_score,
    }
    score_fns = {}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = (
            score_functions[score_name](device)
            if "device" in score_functions[score_name].__code__.co_varnames
            else score_functions[score_name]()
        )

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, only_strict=True):
        total_scores = []
        score_details = {}

        for score_name, weight in score_dict.items():
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](
                    images, prompts, metadata, only_strict
                )
                score_details["accuracy"] = rewards
                score_details["strict_accuracy"] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f"{key}_strict_accuracy"] = value
                for key, value in group_rewards.items():
                    score_details[f"{key}_accuracy"] = value
            else:
                scores, rewards = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

        score_details["avg"] = total_scores
        return score_details, {}

    return _fn


def main():
    import torchvision.transforms as transforms

    image_paths = [
        "test_cases/nasa.jpg",
    ]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert to tensor
        ]
    )

    images = torch.stack([transform(Image.open(image_path).convert("RGB")) for image_path in image_paths])
    prompts = [
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    metadata = {}  # Example metadata
    score_dict = {"ocr": 1.0}
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores, _ = scoring_fn(images, prompts, metadata)
    # Print the scores
    print("Scores:", scores)


if __name__ == "__main__":
    main()
