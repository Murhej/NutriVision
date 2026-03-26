"""
Download the configured Kaggle datasets with kagglehub.
"""

from __future__ import annotations

DATASET_IDS = [
    "rkuo2000/uecfood256",
    "moltean/fruits",
    "kritikseth/fruit-and-vegetable-image-recognition",
    "zhaoyj688/vegfru",
    "harishkumardatalab/food-image-classification-dataset",
    "iamsouravbanerjee/indian-food-images-dataset",
    "rizkyyk/dataset-food-classification",
]


def main() -> None:
    import kagglehub

    failed = []
    for dataset_id in DATASET_IDS:
        try:
            path = kagglehub.dataset_download(dataset_id)
        except Exception as exc:
            failed.append((dataset_id, str(exc)))
            print(f"{dataset_id}: FAILED ({exc})")
            continue
        print(f"{dataset_id}: {path}")

    if failed:
        print("\nSome downloads failed:")
        for dataset_id, error_text in failed:
            print(f"  - {dataset_id}: {error_text}")


if __name__ == "__main__":
    main()
