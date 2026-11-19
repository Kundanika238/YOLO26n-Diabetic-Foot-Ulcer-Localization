
from pathlib import Path
from ultralytics import YOLO
import torch

# Project directory
PROJECT = Path(r"D:\Diabetic_Foot_Ulcer_Project")

# Screened research dataset
DATA = (
    PROJECT
    / "03_YOLO_Dataset"
    / "DFU_Research_SCREENED_DRAFT"
    / "data.yaml"
)

# Original pretrained YOLO26n weights
MODEL = PROJECT / "yolo26n.pt"

# Separate folder for new research experiments
OUTPUT = PROJECT / "05_Models"

RUN_NAME = "yolo26n_research_baseline_50epochs_v2"


def main():

    print("=" * 60)
    print("DFU RESEARCH - YOLO26n BASELINE TRAINING")
    print("=" * 60)

    # Check important files
    if not DATA.is_file():
        raise FileNotFoundError(DATA)

    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)

    # Avoid reusing an existing experiment
    run_directory = OUTPUT / RUN_NAME

    if run_directory.exists():
        raise FileExistsError(
            f"Training folder already exists: {run_directory}"
        )

    # Verify GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    print("\nPyTorch:", torch.__version__)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Dataset:", DATA)
    print("Pretrained model:", MODEL)
    print("Output:", run_directory)

    # Load original pretrained model
    model = YOLO(str(MODEL))

    # Train a fresh research baseline
    model.train(
        data=str(DATA),
        epochs=50,
        imgsz=512,
        batch=4,
        device=0,
        workers=0,
        optimizer="auto",
        seed=42,
        deterministic=True,
        pretrained=True,
        resume=False,
        amp=True,
        patience=100,
        close_mosaic=10,
        cache=False,
        save=True,
        plots=True,
        val=True,
        project=str(OUTPUT),
        name=RUN_NAME,
        exist_ok=False,
    )

    print("\n" + "=" * 60)
    print("BASELINE TRAINING FINISHED")
    print("=" * 60)

    print("\nResults folder:")
    print(run_directory)

    print("\nBest model:")
    print(run_directory / "weights" / "best.pt")

    print("\nTraining results:")
    print(run_directory / "results.csv")


if __name__ == "__main__":
    main()
