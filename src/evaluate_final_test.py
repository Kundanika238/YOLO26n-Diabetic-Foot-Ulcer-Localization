
from pathlib import Path
import argparse

from ultralytics import YOLO


def main():
    root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Evaluate the final YOLO26n DFU checkpoint."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to the local final screened dataset YAML."
    )

    parser.add_argument(
        "--model",
        default=str(root / "models" / "best.pt"),
        help="Path to the trained YOLO26n checkpoint."
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="Device: cpu or 0 for the first CUDA GPU."
    )

    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    model = YOLO(str(model_path))

    results = model.val(
        data=str(data_path),
        split="test",
        imgsz=512,
        batch=4,
        device=args.device,
        workers=0,
        conf=0.001,
        iou=0.70,
        plots=False,
        save_json=False
    )

    precision = float(results.box.mp)
    recall = float(results.box.mr)
    map50 = float(results.box.map50)
    map95 = float(results.box.map)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    print("\nFINAL YOLO26n TEST RESULTS")
    print("=" * 40)
    print(f"Precision:      {precision * 100:.2f}%")
    print(f"Recall:         {recall * 100:.2f}%")
    print(f"F1-score:       {f1 * 100:.2f}%")
    print(f"mAP@0.5:        {map50 * 100:.2f}%")
    print(f"mAP@0.5:0.95:   {map95 * 100:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()
