"""
2-Stage cascade inference on a single X-ray image.
────────────────────────────────────────────────────
Run from project root:
    python src/inference.py --image path/to/xray.jpg

Returns one of:
    normal | hairline | major_crack
"""

import os
import sys
import argparse
import torch
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from model_stage1 import build_stage1_model
from model_stage2 import build_stage2_model

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")

# Stage 1 threshold: lower than 0.5 to favour recall (catch more fractures)
# Means: predict fracture if P(fracture) > 0.35
FRACTURE_THRESHOLD = 0.35

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ─── Load models (cached for repeated calls) ──────────────────────────────────

_model1 = None
_model2 = None

def load_models():
    global _model1, _model2

    if _model1 is None:
        _model1 = build_stage1_model(pretrained=False).to(DEVICE)
        ckpt1   = torch.load(os.path.join(CKPT_DIR, "model1_best.pth"), map_location=DEVICE)
        _model1.load_state_dict(ckpt1["model_state"])
        _model1.eval()
        print("Stage 1 model loaded.")

    if _model2 is None:
        _model2 = build_stage2_model(num_classes=2, pretrained=False).to(DEVICE)
        ckpt2   = torch.load(os.path.join(CKPT_DIR, "model2_best.pth"), map_location=DEVICE)
        _model2.load_state_dict(ckpt2["model_state"])
        _model2.eval()
        print("Stage 2 model loaded.")

    return _model1, _model2


# ─── Core prediction ──────────────────────────────────────────────────────────

def predict(image_path: str, verbose: bool = True) -> dict:
    """
    Args:
        image_path : path to X-ray image file
        verbose    : print result to console

    Returns:
        {
          "final_label"    : str   e.g. "hairline"
          "stage1_label"   : str   "normal" or "fracture"
          "stage1_conf"    : float
          "stage2_label"   : str | None
          "stage2_conf"    : float | None
          "stage2_probs"   : dict  | None  e.g. {"hairline": 0.82, "major_crack": 0.18}
        }
    """
    model1, model2 = load_models()

    # Load and preprocess
    img    = Image.open(image_path).convert("RGB")
    tensor = PREPROCESS(img).unsqueeze(0).to(DEVICE)

    result = {}

    with torch.no_grad():
        # ── Stage 1 ──────────────────────────────────────────────────────────
        logits1 = model1(tensor)
        probs1  = torch.softmax(logits1, dim=1)[0]
        # probs1[0] = normal, probs1[1] = fracture
        fracture_prob = probs1[1].item()

        if fracture_prob <= FRACTURE_THRESHOLD:
            result = {
                "final_label"  : "normal",
                "stage1_label" : "normal",
                "stage1_conf"  : probs1[0].item(),
                "stage2_label" : None,
                "stage2_conf"  : None,
                "stage2_probs" : None,
            }
        else:
            result["stage1_label"] = "fracture"
            result["stage1_conf"]  = fracture_prob

            # ── Stage 2 ──────────────────────────────────────────────────────
            logits2 = model2(tensor)
            probs2  = torch.softmax(logits2, dim=1)[0]
            # probs2[0] = hairline, probs2[1] = major_crack
            class_names = ["hairline", "major_crack"]
            pred2       = probs2.argmax().item()

            result["stage2_label"]  = class_names[pred2]
            result["stage2_conf"]   = probs2[pred2].item()
            result["stage2_probs"]  = {cls: probs2[i].item() for i, cls in enumerate(class_names)}
            result["final_label"]   = class_names[pred2]

    if verbose:
        print(f"\n{'─'*45}")
        print(f"  Image       : {os.path.basename(image_path)}")
        print(f"  Stage 1     : {result['stage1_label']}  (conf: {result['stage1_conf']:.4f})")
        if result["stage2_label"]:
            print(f"  Stage 2     : {result['stage2_label']}  (conf: {result['stage2_conf']:.4f})")
            for cls, p in result["stage2_probs"].items():
                print(f"    {cls:20s}: {p:.4f}")
        print(f"  ► FINAL     : {result['final_label'].upper()}")
        print(f"{'─'*45}\n")

    return result


# ─── Batch inference ──────────────────────────────────────────────────────────

def predict_folder(folder_path: str) -> list:
    """Run inference on all images in a folder. Returns list of result dicts."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    results    = []
    images     = [f for f in os.listdir(folder_path)
                  if os.path.splitext(f)[1].lower() in extensions]

    print(f"\nRunning inference on {len(images)} images in {folder_path}")
    for fname in sorted(images):
        path = os.path.join(folder_path, fname)
        res  = predict(path, verbose=False)
        res["filename"] = fname
        results.append(res)
        print(f"  {fname:40s} → {res['final_label']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  type=str, help="Path to a single X-ray image")
    parser.add_argument("--folder", type=str, help="Path to folder of X-ray images")
    args = parser.parse_args()

    if args.image:
        predict(args.image, verbose=True)
    elif args.folder:
        predict_folder(args.folder)
    else:
        parser.print_help()
