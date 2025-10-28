import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import base64
import io

# --- auto-download models from Google Drive if missing ---
try:
    import gdown
except Exception:
    gdown = None

# ensure model folder exists
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODEL_DIR = os.path.join(BACKEND_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Google Drive file IDs\ n
LEAF_FILE_ID = "12U8nEDWS4chnW71VaUMYwjP9VNXqIMqM"
BARK_FILE_ID = "1G8-fXTN0DWcBKfKysxzlBN8zLGvYeKyc"

def _gdrive_download(file_id: str, out_path: str):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True
    try:
        if gdown is None:
            import gdown as _g
            _g.download(f"https://drive.google.com/uc?id={file_id}&export=download", out_path, quiet=False)
        else:
            url = f"https://drive.google.com/uc?id={file_id}&export=download"
            gdown.download(url, out_path, quiet=False)
        return True
    except Exception:
        return False

leaf_out = os.path.join(MODEL_DIR, "resnet101_leaf_classifier.pth")
bark_out = os.path.join(MODEL_DIR, "resnet101_final.pth")
_gdrive_download(LEAF_FILE_ID, leaf_out)
_gdrive_download(BARK_FILE_ID, bark_out)

# =========================
# DEVICE CONFIGURATION
# =========================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =========================
# # PATH CONFIGURATION
# # =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))

LEAF_MODEL_PATH = next((p for p in [
    os.path.join(BACKEND_DIR, "model", "resnet101_leaf_classifier.pth"),
    os.path.join(PROJECT_ROOT, "backend", "model", "resnet101_leaf_classifier.pth"),
    os.path.join(PROJECT_ROOT, "model", "resnet101_leaf_classifier.pth")
] if os.path.exists(p)), None)

LEAF_CSV_PATH = next((p for p in [
    os.path.join(PROJECT_ROOT, "backend", "data", "Leaf1", "Leaf1", "train.csv"),
    os.path.join(BACKEND_DIR, "data", "Leaf1", "Leaf1", "train.csv")
] if os.path.exists(p)), None)

NEW_BARK_MODEL_PATH = os.path.join(BACKEND_DIR, "model", "resnet101_final.pth")
if not os.path.exists(NEW_BARK_MODEL_PATH):
    NEW_BARK_MODEL_PATH = None

BARK_CSV_PATH = next((p for p in [
    os.path.join(PROJECT_ROOT, "backend", "data", "tree-bark", "Bark.csv"),
    os.path.join(BACKEND_DIR, "data", "tree-bark", "Bark.csv")
] if os.path.exists(p)), None)

# =========================
# HELPER FUNCTION: LOAD CSV
# =========================
def load_csv(path):
    encodings = ("utf-8", "latin1", "cp1252")
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, engine="python")
        except Exception:
            continue
    with open(path, "rb") as f:
        raw = f.read().decode("utf-8", errors="replace")
    return pd.read_csv(pd.io.common.StringIO(raw), engine="python")

# =========================
# LOAD LEAF MODEL & DATA
# =========================
leaf_data = load_csv(LEAF_CSV_PATH)
leaf_data.columns = leaf_data.columns.str.strip().str.lower().str.replace(" ", "_")
class_names_leaf = sorted(leaf_data["label"].unique())
idx_to_class_leaf = {i: name for i, name in enumerate(class_names_leaf)}
NUM_CLASSES_LEAF = len(class_names_leaf)

leaf_model = models.resnet101(weights=None)
leaf_model.fc = nn.Linear(leaf_model.fc.in_features, NUM_CLASSES_LEAF)

loaded_leaf = torch.load(LEAF_MODEL_PATH, map_location="cpu")
if isinstance(loaded_leaf, nn.Module):
    leaf_model = loaded_leaf
else:
    def _strip_module_prefix(state_dict):
        return { (k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items() }

    def _extract_state_dict(obj):
        if isinstance(obj, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in obj and isinstance(obj[key], dict):
                    return _strip_module_prefix(obj[key])
            return _strip_module_prefix(obj)
        return None

    sd = _extract_state_dict(loaded_leaf)
    if sd is None:
        raise RuntimeError("Leaf checkpoint unusable")

    try:
        leaf_model.load_state_dict(sd)
    except RuntimeError:
        leaf_model.load_state_dict(sd, strict=False)

leaf_model.eval()

image_size = 224
predict_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# HELPER: IMAGE TO BASE64
# =========================
def image_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# =========================
# LEAF PREDICTION (unchanged)
# =========================
def predict_leaf(image: Image.Image) -> dict:
    if leaf_model is None:
        return { "error": "Model not loaded. Please ensure the model is initialized correctly." }

    try:
        img = image.convert("RGB")
        img_t = predict_transforms(img)
        batch_t = torch.unsqueeze(img_t, 0).to(device)

        leaf_model.eval()
        leaf_model.to(device)
        with torch.no_grad():
            outputs = leaf_model(batch_t)

        probabilities = torch.softmax(outputs, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()

        predicted_class_name = idx_to_class_leaf[predicted_index] if predicted_index < len(idx_to_class_leaf) else f"Unknown (Index {predicted_index})"

        leaf_info = leaf_data[leaf_data["label"] == predicted_class_name]
        if not leaf_info.empty:
            leaf_info = leaf_info.iloc[0]
            scientific_name = predicted_class_name
            common_name     = leaf_info.get("common_name", "N/A")
            uses            = leaf_info.get("uses", "N/A")
            origin          = leaf_info.get("origin", "N/A")
        else:
            scientific_name, common_name, uses, origin = "Unknown", "N/A", "N/A", "N/A"

        return {
            "scientific_name": scientific_name,
            "common_name": common_name,
            "uses": uses,
            "origin": origin,
            "image": image_to_base64(img)
        }

    except Exception as e:
        return { "error": f"An error occurred during prediction: {str(e)}" }

# =========================
# LOAD BARK MODEL & DATA  (COLAB-CORRECT & ROBUST)
# =========================
bark_data = load_csv(BARK_CSV_PATH)
bark_data.columns = bark_data.columns.str.strip().str.lower().str.replace(" ", "_")

# helper functions reused for bark
def _strip_module_prefix(state_dict):
    return { (k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items() }

def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return _strip_module_prefix(obj[key])
        # sometimes the dict *is* the state_dict
        return _strip_module_prefix(obj)
    return None

# Load checkpoint (can be full model, state_dict, or dict with model_state_dict+classes)
loaded_bark = None
if NEW_BARK_MODEL_PATH is not None:
    loaded_bark = torch.load(NEW_BARK_MODEL_PATH, map_location="cpu")
else:
    raise RuntimeError(f"Bark model not found at path: {NEW_BARK_MODEL_PATH}")

bark_model = models.resnet101(weights=None)

# Determine bark_classes first (from checkpoint if present, else CSV)
bark_classes = None
sd = None

if isinstance(loaded_bark, nn.Module):
    # someone saved the whole module
    bark_model = loaded_bark
    # try to extract classes attribute if present
    if hasattr(loaded_bark, "classes"):
        bark_classes = getattr(loaded_bark, "classes")
else:
    # try dict-like checkpoint with model_state_dict and classes
    if isinstance(loaded_bark, dict) and 'classes' in loaded_bark:
        bark_classes = loaded_bark['classes']

    # extract state_dict if possible
    sd = _extract_state_dict(loaded_bark)

# fallback: if classes not found, fallback to CSV sorted order
if bark_classes is None:
    try:
        bark_classes = sorted(bark_data['scientific_name'].unique())
    except Exception:
        bark_classes = []

# Now ensure the model head matches training architecture: Sequential(Dropout, Linear)
in_features = bark_model.fc.in_features
bark_model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features, len(bark_classes) if bark_classes else 1)
)

# Load weights into model
try:
    if sd is not None:
        # load the extracted state_dict; allow strict=False to tolerate small mismatches
        bark_model.load_state_dict(sd, strict=False)
    elif isinstance(loaded_bark, nn.Module):
        # already loaded above
        pass
    else:
        # as a final fallback, try to load the checkpoint directly (torch.load didn't return dict)
        try:
            bark_model.load_state_dict(loaded_bark, strict=False)
        except Exception:
            # last resort: nothing else to do
            pass
except Exception as e:
    # If something went wrong, raise a clearer error
    raise RuntimeError(f"Failed to load bark model weights cleanly: {e}")

bark_model.eval()

bark_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# =========================
# BARK PREDICTION       
# =========================
def predict_bark(image: Image.Image) -> dict:
    try:
        img = image.convert("RGB")
        tensor = bark_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = bark_model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_index = torch.argmax(probabilities).item()

        predicted_class = bark_classes[predicted_index]  # EXACT TRAINING ORDER

        row = bark_data[bark_data["scientific_name"].str.lower() == predicted_class.lower()]

        scientific_name = predicted_class
        common_name = row.iloc[0].get("general_name", "N/A") if not row.empty else "N/A"
        origin = row.iloc[0].get("origin", "N/A") if not row.empty else "N/A"
        uses = row.iloc[0].get("uses", "N/A") if not row.empty else "N/A"

        return {
            "scientific_name": scientific_name,
            "common_name": common_name,
            "uses": uses,
            "origin": origin,
            "image": image_to_base64(img)
        }

    except Exception as e:
        return { "error": f"An error occurred during bark prediction: {str(e)}" }


# =========================
# TEST LOADING (for debug)
# =========================
if __name__ == "__main__":
    print("✅ Leaf model loaded:", LEAF_MODEL_PATH)
    print("✅ Leaf CSV loaded:", LEAF_CSV_PATH)
    print("✅ Bark model loaded:", NEW_BARK_MODEL_PATH)
    print("✅ Bark CSV loaded:", BARK_CSV_PATH)