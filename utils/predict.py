# # # # import os
# # # # import torch
# # # # import torch.nn as nn
# # # # from torchvision import models, transforms
# # # # from PIL import Image
# # # # import pandas as pd
# # # # import base64
# # # # import io

# # # # # --- auto-download models from Google Drive if missing ---
# # # # try:
# # # #     import gdown
# # # # except Exception:
# # # #     gdown = None

# # # # # ensure model folder exists
# # # # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # # # BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
# # # # MODEL_DIR = os.path.join(BACKEND_DIR, "model")
# # # # os.makedirs(MODEL_DIR, exist_ok=True)

# # # # # Google Drive file IDs\ n
# # # # LEAF_FILE_ID = "12U8nEDWS4chnW71VaUMYwjP9VNXqIMqM"
# # # # BARK_FILE_ID = "1G8-fXTN0DWcBKfKysxzlBN8zLGvYeKyc"

# # # # def _gdrive_download(file_id: str, out_path: str):
# # # #     if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
# # # #         return True
# # # #     try:
# # # #         if gdown is None:
# # # #             import gdown as _g
# # # #             _g.download(f"https://drive.google.com/uc?id={file_id}&export=download", out_path, quiet=False)
# # # #         else:
# # # #             url = f"https://drive.google.com/uc?id={file_id}&export=download"
# # # #             gdown.download(url, out_path, quiet=False)
# # # #         return True
# # # #     except Exception:
# # # #         return False

# # # # leaf_out = os.path.join(MODEL_DIR, "resnet101_leaf_classifier.pth")
# # # # bark_out = os.path.join(MODEL_DIR, "resnet101_final.pth")
# # # # _gdrive_download(LEAF_FILE_ID, leaf_out)
# # # # _gdrive_download(BARK_FILE_ID, bark_out)

# # # # # =========================
# # # # # DEVICE CONFIGURATION
# # # # # =========================
# # # # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # # # # =========================
# # # # # # PATH CONFIGURATION
# # # # # # =========================
# # # # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # # # BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
# # # # PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))

# # # # LEAF_MODEL_PATH = next((p for p in [
# # # #     os.path.join(BACKEND_DIR, "model", "resnet101_leaf_classifier.pth"),
# # # #     os.path.join(PROJECT_ROOT, "backend", "model", "resnet101_leaf_classifier.pth"),
# # # #     os.path.join(PROJECT_ROOT, "model", "resnet101_leaf_classifier.pth")
# # # # ] if os.path.exists(p)), None)

# # # # LEAF_CSV_PATH = next((p for p in [
# # # #     os.path.join(PROJECT_ROOT, "backend", "data", "Leaf1", "Leaf1", "train.csv"),
# # # #     os.path.join(BACKEND_DIR, "data", "Leaf1", "Leaf1", "train.csv")
# # # # ] if os.path.exists(p)), None)

# # # # NEW_BARK_MODEL_PATH = os.path.join(BACKEND_DIR, "model", "resnet101_final.pth")
# # # # if not os.path.exists(NEW_BARK_MODEL_PATH):
# # # #     NEW_BARK_MODEL_PATH = None

# # # # BARK_CSV_PATH = next((p for p in [
# # # #     os.path.join(PROJECT_ROOT, "backend", "data", "tree-bark", "Bark.csv"),
# # # #     os.path.join(BACKEND_DIR, "data", "tree-bark", "Bark.csv")
# # # # ] if os.path.exists(p)), None)

# # # # # =========================
# # # # # HELPER FUNCTION: LOAD CSV
# # # # # =========================
# # # # def load_csv(path):
# # # #     encodings = ("utf-8", "latin1", "cp1252")
# # # #     for enc in encodings:
# # # #         try:
# # # #             return pd.read_csv(path, encoding=enc, engine="python")
# # # #         except Exception:
# # # #             continue
# # # #     with open(path, "rb") as f:
# # # #         raw = f.read().decode("utf-8", errors="replace")
# # # #     return pd.read_csv(pd.io.common.StringIO(raw), engine="python")

# # # # # =========================
# # # # # LOAD LEAF MODEL & DATA
# # # # # =========================
# # # # leaf_data = load_csv(LEAF_CSV_PATH)
# # # # leaf_data.columns = leaf_data.columns.str.strip().str.lower().str.replace(" ", "_")
# # # # class_names_leaf = sorted(leaf_data["label"].unique())
# # # # idx_to_class_leaf = {i: name for i, name in enumerate(class_names_leaf)}
# # # # NUM_CLASSES_LEAF = len(class_names_leaf)

# # # # leaf_model = models.resnet101(weights=None)
# # # # leaf_model.fc = nn.Linear(leaf_model.fc.in_features, NUM_CLASSES_LEAF)

# # # # loaded_leaf = torch.load(LEAF_MODEL_PATH, map_location="cpu")
# # # # if isinstance(loaded_leaf, nn.Module):
# # # #     leaf_model = loaded_leaf
# # # # else:
# # # #     def _strip_module_prefix(state_dict):
# # # #         return { (k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items() }

# # # #     def _extract_state_dict(obj):
# # # #         if isinstance(obj, dict):
# # # #             for key in ("state_dict", "model_state_dict", "model"):
# # # #                 if key in obj and isinstance(obj[key], dict):
# # # #                     return _strip_module_prefix(obj[key])
# # # #             return _strip_module_prefix(obj)
# # # #         return None

# # # #     sd = _extract_state_dict(loaded_leaf)
# # # #     if sd is None:
# # # #         raise RuntimeError("Leaf checkpoint unusable")

# # # #     try:
# # # #         leaf_model.load_state_dict(sd)
# # # #     except RuntimeError:
# # # #         leaf_model.load_state_dict(sd, strict=False)

# # # # leaf_model.eval()

# # # # image_size = 224
# # # # predict_transforms = transforms.Compose([
# # # #     transforms.Resize(image_size),
# # # #     transforms.CenterCrop(image_size),
# # # #     transforms.ToTensor(),
# # # #     transforms.Normalize([0.485, 0.456, 0.406],
# # # #                          [0.229, 0.224, 0.225])
# # # # ])

# # # # # =========================
# # # # # HELPER: IMAGE TO BASE64
# # # # # =========================
# # # # def image_to_base64(img: Image.Image) -> str:
# # # #     buffered = io.BytesIO()
# # # #     img.save(buffered, format="PNG")
# # # #     return base64.b64encode(buffered.getvalue()).decode("utf-8")

# # # # # =========================
# # # # # LEAF PREDICTION (unchanged)
# # # # # =========================
# # # # def predict_leaf(image: Image.Image) -> dict:
# # # #     if leaf_model is None:
# # # #         return { "error": "Model not loaded. Please ensure the model is initialized correctly." }

# # # #     try:
# # # #         img = image.convert("RGB")
# # # #         img_t = predict_transforms(img)
# # # #         batch_t = torch.unsqueeze(img_t, 0).to(device)

# # # #         leaf_model.eval()
# # # #         leaf_model.to(device)
# # # #         with torch.no_grad():
# # # #             outputs = leaf_model(batch_t)

# # # #         probabilities = torch.softmax(outputs, dim=1)
# # # #         predicted_index = torch.argmax(probabilities, dim=1).item()

# # # #         predicted_class_name = idx_to_class_leaf[predicted_index] if predicted_index < len(idx_to_class_leaf) else f"Unknown (Index {predicted_index})"

# # # #         leaf_info = leaf_data[leaf_data["label"] == predicted_class_name]
# # # #         if not leaf_info.empty:
# # # #             leaf_info = leaf_info.iloc[0]
# # # #             scientific_name = predicted_class_name
# # # #             common_name     = leaf_info.get("common_name", "N/A")
# # # #             uses            = leaf_info.get("uses", "N/A")
# # # #             origin          = leaf_info.get("origin", "N/A")
# # # #         else:
# # # #             scientific_name, common_name, uses, origin = "Unknown", "N/A", "N/A", "N/A"

# # # #         return {
# # # #             "scientific_name": scientific_name,
# # # #             "common_name": common_name,
# # # #             "uses": uses,
# # # #             "origin": origin,
# # # #             "image": image_to_base64(img)
# # # #         }

# # # #     except Exception as e:
# # # #         return { "error": f"An error occurred during prediction: {str(e)}" }

# # # # # =========================
# # # # # LOAD BARK MODEL & DATA  (COLAB-CORRECT & ROBUST)
# # # # # =========================
# # # # bark_data = load_csv(BARK_CSV_PATH)
# # # # bark_data.columns = bark_data.columns.str.strip().str.lower().str.replace(" ", "_")

# # # # # helper functions reused for bark
# # # # def _strip_module_prefix(state_dict):
# # # #     return { (k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items() }

# # # # def _extract_state_dict(obj):
# # # #     if isinstance(obj, dict):
# # # #         for key in ("state_dict", "model_state_dict", "model"):
# # # #             if key in obj and isinstance(obj[key], dict):
# # # #                 return _strip_module_prefix(obj[key])
# # # #         # sometimes the dict *is* the state_dict
# # # #         return _strip_module_prefix(obj)
# # # #     return None

# # # # # Load checkpoint (can be full model, state_dict, or dict with model_state_dict+classes)
# # # # loaded_bark = None
# # # # if NEW_BARK_MODEL_PATH is not None:
# # # #     loaded_bark = torch.load(NEW_BARK_MODEL_PATH, map_location="cpu")
# # # # else:
# # # #     raise RuntimeError(f"Bark model not found at path: {NEW_BARK_MODEL_PATH}")

# # # # bark_model = models.resnet101(weights=None)

# # # # # Determine bark_classes first (from checkpoint if present, else CSV)
# # # # bark_classes = None
# # # # sd = None

# # # # if isinstance(loaded_bark, nn.Module):
# # # #     # someone saved the whole module
# # # #     bark_model = loaded_bark
# # # #     # try to extract classes attribute if present
# # # #     if hasattr(loaded_bark, "classes"):
# # # #         bark_classes = getattr(loaded_bark, "classes")
# # # # else:
# # # #     # try dict-like checkpoint with model_state_dict and classes
# # # #     if isinstance(loaded_bark, dict) and 'classes' in loaded_bark:
# # # #         bark_classes = loaded_bark['classes']

# # # #     # extract state_dict if possible
# # # #     sd = _extract_state_dict(loaded_bark)

# # # # # fallback: if classes not found, fallback to CSV sorted order
# # # # if bark_classes is None:
# # # #     try:
# # # #         bark_classes = sorted(bark_data['scientific_name'].unique())
# # # #     except Exception:
# # # #         bark_classes = []

# # # # # Now ensure the model head matches training architecture: Sequential(Dropout, Linear)
# # # # in_features = bark_model.fc.in_features
# # # # bark_model.fc = nn.Sequential(
# # # #     nn.Dropout(p=0.4),
# # # #     nn.Linear(in_features, len(bark_classes) if bark_classes else 1)
# # # # )

# # # # # Load weights into model
# # # # try:
# # # #     if sd is not None:
# # # #         # load the extracted state_dict; allow strict=False to tolerate small mismatches
# # # #         bark_model.load_state_dict(sd, strict=False)
# # # #     elif isinstance(loaded_bark, nn.Module):
# # # #         # already loaded above
# # # #         pass
# # # #     else:
# # # #         # as a final fallback, try to load the checkpoint directly (torch.load didn't return dict)
# # # #         try:
# # # #             bark_model.load_state_dict(loaded_bark, strict=False)
# # # #         except Exception:
# # # #             # last resort: nothing else to do
# # # #             pass
# # # # except Exception as e:
# # # #     # If something went wrong, raise a clearer error
# # # #     raise RuntimeError(f"Failed to load bark model weights cleanly: {e}")

# # # # bark_model.eval()

# # # # bark_transform = transforms.Compose([
# # # #     transforms.Resize((224, 224)),
# # # #     transforms.ToTensor(),
# # # #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
# # # #                          std=[0.229, 0.224, 0.225])
# # # # ])
# # # # # =========================
# # # # # BARK PREDICTION       
# # # # # =========================
# # # # def predict_bark(image: Image.Image) -> dict:
# # # #     try:
# # # #         img = image.convert("RGB")
# # # #         tensor = bark_transform(img).unsqueeze(0).to(device)

# # # #         with torch.no_grad():
# # # #             outputs = bark_model(tensor)
# # # #             probabilities = torch.softmax(outputs, dim=1)[0]
# # # #             predicted_index = torch.argmax(probabilities).item()

# # # #         predicted_class = bark_classes[predicted_index]  # EXACT TRAINING ORDER

# # # #         row = bark_data[bark_data["scientific_name"].str.lower() == predicted_class.lower()]

# # # #         scientific_name = predicted_class
# # # #         common_name = row.iloc[0].get("general_name", "N/A") if not row.empty else "N/A"
# # # #         origin = row.iloc[0].get("origin", "N/A") if not row.empty else "N/A"
# # # #         uses = row.iloc[0].get("uses", "N/A") if not row.empty else "N/A"

# # # #         return {
# # # #             "scientific_name": scientific_name,
# # # #             "common_name": common_name,
# # # #             "uses": uses,
# # # #             "origin": origin,
# # # #             "image": image_to_base64(img)
# # # #         }

# # # #     except Exception as e:
# # # #         return { "error": f"An error occurred during bark prediction: {str(e)}" }


# # # # # =========================
# # # # # TEST LOADING (for debug)
# # # # # =========================
# # # # if __name__ == "__main__":
# # # #     print("✅ Leaf model loaded:", LEAF_MODEL_PATH)
# # # #     print("✅ Leaf CSV loaded:", LEAF_CSV_PATH)
# # # #     print("✅ Bark model loaded:", NEW_BARK_MODEL_PATH)

# # # #     print("✅ Bark CSV loaded:", BARK_CSV_PATH)

# # # utils/predict.py
# # # lazy download models and load CSV from data/train.csv and data/Bark.csv
# # import os, torch, torch.nn as nn
# # from torchvision import models, transforms
# # from PIL import Image
# # import pandas as pd, base64, io
# # try:
# #     import gdown
# # except Exception:
# #     gdown = None
# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
# # DATA_DIR = os.path.join(BACKEND_DIR, "data")
# # MODEL_DIR = os.path.join(BACKEND_DIR, "model")
# # os.makedirs(MODEL_DIR, exist_ok=True)
# # LEAF_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
# # BARK_CSV_PATH = os.path.join(DATA_DIR, "Bark.csv")
# # LEAF_FILE_ID = "12U8nEDWS4chnW71VaUMYwjP9VNXqIMqM"
# # BARK_FILE_ID = "1G8-fXTN0DWcBKfKysxzlBN8zLGvYeKyc"
# # leaf_model = None
# # bark_model = None
# # leaf_data = None
# # bark_data = None
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # def _gdrive_download(file_id, out_path):
# #     if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
# #         return
# #     if gdown is None:
# #         import gdown as _g; _g.download(f"https://drive.google.com/uc?id={file_id}&export=download", out_path, quiet=False)
# #     else:
# #         gdown.download(f"https://drive.google.com/uc?id={file_id}&export=download", out_path, quiet=False)

# # def _load_leaf():
# #     global leaf_model, leaf_data
# #     _gdrive_download(LEAF_FILE_ID, os.path.join(MODEL_DIR, "resnet101_leaf_classifier.pth"))
# #     leaf_data = pd.read_csv(LEAF_CSV_PATH)
# #     leaf_data.columns = leaf_data.columns.str.strip().str.lower().str.replace(" ", "_")
# #     classes = sorted(leaf_data["label"].unique())
# #     model = models.resnet101(weights=None)
# #     model.fc = nn.Linear(model.fc.in_features, len(classes))
# #     sd = torch.load(os.path.join(MODEL_DIR, "resnet101_leaf_classifier.pth"), map_location="cpu")
# #     try: model.load_state_dict(sd)
# #     except: model.load_state_dict(sd, strict=False)
# #     model.eval(); leaf_model = model

# # def _load_bark():
# #     global bark_model, bark_data, bark_classes
# #     _gdrive_download(BARK_FILE_ID, os.path.join(MODEL_DIR, "resnet101_final.pth"))
# #     bark_data = pd.read_csv(BARK_CSV_PATH)
# #     bark_data.columns = bark_data.columns.str.strip().str.lower().str.replace(" ", "_")
# #     bark_classes = sorted(bark_data['scientific_name'].unique())
# #     model = models.resnet101(weights=None)
# #     model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, len(bark_classes)))
# #     sd = torch.load(os.path.join(MODEL_DIR, "resnet101_final.pth"), map_location="cpu")
# #     try: model.load_state_dict(sd)
# #     except: model.load_state_dict(sd, strict=False)
# #     model.eval(); bark_model = model

# # predict_transforms = transforms.Compose([
# #     transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
# #     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# # ])
# # bark_transform = predict_transforms

# # def image_to_base64(img):
# #     buf = io.BytesIO(); img.save(buf, format="PNG"); return base64.b64encode(buf.getvalue()).decode()

# # def predict_leaf(image):
# #     if leaf_model is None: _load_leaf()
# #     img = image.convert("RGB"); t = predict_transforms(img).unsqueeze(0).to(device)
# #     with torch.no_grad(): out = leaf_model(t); idx = torch.argmax(torch.softmax(out,1)).item()
# #     cls = sorted(leaf_data["label"].unique())[idx]; row = leaf_data[leaf_data["label"]==cls].iloc[0]
# #     return {"scientific_name":cls,"common_name":row.get("common_name","N/A"),"uses":row.get("uses","N/A"),"origin":row.get("origin","N/A"),"image":image_to_base64(img)}

# # def predict_bark(image):
# #     if bark_model is None: _load_bark()
# #     img = image.convert("RGB"); t = bark_transform(img).unsqueeze(0).to(device)
# #     with torch.no_grad(): out = bark_model(t); idx = torch.argmax(torch.softmax(out,1)).item()
# #     cls = bark_classes[idx]; row = bark_data[bark_data["scientific_name"].str.lower()==cls.lower()].iloc[0]
# #     return {"scientific_name":cls,"common_name":row.get("general_name","N/A"),"uses":row.get("uses","N/A"),"origin":row.get("origin","N/A"),"image":image_to_base64(img)}

# # utils/predict.py with top-5 predictions
# # lazy download models and load CSV from data/train.csv and data/Bark.csv
# # import os, torch, torch.nn as nn
# # from torchvision import models, transforms
# # from PIL import Image
# # import pandas as pd, base64, io
# # try:
# #     import gdown
# # except Exception:
# #     gdown = None
# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
# # DATA_DIR = os.path.join(BACKEND_DIR, "data")
# # MODEL_DIR = os.path.join(BACKEND_DIR, "model")
# # os.makedirs(MODEL_DIR, exist_ok=True)
# # LEAF_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
# # BARK_CSV_PATH = os.path.join(DATA_DIR, "Bark.csv")
# # LEAF_FILE_ID = "12U8nEDWS4chnW71VaUMYwjP9VNXqIMqM"
# # BARK_FILE_ID = "1G8-fXTN0DWcBKfKysxzlBN8zLGvYeKyc"
# # leaf_model = None
# # bark_model = None
# # leaf_data = None
# # bark_data = None
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # def _gdrive_download(file_id, out_path):
# #     if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
# #         return
# #     if gdown is None:
# #         import gdown as _g; _g.download(f"https://drive.google.com/uc?id={file_id}&export=download", out_path, quiet=False)
# #     else:
# #         gdown.download(f"https://drive.google.com/uc?id={file_id}&export=download", out_path, quiet=False)


# # # load functions unchanged ...

# # predict_transforms = transforms.Compose([
# #     transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
# #     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# # ])
# # bark_transform = predict_transforms

# # def image_to_base64(img):
# #     buf = io.BytesIO(); img.save(buf, format="PNG"); return base64.b64encode(buf.getvalue()).decode()

# # def predict_leaf(image):
# #     if leaf_model is None: _load_leaf()
# #     img = image.convert("RGB"); t = predict_transforms(img).unsqueeze(0).to(device)
# #     with torch.no_grad():
# #         out = leaf_model(t)
# #         probs = torch.softmax(out,1).cpu().numpy()[0]
# #     classes = sorted(leaf_data["label"].unique())
# #     top5_idx = probs.argsort()[-5:][::-1]
# #     predictions = [
# #         {"scientific_name": classes[i], "confidence": float(probs[i])}
# #         for i in top5_idx
# #     ]
# #     best = predictions[0]
# #     row = leaf_data[leaf_data["label"]==best["scientific_name"]].iloc[0]
# #     return {"predictions": predictions,
# #             "best_prediction": {
# #                 "scientific_name": best["scientific_name"],
# #                 "common_name": row.get("common_name","N/A"),
# #                 "uses": row.get("uses","N/A"),
# #                 "origin": row.get("origin","N/A"),
# #                 "image": image_to_base64(img)
# #             }}

# # def predict_bark(image):
# #     if bark_model is None: _load_bark()
# #     img = image.convert("RGB"); t = bark_transform(img).unsqueeze(0).to(device)
# #     with torch.no_grad():
# #         out = bark_model(t)
# #         probs = torch.softmax(out,1).cpu().numpy()[0]
# #     top5_idx = probs.argsort()[-5:][::-1]
# #     predictions = [
# #         {"scientific_name": bark_classes[i], "confidence": float(probs[i])}
# #         for i in top5_idx
# #     ]
# #     best = predictions[0]
# #     row = bark_data[bark_data["scientific_name"].str.lower()==best["scientific_name"].lower()].iloc[0]
# #     return {"predictions": predictions,
# #             "best_prediction": {
# #                 "scientific_name": best["scientific_name"],
# #                 "common_name": row.get("general_name","N/A"),
# #                 "uses": row.get("uses","N/A"),
# #                 "origin": row.get("origin","N/A"),
# #                 "image": image_to_base64(img)
# #             }}
# import os
# import io
# import base64
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import pandas as pd
# import requests
# from flask import Flask, request, jsonify

# # ============================================================
# # DEVICE CONFIGURATION
# # ============================================================
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # ============================================================
# # GOOGLE DRIVE FILE IDS
# # ============================================================
# LEAF_FILE_ID = "12U8nEDWS4chnW71VaUMYwjP9VNXqIMqM"  # Leaf model
# BARK_FILE_ID = "1G8-fXTN0DWcBKfKysxzlBN8zLGvYeKyc"  # Bark model

# # ============================================================
# # GOOGLE DRIVE DOWNLOAD FUNCTION
# # ============================================================
# def download_from_drive(file_id: str, dest_path: str):
#     if os.path.exists(dest_path):
#         print(f"✅ Model already exists: {dest_path}")
#         return

#     def _get_confirm_token(response):
#         for key, value in response.cookies.items():
#             if key.startswith("download_warning"):
#                 return value
#         return None

#     def _save_response_content(response, destination):
#         CHUNK_SIZE = 32768
#         with open(destination, "wb") as f:
#             for chunk in response.iter_content(CHUNK_SIZE):
#                 if chunk:
#                     f.write(chunk)

#     print(f"⬇️ Downloading model from Google Drive → {dest_path}")
#     URL = "https://docs.google.com/uc?export=download"
#     session = requests.Session()
#     response = session.get(URL, params={"id": file_id}, stream=True)
#     token = _get_confirm_token(response)
#     if token:
#         params = {"id": file_id, "confirm": token}
#         response = session.get(URL, params=params, stream=True)

#     if "html" in response.headers.get("Content-Type", "").lower():
#         raise RuntimeError(
#             f"❌ Invalid download for {dest_path}. Check Drive sharing permissions."
#         )

#     _save_response_content(response, dest_path)
#     print(f"✅ Downloaded: {dest_path}")

# # ============================================================
# # PATH CONFIGURATION
# # ============================================================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(BASE_DIR, "model")
# DATA_DIR = os.path.join(BASE_DIR, "data")
# os.makedirs(MODEL_DIR, exist_ok=True)

# LEAF_MODEL_PATH = os.path.join(MODEL_DIR, "resnet101_leaf_classifier.pth")
# BARK_MODEL_PATH = os.path.join(MODEL_DIR, "resnet101_final.pth")

# download_from_drive(LEAF_FILE_ID, LEAF_MODEL_PATH)
# download_from_drive(BARK_FILE_ID, BARK_MODEL_PATH)

# LEAF_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
# BARK_CSV_PATH = os.path.join(DATA_DIR, "Bark.csv")

# # ============================================================
# # HELPER FUNCTIONS
# # ============================================================
# def load_csv(path):
#     encodings = ["utf-8", "latin1", "cp1252"]
#     for enc in encodings:
#         try:
#             return pd.read_csv(path, encoding=enc)
#         except Exception:
#             continue
#     raise RuntimeError(f"Failed to read CSV: {path}")

# def _strip_module_prefix(state_dict):
#     return {
#         (k[len("module."):] if k.startswith("module.") else k): v
#         for k, v in state_dict.items()
#     }

# def _extract_state_dict(obj):
#     if isinstance(obj, dict):
#         for key in ("state_dict", "model_state_dict", "model"):
#             if key in obj and isinstance(obj[key], dict):
#                 return _strip_module_prefix(obj[key])
#         return _strip_module_prefix(obj)
#     return None

# # ============================================================
# # LOAD MODELS AND LABELS
# # ============================================================
# leaf_data = load_csv(LEAF_CSV_PATH)
# leaf_data.columns = leaf_data.columns.str.strip().str.lower().str.replace(" ", "_")
# class_names_leaf = list(leaf_data["label"].unique())  # ✅ no sorting

# bark_data = load_csv(BARK_CSV_PATH)
# bark_data.columns = bark_data.columns.str.strip().str.lower().str.replace(" ", "_")
# bark_classes = list(bark_data["scientific_name"].unique())  # ✅ no sorting

# # ---- Leaf model ----
# NUM_CLASSES_LEAF = len(class_names_leaf)
# leaf_model = models.resnet101(weights=None)
# leaf_model.fc = nn.Linear(leaf_model.fc.in_features, NUM_CLASSES_LEAF)
# try:
#     ckpt = torch.load(LEAF_MODEL_PATH, map_location="cpu", weights_only=False)
#     sd = _extract_state_dict(ckpt)
#     if sd:
#         leaf_model.load_state_dict(sd, strict=False)
#     elif isinstance(ckpt, nn.Module):
#         leaf_model = ckpt
# except Exception as e:
#     raise RuntimeError(f"Leaf model load failed: {e}")
# leaf_model.to(device).eval()

# # ---- Bark model ----
# bark_model = models.resnet101(weights=None)
# bark_model.fc = nn.Sequential(
#     nn.Dropout(0.4),
#     nn.Linear(bark_model.fc.in_features, len(bark_classes)),
# )
# try:
#     ckpt = torch.load(BARK_MODEL_PATH, map_location="cpu", weights_only=False)
#     sd = _extract_state_dict(ckpt)
#     if sd:
#         bark_model.load_state_dict(sd, strict=False)
#     elif isinstance(ckpt, nn.Module):
#         bark_model = ckpt
# except Exception as e:
#     raise RuntimeError(f"Bark model load failed: {e}")
# bark_model.to(device).eval()

# # ============================================================
# # IMAGE TRANSFORMS
# # ============================================================
# leaf_transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225]),
# ])

# bark_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])

# # ============================================================
# # FLASK APP
# # ============================================================
# app = Flask(__name__)

# def image_to_base64(img: Image.Image) -> str:
#     buf = io.BytesIO()
#     img.save(buf, format="PNG")
#     return base64.b64encode(buf.getvalue()).decode("utf-8")

# @app.route("/")
# def home():
#     return jsonify({"message": "🌿 Plant Identification API is running!"})

# @app.route("/predict/leaf", methods=["POST"])
# def predict_leaf():
#     if "image" not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400

#     image = Image.open(request.files["image"]).convert("RGB")
#     img_t = leaf_transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         outputs = leaf_model(img_t)
#         probs = torch.softmax(outputs, dim=1)[0]

#     top5_prob, top5_idx = torch.topk(probs, 5)
#     top5 = [
#         {"scientific_name": class_names_leaf[i.item()], "probability": round(p.item() * 100, 2)}
#         for i, p in zip(top5_idx, top5_prob)
#     ]

#     best_class = class_names_leaf[top5_idx[0].item()]
#     row = leaf_data[leaf_data["label"] == best_class]
#     best_prediction = {
#         "scientific_name": best_class,
#         "common_name": row.iloc[0].get("common_name", "N/A") if not row.empty else "N/A",
#         "uses": row.iloc[0].get("uses", "N/A") if not row.empty else "N/A",
#         "origin": row.iloc[0].get("origin", "N/A") if not row.empty else "N/A",
#     }

#     return jsonify({"best_prediction": best_prediction, "top_5_predictions": top5})

# @app.route("/predict/bark", methods=["POST"])
# def predict_bark():
#     if "image" not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400

#     image = Image.open(request.files["image"]).convert("RGB")
#     tensor = bark_transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         outputs = bark_model(tensor)
#         probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

#     top5_idx = probs.argsort()[-5:][::-1]
#     top5 = [
#         {"scientific_name": bark_classes[i], "probability": round(probs[i] * 100, 2)}
#         for i in top5_idx
#     ]

#     best_class = bark_classes[top5_idx[0]]
#     row = bark_data[bark_data["scientific_name"].str.lower() == best_class.lower()]
#     best_prediction = {
#         "scientific_name": best_class,
#         "common_name": row.iloc[0].get("general_name", "N/A") if not row.empty else "N/A",
#         "uses": row.iloc[0].get("uses", "N/A") if not row.empty else "N/A",
#         "origin": row.iloc[0].get("origin", "N/A") if not row.empty else "N/A",
#     }

#     return jsonify({"best_prediction": best_prediction, "top_5_predictions": top5})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


# import os
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import pandas as pd
# import base64
# import io

# # --- Try importing gdown for Google Drive model auto-download ---
# try:
#     import gdown
# except ImportError:
#     gdown = None

# # --- Paths ---
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
# DATA_DIR = os.path.join(BACKEND_DIR, "data")
# MODEL_DIR = os.path.join(BACKEND_DIR, "model")

# os.makedirs(MODEL_DIR, exist_ok=True)

# # --- Google Drive File IDs (update if changed) ---
# LEAF_FILE_ID = "12U8nEDWS4chnW71VaUMYwjP9VNXqIMqM"
# BARK_FILE_ID = "1G8-fXTN0DWcBKfKysxzlBN8zLGvYeKyc"

# # --- Model Paths ---
# LEAF_MODEL_PATH = os.path.join(MODEL_DIR, "resnet101_leaf_classifier.pth")
# BARK_MODEL_PATH = os.path.join(MODEL_DIR, "resnet101_final.pth")

# # --- Google Drive Download Function ---
# def _gdrive_download(file_id: str, out_path: str):
#     """Download model weights from Google Drive if missing."""
#     if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
#         return
#     if gdown is None:
#         print("⚠️ gdown not installed; cannot download model automatically.")
#         return
#     try:
#         url = f"https://drive.google.com/uc?id={file_id}&export=download"
#         print(f"⬇️ Downloading model: {out_path}")
#         gdown.download(url, out_path, quiet=False)
#     except Exception as e:
#         print(f"⚠️ Failed to download {out_path}: {e}")

# # --- Auto-download if needed ---
# _gdrive_download(LEAF_FILE_ID, LEAF_MODEL_PATH)
# _gdrive_download(BARK_FILE_ID, BARK_MODEL_PATH)

# # --- Device Setup ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # --- CSV Paths ---
# LEAF_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
# BARK_CSV_PATH = os.path.join(DATA_DIR, "Bark.csv")

# # --- CSV Loader ---
# def load_csv(path):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"CSV file not found: {path}")
#     for enc in ("utf-8", "latin1", "cp1252"):
#         try:
#             return pd.read_csv(path, encoding=enc)
#         except Exception:
#             continue
#     raise RuntimeError(f"Unable to read CSV: {path}")

# # --- Image Transforms ---
# leaf_transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# bark_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# # --- Globals (cache loaded models & CSVs) ---
# leaf_model = None
# bark_model = None
# leaf_data = None
# bark_data = None
# idx_to_class_leaf = None
# bark_classes = None

# # --- Load Leaf Model ---
# def load_leaf_model():
#     global leaf_model, leaf_data, idx_to_class_leaf
#     if leaf_model is not None:
#         return leaf_model

#     # Load CSV
#     leaf_data = load_csv(LEAF_CSV_PATH)
#     leaf_data.columns = leaf_data.columns.str.strip().str.lower().str.replace(" ", "_")

#     class_names = sorted(leaf_data["label"].unique())
#     idx_to_class_leaf = {i: name for i, name in enumerate(class_names)}

#     # Initialize model
#     model = models.resnet101(weights=None)
#     model.fc = nn.Linear(model.fc.in_features, len(class_names))

#     # Load weights
#     loaded = torch.load(LEAF_MODEL_PATH, map_location="cpu")
#     if isinstance(loaded, dict):
#         state_dict = loaded.get("state_dict", loaded)
#         model.load_state_dict(state_dict, strict=False)
#     elif isinstance(loaded, nn.Module):
#         model = loaded
#     else:
#         raise RuntimeError("Leaf model checkpoint format not recognized.")

#     model.eval()
#     leaf_model = model.to(device)
#     return model

# # --- Load Bark Model ---
# def load_bark_model():
#     global bark_model, bark_data, bark_classes
#     if bark_model is not None:
#         return bark_model

#     # Load CSV
#     bark_data = load_csv(BARK_CSV_PATH)
#     bark_data.columns = bark_data.columns.str.strip().str.lower().str.replace(" ", "_")

#     train_dir = os.path.join(DATA_DIR, "Bark")
#     if os.path.exists(train_dir):
#         bark_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
#     else:
#         bark_classes = sorted(bark_data["scientific_name"].unique())

#     # Initialize model
#     model = models.resnet101(weights=None)
#     in_features = model.fc.in_features
#     model.fc = nn.Sequential(
#         nn.Dropout(p=0.4),
#         nn.Linear(in_features, len(bark_classes))
#     )

#     # Load weights
#     loaded = torch.load(BARK_MODEL_PATH, map_location="cpu")
#     if isinstance(loaded, dict):
#         state_dict = loaded.get("state_dict", loaded)
#         model.load_state_dict(state_dict, strict=False)
#     elif isinstance(loaded, nn.Module):
#         model = loaded
#     else:
#         raise RuntimeError("Bark model checkpoint format not recognized.")

#     model.eval()
#     bark_model = model.to(device)
#     return model

# # --- Convert Image to Base64 (for return) ---
# def image_to_base64(img: Image.Image) -> str:
#     buffered = io.BytesIO()
#     img.save(buffered, format="PNG")
#     return base64.b64encode(buffered.getvalue()).decode("utf-8")

# # --- Predict Leaf ---
# def predict_leaf(image: Image.Image) -> dict:
#     try:
#         model = load_leaf_model()
#         model.eval()

#         img = image.convert("RGB")
#         tensor = leaf_transform(img).unsqueeze(0).to(device)

#         with torch.no_grad():
#             outputs = model(tensor)
#             probs = torch.softmax(outputs, dim=1)[0]
#             top5_probs, top5_idx = torch.topk(probs, 5)

#         leaf_data["label_lower"] = leaf_data["label"].astype(str).str.lower()

#         results = []
#         for p, idx in zip(top5_probs.tolist(), top5_idx.tolist()):
#             cls = idx_to_class_leaf[idx]
#             row = leaf_data[leaf_data["label_lower"] == cls.lower()]
#             info = row.iloc[0] if not row.empty else {}

#             results.append({
#                 "scientific_name": info.get("label", cls),
#                 "common_name": info.get("common_name", "N/A"),
#                 "uses": info.get("uses", "N/A"),
#                 "origin": info.get("origin", "N/A"),
#                 "confidence": round(float(p) * 100, 2)
#             })

#         # Normalize to 100%
#         total = sum(r["confidence"] for r in results)
#         if total > 0:
#             for r in results:
#                 r["confidence"] = round(r["confidence"] / total * 100, 2)

#         return {
#             "best_prediction": results[0] if results else {},
#             "top5_predictions": results,
#             "image": image_to_base64(img)
#         }

#     except Exception as e:
#         return {"error": f"Prediction failed: {str(e)}"}

# # --- Predict Bark ---
# def predict_bark(image: Image.Image) -> dict:
#     try:
#         model = load_bark_model()
#         model.eval()

#         img = image.convert("RGB")
#         tensor = bark_transform(img).unsqueeze(0).to(device)

#         with torch.no_grad():
#             outputs = model(tensor)
#             probs = torch.softmax(outputs, dim=1)[0]
#             top5_probs, top5_idx = torch.topk(probs, 5)

#         bark_data["scientific_name_lower"] = bark_data["scientific_name"].astype(str).str.lower()

#         results = []
#         for p, idx in zip(top5_probs.tolist(), top5_idx.tolist()):
#             cls = bark_classes[idx]
#             row = bark_data[bark_data["scientific_name_lower"] == cls.lower()]
#             info = row.iloc[0] if not row.empty else {}

#             results.append({
#                 "scientific_name": info.get("scientific_name", cls),
#                 "common_name": info.get("general_name", "N/A"),
#                 "uses": info.get("uses", "N/A"),
#                 "origin": info.get("origin", "N/A"),
#                 "confidence": round(float(p) * 100, 2)
#             })

#         # Normalize to 100%
#         total = sum(r["confidence"] for r in results)
#         if total > 0:
#             for r in results:
#                 r["confidence"] = round(r["confidence"] / total * 100, 2)

#         return {
#             "best_prediction": results[0] if results else {},
#             "top5_predictions": results,
#             "image": image_to_base64(img)
#         }

#     except Exception as e:
#         return {"error": f"Prediction failed: {str(e)}"}

# # --- Debug Check (optional) ---
# if __name__ == "__main__":
#     print("✅ Leaf model path:", LEAF_MODEL_PATH)
#     print("✅ Bark model path:", BARK_MODEL_PATH)
#     print("✅ Leaf CSV path:", LEAF_CSV_PATH)
#     print("✅ Bark CSV path:", BARK_CSV_PATH)

# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision import models
# from PIL import Image
# import os
# import logging
# import requests
# from io import BytesIO

# # Initialize logger
# logger = logging.getLogger("plantid-predict")

# # Directories
# MODEL_DIR = os.path.join(os.path.dirname(__file__), "../model")
# os.makedirs(MODEL_DIR, exist_ok=True)

# # Google Drive model links
# MODEL_URLS = {
#     "leaf": "https://drive.google.com/uc?id=12U8nEDWS4chnW71VaUMYwjP9VNXqIMqM",
#     "bark": "https://drive.google.com/uc?id=1G8-fXTN0DWcBKfKysxzlBN8zLGvYeKyc"
# }

# # Model paths
# LEAF_MODEL_PATH = os.path.join(MODEL_DIR, "resnet101_leaf_classifier.pth")
# BARK_MODEL_PATH = os.path.join(MODEL_DIR, "resnet101_final.pth")

# # Download model if not exists
# def download_model(url, path):
#     if not os.path.exists(path):
#         logger.info(f"⬇️ Downloading model from Google Drive → {path}")
#         response = requests.get(url, stream=True)
#         response.raise_for_status()
#         with open(path, "wb") as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 f.write(chunk)
#         logger.info(f"✅ Model downloaded successfully: {path}")

# download_model(MODEL_URLS["leaf"], LEAF_MODEL_PATH)
# download_model(MODEL_URLS["bark"], BARK_MODEL_PATH)

# # Device config
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Transforms
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# # Class labels (adjust as per your training)
# LEAF_CLASSES = ["Neem", "Mango", "Guava", "Banana", "Jasmine", "Tulsi"]
# BARK_CLASSES = ["Teak", "Neem", "Mango", "Guava", "Eucalyptus", "Banyan"]

# # Load model
# def load_model(model_path, num_classes):
#     model = models.resnet101(weights=None)
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     try:
#         model.load_state_dict(torch.load(model_path, map_location=DEVICE))
#         model.to(DEVICE)
#         model.eval()
#         logger.info(f"✅ Loaded model: {model_path}")
#         return model
#     except Exception as e:
#         logger.error(f"❌ Error loading model {model_path}: {e}")
#         return None

# # Initialize models
# leaf_model = load_model(LEAF_MODEL_PATH, len(LEAF_CLASSES))
# bark_model = load_model(BARK_MODEL_PATH, len(BARK_CLASSES))

# # Load image (from file, bytes, or URL)
# def load_image(image_input):
#     try:
#         if isinstance(image_input, Image.Image):
#             return image_input.convert("RGB")
#         elif isinstance(image_input, str) and image_input.startswith("http"):
#             response = requests.get(image_input)
#             response.raise_for_status()
#             return Image.open(BytesIO(response.content)).convert("RGB")
#         else:
#             return Image.open(BytesIO(image_input)).convert("RGB")
#     except Exception as e:
#         raise ValueError(f"Invalid image or URL: {e}")

# # Predict helper
# def predict(model, image, class_names):
#     try:
#         image_t = transform(image).unsqueeze(0).to(DEVICE)
#         with torch.no_grad():
#             outputs = model(image_t)
#             _, pred = torch.max(outputs, 1)
#             return {"prediction": class_names[pred.item()]}
#     except Exception as e:
#         logger.exception("Prediction failed")
#         return {"error": f"Prediction failed: {e}"}

# # Prediction endpoints
# def predict_leaf(image_input):
#     if leaf_model is None:
#         return {"error": "Leaf model not loaded"}
#     try:
#         image = load_image(image_input)
#         return predict(leaf_model, image, LEAF_CLASSES)
#     except Exception as e:
#         return {"error": f"Leaf prediction failed: {e}"}

# def predict_bark(image_input):
#     if bark_model is None:
#         return {"error": "Bark model not loaded"}
#     try:
#         image = load_image(image_input)
#         return predict(bark_model, image, BARK_CLASSES)
#     except Exception as e:
        # return {"error": f"Bark prediction failed: {e}"}


# utils/predict.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import os
from PIL import Image
import logging

# Set up logging
logger = logging.getLogger("plantid-predict")

# Paths to models
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
LEAF_MODEL_PATH = os.path.join(MODEL_DIR, "resnet101_leaf_classifier.pth")
BARK_MODEL_PATH = os.path.join(MODEL_DIR, "resnet101_final.pth")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load model safely with fallback for PyTorch 2.6+
def load_model(model_path, num_classes):
    model = models.resnet101(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if not os.path.exists(model_path):
        logger.error("Model not found: %s", model_path)
        return None

    try:
        # Try new PyTorch behavior (safe)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        # Older versions of torch may not support weights_only param
        state_dict = torch.load(model_path, map_location=device)
    except Exception as e:
        logger.warning("⚠️ Safe load failed (%s), retrying with weights_only=False", e)
        # Force allow full pickle load (only if you trust the file)
        state_dict = torch.load(model_path, map_location=device, weights_only=False)

    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


# Lazy load models (only once)
_leaf_model = None
_bark_model = None


def get_leaf_model():
    global _leaf_model
    if _leaf_model is None:
        logger.info("Loading leaf model from %s", LEAF_MODEL_PATH)
        _leaf_model = load_model(LEAF_MODEL_PATH, num_classes=5)  # adjust num_classes
    return _leaf_model


def get_bark_model():
    global _bark_model
    if _bark_model is None:
        logger.info("Loading bark model from %s", BARK_MODEL_PATH)
        _bark_model = load_model(BARK_MODEL_PATH, num_classes=5)  # adjust num_classes
    return _bark_model


# Predict functions
def predict_leaf(image: Image.Image):
    model = get_leaf_model()
    if model is None:
        return {"error": "Leaf model not available"}
    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)
    return {"prediction": f"Leaf class {pred.item()}"}


def predict_bark(image: Image.Image):
    model = get_bark_model()
    if model is None:
        return {"error": "Bark model not available"}
    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)
    return {"prediction": f"Bark class {pred.item()}"}


