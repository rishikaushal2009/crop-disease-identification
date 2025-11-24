import json
import os
from pathlib import Path
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import gradio as gr

MODEL_PATH = Path("models/disease_model.pth")
CLASS_INDEX_PATH = Path("models/class_index.json")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_class_names = []

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_artifacts():
    global _model, _class_names
    # Load class names first to know output dimension when building architecture
    if CLASS_INDEX_PATH.exists():
        with open(CLASS_INDEX_PATH, 'r', encoding='utf-8') as f:
            _class_names = json.load(f)
    else:
        raise FileNotFoundError(f"Class index file not found: {CLASS_INDEX_PATH}.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Train and save first.")

    # Try safe weights-only load first (state_dict). Fallback to full model load if necessary.
    try:
        obj = torch.load(MODEL_PATH, map_location=_device, weights_only=True)
        if isinstance(obj, dict):
            # Build the architecture and load weights
            model = models.mobilenet_v3_small(weights=None)
            model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(_class_names))
            model.load_state_dict(obj, strict=False)
            _model = model.to(_device)
        else:
            _model = obj
    except Exception:
        # Allowlist MobileNetV3 for safe unpickling, then load with weights_only=False
        try:
            from torchvision.models.mobilenetv3 import MobileNetV3
            torch.serialization.add_safe_globals([MobileNetV3])
        except Exception:
            pass
        _model = torch.load(MODEL_PATH, map_location=_device, weights_only=False)

    if hasattr(_model, 'eval'):
        _model.eval()

@torch.no_grad()
def predict(image: Image.Image):
    if _model is None:
        load_artifacts()
    tensor = transform(image).unsqueeze(0).to(_device)
    outputs = _model(tensor)
    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    return {cls: float(prob) for cls, prob in zip(_class_names, probs)}

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Crop Disease Classifier",
    description="Upload a leaf image to classify disease (demo)."
)

if __name__ == "__main__":
    if not MODEL_PATH.exists() or not CLASS_INDEX_PATH.exists():
        print("Artifacts missing. Please train the model and generate 'models/disease_model.pth' and 'models/class_index.json'.")
    iface.launch()
