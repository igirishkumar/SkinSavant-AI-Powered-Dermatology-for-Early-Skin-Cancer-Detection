# ai_core.py (Corrected with Proper Risk Levels)

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os

# --- Your Model Configuration Code ---
CLASS_NAMES = [
    'akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'
]
CLASS_FULL_NAMES = {
    'akiec': 'Actinic Keratoses / Bowen\'s Disease', 
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis', 
    'df': 'Dermatofibroma',
    'mel': 'Melanoma', 
    'nv': 'Melanocytic Nevi (Mole)', 
    'vasc': 'Vascular Lesions'
}

# FIXED: Proper medical risk levels (Low/Medium/High only, no "Critical")
CLASS_RISK = {
    'akiec': 'High',      # Actinic Keratoses - Precancerous lesion
    'bcc': 'High',        # Basal Cell Carcinoma - Most common skin cancer
    'bkl': 'Low',         # Benign Keratosis - Non-cancerous growth
    'df': 'Low',          # Dermatofibroma - Benign fibrous nodule
    'mel': 'High',        # Melanoma - Most dangerous skin cancer
    'nv': 'Low',          # Melanocytic Nevi - Common mole, usually benign
    'vasc': 'Low'         # Vascular Lesions - Blood vessel abnormalities, usually benign
}

# --- Your Loading Functions ---
def load_model(model_path, device='cpu', num_classes=7):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint: state_dict = checkpoint['model']
        else: state_dict = checkpoint
    else:
        if hasattr(checkpoint, 'eval'): checkpoint.eval(); return checkpoint.to(device)
        state_dict = checkpoint

    keys = list(state_dict.keys())
    first_key = keys[0] if keys else ""
    model = None

    if any('0.0.' in k for k in keys[:10]):
        print("Detected fastai-style model")
        model = load_fastai_model(state_dict, num_classes, device)
    elif 'features.denseblock' in first_key or any('denseblock' in k for k in keys[:20]):
        if any('denseblock4.denselayer32' in k for k in keys): print("Detected DenseNet161"); model = models.densenet161(weights=None)
        elif any('denseblock4.denselayer24' in k for k in keys): print("Detected DenseNet169"); model = models.densenet169(weights=None)
        elif any('denseblock4.denselayer16' in k for k in keys): print("Detected DenseNet121"); model = models.densenet121(weights=None)
        else: print("Detected DenseNet201"); model = models.densenet201(weights=None)
        num_features = model.classifier.in_features; model.classifier = nn.Linear(num_features, num_classes)
    elif 'layer4' in str(keys) and 'fc' in str(keys):
        print("Detected ResNet50/101/152"); model = models.resnet50(weights=None)
        num_features = model.fc.in_features; model.fc = nn.Linear(num_features, num_classes)
    else:
        print("Unknown architecture, trying DenseNet121"); model = models.densenet121(weights=None)
        num_features = model.classifier.in_features; model.classifier = nn.Linear(num_features, num_classes)

    try: model.load_state_dict(state_dict, strict=True); print("Model loaded with strict=True")
    except RuntimeError as e:
        print(f"Strict loading failed: {e}"); model.load_state_dict(state_dict, strict=False); print("Model loaded with strict=False")
    
    model = model.to(device); model.eval()
    return model

def load_fastai_model(state_dict, num_classes, device):
    model = models.densenet169(weights=None); num_features = model.classifier.in_features; model.classifier = nn.Linear(num_features, num_classes)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith('0.'): new_key = key[2:]
        if key.startswith('1.'): new_key = 'classifier' + key[1:]
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict, strict=False); return model

# --- Your Wrapper Class ---
class SkinCancerClassifier:
    def __init__(self, model_path, device=None, num_classes=7):
        if device is None: self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else: self.device = torch.device(device)
        self.model = load_model(model_path, self.device, num_classes)
        self.class_names = CLASS_NAMES; self.class_full_names = CLASS_FULL_NAMES
        self.class_risk = CLASS_RISK; self.num_classes = num_classes
        self.model_type = type(self.model).__name__

    def predict(self, image_tensor):
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device); outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1); top_prob, top_idx = torch.max(probabilities, dim=1)
            idx = top_idx.item()
            predicted_class = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
            all_probs = {self.class_names[i]: probabilities[0][i].item() for i in range(min(len(self.class_names), probabilities.shape[1]))}
            return {
                'predicted_class': predicted_class, 
                'predicted_class_full': self.class_full_names.get(predicted_class, predicted_class),
                'confidence': top_prob.item(), 
                'risk_level': self.class_risk.get(predicted_class, 'Unknown'), 
                'all_probabilities': all_probs
            }
    def get_model(self): return self.model

# --- Grad-CAM Implementation (Corrected) ---
class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model; self.target_layer_name = target_layer_name
        self.gradients = None; self.activations = None; self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output): self.activations = output
        def backward_hook(module, grad_input, grad_output): self.gradients = grad_output[0]
        
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx):
        output = self.model(input_tensor); self.model.zero_grad(); class_score = output[0, class_idx]; class_score.backward()
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(gradients, dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        cam = np.maximum(cam.cpu().detach().numpy(), 0)
        cam = cv2.resize(cam, (224, 224))
        
        if np.max(cam) > np.min(cam):
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        else:
            cam = np.zeros_like(cam)
            
        return np.uint8(cam * 255)

# --- Function to Generate Multiple Grad-CAMs ---
def generate_multiple_gradcams(model, input_tensor, image_path, class_names, target_layer_name):
    """Generates Grad-CAMs for the top 3 predicted classes."""
    gradcam_paths = []
    original_image_cv = cv2.imread(image_path)
    if original_image_cv is None:
        print(f"[ERROR] Could not read image for Grad-CAM overlay: {image_path}")
        return []
    original_image_cv = cv2.resize(original_image_cv, (224, 224))
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top3_probs, top3_indices = torch.topk(probabilities, 3)

    for i in range(3):
        class_idx = top3_indices[0][i].item()
        class_name = class_names[class_idx]
        
        grad_cam = GradCAM(model, target_layer_name)
        cam = grad_cam.generate_cam(input_tensor, class_idx)
        
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(original_image_cv, 0.6, heatmap, 0.4, 0)

        gradcam_output_dir = 'static/gradcam_outputs'
        os.makedirs(gradcam_output_dir, exist_ok=True)
        
        base_name = os.path.basename(image_path).split('.')[0]
        gradcam_filename = f"{base_name}_gradcam_{i+1}.jpg"
        
        gradcam_path = os.path.join(gradcam_output_dir, gradcam_filename)
        cv2.imwrite(gradcam_path, superimposed_img)
        gradcam_paths.append(gradcam_path)
        print(f"[DEBUG] Saved Grad-CAM: {gradcam_path}")

    return gradcam_paths

# --- Main Prediction Function ---
def predict_and_explain(image_path):
    try:
        print(f"[DEBUG] Starting analysis for: {image_path}")
        
        MODEL_PATH = 'best_model.pth'
        classifier = SkinCancerClassifier(MODEL_PATH)
        
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(image).unsqueeze(0)

        prediction_results = classifier.predict(input_tensor)
        risk_level = prediction_results['risk_level']
        confidence_score = prediction_results['confidence']
        predicted_class = prediction_results['predicted_class']
        full_class_name = prediction_results['predicted_class_full']

        model = classifier.get_model()
        target_layer = 'features.denseblock4'
        if 'ResNet' in classifier.model_type:
            target_layer = 'layer4'
        
        print("[DEBUG] Generating multiple Grad-CAMs...")
        gradcam_paths = generate_multiple_gradcams(model, input_tensor, image_path, CLASS_NAMES, target_layer)
        print(f"[DEBUG] Generated Grad-CAMs: {gradcam_paths}")
        
        # Determine simple risk for CSS (lowercase for CSS classes)
        simple_risk = risk_level.lower()  # Will be 'low', 'medium', or 'high'
        
        print(f"[DEBUG] Analysis successful. Risk: {risk_level}, Simple Risk: {simple_risk}, Confidence: {confidence_score}")
        return predicted_class, confidence_score, gradcam_paths, full_class_name, simple_risk

    except Exception as e:
        print(f"[FATAL ERROR] during prediction/explanation: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None