import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model import get_model

class PlantAPI:
    def __init__(self, weights='./weights/model.pth', device=None):
        if device is None:
            device = torch.device("cpu")
        else: self.device = torch.device(device)

        # I WILL NOT DIE, I'LL WAIT HERE FOR YOU, IN MY TIME OF DYIIIING
        if not os.path.exists(weights):
            raise FileNotFoundError(f"No .pts in <{weights}>")

        checkpoint = torch.load(weights, map_location=self.device)
        self.class_names = checkpoint['class_names']

        self.model = get_model(classes = len(self.class_names), pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image):
        if isinstance(image, str):
            sample = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            sample = image.convert("RGB")
        
        tensor = self.transform(sample).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probs = F.softmax(output, dim=1)[0]

        conf, pred_idx = torch.max(probs, dim=0)
        top_classes = self.class_names[pred_idx.item()]
        top_conf = conf.item() * 100

        top_probs, top_idx = torch.topk(probs, 3)
        top_res = [
            {"class" : self.class_names[idx.item()], "conf" : round(prob.item() * 100, 2)}
            for prob, idx in zip(top_probs, top_idx)
        ]
        
        return {
            "class" : top_classes,
            "confidence" : round(top_conf, 2),
            "top_results" : top_res
        }

_classifier_instance = None # Singleton

def plant_api(image : str, weights : str = "./model/weights/model.pth"):
    global _classifier_instance

    if _classifier_instance is None:
        _classifier_instance = PlantAPI(weights=weights)

    return _classifier_instance.predict(image)