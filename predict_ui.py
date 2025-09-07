import gradio as gr
import torch
from torchvision import transforms, models
from PIL import Image

# 1️⃣ Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_classes = 10  
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('models/transfer_model.pth', map_location=device))
model.to(device)
model.eval()

# 2️⃣ Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

# 3️⃣ Prediction function
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']  # CIFAR-10 classes
def predict(img):
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)
    return class_names[pred.item()]

# 4️⃣ Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=1),  
    title="CIFAR-10 Image Classifier",
    description="Upload an image and see which CIFAR-10 class it belongs to!"
)

iface.launch(share=True)