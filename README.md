# CIFAR-10 Image Classifier 🖼️🚀

This project uses **Transfer Learning** with **ResNet18** to classify images from the CIFAR-10 dataset. It includes a **Gradio interface** for easy testing and interaction.

---

## Classes

The CIFAR-10 dataset contains 10 classes:

- airplane  
- automobile  
- bird  
- cat  
- deer  
- dog  
- frog  
- horse  
- ship  
- truck  

---

## How to Run

### 1️⃣ Install dependencies

```bash
pip install torch torchvision gradio Pillow
```

### 2️⃣ Run train_model.py

```bash
python train_model.py
```

### 3️⃣ Launch the Gradio interface

```bash
python app.py
```

## Folder Structure

cifar10-classifier/

├─ train_model.py

├─ app.py

└─ README.md

## Notes

This project is for educational and portfolio purposes.

You can re-train the model using train_model.py if you want to experiment with different settings.
