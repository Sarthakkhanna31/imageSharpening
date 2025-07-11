
import streamlit as st
import torch
import torchvision.transforms as T
import cv2
from PIL import Image
import numpy as np
import tempfile

from model import HookedDnCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HookedDnCNN().to(device)
model.load_state_dict(torch.load('student_motion_model_2.pth', map_location=device))
model.eval()

def preprocess_image(img):
    img = img.convert('RGB').resize((128, 128))
    tensor = T.ToTensor()(img).unsqueeze(0).to(device)
    return tensor

def postprocess_image(tensor):
    tensor = tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    tensor = np.clip(tensor, 0, 1)
    return (tensor * 255).astype(np.uint8)

def enhance_image(uploaded_file):
    img = Image.open(uploaded_file)
    inp = preprocess_image(img)
    with torch.no_grad():
        out = model(inp)
    sharp = postprocess_image(out)
    return sharp

def enhance_video(uploaded_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inp = preprocess_image(img)
        with torch.no_grad():
            out = model(inp)
        sharp = postprocess_image(out)
        frames.append(sharp)

    cap.release()
    return frames

st.title("🔍 Real-Time Image/Video Sharpening using Knowledge Distillation")
option = st.radio("Choose input type:", ["Image", "Video"])

if option == "Image":
    file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if file:
        sharp_img = enhance_image(file)
        st.image(sharp_img, caption="Enhanced Image", use_column_width=True)

elif option == "Video":
    file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if file:
        st.text("Processing video...")
        sharp_frames = enhance_video(file)
        st.video(file)
        st.text("Preview of enhanced frames:")
        for frame in sharp_frames[:5]:
            st.image(frame, use_column_width=True)
