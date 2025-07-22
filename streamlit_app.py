import streamlit as st
import torch
import torchvision.transforms as T
import cv2
from PIL import Image
import numpy as np
import tempfile
import time
from io import BytesIO
from model import HookedDnCNN

# Page configuration
st.set_page_config(
    page_title="AI Image Enhancer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .stats-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #667eea;
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .processing-animation {
        text-align: center;
        padding: 2rem;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HookedDnCNN().to(device)
    model.load_state_dict(torch.load('student_motion_model_2.pth', map_location=device))
    model.eval()
    return model, device

def preprocess_image(img):
    img = img.convert('RGB').resize((128, 128))
    tensor = T.ToTensor()(img).unsqueeze(0).to(st.session_state.device)
    return tensor

def postprocess_image(tensor):
    tensor = tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    tensor = np.clip(tensor, 0, 1)
    return (tensor * 255).astype(np.uint8)

def enhance_image(uploaded_file):
    img = Image.open(uploaded_file)
    original_size = img.size
    inp = preprocess_image(img)
    
    with torch.no_grad():
        out = st.session_state.model(inp)
    
    sharp = postprocess_image(out)
    # Resize back to original dimensions
    sharp_img = Image.fromarray(sharp).resize(original_size, Image.LANCZOS)
    return np.array(sharp_img), img

def enhance_video(uploaded_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inp = preprocess_image(img)
        
        with torch.no_grad():
            out = st.session_state.model(inp)
        
        sharp = postprocess_image(out)
        frames.append(sharp)
        
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
        
        # Limit processing for demo purposes
        if frame_count >= 30:  # Process only first 30 frames
            break
    
    cap.release()
    progress_bar.empty()
    status_text.empty()
    return frames, fps

# Initialize session state
if 'model' not in st.session_state:
    with st.spinner('Loading AI model...'):
        st.session_state.model, st.session_state.device = load_model()

# Header
st.markdown("""
<div class="main-header">
    <h1>🔍 AI-Powered Image & Video Enhancer</h1>
    <p style="font-size: 1.2rem; margin: 0;">Real-Time Sharpening using Knowledge Distillation</p>
    <p style="font-size: 1rem; margin-top: 0.5rem; opacity: 0.9;">Transform blurry images and videos into crystal-clear content</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.markdown("### 🎯 About This Project")
    st.markdown("""
    <div class="feature-card">
        <h4>🧠 Deep Learning Model</h4>
        <p>Uses Knowledge Distillation with DnCNN architecture for efficient real-time processing</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4>⚡ Key Features</h4>
        <ul>
            <li>Real-time image enhancement</li>
            <li>Video frame processing</li>
            <li>GPU acceleration support</li>
            <li>Batch processing capability</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Model info
    device_info = "🚀 GPU" if torch.cuda.is_available() else "💻 CPU"
    st.markdown(f"""
    <div class="stats-container">
        <h4>System Status</h4>
        <p><strong>Device:</strong> {device_info}</p>
        <p><strong>Model:</strong> Loaded ✅</p>
        <p><strong>Status:</strong> Ready for processing</p>
    </div>
    """, unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📂 Input Selection")
    
    # Input type selection with better styling
    option = st.radio(
        "Choose your input type:",
        ["📸 Image Enhancement", "🎬 Video Enhancement"],
        help="Select whether you want to enhance a single image or process video frames"
    )
    
    if "Image" in option:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("#### Upload Your Image")
        st.markdown("*Supported formats: PNG, JPG, JPEG*")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg"],
            help="Upload a blurry or low-quality image to enhance"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            with st.spinner('🔄 Enhancing your image...'):
                start_time = time.time()
                enhanced_img, original_img = enhance_image(uploaded_file)
                processing_time = time.time() - start_time
            
            with col2:
                st.markdown("### 🎯 Results")
                
                # Show processing stats
                st.markdown(f"""
                <div class="success-message">
                    <strong>✅ Enhancement Complete!</strong><br>
                    Processing time: {processing_time:.2f} seconds<br>
                    Resolution: {original_img.size[0]}x{original_img.size[1]}
                </div>
                """, unsafe_allow_html=True)
                
                # Comparison tabs
                tab1, tab2, tab3 = st.tabs(["🔍 Enhanced", "📷 Original", "🔄 Comparison"])
                
                with tab1:
                    st.image(enhanced_img, caption="Enhanced Image", use_column_width=True)
                    
                with tab2:
                    st.image(original_img, caption="Original Image", use_column_width=True)
                    
                with tab3:
                    col_before, col_after = st.columns(2)
                    with col_before:
                        st.markdown("**Before**")
                        st.image(original_img, use_column_width=True)
                    with col_after:
                        st.markdown("**After**")
                        st.image(enhanced_img, use_column_width=True)
                
                # Download button
                enhanced_pil = Image.fromarray(enhanced_img)
                buf = BytesIO()
                enhanced_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 Download Enhanced Image",
                    data=byte_im,
                    file_name="enhanced_image.png",
                    mime="image/png"
                )
    
    else:  # Video option
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("#### Upload Your Video")
        st.markdown("*Supported formats: MP4, AVI, MOV*")
        
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov"],
            help="Upload a video to enhance its frames"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            with col2:
                st.markdown("### 🎬 Video Processing")
                
                # Original video preview
                st.markdown("#### Original Video")
                st.video(uploaded_file)
                
                if st.button("🚀 Start Processing", type="primary"):
                    with st.spinner('🔄 Processing video frames...'):
                        start_time = time.time()
                        enhanced_frames, fps = enhance_video(uploaded_file)
                        processing_time = time.time() - start_time
                    
                    st.markdown(f"""
                    <div class="success-message">
                        <strong>✅ Video Processing Complete!</strong><br>
                        Processed {len(enhanced_frames)} frames<br>
                        Processing time: {processing_time:.2f} seconds<br>
                        Average: {len(enhanced_frames)/processing_time:.1f} FPS
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("#### 🖼️ Enhanced Frame Samples")
                    
                    # Display enhanced frames in a grid
                    cols = st.columns(3)
                    for i, frame in enumerate(enhanced_frames[:9]):  # Show first 9 frames
                        with cols[i % 3]:
                            st.image(
                                frame, 
                                caption=f"Frame {i+1}",
                                use_column_width=True
                            )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #666;">
    <p>🔬 <strong>Built with:</strong> PyTorch • Streamlit • Computer Vision • Deep Learning</p>
    <p>💡 <strong>Technology:</strong> Knowledge Distillation • DnCNN Architecture • Real-time Processing</p>
</div>
""", unsafe_allow_html=True)