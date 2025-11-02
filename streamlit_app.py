"""
Streamlit App for Pseudocode to Code Converter
==============================================
This app uses a fine-tuned GPT-2 model to convert pseudocode to code.
"""

import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Pseudocode to Code Converter",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stCodeBlock {
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the model and tokenizer with caching"""
    try:
        # Model is in the current directory
        model_dir = Path(".")
        
        # Check if required model files exist
        required_files = ["config.json", "tokenizer_config.json", "vocab.json"]
        missing_files = [f for f in required_files if not (model_dir / f).exists()]
        
        if missing_files:
            st.error(f"❌ Missing required files: {', '.join(missing_files)}")
            st.error("Please make sure all model files are in the current directory.")
            return None, None, None
        
        # Check for model file (could be .safetensors or .bin)
        model_file = None
        if (model_dir / "model.safetensors").exists():
            model_file = "model.safetensors"
        elif (model_dir / "pytorch_model.bin").exists():
            model_file = "pytorch_model.bin"
        
        if not model_file:
            st.warning("⚠️ Model weight file not found. Checking if files exist in directory...")
            model_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
            if model_files:
                st.info(f"Found model files: {[f.name for f in model_files]}")
            return None, None, None
        
        st.info("🔄 Loading model and tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        model = GPT2LMHeadModel.from_pretrained(model_dir)
        
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        st.success(f"✅ Model loaded successfully on {device.upper()}!")
        return tokenizer, model, device
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {str(e)}")
        st.error("Please make sure all model files are in the current directory.")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.exception(e)
        return None, None, None

def generate_code(pseudo_code, tokenizer, model, device, max_length=150, temperature=0.7):
    """
    Generate code from pseudocode using the trained model
    
    Args:
        pseudo_code: Input pseudocode string
        tokenizer: GPT2Tokenizer instance
        model: GPT2LMHeadModel instance
        device: Device to run inference on
        max_length: Maximum length of generated sequence
        temperature: Sampling temperature
    
    Returns:
        Generated code string
    """
    prompt = f"### PSEUDOCODE:\n{pseudo_code}\n### PYTHON CODE:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the code part
    if "### PYTHON CODE:" in generated:
        code = generated.split("### PYTHON CODE:")[1].strip()
        # Remove any trailing markers
        if "###" in code:
            code = code.split("###")[0].strip()
        return code
    
    return generated

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">💻 Pseudocode to Code Converter</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Convert your pseudocode to code using AI-powered GPT-2 model</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown("---")
        
        max_length = st.slider("Max Length", min_value=50, max_value=300, value=150, step=10,
                              help="Maximum length of generated code")
        temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, value=0.7, step=0.1,
                               help="Controls randomness: lower = more deterministic, higher = more creative")
        
        st.markdown("---")
        st.header("📝 Example Inputs")
        example_inputs = [
            "create integer variable x",
            "read input from user",
            "for i from 0 to 10 print i",
            "if x greater than 5 print yes",
            "create list numbers",
            "set x to 10",
            "print sum of a and b"
        ]
        
        for example in example_inputs:
            if st.button(f"📌 {example}", key=example, use_container_width=True):
                st.session_state.example_input = example
    
    # Load model (cached)
    tokenizer, model, device = load_model()
    
    if tokenizer is None or model is None:
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📥 Input Pseudocode")
        
        # Check if example was clicked
        default_value = st.session_state.get('example_input', '')
        if default_value:
            st.session_state.example_input = ''  # Clear after use
        
        pseudo_input = st.text_area(
            "Enter your pseudocode:",
            value=default_value,
            height=200,
            placeholder="e.g., create integer variable x\nread input from user\nprint x",
            key="pseudo_input"
        )
        
        generate_button = st.button("🚀 Generate Code", type="primary", use_container_width=True)
    
    with col2:
        st.header("📤 Generated Code")
        
        if generate_button:
            if not pseudo_input.strip():
                st.warning("⚠️ Please enter some pseudocode first!")
            else:
                with st.spinner("🤖 Generating code..."):
                    try:
                        generated_code = generate_code(
                            pseudo_input,
                            tokenizer,
                            model,
                            device,
                            max_length=max_length,
                            temperature=temperature
                        )
                        
                        st.code(generated_code, language=None)
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Code",
                            data=generated_code,
                            file_name="generated_code.txt",
                            mime="text/plain"
                        )
                        
                        # Copy button (using st.code which already has copy functionality)
                        st.info("💡 Tip: Click the copy icon in the code block above to copy the code!")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating code: {str(e)}")
                        st.exception(e)
        else:
            st.info("👈 Enter pseudocode and click 'Generate Code' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Built with ❤️ using Streamlit and GPT-2</p>
        <p>Model: Fine-tuned GPT-2 on SPOC dataset</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

