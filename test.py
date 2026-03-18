from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "/content/drive/MyDrive/Colab Notebooks/Apollo2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

!pip install transformers==4.51.3 -U
pip install onnx_ir
#!git clone https://github.com/casper-hansen/AutoAWQ
#!cd AutoAWQ
#!pip install -e .
!pip install autoawq
!pip install onnxruntime-genai-cuda
!curl https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/awq-quantized-model.py -o awq-quantized-model.py

!python awq-quantized-model.py --model_path /Apollo2 --quant_path ./awq-out/ --output_path ./onnx-out/ --execution_provider cuda
