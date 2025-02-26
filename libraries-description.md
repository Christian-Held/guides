# **🚀 Guide: Understanding CUDA, PyTorch, Transformers, PEFT, and More!**
This guide explains **all the key technologies** we installed for fine-tuning DeepSeek. It focuses on **why we need them, what they do, and how they affect training.**  

---

## **📌 1️⃣ CUDA (Compute Unified Device Architecture)**
✅ **What is it?**  
CUDA is a **parallel computing platform** by NVIDIA that allows **GPUs** to accelerate deep learning. It provides **low-level access** to the GPU for AI computations.

✅ **Why do we need it?**  
Without CUDA, deep learning would run on **CPU**, which is **100x slower** for training. CUDA enables models like DeepSeek to **train efficiently**.

✅ **How does it impact training?**  
- **More VRAM → Larger models fit in memory.**  
- **More CUDA cores → Faster matrix calculations.**  
- **Faster tensor operations → Reduces training time.**  

✅ **What happens if CUDA is missing?**  
- PyTorch **falls back to CPU** (VERY slow training).  
- DeepSeek **might not run at all** on large models.  

✅ **Installed with:**  
```bash
sudo sh cuda_12.1.1_530.30.02_linux.run
```
✅ **Check if CUDA is working:**  
```bash
nvidia-smi
```

---

## **📌 2️⃣ cuDNN (CUDA Deep Neural Network Library)**
✅ **What is it?**  
A **GPU-accelerated library** that optimizes **neural networks** for NVIDIA GPUs. It speeds up deep learning **by optimizing low-level tensor computations**.

✅ **Why do we need it?**  
- Makes training **much faster** by using optimized GPU operations.  
- Reduces **latency** in inference (faster responses).  
- Works **with PyTorch** to improve performance.

✅ **What happens if cuDNN is missing?**  
- Training **still works**, but **slower** because PyTorch uses **unoptimized** GPU code.  
- Some **large models may fail to run.**

✅ **Installed automatically with CUDA.**

---

## **📌 3️⃣ PyTorch (Machine Learning Framework)**
✅ **What is it?**  
A **deep learning framework** developed by Meta (Facebook). It allows us to **train, fine-tune, and deploy AI models efficiently**.

✅ **Why do we need it?**  
- **Runs neural networks** using CUDA & GPU acceleration.  
- Allows **fine-tuning** of pre-trained models.  
- Works with **Transformers, PEFT, LoRA**, and more.  

✅ **Key features:**  
- **Dynamic computation graphs** (flexible training).  
- **Optimized tensor operations** (fast GPU math).  
- **Autograd (automatic differentiation)** → Simplifies backpropagation.

✅ **Installed with:**  
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
✅ **Check if PyTorch is using CUDA:**  
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
✅ **Output should be:**  
```bash
True
```
If **False**, CUDA isn't working correctly.

---

## **📌 4️⃣ Hugging Face Transformers (Pre-trained AI Models)**
✅ **What is it?**  
A **library for NLP models**, including **DeepSeek, GPT, LLaMA, and BERT**. It simplifies working with **pre-trained** AI models.

✅ **Why do we need it?**  
- Loads **pre-trained DeepSeek models**.  
- Converts **text into tensors** for the AI to understand.  
- Provides **ready-to-use architectures** for training.  

✅ **How does it impact fine-tuning?**  
- Allows **modifying** pre-trained models with our **own dataset**.  
- Uses **attention mechanisms** for text processing.  
- Supports **LoRA & quantization** to optimize large models.  

✅ **Installed with:**  
```bash
pip install transformers
```

✅ **Example usage:**  
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

input_text = "Who is Christian Held?"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
output = model.generate(**inputs)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
---

## **📌 5️⃣ PEFT (Parameter Efficient Fine-Tuning)**
✅ **What is it?**  
PEFT is a **library for fine-tuning large models** without **modifying all weights**.

✅ **Why do we need it?**  
- Fine-tunes **only small adapter layers** instead of the whole model.  
- **Uses less VRAM** (needed for RTX 2060 Super).  
- Works well with **LoRA** to speed up training.

✅ **Installed with:**  
```bash
pip install peft
```

✅ **How it helps?**  
- Instead of updating **billions** of parameters, it only updates **millions** → **Much faster & less memory usage.**  
- Keeps **pre-trained knowledge** while adapting to new data.  

✅ **Example usage:**  
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```
---

## **📌 6️⃣ BitsAndBytes (Quantization for Large Models)**
✅ **What is it?**  
A library that allows models to **use less VRAM** by converting weights to **4-bit or 8-bit precision**.

✅ **Why do we need it?**  
- Makes **big models fit in GPU memory**.  
- Allows training **on consumer GPUs** (like RTX 2060).  
- **Reduces accuracy slightly**, but speeds up training.

✅ **Installed with:**  
```bash
pip install bitsandbytes
```

✅ **How does it work?**  
- Uses **"quantization"** to store numbers with **fewer bits**.  
- **8-bit quantization** → **Faster, slightly less accurate.**  
- **4-bit quantization** → **Even more memory savings, but lower precision.**  

✅ **Example usage:**  
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Use 8-bit precision
    bnb_8bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", quantization_config=bnb_config)
```
---

## **📌 7️⃣ Datasets (Handling Large Training Data)**
✅ **What is it?**  
A library to load, process, and store **large-scale datasets** efficiently.

✅ **Why do we need it?**  
- Stores training data in **efficient formats** (Arrow, JSONL, etc.).  
- **Speeds up data loading** for fine-tuning.  
- Handles **text tokenization & padding** automatically.

✅ **Installed with:**  
```bash
pip install datasets
```

✅ **Example usage:**  
```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="dataset_christian_held_002.json")
```

---

## **📌 8️⃣ Accelerate (Optimized Training)**
✅ **What is it?**  
A Hugging Face library that **optimizes training** across multiple GPUs and CPUs.

✅ **Why do we need it?**  
- **Optimizes tensor operations** to use **less memory**.  
- Helps distribute training **across multiple GPUs**.  
- Works well with **LoRA & quantization**.

✅ **Installed with:**  
```bash
pip install accelerate
```

✅ **Example usage:**  
```bash
accelerate launch train.py
```

---

## **🎯 Final Summary: What Each Library Does**
| **Library**     | **Purpose** |
|---------------|------------|
| **CUDA**  | Enables GPU acceleration |
| **cuDNN** | Optimizes deep learning performance |
| **PyTorch** | Core deep learning framework |
| **Transformers** | Loads & fine-tunes AI models |
| **PEFT** | Efficient fine-tuning (LoRA) |
| **BitsAndBytes** | Reduces VRAM usage (quantization) |
| **Datasets** | Handles training data efficiently |
| **Accelerate** | Optimizes multi-GPU training |

