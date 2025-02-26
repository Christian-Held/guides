## **🔥 Fine-Tuning Command Breakdown**
Example command:
```bash
nohup python finetune_christian_held.py \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --train_data "dataset_christian_held_002.json" \
    --output_dir "/mnt/externe-platte/christian_held_finetuned_002" \
    --epochs 50 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --context_length 256 \
    --quantize 8bit \
    --learning_rate 1e-5 \
    --warmup_ratio 0.2 \
    --device cuda \
    --eval > training_002.log 2>&1 &
```

### **📌 1. `--model_path`**
📢 **What it does:**  
Defines **which base model** you are fine-tuning.  
✅ **Example:** `"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"`  
❗ **Trade-off:** Choosing a **bigger model (e.g., 7B or 13B)** **increases VRAM usage**.

📊 **Best Practices:**  
- **Small GPUs (≤ 8GB VRAM):** Use **1.5B models** with 8-bit/4-bit quantization.  
- **Larger GPUs (≥ 24GB VRAM):** You can try **bigger models** like 7B or 13B.  

---

### **📌 2. `--train_data`**
📢 **What it does:**  
Specifies the **training dataset file**.  
✅ **Example:** `"dataset_christian_held_002.json"`  

📊 **Best Practices:**  
- Use **.json** or **.jsonl** for structured data.  
- Make sure data **is tokenized and padded** properly.  

---

### **📌 3. `--output_dir`**
📢 **What it does:**  
Defines where the **fine-tuned model** will be saved.  
✅ **Example:** `"/mnt/externe-platte/christian_held_finetuned_002"`

📊 **Trade-off:**  
- Ensure the directory **has enough storage space** (~10GB per checkpoint).  

---

### **📌 4. `--epochs`**
📢 **What it does:**  
Defines **how many times** the entire dataset is processed.  
✅ **Example:** `--epochs 50`  

📊 **Effects:**  
- **Higher epochs (e.g., 100+)** → **Better learning**, but overfitting risk.
- **Lower epochs (e.g., 10-20)** → **Faster training**, but may **not generalize well**.

📊 **Best Practices:**  
- **Start with 20-50 epochs** for small datasets.  
- **Use early stopping** if possible to **prevent overfitting**.  

---

### **📌 5. `--batch_size`**
📢 **What it does:**  
Defines **how many examples** are processed at once.  
✅ **Example:** `--batch_size 1`  

📊 **Trade-offs:**  
- **Larger batch sizes (4, 8, 16+)** → **Faster training**, but needs **more VRAM**.  
- **Smaller batch sizes (1, 2)** → **Less VRAM usage**, but **slower learning**.  

📊 **Best Practices:**  
- **If VRAM is low (≤ 8GB):** Use **batch_size = 1** with **gradient accumulation**.  
- **If VRAM is high (≥ 24GB):** Use batch_size **4+** for **faster convergence**.  

---

### **📌 6. `--gradient_accumulation_steps`**
📢 **What it does:**  
Simulates **a larger batch size** by accumulating gradients.  
✅ **Example:** `--gradient_accumulation_steps 16`  

📊 **Effects:**  
- **Higher values (8-32)** → Simulates **larger batch size**, but slower updates.  
- **Lower values (1-4)** → Faster updates, but may **not generalize well**.  

📊 **Best Practices:**  
- **For small VRAM:** Set **16 or higher**.  
- **For larger VRAM:** Keep at **4-8** for balance.  

---

### **📌 7. `--context_length`**
📢 **What it does:**  
Defines the **maximum token length** per input.  
✅ **Example:** `--context_length 256`  

📊 **Trade-offs:**  
- **Higher values (512-1024)** → Better understanding of **long text**, but **more VRAM usage**.  
- **Lower values (128-256)** → Fits in **low VRAM**, but may **cut important context**.  

📊 **Best Practices:**  
- **RTX 2060 (6-8GB VRAM):** Keep **≤ 256 tokens**.  
- **RTX 3090+ (24GB VRAM):** Can use **512+ tokens**.  

---

### **📌 8. `--quantize`**
📢 **What it does:**  
Reduces model precision to **save VRAM**.  
✅ **Options:**  
- `"none"` → No quantization (best quality, max VRAM usage).  
- `"8bit"` → 8-bit precision (**50% less VRAM**, slight loss in quality).  
- `"4bit"` → 4-bit precision (**75% less VRAM**, more quality loss).  

📊 **Best Practices:**  
- **Low VRAM (≤ 8GB):** Use `"8bit"` or `"4bit"`.  
- **If you want best accuracy:** Use `"none"`, but needs **big GPU**.  

---

### **📌 9. `--learning_rate`**
📢 **What it does:**  
Defines **how fast** weights are updated.  
✅ **Example:** `--learning_rate 1e-5`  

📊 **Effects:**  
- **Higher (3e-5 - 1e-4)** → Faster learning, but **risk of instability**.  
- **Lower (1e-5 - 3e-6)** → More **stable**, but slower convergence.  

📊 **Best Practices:**  
- **If dataset is small** → Use **lower learning rate (1e-5)**.  
- **If dataset is large** → Use **higher learning rate (3e-5)**.  

---

### **📌 10. `--warmup_ratio`**
📢 **What it does:**  
Gradually increases learning rate at the start.  
✅ **Example:** `--warmup_ratio 0.2`  

📊 **Effects:**  
- **Higher (0.2 - 0.3)** → **More stable training** (less chance of exploding gradients).  
- **Lower (0.05 - 0.1)** → Faster adaptation, but **risk of bad initialization**.  

📊 **Best Practices:**  
- **For small datasets:** Use **0.1-0.2**.  
- **For large datasets:** Use **0.05**.  

---

### **📌 11. `--device`**
📢 **What it does:**  
Specifies where to run training.  
✅ **Options:** `"cuda"` or `"cpu"`  

📊 **Trade-offs:**  
- **CUDA (GPU)** → **Faster**, but needs **VRAM**.  
- **CPU** → **Much slower**, but can run on any system.  

📊 **Best Practices:**  
- Always use **CUDA** if you have a **GPU**.  

---

### **📌 12. `--eval`**
📢 **What it does:**  
Runs **evaluation** during training.  
✅ **Example:** `--eval`  

📊 **Trade-offs:**  
- **Enabled (`--eval`)** → Checks validation loss, but **needs more compute**.  
- **Disabled (remove `--eval`)** → Faster training, but **no validation feedback**.  

📊 **Best Practices:**  
- Use `--eval` for better monitoring.  

---

## **🚀 Conclusion**
| **Argument**        | **Effect**                   | **Trade-Offs** |
|---------------------|----------------------------|----------------|
| `--batch_size 1`   | Uses **less VRAM**, slow | Slower training |
| `--gradient_accumulation_steps 16` | Simulates **larger batch**, better learning | Slower updates |
| `--context_length 256` | Smaller VRAM usage | May cut important context |
| `--quantize 8bit` | Saves **50% VRAM** | Slight quality loss |

Hope this helps! 🚀🔥
