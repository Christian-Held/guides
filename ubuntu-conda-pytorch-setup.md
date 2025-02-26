Here is a **cleaned-up, well-structured, and streamlined guide** that ensures a **stable DeepSeek fine-tuning setup** on **Ubuntu 22.04 LTS**. I have removed **redundancies**, **fixed formatting issues**, and **optimized** each step. 🚀

---

# **🔥 Ultimate Guide: Fine-Tuning DeepSeek on Ubuntu 22.04 (Stable Setup)**  
**Goal:** Set up a **stable and optimized environment** for **fine-tuning DeepSeek** with CUDA, PyTorch, and PEFT (LoRA).  

---

## **📌 1️⃣ Install Anaconda (Recommended)**
Using Anaconda ensures **environment isolation**, preventing conflicts.

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-0-Linux-x86_64.sh
bash Anaconda3-2023.07-0-Linux-x86_64.sh

export PATH="$HOME/anaconda3/bin:$PATH"
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

conda update -n base -c defaults conda -y
```

---

## **📌 2️⃣ Create a New Conda Environment**
```bash
conda create --name deepseek-test python=3.10.12 -y
conda activate deepseek-test
```

---

## **📌 3️⃣ Remove Old NVIDIA Drivers & CUDA (Optional, if issues exist)**
If your system has **mismatched drivers or CUDA installations**, remove them first.

```bash
sudo apt purge '*nvidia*' '*cuda*' -y
sudo apt autoremove -y
sudo rm -rf /usr/local/cuda*
sudo rm -rf /etc/modprobe.d/nvidia.conf
sudo rmmod nvidia_uvm
```

---

## **📌 4️⃣ Install NVIDIA Driver 550 (Stable Version)**
Check the recommended driver version for your GPU:
```bash
ubuntu-drivers devices
```
Then, install it:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y nvidia-driver-550
sudo reboot
```

---

## **📌 5️⃣ Install CUDA 12.1 (Compatible & Stable)**
Avoid `apt install nvidia-cuda-toolkit`, as it installs an **older version**.

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
sudo sh cuda_12.1.1_530.30.02_linux.run
```
### **🛠️ Installation Notes**
- **Say "NO"** when asked to install the NVIDIA driver (already installed).  
- **Say "YES"** to installing CUDA.

---

## **📌 6️⃣ Configure CUDA Environment Variables**
```bash
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify CUDA installation:
```bash
nvcc --version
```
✅ Expected Output:
```
Cuda compilation tools, release 12.1, V12.1.105
```

Check GPU status:
```bash
nvidia-smi
```
✅ Expected CUDA Version: **12.1**

---

## **📌 7️⃣ Install PyTorch with CUDA 12.1**
**(Use Conda or Pip, not both!)**

### **👉 If using Conda (Recommended)**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### **👉 If using Pip**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

✅ Verify installation:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.__version__)"
```
Expected Output:
```
True NVIDIA GeForce RTX 2060 SUPER 2.5.1+cu121
```

---

## **📌 8️⃣ Install Required Deep Learning Libraries**
```bash
pip install transformers datasets accelerate peft bitsandbytes
```
For **Flash Attention (Optional, if supported)**:
```bash
pip install flash-attn
```

---

## **📌 9️⃣ Restart & Verify Environment**
```bash
conda deactivate
conda activate deepseek-test
```

Check if GPU is working:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
✅ Expected Output:
```
True
```

---

## **🎯 Final Summary**
| **Step**        | **Command**  | **Purpose** |
|----------------|-------------|-------------|
| **Install Anaconda** | `wget ...` | Isolate Python environment |
| **Create Conda Env** | `conda create --name deepseek-test python=3.10.12 -y` | Manage dependencies |
| **Remove Old Drivers (Optional)** | `sudo apt purge '*nvidia*'` | Fix conflicts |
| **Install NVIDIA Driver** | `sudo apt install -y nvidia-driver-550` | Stable GPU support |
| **Install CUDA 12.1** | `sudo sh cuda_12.1.1_530.30.02_linux.run` | CUDA version control |
| **Set CUDA Path** | `export PATH=/usr/local/cuda-12.1/bin:$PATH` | Enable CUDA tools |
| **Install PyTorch** | `pip install torch torchvision ...` | Install AI framework |
| **Install Libraries** | `pip install transformers datasets accelerate peft bitsandbytes` | Training dependencies |
| **Restart & Verify** | `python -c "import torch; print(torch.cuda.is_available())"` | Ensure proper installation |

---

## **✅ Next Steps**
You are now **fully set up** to **fine-tune DeepSeek** with PyTorch & CUDA! 🎉🔥
