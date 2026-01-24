# WALL-E Setup Guide for Jetson Orin Nano

Complete guide for deploying WALL-E conversational robot on NVIDIA Jetson Orin Nano with 8GB VRAM.

## Hardware Requirements

- **NVIDIA Jetson Orin Nano** (8GB version) - or Jetson Orin NX/AGX
- **Storage**: Minimum 32GB SD card or NVMe SSD (64GB+ recommended)
- **Power**: 5V/4A power supply (or appropriate for your Jetson model)
- **Robot Body**: Compatible with serial communication (optional)
- **USB Serial Adapter**: If connecting to robot hardware

## Software Requirements

- **JetPack 6.1 or later** (includes CUDA, cuDNN, TensorRT)
- **Python 3.10+**
- **TensorRT-LLM v0.12.0-jetson or later**

---

## 🚀 Quick Start

### Step 1: Install JetPack

1. Flash JetPack 6.1+ to your Jetson Orin Nano:
   ```bash
   # Follow NVIDIA's official guide:
   # https://developer.nvidia.com/embedded/jetpack
   ```

2. Verify CUDA installation:
   ```bash
   nvcc --version
   # Should show CUDA 12.x
   ```

3. Check available memory:
   ```bash
   free -h
   nvidia-smi
   # Should show ~8GB GPU memory
   ```

### Step 2: Install TensorRT-LLM

```bash
# Install TensorRT-LLM (Jetson-specific wheel)
pip3 install --no-cache-dir \
  tensorrt_llm==0.12.0 \
  -f https://nvidia.github.io/TensorRT-LLM/wheels/jetson/

# Verify installation
python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
```

**Note**: If the wheel is not available, follow NVIDIA's guide to build from source:
```bash
git clone -b v0.12.0-jetson https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
# Follow README4Jetson.md for build instructions
```

### Step 3: Clone WALL-E Repository

```bash
cd /workspace  # or your preferred directory
git clone <your-repo-url> memory
cd memory
```

### Step 4: Install Python Dependencies

```bash
# Install core dependencies
pip3 install -r requirements.txt

# This will install:
# - transformers (for tokenizers)
# - sentence-transformers (for embeddings)
# - duckduckgo-search (for web search)
# - pyserial (for robot control)
# - openai (for compatibility layer)
```

### Step 5: Download and Convert Qwen3-4B Model

This is the most critical step. The model needs to be converted to TensorRT format.

```bash
# Run the model preparation script
python3 prepare_model.py \
  --model_id "Qwen/Qwen3-4B-Instruct-2507-FP8" \
  --output_dir "/workspace/models/qwen3-4b-fp8" \
  --quantization int4_awq \
  --max_input_len 2048 \
  --max_output_len 512

# This will:
# 1. Download the model from HuggingFace (~8GB)
# 2. Convert checkpoint to TensorRT format
# 3. Build optimized engine (~15-30 minutes)
# 4. Save to /workspace/models/qwen3-4b-fp8/
```

**Important Notes**:
- **Quantization**: Use `int4_awq` for stability. FP8 has [known issues](https://github.com/NVIDIA/TensorRT-LLM/issues/8570) as of October 2025.
- **Build Time**: Engine building takes 15-30 minutes depending on model size.
- **Disk Space**: Requires ~16GB free space during conversion.

### Step 6: Configure WALL-E

The configuration is already optimized for Jetson Orin in `config.py`. Verify these settings:

```python
# config.py - Key Settings for Jetson Orin Nano

BACKEND = "tensorrt-llm"  # Use TensorRT-LLM backend
MODEL_PATH = "/workspace/models/qwen3-4b-fp8/engine"
TOKENIZER_PATH = "/workspace/models/qwen3-4b-fp8/engine/tokenizer"
QUANTIZATION = "int4_awq"  # INT4 quantization for 8GB VRAM

# Memory optimization
MAX_INPUT_LEN = 2048
MAX_OUTPUT_LEN = 512
MAX_BATCH_SIZE = 1
GPU_MEMORY_FRACTION = 0.9  # Use 90% of GPU memory

# Enable semantic search with lightweight embeddings
USE_SEMANTIC_SEARCH = True
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 80MB model

# Robot serial port (optional)
SERIAL_PORT = None  # Set to "/dev/ttyUSB0" if robot is connected
BAUD_RATE = 9600
```

### Step 7: Run WALL-E

```bash
python3 walle_enhanced.py
```

You should see:
```
🚀 Initializing TensorRT-LLM backend...
🔧 Loading tokenizer from /workspace/models/qwen3-4b-fp8/engine/tokenizer...
🔧 Loading TensorRT engine from /workspace/models/qwen3-4b-fp8/engine...
✅ TensorRT-LLM initialized successfully
🔧 Loading embedding model: sentence-transformers/all-MiniLM-L6-v2 on cuda...
✅ Embedding model loaded
🤖 WALL-E Online v3.0 - TensorRT-LLM (Qwen3-4B-Instruct-2507-FP8)
   Memory: Semantic Search | Embedding: sentence-transformers/all-MiniLM-L6-v2
✅ TensorRT-LLM validation passed

You: Hello!
```

---

## 🔧 Advanced Configuration

### Optimizing for Memory

If you encounter OOM (Out of Memory) errors:

1. **Reduce context length**:
   ```python
   MAX_INPUT_LEN = 1024  # Reduce from 2048
   MAX_OUTPUT_LEN = 256  # Reduce from 512
   ```

2. **Disable semantic search** (saves ~100MB):
   ```python
   USE_SEMANTIC_SEARCH = False
   ```

3. **Use CPU for embeddings**:
   ```python
   EMBEDDING_DEVICE = "cpu"
   ```

4. **Reduce KV cache**:
   ```python
   MAX_KV_CACHE_LENGTH = 1024  # Reduce from 2048
   ```

### Connecting Robot Hardware

1. Connect your robot controller via USB
2. Find the serial port:
   ```bash
   ls /dev/ttyUSB* /dev/ttyACM*
   # Usually /dev/ttyUSB0 or /dev/ttyACM0
   ```

3. Update config:
   ```python
   SERIAL_PORT = "/dev/ttyUSB0"
   BAUD_RATE = 9600  # Match your robot's baud rate
   ```

4. Add user to dialout group (for serial access):
   ```bash
   sudo usermod -a -G dialout $USER
   # Log out and back in for changes to take effect
   ```

### Performance Tuning

**Enable CUDA Graphs** (reduces latency):
```python
ENABLE_CUDA_GRAPH = True  # Already enabled by default
```

**Monitor GPU usage**:
```bash
# In another terminal
watch -n 1 nvidia-smi
```

**Check memory usage**:
```bash
# During WALL-E operation
tegrastats
```

---

## 📊 Expected Performance

With Jetson Orin Nano 8GB + Qwen3-4B-INT4:

- **Time to First Token (TTFT)**: 200-500ms
- **Throughput**: 15-25 tokens/second
- **GPU Memory Usage**: ~5-6GB
- **CPU Memory Usage**: ~2GB

---

## 🐛 Troubleshooting

### Issue: "TensorRT-LLM not available"

**Solution**:
```bash
pip3 install tensorrt_llm -f https://nvidia.github.io/TensorRT-LLM/wheels/jetson/
```

If not available, build from source following [NVIDIA's guide](https://github.com/NVIDIA/TensorRT-LLM/blob/v0.12.0-jetson/README4Jetson.md).

### Issue: "CUDA out of memory"

**Solutions**:
1. Reduce `MAX_INPUT_LEN` and `MAX_OUTPUT_LEN` in config.py
2. Disable semantic search: `USE_SEMANTIC_SEARCH = False`
3. Close other GPU applications
4. Reboot Jetson to clear GPU memory

### Issue: "Model validation failed"

**Check**:
```bash
ls -la /workspace/models/qwen3-4b-fp8/engine/
# Should contain .engine files and tokenizer directory
```

If missing, re-run `prepare_model.py`.

### Issue: "Slow inference"

**Solutions**:
1. Verify CUDA graphs are enabled: `ENABLE_CUDA_GRAPH = True`
2. Check GPU is being used: `nvidia-smi` should show WALL-E process
3. Use INT4 quantization (faster than FP8/FP16)
4. Reduce beam width: `MAX_BEAM_WIDTH = 1`

### Issue: "Import error: sentence-transformers"

**Solution**:
```bash
pip3 install sentence-transformers
```

### Issue: "Serial port permission denied"

**Solution**:
```bash
sudo usermod -a -G dialout $USER
# Then log out and back in
```

---

## 🔄 Switching Between Backends

### Use Ollama as Fallback

If TensorRT-LLM is not working, you can use Ollama:

1. Install Ollama:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. Pull model:
   ```bash
   ollama pull qwen2.5:3b
   ```

3. Update config:
   ```python
   BACKEND = "ollama"
   OLLAMA_MODEL = "qwen2.5:3b"
   ```

4. Run WALL-E:
   ```bash
   python3 walle_enhanced.py
   ```

---

## 📝 Best Practices for Conversational Robot

### 1. Memory Management

- **Core Memory**: Updated automatically by LLM to remember key facts about user
- **Recall Memory**: Recent conversation history (last 40 messages)
- **Archival Memory**: Long-term storage for important facts

The system automatically compresses old memories to archival storage.

### 2. Context Awareness

The system includes:
- **Vision context**: Simulated or from camera
- **Environment context**: Battery level, location
- **Interaction context**: Who is speaking, when last spoke

### 3. Tool Usage

WALL-E can:
- **Search the web** for current information
- **Control robot body** via serial commands
- **Manage memory** to remember conversations
- **Express emotions** through movements

### 4. Personality System

Adjust personality traits in `personality.json`:
```json
{
  "humor": 50,    // 0-100: How funny responses are
  "honesty": 90,  // 0-100: How direct/honest
  "sass": 20      // 0-100: How sassy/playful
}
```

---

## 🚀 Optimization Tips for Production

### 1. Use NVMe SSD
Swap to NVMe for faster model loading:
```bash
# Mount NVMe as /workspace
sudo mkdir -p /workspace
sudo mount /dev/nvme0n1p1 /workspace
```

### 2. Enable Jetson Performance Mode
```bash
sudo nvpmodel -m 0  # Max performance
sudo jetson_clocks  # Lock clocks to max
```

### 3. Reduce Logging
In production, reduce print statements for better performance.

### 4. Use systemd for Auto-Start
Create `/etc/systemd/system/walle.service`:
```ini
[Unit]
Description=WALL-E Conversational Robot
After=network.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/workspace/memory
ExecStart=/usr/bin/python3 /workspace/memory/walle_enhanced.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable walle
sudo systemctl start walle
```

---

## 📚 Additional Resources

- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- [Jetson AI Lab - TensorRT-LLM](https://www.jetson-ai-lab.com/tensorrt_llm)
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8)
- [NVIDIA Jetson Orin Documentation](https://developer.nvidia.com/embedded/jetson-orin)

---

## ⚠️ Important Hardware Note

**Clarification on "Jetson Nano Orion"**:
- TensorRT-LLM supports **Jetson Orin** devices (Orin Nano, Orin NX, Orin AGX)
- The original **Jetson Nano** is NOT supported by TensorRT-LLM
- If you have a Jetson Nano (not Orin), use the Ollama backend instead

To check your device:
```bash
cat /etc/nv_tegra_release
# Should show "Orin" in the name for TensorRT-LLM support
```

---

## 🆘 Support

If you encounter issues:

1. Check the logs in the terminal output
2. Verify GPU memory with `nvidia-smi`
3. Check system resources with `tegrastats`
4. Review this troubleshooting guide
5. Open an issue on GitHub with full error logs

---

## 🎉 You're Ready!

WALL-E is now configured and optimized for your Jetson Orin Nano. The system features:

✅ TensorRT-LLM for fast, efficient inference
✅ INT4 quantization fitting in 8GB VRAM
✅ Semantic memory with lightweight embeddings
✅ Web search for current information
✅ Robot control via serial communication
✅ MemGPT-style persistent memory

Enjoy your conversational robot companion! 🤖
