# WALL-E: MemGPT-Inspired Robot Brain 🤖

**WALL-E v3.0** is an autonomous agent framework designed for local execution on NVIDIA Jetson devices and development machines. It transforms a standard LLM into a persistent personality with a "physical" body, featuring a three-tiered memory system (Core, Recall, Archival), a multi-modal context manager (simulating vision/sensors), and direct hardware control.

**New in v3.0**: TensorRT-LLM backend for optimized inference on Jetson Orin Nano (8GB VRAM), with full backward compatibility for Ollama.

## 🌟 Key Features

* **🚀 Dual Backend Support**:
    * **TensorRT-LLM**: Optimized for NVIDIA Jetson Orin (2-3x faster inference)
    * **Ollama**: Cross-platform development and testing
* **🧠 Three-Tier Memory System**:
    * **Core Memory**: Persistent instructions and persona residing *inside* the context window. Editable by the LLM.
    * **Recall Memory**: Vector-backed conversation history with semantic search.
    * **Archival Memory**: Long-term storage for facts and deep history.
* **💓 Heartbeat / Chain-of-Thought**: A mechanism allowing the robot to "think" or perform multiple actions (e.g., "Look left" -> "Scan" -> "Speak") without user interruption.
* **👀 Context Awareness**: Simulates sensory inputs (Vision, Battery, Environment) injected dynamically into the prompt.
* **🦾 Robotics Control**: Direct serial port integration for motor control, head rotation, and emotional expression.
* **🌍 Internet Connected**: Tools to fetch real-time data (weather, news) using DuckDuckGo.
* **⚡ Memory Optimized**: Efficient embeddings using sentence-transformers, optimized for 8GB VRAM.

## 🛠️ Prerequisites

### For Jetson Orin Nano (Recommended for Production)
* **NVIDIA Jetson Orin Nano** (8GB) or Orin NX/AGX
* **JetPack 6.1+** (includes CUDA, cuDNN, TensorRT)
* **Python 3.10+**
* **TensorRT-LLM v0.12.0-jetson+**
* **Hardware (Optional)**: An Arduino/Serial robot. The system defaults to "Simulation Mode" if no device is found.

### For Development (Any Platform)
* **Python 3.10+**
* **Ollama** running locally ([Download](https://ollama.ai))
* **Hardware (Optional)**: An Arduino/Serial robot for testing

## 🚀 Installation

### Option A: TensorRT-LLM (Jetson Orin Only)

**Complete setup guide**: See [JETSON_SETUP.md](JETSON_SETUP.md) for detailed instructions.

**Quick start**:

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url> memory
   cd memory
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install TensorRT-LLM**:
   ```bash
   pip install tensorrt_llm -f https://nvidia.github.io/TensorRT-LLM/wheels/jetson/
   ```

4. **Prepare Model** (15-30 minutes):
   ```bash
   python prepare_model.py \
     --model_id "Qwen/Qwen3-4B-Instruct-2507-FP8" \
     --output_dir "/workspace/models/qwen3-4b-fp8" \
     --quantization int4_awq
   ```

5. **Run WALL-E**:
   ```bash
   python walle_enhanced.py
   ```

### Option B: Ollama (Development/Cross-Platform)

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url> memory
   cd memory
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Ollama**:
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh

   # Pull models
   ollama pull qwen2.5:3b
   ```

4. **Configure Backend**:
   Edit `config.py`:
   ```python
   BACKEND = "ollama"
   OLLAMA_MODEL = "qwen2.5:3b"
   ```

5. **Run WALL-E**:
   ```bash
   python walle_enhanced.py
   ```

## 🏃 Usage

### Start the Brain
```bash
python walle_enhanced.py
````

*This initializes the cognitive loop, connects to databases, and starts the simulation.*

### Memory Inspector

To see what is actually stored in the database (Core blocks, Recall vectors, etc.):

```bash
python memory_inspector.py
```

## ⚙️ Configuration

Edit `config.py` to adjust:

### Backend Selection
* **`BACKEND`**: Choose `"tensorrt-llm"` (Jetson Orin) or `"ollama"` (cross-platform)

### TensorRT-LLM Settings (Jetson)
* **`MODEL_PATH`**: Path to TensorRT engine files
* **`QUANTIZATION`**: `"int4_awq"` (recommended), `"int8"`, or `"fp8"`
* **`MAX_INPUT_LEN`**: Maximum input tokens (default: 2048)
* **`MAX_OUTPUT_LEN`**: Maximum output tokens (default: 512)
* **`GPU_MEMORY_FRACTION`**: GPU memory usage (default: 0.9)

### Ollama Settings
* **`OLLAMA_MODEL`**: Ollama model name (e.g., `"qwen2.5:3b"`)
* **`OLLAMA_BASE_URL`**: Ollama server URL (default: `http://localhost:11434`)

### Memory Settings
* **`USE_SEMANTIC_SEARCH`**: Enable vector search (default: True)
* **`EMBEDDING_MODEL`**: sentence-transformers model (default: `all-MiniLM-L6-v2`)
* **`RECALL_MEMORY_LIMIT`**: Messages before archival (default: 40)

### Robot Control
* **`SERIAL_PORT`**: Arduino port (e.g., `"/dev/ttyUSB0"`) or `None` for simulation
* **`BAUD_RATE`**: Serial baud rate (default: 9600)

### Search Settings
* **`SEARCH_REGIONS`**: DuckDuckGo regions (default: `["wt-wt", "us-en"]`)
* **`MAX_SEARCH_RESULTS`**: Results per query (default: 5)

## 📊 Performance

### Jetson Orin Nano 8GB with Qwen3-4B-INT4

| Metric | Value |
|--------|-------|
| Time to First Token | 200-500ms |
| Throughput | 15-25 tokens/sec |
| GPU Memory | ~5-6GB |
| CPU Memory | ~2GB |

### Comparison: TensorRT-LLM vs Ollama

| Backend | TTFT | Throughput | Best For |
|---------|------|------------|----------|
| TensorRT-LLM | 200-500ms | 15-25 tok/s | Production (Jetson) |
| Ollama | 800-1200ms | 8-12 tok/s | Development (Any) |

## 🧪 Quick System Tests

Run these steps to verify your installation:

1.  **Configuration Check**:
    Run `python config.py`. Expect no errors.

2.  **Memory Write Test**:

      * Start the bot: `python walle_enhanced.py`
      * Say: *"My name is [Your Name]. I am a Python developer."*
      * Exit and run `python memory_inspector.py`. Check the **Core Memory** section; the `<human>` block should now contain your name.

3.  **Tool Use Test**:

      * Ask: *"What is the price of Bitcoin?"* -\> Should trigger `consult_internet_for_facts`.
      * Command: *"Look left and wave."* -\> Should trigger `set_head_rotation` and `wave_hello`.

## 📚 Documentation

- **[JETSON_SETUP.md](JETSON_SETUP.md)** - Complete guide for deploying on Jetson Orin Nano
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Migration from v2.6 (Ollama) to v3.0 (TensorRT-LLM)
- **[TECHNICAL_MANUAL.md](TECHNICAL_MANUAL.md)** - Architecture and technical details
- **[prepare_model.py](prepare_model.py)** - Model preparation script for TensorRT-LLM

## 🤝 Contributing

Contributions are welcome! Please follow best practices for AI agents with memory:

- Maintain separation between Core/Recall/Archival memory tiers
- Optimize for low-memory devices (Jetson Orin 8GB)
- Preserve backward compatibility with existing databases
- Test on both TensorRT-LLM and Ollama backends

## 📝 License

This project is open source. See LICENSE for details.

## 🙏 Acknowledgments

- **MemGPT** - Inspiration for the memory architecture
- **NVIDIA TensorRT-LLM** - Optimized inference engine
- **Ollama** - Easy local LLM deployment
- **Qwen Team** - Qwen3-4B-Instruct model

## ⚠️ Important Notes

### Hardware Compatibility

- **TensorRT-LLM** requires NVIDIA Jetson **Orin** devices (Orin Nano, Orin NX, Orin AGX)
- The original **Jetson Nano** is NOT supported by TensorRT-LLM
- For non-Jetson hardware, use the Ollama backend

Check your device:
```bash
cat /etc/nv_tegra_release
# Should show "Orin" for TensorRT-LLM support
```

### Quantization Recommendations

- **INT4 AWQ**: Recommended for Jetson Orin 8GB (most stable, good performance)
- **INT8**: Alternative if INT4 quality is insufficient
- **FP8**: Known [bugs](https://github.com/NVIDIA/TensorRT-LLM/issues/8570) with Qwen3-4B (October 2025)

## 📞 Support

- **Issues**: Open a GitHub issue with detailed logs
- **Questions**: Check [JETSON_SETUP.md](JETSON_SETUP.md) and [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- **Performance**: See troubleshooting in [JETSON_SETUP.md](JETSON_SETUP.md#-troubleshooting)

---

**WALL-E v3.0** - An AI agent with memory, optimized for edge deployment on NVIDIA Jetson devices. 🤖