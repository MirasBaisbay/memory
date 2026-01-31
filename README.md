# WALL-E: Memory-Augmented Robot Brain

**WALL-E v3.1** is an autonomous agent framework designed for local execution on NVIDIA Jetson devices. It transforms a standard LLM into a persistent personality with a "physical" body, featuring a three-tiered memory system with FAISS-accelerated search, importance decay for fact prioritization, and direct hardware control.

**New in v3.1**:
- FAISS integration for 5-10x faster semantic search
- Importance decay for intelligent fact prioritization
- Ollama with Qwen3-4B as recommended backend (easier setup, excellent tool calling)

## Key Features

* **Dual Backend Support**:
    * **Ollama** (Recommended): Simple setup, excellent tool calling with Qwen3-4B
    * **TensorRT-LLM**: Optional for maximum performance on Jetson Orin
* **Three-Tier Memory System**:
    * **Core Memory**: Persistent persona and user profile in context window (LLM-editable)
    * **Recall Memory**: FAISS-accelerated conversation history with semantic search
    * **Archival Memory**: Long-term facts with importance decay
* **FAISS Vector Search**: O(log n) semantic search instead of O(n) naive cosine similarity
* **Importance Decay**: Recent facts prioritized over older ones using exponential decay
* **Heartbeat / Chain-of-Thought**: Multi-step reasoning without user interruption
* **Context Awareness**: Simulated sensory inputs (Vision, Battery, Environment)
* **Robotics Control**: Serial port integration for motor control and expressions
* **Internet Connected**: Real-time data fetching via DuckDuckGo

## Quick Start (Recommended: Ollama)

### 1. Clone and Install

```bash
git clone <your-repo-url> memory
cd memory
pip install -r requirements.txt
```

### 2. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 3. Pull Qwen3-4B Model

```bash
ollama pull qwen3:4b
```

### 4. Run WALL-E

```bash
python walle_enhanced.py
```

That's it! The system uses Ollama with Qwen3-4B by default.

## Alternative: TensorRT-LLM (Jetson Orin Only)

For maximum performance on Jetson Orin devices, see [JETSON_SETUP.md](JETSON_SETUP.md).

```python
# In config.py, change:
BACKEND = "tensorrt-llm"
```

## Configuration

Edit `config.py` to customize:

### Backend Selection
```python
BACKEND = "ollama"           # "ollama" (recommended) or "tensorrt-llm"
OLLAMA_MODEL = "qwen3:4b"    # Best tool calling among small models
```

### Memory Settings
```python
USE_SEMANTIC_SEARCH = True   # Enable FAISS-accelerated vector search
USE_FAISS = True             # Enable FAISS (falls back to naive cosine if False)
RECALL_MEMORY_LIMIT = 40     # Messages before compression to archival
```

### Importance Decay
```python
IMPORTANCE_DECAY_HALF_LIFE = 30.0  # Days until recency score halves
IMPORTANCE_STATIC_WEIGHT = 0.7     # Weight for static importance
IMPORTANCE_RECENCY_WEIGHT = 0.3    # Weight for recency (newer = higher)
```

### Robot Control
```python
SERIAL_PORT = None           # e.g., "/dev/ttyUSB0" or None for simulation
BAUD_RATE = 9600
```

## Architecture

```
+------------------------------------------------------------------+
|                      WALL-E Memory System                         |
+------------------------------------------------------------------+
|                                                                   |
|  +----------------+  +------------------+  +-------------------+  |
|  |  Core Memory   |  |  Recall Memory   |  |  Archival Memory  |  |
|  |   (SQLite)     |  | (SQLite + FAISS) |  | (SQLite + FAISS)  |  |
|  |                |  |                  |  |                   |  |
|  | - persona      |  | - 40 messages    |  | - Facts/summaries |  |
|  | - human        |  | - FAISS index    |  | - Importance decay|  |
|  | - system       |  | - Semantic search|  | - Categories      |  |
|  +-------+--------+  +--------+---------+  +---------+---------+  |
|          |                    |                      |            |
|          | Always in          | O(log n)             | Decay-     |
|          | context            | search               | weighted   |
|          +--------------------+----------------------+            |
|                               |                                   |
|                    +----------v-----------+                       |
|                    |    Ollama / LLM      |                       |
|                    |    (qwen3:4b)        |                       |
|                    +----------------------+                       |
+-------------------------------------------------------------------+
```

## Performance

### Search Performance (FAISS vs Naive)

| Method | Complexity | 100 msgs | 1000 msgs |
|--------|------------|----------|-----------|
| Naive Cosine | O(n) | ~30ms | ~300ms |
| FAISS | O(log n) | ~5ms | ~15ms |

### Importance Decay Example

```
Fact: "User likes coffee" (importance=8, age=0 days)
  -> effective_importance = 8 * 0.7 + 1.0 * 10 * 0.3 = 8.6

Fact: "User liked tea" (importance=9, age=60 days)
  -> recency_score = exp(-60/30) = 0.135
  -> effective_importance = 9 * 0.7 + 0.135 * 10 * 0.3 = 6.7

Result: Recent "coffee" fact ranks higher than older "tea" fact
```

## Project Structure

```
memory/
├── walle_enhanced.py      # Main cognitive loop
├── config.py              # Configuration settings
├── memory_system.py       # 3-tier memory + FAISS
├── memory_tools.py        # LLM memory tools
├── memory_inspector.py    # Debug/inspect memory
├── context_manager.py     # Sensor simulation
├── robot_tools.py         # Hardware control
├── knowledge_tools.py     # Internet search
├── personality_system.py  # Personality traits
├── heartbeat.py           # Multi-step reasoning
├── tensorrt_client.py     # TensorRT-LLM client (optional)
├── prepare_model.py       # Model preparation (TensorRT)
└── requirements.txt       # Dependencies
```

## Memory Inspector

View stored memories:

```bash
python memory_inspector.py
```

## System Tests

1. **Memory Write Test**:
   - Start: `python walle_enhanced.py`
   - Say: "My name is Alex. I'm a Python developer."
   - Exit and run: `python memory_inspector.py`
   - Check: `<human>` block should contain your info

2. **Tool Use Test**:
   - Ask: "What's the weather in Tokyo?"
   - Should trigger: `consult_internet_for_facts`

3. **FAISS Search Test**:
   - Have a few conversations
   - Ask about something mentioned earlier
   - Should retrieve relevant context quickly

## Rollback Options

Disable features without code changes:

```python
# In config.py:

# Disable FAISS (use naive cosine similarity)
USE_FAISS = False

# Disable importance decay
IMPORTANCE_RECENCY_WEIGHT = 0.0

# Switch to TensorRT-LLM
BACKEND = "tensorrt-llm"
```

## Documentation

- **[JETSON_SETUP.md](JETSON_SETUP.md)** - TensorRT-LLM deployment on Jetson Orin
- **[TECHNICAL_MANUAL.md](TECHNICAL_MANUAL.md)** - Architecture details
- **[ARCHITECTURE_ANALYSIS.md](ARCHITECTURE_ANALYSIS.md)** - Design decisions and comparisons

## Requirements

- Python 3.10+
- Ollama (recommended) or TensorRT-LLM (Jetson Orin)
- ~4-5GB RAM for Qwen3-4B
- Optional: Arduino/Serial robot for hardware control

## Acknowledgments

- **MemGPT** - Inspiration for memory architecture
- **FAISS** - Fast similarity search
- **Ollama** - Easy local LLM deployment
- **Qwen Team** - Qwen3 models with excellent tool calling

## License

This project is open source. See LICENSE for details.

---

**WALL-E v3.1** - Memory-augmented AI agent with FAISS search and importance decay.
