import os
from dataclasses import dataclass, field
from typing import List, Literal
from pathlib import Path

@dataclass
class Config:
    # --- Backend Selection ---
    BACKEND: Literal["tensorrt-llm", "ollama"] = "tensorrt-llm"

    # --- TensorRT-LLM Settings (for Jetson Orin Nano) ---
    # Model from HuggingFace: Qwen/Qwen3-4B-Instruct-2507-FP8
    MODEL_NAME: str = "Qwen3-4B-Instruct-2507-FP8"
    MODEL_PATH: str = "/workspace/models/qwen3-4b-fp8"  # Path to converted TensorRT engine
    TOKENIZER_PATH: str = "/workspace/models/qwen3-4b-fp8/tokenizer"

    # TensorRT-LLM Runtime Settings (Optimized for Jetson Orin 8GB VRAM)
    MAX_INPUT_LEN: int = 2048  # Reduced for memory constraints
    MAX_OUTPUT_LEN: int = 512
    MAX_BATCH_SIZE: int = 1    # Single batch for low memory
    MAX_BEAM_WIDTH: int = 1

    # Quantization Settings
    QUANTIZATION: Literal["fp8", "int4_awq", "int8"] = "int4_awq"  # INT4 more stable than FP8
    USE_GPTQ: bool = False

    # KV Cache Settings (Critical for memory management)
    MAX_KV_CACHE_LENGTH: int = 2048
    ENABLE_KV_CACHE_REUSE: bool = True

    # Embedding Model (Lightweight for Jetson)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # 80MB model
    EMBEDDING_DEVICE: str = "cuda"  # Use GPU for embeddings
    EMBEDDING_BATCH_SIZE: int = 8

    # --- Ollama Settings (Legacy/Fallback) ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "walle-brain"

    # --- Memory Settings ---
    MAX_CONTEXT_MESSAGES: int = 10  # Reduced for TensorRT-LLM memory constraints
    USE_SEMANTIC_SEARCH: bool = True  # Enable with lightweight embeddings
    RECALL_MEMORY_LIMIT: int = 40     # Reduced for faster processing

    # --- Search Settings ---
    MAX_SEARCH_RESULTS: int = 5
    SEARCH_REGIONS: List[str] = field(default_factory=lambda: ["wt-wt", "us-en"])

    # --- Robot Settings ---
    SERIAL_PORT: str = None
    BAUD_RATE: int = 9600

    # --- Performance Settings (Jetson Optimization) ---
    GPU_MEMORY_FRACTION: float = 0.9  # Use 90% of GPU memory
    ENABLE_CUDA_GRAPH: bool = True     # Optimize inference latency
    STREAMING_LLM: bool = True          # Enable streaming for better UX

    def validate(self) -> bool:
        """Validates configuration based on selected backend."""
        if self.BACKEND == "tensorrt-llm":
            return self._validate_tensorrt()
        else:
            return self._validate_ollama()

    def _validate_tensorrt(self) -> bool:
        """Checks if TensorRT-LLM model and dependencies are available."""
        try:
            # Check if model path exists
            model_path = Path(self.MODEL_PATH)
            if not model_path.exists():
                print(f"⚠️ Warning: TensorRT model path not found: {self.MODEL_PATH}")
                print(f"   Please run the model preparation script first.")
                return False

            # Check for engine file
            engine_files = list(model_path.glob("*.engine"))
            if not engine_files:
                print(f"⚠️ Warning: No .engine files found in {self.MODEL_PATH}")
                print(f"   Please build the TensorRT engine first.")
                return False

            # Check tokenizer
            tokenizer_path = Path(self.TOKENIZER_PATH)
            if not tokenizer_path.exists():
                print(f"⚠️ Warning: Tokenizer not found at {self.TOKENIZER_PATH}")
                return False

            print(f"✅ TensorRT-LLM validation passed")
            print(f"   Model: {self.MODEL_NAME}")
            print(f"   Engine: {engine_files[0].name}")
            print(f"   Quantization: {self.QUANTIZATION}")
            return True

        except Exception as e:
            print(f"❌ TensorRT-LLM validation failed: {e}")
            return False

    def _validate_ollama(self) -> bool:
        """Checks if Ollama is running and model exists."""
        try:
            import requests
            # Check connection
            res = requests.get(self.OLLAMA_BASE_URL, timeout=5)
            if res.status_code != 200:
                print(f"⚠️ Warning: Ollama server not responding")
                return False

            # Check model
            res = requests.get(f"{self.OLLAMA_BASE_URL}/api/tags")
            models = [m['name'] for m in res.json()['models']]
            if self.OLLAMA_MODEL not in models and f"{self.OLLAMA_MODEL}:latest" not in models:
                print(f"⚠️ Warning: Model '{self.OLLAMA_MODEL}' not found. Available: {models}")
                return False

            print(f"✅ Ollama validation passed")
            return True
        except Exception as e:
            print(f"❌ Ollama validation failed: {e}")
            return False

    @classmethod
    def load(cls):
        return cls()

conf = Config.load()