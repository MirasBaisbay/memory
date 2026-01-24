#!/usr/bin/env python3
"""
Model Preparation Script for WALL-E TensorRT-LLM
Downloads and converts Qwen3-4B-Instruct-2507-FP8 for Jetson Orin Nano

Usage:
    python prepare_model.py --quantization int4_awq
    python prepare_model.py --quantization fp8  # If FP8 bug is fixed

Requirements:
    - JetPack 6.1 or later
    - TensorRT-LLM v0.12.0-jetson or later
    - At least 16GB disk space for model conversion
"""

import os
import sys
import argparse
import shutil
from pathlib import Path


def check_requirements():
    """Check if required packages are installed"""
    print("🔍 Checking requirements...")

    required_packages = {
        'tensorrt_llm': 'TensorRT-LLM',
        'transformers': 'Transformers',
        'torch': 'PyTorch',
    }

    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {name} installed")
        except ImportError:
            print(f"  ❌ {name} not installed")
            missing.append(package)

    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install tensorrt_llm transformers torch")
        return False

    # Check CUDA availability
    import torch
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available. Please check your installation.")
        return False

    print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  ✅ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    return True


def download_model(model_id: str, output_dir: Path):
    """Download model from HuggingFace"""
    print(f"\n📥 Downloading model: {model_id}")
    print(f"   Output: {output_dir}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download model
    print("   Downloading model weights...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype='auto',  # Use model's native dtype
        )
        model.save_pretrained(output_dir / "hf_model")
        print("   ✅ Model downloaded")
    except Exception as e:
        print(f"   ❌ Failed to download model: {e}")
        return False

    # Download tokenizer
    print("   Downloading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        tokenizer.save_pretrained(output_dir / "tokenizer")
        print("   ✅ Tokenizer downloaded")
    except Exception as e:
        print(f"   ❌ Failed to download tokenizer: {e}")
        return False

    return True


def convert_checkpoint(hf_model_dir: Path, output_dir: Path, quantization: str):
    """Convert HuggingFace checkpoint to TensorRT-LLM format"""
    print(f"\n🔧 Converting checkpoint to TensorRT-LLM format...")
    print(f"   Quantization: {quantization}")

    checkpoint_dir = output_dir / "trt_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Build conversion command
    convert_script = "python -m tensorrt_llm.commands.convert_checkpoint"

    cmd = [
        sys.executable, "-m", "tensorrt_llm.commands.convert_checkpoint",
        "--model_dir", str(hf_model_dir),
        "--output_dir", str(checkpoint_dir),
        "--dtype", "float16",  # Use FP16 as base dtype
    ]

    # Add quantization
    if quantization == "int4_awq":
        cmd.extend(["--use_weight_only", "--weight_only_precision", "int4_awq"])
    elif quantization == "int8":
        cmd.extend(["--use_weight_only", "--weight_only_precision", "int8"])
    elif quantization == "fp8":
        cmd.extend(["--quantization", "fp8"])
    elif quantization == "none":
        pass  # No quantization
    else:
        print(f"   ❌ Unknown quantization: {quantization}")
        return False

    # Run conversion
    print(f"   Running: {' '.join(cmd)}")
    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"   ❌ Conversion failed:")
        print(result.stderr)
        return False

    print("   ✅ Checkpoint converted")
    return True


def build_engine(checkpoint_dir: Path, output_dir: Path, max_input_len: int, max_output_len: int):
    """Build TensorRT engine from checkpoint"""
    print(f"\n🏗️ Building TensorRT engine...")
    print(f"   Max input length: {max_input_len}")
    print(f"   Max output length: {max_output_len}")
    print(f"   This may take 10-30 minutes...")

    engine_dir = output_dir / "engine"
    engine_dir.mkdir(parents=True, exist_ok=True)

    # Build engine command
    cmd = [
        sys.executable, "-m", "tensorrt_llm.commands.build",
        "--checkpoint_dir", str(checkpoint_dir),
        "--output_dir", str(engine_dir),
        "--max_batch_size", "1",
        "--max_input_len", str(max_input_len),
        "--max_seq_len", str(max_input_len + max_output_len),
        "--max_beam_width", "1",
        "--gemm_plugin", "auto",
        "--gpt_attention_plugin", "auto",
        "--context_fmha", "enable",  # Flash attention
        "--paged_kv_cache", "enable",  # Paged KV cache for memory efficiency
        "--remove_input_padding", "enable",  # Memory optimization
    ]

    # Run build
    print(f"   Running: {' '.join(cmd)}")
    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"   ❌ Build failed:")
        print(result.stderr)
        return False

    print("   ✅ Engine built successfully")
    return True


def create_config_file(output_dir: Path, model_id: str, quantization: str):
    """Create configuration file for the model"""
    config = {
        "model_id": model_id,
        "quantization": quantization,
        "max_input_len": 2048,
        "max_output_len": 512,
        "created_at": str(Path(output_dir).stat().st_mtime),
    }

    config_path = output_dir / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n📝 Configuration saved to: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Qwen3-4B model for TensorRT-LLM")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507-FP8",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/models/qwen3-4b-fp8",
        help="Output directory for converted model"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["int4_awq", "int8", "fp8", "none"],
        default="int4_awq",
        help="Quantization method (int4_awq recommended for Jetson)"
    )
    parser.add_argument(
        "--max_input_len",
        type=int,
        default=2048,
        help="Maximum input sequence length"
    )
    parser.add_argument(
        "--max_output_len",
        type=int,
        default=512,
        help="Maximum output length"
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip model download (use existing)"
    )
    parser.add_argument(
        "--skip_build",
        action="store_true",
        help="Skip engine build (convert checkpoint only)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("WALL-E TensorRT-LLM Model Preparation")
    print("=" * 60)
    print(f"Model: {args.model_id}")
    print(f"Output: {args.output_dir}")
    print(f"Quantization: {args.quantization}")
    print("=" * 60)

    # Check requirements
    if not check_requirements():
        return 1

    output_dir = Path(args.output_dir)

    # Step 1: Download model
    if not args.skip_download:
        if not download_model(args.model_id, output_dir):
            print("\n❌ Model download failed")
            return 1
    else:
        print("\n⏭️ Skipping model download")

    # Step 2: Convert checkpoint
    hf_model_dir = output_dir / "hf_model"
    if not hf_model_dir.exists():
        print(f"\n❌ Model directory not found: {hf_model_dir}")
        print("   Run without --skip_download first")
        return 1

    if not convert_checkpoint(hf_model_dir, output_dir, args.quantization):
        print("\n❌ Checkpoint conversion failed")
        return 1

    # Step 3: Build engine
    if not args.skip_build:
        checkpoint_dir = output_dir / "trt_checkpoint"
        if not build_engine(checkpoint_dir, output_dir, args.max_input_len, args.max_output_len):
            print("\n❌ Engine build failed")
            return 1
    else:
        print("\n⏭️ Skipping engine build")

    # Step 4: Copy tokenizer to engine directory
    tokenizer_src = output_dir / "tokenizer"
    tokenizer_dst = output_dir / "engine" / "tokenizer"
    if tokenizer_src.exists() and not tokenizer_dst.exists():
        shutil.copytree(tokenizer_src, tokenizer_dst)
        print(f"   ✅ Tokenizer copied to engine directory")

    # Step 5: Create config file
    import json
    create_config_file(output_dir, args.model_id, args.quantization)

    # Final instructions
    print("\n" + "=" * 60)
    print("✅ Model preparation complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Update config.py with the following:")
    print(f"   MODEL_PATH = '{output_dir / 'engine'}'")
    print(f"   TOKENIZER_PATH = '{output_dir / 'engine' / 'tokenizer'}'")
    print(f"   QUANTIZATION = '{args.quantization}'")
    print("\n2. Run WALL-E:")
    print("   python walle_enhanced.py")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
