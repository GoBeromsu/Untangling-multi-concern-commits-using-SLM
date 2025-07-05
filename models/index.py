#!/usr/bin/env python3
"""
LLaMA Model Converter for LM Studio
Converts .pth based LLaMA/CodeLLaMA models to GGUF format for LM Studio usage on macOS M3 Pro Max
"""

# Standard library imports
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

# Configuration constants
DEPENDENCIES = [
    "torch",
    "transformers",
    "sentencepiece",
    "numpy",
    "accelerate",
    "protobuf",
    "safetensors",
]

MODEL_ARCHITECTURES = ["LlamaForCausalLM"]
LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp"
DEFAULT_FFN_DIM_MULTIPLIER = 2.6666666666666665
DEFAULT_VOCAB_SIZE = 32000
DEFAULT_MAX_SEQ_LEN = 4096
DEFAULT_NORM_EPS = 1e-6
DEFAULT_ROPE_THETA = 10000.0

# Model size detection mapping based on hidden_size and num_layers
MODEL_SIZE_MAP = {
    (4096, 32): "7B",
    (5120, 40): "13B",
    (6656, 60): "30B",
    (8192, 80): "65B",
}


def check_dependencies() -> bool:
    """Check and install missing Python dependencies"""
    missing_deps = []

    for dep in DEPENDENCIES:
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            missing_deps.append(dep)

    if not missing_deps:
        print("📦 All dependencies already installed")
        return True

    print(f"📦 Installing {len(missing_deps)} missing dependencies...")

    for dep in missing_deps:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", dep], capture_output=True
        )
        if result.returncode != 0:
            print(f"  ❌ Failed to install {dep}")
            return False
        print(f"  ✅ {dep}")

    return True


def clone_llama_cpp(dest_dir: Path) -> bool:
    if dest_dir.exists():
        return True

    print("📥 Cloning llama.cpp...")
    result = subprocess.run(
        ["git", "clone", LLAMA_CPP_REPO, str(dest_dir)],
        capture_output=True,
    )

    if result.returncode != 0:
        print("  ❌ Failed to clone llama.cpp")
        return False

    print("  ✅ llama.cpp cloned")
    return True


def build_llama_cpp(llama_cpp_dir: Path) -> bool:
    """Build llama.cpp with Metal acceleration for optimal M3 performance"""
    print("🔨 Building llama.cpp for macOS M3...")

    result = subprocess.run(["make"], cwd=llama_cpp_dir, capture_output=True)

    if result.returncode != 0:
        print("  ❌ Failed to build llama.cpp")
        return False

    print("  ✅ llama.cpp built successfully")
    return True


def setup_shared_environment() -> Optional[Path]:
    """Setup shared llama.cpp environment"""
    print("🔧 Setting up conversion environment...")

    if not check_dependencies():
        return None

    shared_dir = Path.home() / ".cache" / "llama-cpp-converter"
    shared_dir.mkdir(parents=True, exist_ok=True)
    llama_cpp_dir = shared_dir / "llama.cpp"

    if not clone_llama_cpp(llama_cpp_dir):
        return None

    if not build_llama_cpp(llama_cpp_dir):
        return None

    print(f"  ✅ Environment ready: {llama_cpp_dir}")
    return llama_cpp_dir


def detect_model_size(params: Dict) -> str:
    dim = params.get("dim", 0)
    n_layers = params.get("n_layers", 0)

    return MODEL_SIZE_MAP.get((dim, n_layers), f"custom_{n_layers}L_{dim}D")


def load_model_params(model_dir: Path) -> Dict:
    params_file = model_dir / "params.json"
    if not params_file.exists():
        raise FileNotFoundError(f"params.json not found in {model_dir}")

    with open(params_file, "r") as f:
        params = json.load(f)

    print(f"  📋 Model params: {params}")
    return params


def create_hf_config(params: Dict):
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=params.get("vocab_size", DEFAULT_VOCAB_SIZE),
        hidden_size=params["dim"],
        intermediate_size=int(
            params["dim"] * params.get("ffn_dim_multiplier", DEFAULT_FFN_DIM_MULTIPLIER)
        ),
        num_hidden_layers=params["n_layers"],
        num_attention_heads=params["n_heads"],
        num_key_value_heads=params.get("n_kv_heads", params["n_heads"]),
        max_position_embeddings=params.get("max_seq_len", DEFAULT_MAX_SEQ_LEN),
        rms_norm_eps=params.get("norm_eps", DEFAULT_NORM_EPS),
        rope_theta=params.get("rope_theta", DEFAULT_ROPE_THETA),
        use_cache=True,
        tie_word_embeddings=False,
        architectures=MODEL_ARCHITECTURES,
    )


def ensure_config_architectures(output_path: Path) -> None:
    """Ensure config.json has architectures field required for GGUF conversion"""
    config_path = output_path / "config.json"

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    if "architectures" not in config_dict:
        config_dict["architectures"] = MODEL_ARCHITECTURES

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        print("    ✅ Added architectures field to config.json")


def detect_llama31_tokenizer(tokenizer_path: Path) -> bool:
    """Check if tokenizer uses Llama3.1 format which requires special handling"""
    if not tokenizer_path.exists():
        return False

    try:
        with open(tokenizer_path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline().strip()
        return "==" in first_line and first_line.split()[-1].isdigit()
    except Exception:
        return False


def copy_tokenizer(input_path: Path, output_path: Path) -> None:
    tokenizer_src = input_path / "tokenizer.model"
    tokenizer_dst = output_path / "tokenizer.model"

    if tokenizer_src.exists():
        if detect_llama31_tokenizer(tokenizer_src):
            print("    ⚠️  Llama3.1 tokenizer detected - may require special handling")

        shutil.copy2(tokenizer_src, tokenizer_dst)


def remap_weight_names(state_dict: Dict) -> Dict:
    """Convert LLaMA weight names to HuggingFace format for compatibility"""
    name_mapping = {
        "tok_embeddings.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }

    layer_mappings = {
        "attention.wq.weight": "self_attn.q_proj.weight",
        "attention.wk.weight": "self_attn.k_proj.weight",
        "attention.wv.weight": "self_attn.v_proj.weight",
        "attention.wo.weight": "self_attn.o_proj.weight",
        "feed_forward.w1.weight": "mlp.gate_proj.weight",
        "feed_forward.w2.weight": "mlp.down_proj.weight",
        "feed_forward.w3.weight": "mlp.up_proj.weight",
        "attention_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
    }

    hf_state_dict = {}

    for key, value in state_dict.items():
        if key in name_mapping:
            hf_key = name_mapping[key]
            if key == "tok_embeddings.weight":
                print(f"    🔍 Mapping {key} -> {hf_key}, shape: {value.shape}")
        elif key.startswith("layers."):
            parts = key.split(".")
            layer_num = parts[1]
            layer_part = ".".join(parts[2:])

            if layer_part in layer_mappings:
                hf_key = f"model.layers.{layer_num}.{layer_mappings[layer_part]}"
            else:
                print(f"    ⚠️  Unknown layer weight: {key}")
                continue
        else:
            print(f"    ⚠️  Unknown weight: {key}")
            continue

        hf_state_dict[hf_key] = value

    return hf_state_dict


def convert_model_weights(input_path: Path, output_path: Path, config) -> None:
    import torch
    from safetensors.torch import save_file

    pth_files = sorted(list(input_path.glob("consolidated.*.pth")))
    if not pth_files:
        raise FileNotFoundError("No consolidated.*.pth files found")

    print(f"  📂 Loading {len(pth_files)} .pth files...")

    state_dict = {}
    for pth_file in pth_files:
        print(f"    Loading {pth_file.name}...")
        weights = torch.load(pth_file, map_location="cpu")
        state_dict.update(weights)

    # Debug: Print key tensor shapes
    key_tensors = ["tok_embeddings.weight", "norm.weight", "output.weight"]
    for key in key_tensors:
        if key in state_dict:
            print(f"    🔍 {key}: {state_dict[key].shape}")

    hf_state_dict = remap_weight_names(state_dict)

    safetensors_path = output_path / "model.safetensors"
    save_file(hf_state_dict, safetensors_path)
    print(f"    ✅ Saved as safetensors: {safetensors_path}")


def convert_pth_to_hf(input_path: Path, output_path: Path, params: Dict) -> bool:
    print(f"🔄 Converting {input_path.name} to HuggingFace format...")

    try:
        from transformers import LlamaConfig
        from safetensors.torch import save_file

        output_path.mkdir(parents=True, exist_ok=True)

        config = create_hf_config(params)
        config.save_pretrained(output_path)

        ensure_config_architectures(output_path)
        copy_tokenizer(input_path, output_path)
        convert_model_weights(input_path, output_path, config)

        print(f"  ✅ Conversion completed: {output_path}")
        return True

    except ImportError as e:
        print(f"  ❌ Missing dependency: {e}")
        return False
    except FileNotFoundError as e:
        print(f"  ❌ File not found: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Conversion failed: {e}")
        return False


def handle_tokenizer_fallback(hf_path: Path) -> Optional[Path]:
    """Handle Llama3.1 tokenizer by temporarily moving problematic file"""
    tokenizer_file = hf_path / "tokenizer.model"

    if not detect_llama31_tokenizer(tokenizer_file):
        return None

    try:
        tokenizer_backup = hf_path / "tokenizer.model.bak"
        shutil.move(str(tokenizer_file), str(tokenizer_backup))
        print("    🔄 Temporarily moved problematic tokenizer file")
        return tokenizer_backup
    except Exception:
        return None


def restore_tokenizer(tokenizer_backup: Optional[Path], original_path: Path) -> None:
    """Restore tokenizer file after conversion attempt"""
    if tokenizer_backup and tokenizer_backup.exists():
        try:
            shutil.move(str(tokenizer_backup), str(original_path))
            print("    🔄 Restored tokenizer file")
        except Exception:
            pass


def execute_gguf_conversion(
    convert_script: Path, hf_path: Path, output_file: Path
) -> subprocess.CompletedProcess:
    """Execute the GGUF conversion subprocess"""
    cmd = [
        sys.executable,
        str(convert_script),
        str(hf_path),
        "--outtype",
        "f16",
        "--outfile",
        str(output_file),
    ]

    print(f"    🚀 Running: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True)


def convert_hf_to_gguf(
    llama_cpp_dir: Path, hf_path: Path, output_path: Path, model_name: str
) -> bool:
    print(f"🔄 Converting {model_name} to GGUF format...")

    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print("  ❌ Conversion script not found in llama.cpp")
        return False

    output_file = output_path / f"{model_name.lower().replace('_', '-')}-f16.gguf"
    tokenizer_file = hf_path / "tokenizer.model"

    tokenizer_backup = handle_tokenizer_fallback(hf_path)
    if tokenizer_backup:
        print("    ⚠️  Using tokenizer fallback for Llama3.1 compatibility")

    try:
        result = execute_gguf_conversion(convert_script, hf_path, output_file)

        if result.returncode == 0:
            print(f"  ✅ GGUF conversion completed: {output_file}")
            return True
        else:
            print(f"  ❌ GGUF conversion failed: {result.stderr}")
            if tokenizer_backup:
                print("    💡 Llama3.1 tokenizer requires special handling")
            return False

    finally:
        restore_tokenizer(tokenizer_backup, tokenizer_file)


def convert_model(model_name: str, models_dir: Path) -> bool:
    """Convert a single model from .pth to GGUF in model_name/gguf/ folder"""
    model_path = models_dir / model_name

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False

    print(f"\n🎯 Converting model: {model_name}")

    try:
        # Setup shared environment
        shared_llama_cpp_dir = setup_shared_environment()
        if not shared_llama_cpp_dir:
            print("❌ Environment setup failed")
            return False

        params = load_model_params(model_path)
        model_size = detect_model_size(params)
        print(f"  📊 Model size: {model_size}")

        # Create work directory in the model folder
        work_dir = model_path / "conversion_work"
        work_dir.mkdir(exist_ok=True)

        hf_path = work_dir / "hf_format"

        if not convert_pth_to_hf(model_path, hf_path, params):
            return False

        if not convert_hf_to_gguf(
            shared_llama_cpp_dir, hf_path, model_path, model_name
        ):
            return False

        print(f"  🎉 Model {model_name} converted successfully!")
        print(f"  📁 GGUF file saved in: {model_path}")

        # Cleanup work directory
        shutil.rmtree(work_dir)
        print(f"  🧹 Cleaned up temporary files")

        return True

    except FileNotFoundError as e:
        print(f"  ❌ Required file missing for {model_name}: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Failed to convert {model_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert LLaMA model to GGUF format for LM Studio"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name to convert (must exist in current directory)",
    )
    parser.add_argument(
        "--models-dir",
        default=".",
        help="Directory containing model folders (default: current directory)",
    )

    args = parser.parse_args()
    models_dir = Path(args.models_dir).resolve()

    print("🦙 LLaMA to GGUF Converter for LM Studio")
    print("=" * 50)
    print(f"📂 Models directory: {models_dir}")
    print(f"🎯 Converting: {args.model}")
    print(f"🖥️  Platform: macOS M3 Pro Max")
    print("=" * 50)

    try:
        success = convert_model(args.model, models_dir)
        if success:
            print(f"\n🎉 Successfully converted {args.model}!")
            print(f"📁 Find your GGUF file in: {models_dir}/{args.model}/")
            print("🎯 Ready for LM Studio import!")
        else:
            print(f"\n❌ Failed to convert {args.model}")
            return 1

    except KeyboardInterrupt:
        print("\n⏹️  Conversion interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
