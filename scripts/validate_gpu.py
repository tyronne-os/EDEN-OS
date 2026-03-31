"""EDEN OS — GPU Validation Script"""
from scripts.setup_models import validate_gpu

if __name__ == "__main__":
    result = validate_gpu()
    print(f"GPU Available: {result['available']}")
    print(f"GPU: {result['gpu']}")
    print(f"VRAM: {result['vram_gb']:.1f} GB")
