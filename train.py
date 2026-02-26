
import torch
import platform

def main():
    print("="*50)
    print("SAR Edge Flood AI Project")
    print("="*50)
    print("Platform:", platform.platform())
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Everything is working correctly.")
    print("="*50)

if __name__ == "__main__":
    main()
