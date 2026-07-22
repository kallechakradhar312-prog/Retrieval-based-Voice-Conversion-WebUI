import os
import zipfile
from huggingface_hub import login, snapshot_download, hf_hub_download

def load_dotenv():
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ[k.strip()] = v.strip()


def main():
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found in environment or .env file.")
    print("Logging into Hugging Face...")
    login(token=token)

    # Make directories
    os.makedirs("assets/hubert_base", exist_ok=True)
    os.makedirs("assets/rmvpe", exist_ok=True)
    os.makedirs("assets/pretrained", exist_ok=True)
    os.makedirs("assets/pretrained_v2", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 1. Download hubert_base
    print("Downloading hubert_base...")
    snapshot_download(
        repo_id="lj1995/VoiceConversionWebUI",
        allow_patterns=["hubert_base/*"],
        local_dir="assets"
    )

    # 2. Download rmvpe.pt and place it under assets/rmvpe/
    print("Downloading rmvpe.pt...")
    hf_hub_download(
        repo_id="lj1995/VoiceConversionWebUI",
        filename="rmvpe.pt",
        local_dir="assets/rmvpe"
    )

    # Download rmvpe.onnx for Windows DirectML execution
    print("Downloading rmvpe.onnx...")
    hf_hub_download(
        repo_id="lj1995/VoiceConversionWebUI",
        filename="rmvpe.onnx",
        local_dir="assets/rmvpe"
    )

    # 3. Download pretrained weights for training
    print("Downloading pretrained base models for v1 and v2 (40k)...")
    snapshot_download(
        repo_id="lj1995/VoiceConversionWebUI",
        allow_patterns=[
            "pretrained/f0G40k.pth",
            "pretrained/f0D40k.pth",
            "pretrained_v2/f0G40k.pth",
            "pretrained_v2/f0D40k.pth"
        ],
        local_dir="assets"
    )

    # 4. Download mute.zip and extract it to logs/
    print("Downloading mute.zip...")
    mute_zip_path = hf_hub_download(
        repo_id="lj1995/VoiceConversionWebUI",
        filename="mute.zip"
    )
    print(f"Extracting mute.zip from {mute_zip_path} to logs/...")
    with zipfile.ZipFile(mute_zip_path, 'r') as zip_ref:
        zip_ref.extractall("logs")

    print("\nAll assets downloaded and structured successfully!")

if __name__ == "__main__":
    main()
