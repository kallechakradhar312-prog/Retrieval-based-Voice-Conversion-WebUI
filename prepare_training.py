import os
import json
import random
from pathlib import Path

def main():
    exp_name = "dummy_voice"
    exp_dir = Path("logs") / exp_name
    
    # 1. Gather all base names from 0_gt_wavs
    gt_wavs_dir = exp_dir / "0_gt_wavs"
    feature_dir = exp_dir / "3_feature768"  # for v2
    f0_dir = exp_dir / "2a_f0"
    f0nsf_dir = exp_dir / "2b-f0nsf"

    names = (
        set([f.name.split(".")[0] for f in gt_wavs_dir.glob("*.wav")])
        & set([f.name.split(".")[0] for f in feature_dir.glob("*.npy")])
        & set([f.name.split(".")[0] for f in f0_dir.glob("*.npy")])
        & set([f.name.split(".")[0] for f in f0nsf_dir.glob("*.npy")])
    )
    
    if not names:
        print("Error: No training clips found! Make sure preprocess, F0 extraction, and feature extraction are complete.")
        return

    print(f"Found {len(names)} training clips.")

    # 2. Build filelist
    opt = []
    spk_id = 0
    for name in sorted(names):
        wav_path = os.path.abspath(gt_wavs_dir / f"{name}.wav")
        npy_path = os.path.abspath(feature_dir / f"{name}.npy")
        f0_path = os.path.abspath(f0_dir / f"{name}.wav.npy")
        f0nsf_path = os.path.abspath(f0nsf_dir / f"{name}.wav.npy")
        
        # Double escape backslashes for Windows path format in training code
        wav_str = wav_path.replace("\\", "\\\\")
        npy_str = npy_path.replace("\\", "\\\\")
        f0_str = f0_path.replace("\\", "\\\\")
        f0nsf_str = f0nsf_path.replace("\\", "\\\\")
        
        opt.append(f"{wav_str}|{npy_str}|{f0_str}|{f0nsf_str}|{spk_id}")

    # 3. Add mute files
    now_dir = os.path.abspath(".")
    fea_dim = 768  # v2
    sr = "40k"
    
    mute_wav = os.path.join(now_dir, "logs", "mute", "0_gt_wavs", f"mute{sr}.wav").replace("\\", "\\\\")
    mute_npy = os.path.join(now_dir, "logs", "mute", f"3_feature{fea_dim}", "mute.npy").replace("\\", "\\\\")
    mute_f0 = os.path.join(now_dir, "logs", "mute", "2a_f0", "mute.wav.npy").replace("\\", "\\\\")
    mute_f0nsf = os.path.join(now_dir, "logs", "mute", "2b-f0nsf", "mute.wav.npy").replace("\\", "\\\\")

    for _ in range(2):
        opt.append(f"{mute_wav}|{mute_npy}|{mute_f0}|{mute_f0nsf}|{spk_id}")

    random.shuffle(opt)
    
    filelist_path = exp_dir / "filelist.txt"
    with open(filelist_path, "w", encoding="utf8") as f:
        f.write("\n".join(opt))
    print(f"filelist.txt generated at {filelist_path}")

    # 4. Copy configuration JSON template
    config_template = Path("configs") / "v1" / "40k.json"  # v2 40k uses v1 40k template as fallback
    config_save_path = exp_dir / "config.json"
    
    with open(config_template, "r", encoding="utf8") as f:
        config_data = json.load(f)
        
    # We can customize model dimensions or training batch size here if needed
    config_data["train"]["batch_size"] = 1 # batch size 1 is required for CPU training to avoid Out of Memory
    
    with open(config_save_path, "w", encoding="utf8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=4, sort_keys=True)
    print(f"config.json written to {config_save_path}")

if __name__ == "__main__":
    main()
