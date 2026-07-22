import os
import sys
import argparse
import numpy as np
import librosa
import soundfile as sf

def main():
    parser = argparse.ArgumentParser(description="Mix converted vocals and instrumental tracks back together.")
    parser.add_argument("-v", "--vocals", type=str, required=True, help="Path to the converted vocal track (.wav)")
    parser.add_argument("-i", "--instrumental", type=str, required=True, help="Path to the instrumental/background track (.wav/.mp3)")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to save the mixed song cover (.wav)")
    parser.add_argument("-vv", "--vocal_volume", type=float, default=1.0, help="Volume multiplier for vocals (e.g. 1.0 = normal, 1.2 = louder, 0.8 = quieter)")
    parser.add_argument("-iv", "--inst_volume", type=float, default=1.0, help="Volume multiplier for instrumental (e.g. 1.0 = normal, 0.8 = quieter)")
    parser.add_argument("-n", "--normalize", action="store_true", default=True, help="Automatically normalize volume to prevent clipping/distortion")

    args = parser.parse_args()

    if not os.path.exists(args.vocals):
        print(f"Error: Vocals file not found: {args.vocals}")
        sys.exit(1)
    if not os.path.exists(args.instrumental):
        print(f"Error: Instrumental file not found: {args.instrumental}")
        sys.exit(1)

    print("Loading instrumental background track...")
    y_inst, sr_inst = librosa.load(args.instrumental, sr=None, mono=False)
    print(f"Instrumental loaded: {y_inst.shape} | Sample Rate: {sr_inst}Hz")

    print("Loading converted vocal track (and resampling to match instrumental)...")
    y_voc, sr_voc = librosa.load(args.vocals, sr=sr_inst, mono=False)
    print(f"Vocals loaded: {y_voc.shape} | Resampled to: {sr_inst}Hz")

    # 1. Align Channels (Mono to Stereo or Stereo to Mono)
    # If one is stereo and the other is mono, match them.
    if y_inst.ndim == 2 and y_voc.ndim == 1:
        print("Matching channels: Converting mono vocals to stereo...")
        y_voc = np.vstack([y_voc, y_voc])
    elif y_inst.ndim == 1 and y_voc.ndim == 2:
        print("Matching channels: Converting stereo vocals to mono...")
        y_voc = np.mean(y_voc, axis=0)

    # 2. Align Lengths (Pad shorter track with silence)
    len_inst = y_inst.shape[-1]
    len_voc = y_voc.shape[-1]
    max_len = max(len_inst, len_voc)

    if len_inst < max_len:
        pad_width = max_len - len_inst
        print(f"Padding instrumental track with {pad_width} silent frames...")
        if y_inst.ndim == 2:
            y_inst = np.pad(y_inst, ((0, 0), (0, pad_width)))
        else:
            y_inst = np.pad(y_inst, (0, pad_width))
    elif len_voc < max_len:
        pad_width = max_len - len_voc
        print(f"Padding vocal track with {pad_width} silent frames...")
        if y_voc.ndim == 2:
            y_voc = np.pad(y_voc, ((0, 0), (0, pad_width)))
        else:
            y_voc = np.pad(y_voc, (0, pad_width))

    # 3. Apply volume multipliers and mix
    print(f"Mixing: Vocal Vol Multiplier = {args.vocal_volume} | Instrumental Vol Multiplier = {args.inst_volume}")
    y_mixed = (y_inst * args.inst_volume) + (y_voc * args.vocal_volume)

    # 4. Normalize to prevent digital clipping/distortion
    if args.normalize:
        max_val = np.max(np.abs(y_mixed))
        if max_val > 1.0:
            print(f"Volume peak is at {max_val:.2f} (clipping). Normalizing down to 0.98 peak to prevent distortion...")
            y_mixed = (y_mixed / max_val) * 0.98
        else:
            print(f"Volume peak is at safe level: {max_val:.2f}")

    # 5. Save the mixed audio (sf.write expects channels last: (length, channels))
    print(f"Saving final mix to: {args.output}...")
    sf.write(args.output, y_mixed.T if y_mixed.ndim == 2 else y_mixed, sr_inst)
    print("Done! Mix completed successfully!")

if __name__ == "__main__":
    main()
