import os
import sys
import argparse
import logging
from pathlib import Path
from scipy.io import wavfile

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    # Make sure we add current directory to python path
    sys.path.append(os.path.abspath("."))

    parser = argparse.ArgumentParser(description="RVC CLI Inference Script (CPU Friendly)")
    parser.add_argument("-m", "--model", type=str, required=True, help="Name of the model file in assets/weights (e.g., dummy_voice.pth or absolute path)")
    parser.add_argument("-i", "--index", type=str, default="", help="Path to the index file (optional)")
    parser.add_argument("-in", "--input", type=str, required=True, help="Path to input audio file")
    parser.add_argument("-out", "--output", type=str, required=True, help="Path to output converted audio file")
    parser.add_argument("-t", "--transpose", type=int, default=0, help="Transpose semitones (pitch change: +12 to go up one octave, -12 down, etc.)")
    parser.add_argument("-f0", "--f0_method", type=str, default="rmvpe", choices=["pm", "rmvpe"], help="F0 pitch extraction method (default: rmvpe)")
    parser.add_argument("-r", "--index_rate", type=float, default=0.75, help="Index retrieval rate (strength of retrieval similarity: 0.0 to 1.0)")
    parser.add_argument("-p", "--protect", type=float, default=0.33, help="Protect voiceless consonants and breath sounds (0.0 to 0.5)")
    parser.add_argument("-s", "--resample_sr", type=int, default=0, help="Resample output sampling rate (0 to use target model sample rate)")
    parser.add_argument("-mix", "--rms_mix_rate", type=float, default=0.25, help="Volume envelope mix rate (0.0 to 1.0)")

    args = parser.parse_args()

    # 1. Setup RVC environment variables
    weight_root = os.path.abspath("assets/weights")
    os.environ["weight_root"] = weight_root
    os.environ["index_root"] = os.path.abspath("assets/indices")
    os.environ["outside_index_root"] = os.path.abspath("assets/indices")
    os.environ["rmvpe_root"] = os.path.abspath("assets/rmvpe")

    # If the model is an absolute path or relative path to a specific folder, move/copy it or resolve it
    model_name = args.model
    if os.path.isabs(model_name) or "/" in model_name or "\\" in model_name:
        model_path = Path(model_name)
        if model_path.exists():
            # If it's a file, we copy it to assets/weights if not already there, or resolve it
            dest = Path(weight_root) / model_path.name
            if not dest.exists():
                import shutil
                logger.info(f"Copying model from {model_path} to {dest}...")
                shutil.copy(model_path, dest)
            model_name = model_path.name
        else:
            logger.error(f"Specified model path does not exist: {model_name}")
            sys.exit(1)

    # 2. Import RVC Config and VC module (with sys.argv mocked to avoid argparse collision)
    old_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        from configs.config import Config
        from infer.vc.modules import VC
        
        # 3. Load configurations (still within sys.argv mock)
        logger.info("Initializing RVC configuration...")
        config = Config()
    except ImportError as e:
        logger.error(f"Failed to import RVC components. Make sure you run from the RVC repository root directory: {e}")
        sys.exit(1)
    finally:
        sys.argv = old_argv
    
    # Force CPU configuration if GPU wasn't found or wanted
    config.device = "cpu"
    config.is_half = False

    # 4. Initialize VC
    logger.info("Loading RVC Model...")
    vc = VC(config)
    
    # Load model weights
    vc.get_vc(model_name)

    # 5. Run conversion (inference)
    logger.info(f"Converting audio: {args.input}...")
    
    # Speaker ID (sid) is 0 for single speaker models
    sid = 0 
    
    status, (tgt_sr, audio_data) = vc.vc_single(
        sid=sid,
        input_audio_path=args.input,
        f0_up_key=args.transpose,
        f0_method=args.f0_method,
        file_index=args.index,
        index_rate=args.index_rate,
        resample_sr=args.resample_sr,
        rms_mix_rate=args.rms_mix_rate,
        protect=args.protect
    )

    logger.info(f"Inference Status: {status}")

    # 6. Save target audio file
    if audio_data is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        wavfile.write(args.output, tgt_sr, audio_data)
        logger.info(f"Success! Converted audio saved to: {args.output}")
    else:
        logger.error("Failed to generate voice conversion output.")
        sys.exit(1)

if __name__ == "__main__":
    main()
