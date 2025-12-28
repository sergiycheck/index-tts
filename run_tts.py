import os
import uuid
import argparse
import sys
from indextts.infer_v2 import IndexTTS2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def validate_args(args):
    if args.target_text and len(args.target_text) > 300:
        raise ValueError("Text is too long (max 300 characters).")

    if not os.path.exists(args.audio_ref):
        raise FileNotFoundError(f"Reference audio not found: {args.audio_ref}")

    if not os.path.exists(args.output_dir):
        raise FileNotFoundError(f"Output directory not found: {args.output_dir}")
        
    if not args.audio_name:
        raise ValueError("Audio name to be generated must be provided.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="IndexTTS text with reference audio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--target_text",
        type=str,
        required=True,
        help="Text to synthesize",
    )

    parser.add_argument(
        "--audio_ref",
        type=str,
        required=True,
        help="Reference audio file",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(BASE_DIR, "outputs"),
        help="Directory to store generated wav",
    )
    
    parser.add_argument(
        "--audio_name",
        type=str,
        default=False,
        help="Name of the generated audio file",
    )

    return parser.parse_args()


def indextts_generation():
    args = parse_args()

    # Normalize paths so subprocess + parent dir works
    args.audio_ref = os.path.abspath(args.audio_ref)
    args.output_dir = os.path.abspath(args.output_dir)

    validate_args(args)

    tts = IndexTTS2(
        cfg_path=os.path.join(BASE_DIR, "checkpoints", "config.yaml"),
        model_dir=os.path.join(BASE_DIR, "checkpoints"),
        use_fp16=False,
        use_cuda_kernel=False,
        use_deepspeed=False,
    )

    output_path = os.path.join(args.output_dir, args.audio_name)

    tts.infer(
        spk_audio_prompt=args.audio_ref,
        text=args.target_text,
        output_path=output_path,
        emo_alpha=0.6,
        use_emo_text=True,
        use_random=False,
        verbose=True,
    )

    print(output_path)

    return output_path

if __name__ == "__main__":
    try:
        indextts_generation()
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
