#!/usr/bin/env python3
"""Run the un-fine-tuned local Qwen2-VL base on one contact-sheet PNG.

This is a baseline-only offline test.  It deliberately loads the base checkpoint
directly and never reads ``Saved_Model`` or applies a LoRA adapter, so its answer
shows what the pre-trained VLM understands before contact-sheet fine-tuning.
"""

import argparse
import time
from pathlib import Path


CONTACT_SHEET_PROMPT = (
    'You are given one contact sheet containing 4 sequential robot-view frames '
    'from the same short video clip. Read tiles in temporal order: top left = '
    't=0.0s, top right = t=0.5s, bottom left = t=1.0s, bottom right = t=1.5s '
    '(now). The tracked person ID is printed in every tile header. Compare that '
    'same person across all four real frames. Is its motion crossing the camera '\
    'view, approaching the camera, moving straight ahead, or unclear? Return only '\
    'a JSON array such as [{"id": 12, "state": "crossing"}].')

SINGLE_FRAME_PROMPT = (
    'You are given one newest robot-view camera frame. Every visible person has a '
    'ByteTrack ID label. Describe the visible pose and scene context for the person '
    'with the displayed ID. Do not invent a past trajectory from this one frame. '
    'Return only a JSON array such as [{"id": 12, "state": "unclear"}].')

STATE_WORD_PROMPT_SUFFIX = (
    '\nReturn exactly one lowercase word from: crossing, approaching, straight, '
    'talking, unclear. Do not return JSON, punctuation, or an explanation.')


def parse_arguments() -> argparse.Namespace:
    """Read one local image and inference limits without any ROS dependency."""
    parser = argparse.ArgumentParser(description=__doc__)
    input_source = parser.add_mutually_exclusive_group(required=True)
    input_source.add_argument('--image', type=Path,
                              help='One local PNG/JPEG to send as one Qwen image input.')
    input_source.add_argument('--images-dir', type=Path,
                              help='Directory of PNG/JPEG images to test sequentially after one model load.')
    parser.add_argument('--input-mode', choices=('contact-sheet', 'single-frame'),
                        default='contact-sheet',
                        help='Prompt contract represented by --image.')
    parser.add_argument('--response-mode', choices=('json', 'state-word'), default='json',
                        help='Generate full JSON or only one state word for latency benchmarking.')
    parser.add_argument('--model', default='unsloth/Qwen2-VL-2B-Instruct-bnb-4bit',
                        help='Local base Qwen2-VL model identifier or cache-resolvable path.')
    parser.add_argument('--device', default='cuda:0',
                        help='CUDA device used for the complete 4-bit base model.')
    parser.add_argument('--max-new-tokens', type=int, default=64,
                        help='Maximum generated tokens; JSON output needs only a small budget.')
    parser.add_argument('--runs', type=int, default=1,
                        help='Sequential inference calls after one model load; run 2+ to measure warm latency.')
    parser.add_argument('--min-pixels', type=int, default=3136,
                        help='Qwen processor minimum image pixels after resize.')
    parser.add_argument('--max-pixels', type=int, default=200704,
                        help='Qwen processor maximum image pixels after resize.')
    parser.add_argument('--result-file', type=Path,
                        help='Optional UTF-8 file receiving image, latency, and raw model response.')
    return parser.parse_args()


def patch_pillow_resampling() -> None:
    """Provide Pillow 9 aliases required by the installed Transformers build."""
    from PIL import Image as PilImageModule
    if not hasattr(PilImageModule, 'Resampling'):
        PilImageModule.Resampling = type('PillowResampling', (), {
            'NEAREST': PilImageModule.NEAREST,
            'LANCZOS': PilImageModule.LANCZOS,
            'BILINEAR': PilImageModule.BILINEAR,
            'BICUBIC': PilImageModule.BICUBIC,
            'BOX': PilImageModule.BOX,
            'HAMMING': PilImageModule.HAMMING,
        })


def format_result(image_paths: list[Path], model_name: str,
                  results: list[tuple[list[float], str]], input_mode: str,
                  response_mode: str) -> str:
    """Create one durable record with cold and warm timings for every image."""
    sections = [
        f'Images: {len(image_paths)}',
        f'Input mode: {input_mode}',
        f'Response mode: {response_mode}',
        f'Base model: {model_name}',
    ]
    for image_path, (latencies_sec, response) in zip(image_paths, results):
        latency_lines = '\n'.join(
            f'  Run {run_index}: {latency_sec:.2f}s'
            for run_index, latency_sec in enumerate(latencies_sec, start=1))
        sections.extend((
            f'Image: {image_path}',
            'Inference latency:',
            latency_lines,
            'Raw response:',
            response,
        ))
    return '\n'.join(sections) + '\n'


def main() -> None:
    """Load base-only Qwen, run one deterministic generation, and print its answer."""
    args = parse_arguments()
    if args.image is not None:
        image_paths = [args.image]
    else:
        image_paths = sorted(
            path for path in args.images_dir.iterdir()
            if path.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    if not image_paths:
        raise FileNotFoundError('No PNG/JPEG input images were found')
    missing_images = [str(path) for path in image_paths if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(f'Image(s) do not exist: {missing_images}')
    if args.min_pixels < 1 or args.max_pixels < args.min_pixels:
        raise ValueError('Require 1 <= min-pixels <= max-pixels')
    if args.max_new_tokens < 1:
        raise ValueError('max-new-tokens must be positive')
    if args.runs < 1:
        raise ValueError('runs must be positive')

    patch_pillow_resampling()
    import torch
    from PIL import Image as PilImage
    from unsloth import FastVisionModel

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; this base-model test requires the configured GPU')
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError(f'--device must be CUDA, received {args.device!r}')
    device_index = 0 if device.index is None else device.index
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(f'{args.device!r} is unavailable')

    # This direct base-model load is intentionally distinct from the ROS LoRA
    # loader.  ``local_files_only`` keeps the baseline reproducible offline.
    model, processor = FastVisionModel.from_pretrained(
        model_name=args.model,
        max_seq_length=4096,
        dtype=torch.float16,
        load_in_4bit=True,
        device_map={'': device_index},
        use_gradient_checkpointing=False,
        use_exact_model_name=True,
        fullgraph=False,
        local_files_only=True,
    )
    FastVisionModel.for_inference(model)
    model.eval()

    prompt = CONTACT_SHEET_PROMPT if args.input_mode == 'contact-sheet' else SINGLE_FRAME_PROMPT
    if args.response_mode == 'state-word':
        prompt = prompt.rsplit('Return only', 1)[0].rstrip() + STATE_WORD_PROMPT_SUFFIX

    results = []
    for image_path in image_paths:
        image = PilImage.open(image_path).convert('RGB')
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': image},
            {'type': 'text', 'text': prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=False,
                           min_pixels=args.min_pixels, max_pixels=args.max_pixels,
                           return_tensors='pt')
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Calls share the already loaded model.  The first call can include
        # CUDA/Triton setup; later images reveal steady-state latency.
        latencies_sec = []
        response = ''
        for _ in range(args.runs):
            torch.cuda.synchronize(device)
            started_at = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            torch.cuda.synchronize(device)
            latencies_sec.append(time.perf_counter() - started_at)
            generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
            response = processor.batch_decode(
                generated_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0].strip()
        results.append((latencies_sec, response))
    result = format_result(image_paths, args.model, results, args.input_mode, args.response_mode)
    if args.result_file is not None:
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(result, encoding='utf-8')
    print(result, end='')


if __name__ == '__main__':
    main()
