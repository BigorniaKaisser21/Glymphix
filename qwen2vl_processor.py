"""
qwen2vl_processor.py
────────────────────
Handwritten-code OCR using Qwen2-VL-2B-Instruct, a vision-language model
that understands images end-to-end without the character-segmentation
limitations of Tesseract / RapidOCR / EasyOCR.

Why Qwen2-VL over the other engines?
  • Understands context — won't split "!=" across two boxes
  • Handles mixed handwriting + printed symbols in one pass
  • No ruled-line preprocessing needed (the model sees colour naturally)
  • 2B parameter variant runs on CPU in ~8–16 GB RAM (slow but usable)

Install:
    pip install torch torchvision transformers>=4.45.0 accelerate qwen-vl-utils

First run downloads ~5 GB of model weights from Hugging Face automatically.
Set HF_HOME or TRANSFORMERS_CACHE to control where they land.
"""

import os
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Prompt engineering
# ─────────────────────────────────────────────────────────────────────────────

# A focused, unambiguous prompt reduces hallucination.  We explicitly forbid
# explanations and Markdown fences so the raw code comes through cleanly.
_SYSTEM_PROMPT = (
    "You are a precise OCR engine specialised in handwritten source code. "
    "Transcribe every character exactly as written. "
    "Preserve indentation using spaces. "
    "Do NOT add explanations, comments, or Markdown code fences. "
    "Output ONLY the transcribed code, nothing else."
)

_USER_PROMPT = (
    "Transcribe all handwritten code visible in this image. "
    "Keep the exact characters, operators, punctuation, and indentation. "
    "One line of handwriting = one line of output."
)

# Max new tokens to generate.  512 covers ~50 lines of code and keeps
# activation memory manageable on CPU.  Raise to 1024 if you have >16 GB RAM.
_MAX_NEW_TOKENS = 512

# Visual-token budget
# Qwen2-VL tiles images into 28x28-pixel patches — each patch = one visual
# token.  A 1920x1440 image produces ~1400 patch tokens; at float32 each
# token costs ~12 MB of activation memory across 28 layers -> >20 GB total.
# max_pixels caps the total pixel area fed to the vision encoder.
# 512x512 = 262144 px -> ~336 patch tokens -> <4 GB activation on CPU.
_MIN_PIXELS = 224 * 224        #  50176 px
_MAX_PIXELS  = 512 * 512       # 262144 px  <- primary memory knob

# Hard image-resize cap applied BEFORE the processor sees the image.
_MAX_IMAGE_SIDE = 768   # px on the longest side


# ─────────────────────────────────────────────────────────────────────────────
# Minimal image preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_image(image_path: str) -> Image.Image:
    """
    Load and resize the image to stay within the visual-token budget.

    Qwen2-VL tiles images into 28x28-pixel patches; more patches = more
    tokens = exponentially more activation memory.  On a CPU-only machine
    with ~16 GB RAM the safe ceiling is ~512x512 pixels total area.

    Unlike Tesseract/EasyOCR we do NOT run ruled-line removal or CLAHE --
    the model sees colour natively and those ops introduce artefacts for VLMs.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    max_side = max(w, h)

    if max_side < 224:
        # Upscale only if truly tiny -- below 224 px detail is lost
        scale = 224 / max_side
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    elif max_side > _MAX_IMAGE_SIDE:
        # Downscale to hard cap -- primary OOM prevention
        scale = _MAX_IMAGE_SIDE / max_side
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    print(f"  Qwen2-VL image size after resize: {img.size[0]}x{img.size[1]} px")
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def _strip_model_wrapper(text: str) -> str:
    """
    Remove artefacts that Qwen2-VL sometimes adds despite the prompt.

    Handles:
      • Markdown code fences   ```python … ```  or  ``` … ```
      • Leading/trailing blank lines
      • "Here is the code:" style preambles (up to the first blank line)
    """
    import re
    # Strip Markdown fences
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())

    # Strip single-line preamble like "Here is the transcribed code:"
    lines = text.splitlines()
    if lines and re.match(
        r"^(here\s+is|below\s+is|the\s+code|transcribed|output)\b",
        lines[0].strip(),
        re.IGNORECASE,
    ):
        # Drop everything up to (and including) the first blank separator line
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == "":
                lines = lines[i + 1 :]
                break
        else:
            lines = lines[1:]  # no blank separator — just drop the first line

    return "\n".join(lines).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Processor class
# ─────────────────────────────────────────────────────────────────────────────

class Qwen2VLProcessor:
    """
    Handwriting/code recognition using Qwen2-VL-2B-Instruct.

    Same interface as EasyOCRProcessor and RapidOCRProcessor:
        processor.extract_text(image_path) → str | None
        processor.process(image_path)      → str | None
        processor.available                → bool
    """

    MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

    def __init__(self):
        print("=" * 50)
        print("LOADING Qwen2-VL-2B-Instruct MODEL...")
        print("=" * 50)

        self.model = None
        self.processor = None
        self.device = None
        self.available = False

        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

            # Prefer CUDA, then MPS (Apple Silicon), fall back to CPU
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

            print(f"  Device: {self.device}")

            # torch_dtype: float16 halves VRAM/RAM usage on GPU/MPS.
            # CPU must stay float32 — float16 is not accelerated on CPU and
            # mixed-precision ops can cause shape mismatches in layer norms.
            dtype = torch.float16 if self.device != "cpu" else torch.float32

            # ── device_map strategy ──────────────────────────────────────────
            # device_map="auto" (accelerate dispatcher) rewrites layer norm
            # calls in-place.  On CPU-only builds this causes:
            #   "got weight of shape [151936, 1536] and normalized_shape=[1280]"
            # because the dispatcher maps the lm_head weight onto the final
            # layer-norm slot.
            #
            # Fix: only use device_map="auto" when a real GPU/MPS is present.
            # On CPU, load normally and move the model manually.
            #
            # attn_implementation="eager" disables flash-attention / SDPA,
            # both of which require CUDA and error out on CPU-only installs.
            load_kwargs = dict(
                torch_dtype=dtype,
                trust_remote_code=True,
                attn_implementation="eager",
            )
            if self.device != "cpu":
                load_kwargs["device_map"] = "auto"

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.MODEL_ID, **load_kwargs
            )

            # Explicit .to() is only needed when device_map was not used
            if self.device == "cpu":
                self.model = self.model.to("cpu")

            self.model.eval()

            # AutoProcessor bundles the tokeniser + image processor.
            # min_pixels / max_pixels enforce a hard visual-token budget so
            # the vision encoder never allocates more memory than expected,
            # even if the image was not resized before calling the processor.
            self.processor = AutoProcessor.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True,
                min_pixels=_MIN_PIXELS,
                max_pixels=_MAX_PIXELS,
            )

            self.available = True
            print(f"✅ Qwen2-VL-2B-Instruct loaded on {self.device}")

        except ImportError as exc:
            print(
                f"❌ Qwen2-VL dependencies not installed: {exc}\n"
                "   Run: pip install torch torchvision transformers>=4.45.0 "
                "accelerate qwen-vl-utils"
            )
        except Exception as exc:
            print(f"❌ Failed to load Qwen2-VL-2B-Instruct: {exc}")

    # ── Core inference ────────────────────────────────────────────────────────

    def extract_text(self, image_path: str) -> str | None:
        """
        Run Qwen2-VL on *image_path* and return the transcribed code.

        Returns None when the model is unavailable or inference fails.
        """
        if not self.available:
            print("⚠️  Qwen2-VL not available")
            return None

        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None

        try:
            import torch

            pil_img = _prepare_image(image_path)

            # Build the conversation payload.  Qwen2-VL uses the standard
            # chat-template format with an <image> placeholder.
            messages = [
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": _USER_PROMPT},
                    ],
                },
            ]

            # Tokenise: apply_chat_template converts messages → input_ids;
            # process_vision_info extracts pixel tensors from PIL images.
            try:
                # qwen-vl-utils provides the canonical helper
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
            except ImportError:
                # Fallback: pass the PIL image directly via the processor
                image_inputs = [pil_img]
                video_inputs = None

            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Build the full model input dict.
            # Passing min_pixels/max_pixels here re-applies the budget even
            # if the image was already resized, guarding against edge cases.
            processor_kwargs = dict(
                text=[text_input],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
                min_pixels=_MIN_PIXELS,
                max_pixels=_MAX_PIXELS,
            )
            if video_inputs is not None:
                processor_kwargs["videos"] = video_inputs

            inputs = self.processor(**processor_kwargs)

            # Move input tensors to GPU/MPS when device_map was not used.
            # On CPU the tensors are already on the right device.
            if self.device != "cpu":
                inputs = {k: v.to(self.device) if hasattr(v, "to") else v
                          for k, v in inputs.items()}

            # Generate — no_grad saves memory; we don't need gradients
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=_MAX_NEW_TOKENS,
                    do_sample=False,        # greedy: deterministic, lower RAM
                    temperature=None,       # must be None when do_sample=False
                    top_p=None,
                    repetition_penalty=1.05,  # mild penalty against repeated lines
                )

            # Trim the prompt tokens — keep only the newly generated tokens
            input_len = inputs["input_ids"].shape[1]
            new_tokens = generated_ids[:, input_len:]
            raw_output = self.processor.batch_decode(
                new_tokens, skip_special_tokens=True
            )[0]

            result = _strip_model_wrapper(raw_output)

            if result:
                from ocr_utils import ocr_quality_score
                score = ocr_quality_score(result)
                print(f"  Qwen2-VL result (score={score}): {repr(result[:100])}")

            return result if result else None

        except Exception as exc:
            print(f"❌ Qwen2-VL inference error: {exc}")
            return None

    def process(self, image_path: str) -> str | None:
        """Alias kept for interface parity with EasyOCRProcessor / RapidOCRProcessor."""
        return self.extract_text(image_path)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_qwen2vl_instance: Qwen2VLProcessor | None = None


def get_qwen2vl_processor() -> Qwen2VLProcessor:
    """Return the module-level Qwen2-VL singleton, creating it on first call."""
    global _qwen2vl_instance
    if _qwen2vl_instance is None:
        _qwen2vl_instance = Qwen2VLProcessor()
    return _qwen2vl_instance
