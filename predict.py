import torch
from diffusers import StableDiffusionXLPipeline
from cog import BasePredictor, Input, Path
from huggingface_hub import hf_hub_download, snapshot_download
import logging
import os

MODEL_NAME = "lykon/dreamshaper-xl-1-0"
MODEL_CACHE = "./dreamshaperxl"

LORA_REPO = "dennis-brinelinestudios/soulcaller-lora"
LORA_FILENAME = "SDXL_Inkdrawing_Directors_Cut_E.safetensors"

logging.basicConfig(level=logging.DEBUG)

class Predictor(BasePredictor):
    def setup(self):
        logging.info("🟡 Loading DreamShaperXL Model...")

        try:
            if not os.path.exists(MODEL_CACHE):
                logging.info("🟡 Downloading DreamShaperXL...")
                model_path = snapshot_download(repo_id=MODEL_NAME, cache_dir=MODEL_CACHE)
            else:
                logging.info(f"🟢 Using cached model: {MODEL_CACHE}")
                model_path = MODEL_CACHE

            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16
            ).to("cuda")

            logging.info("🟢 DreamShaperXL model loaded successfully.")

        except Exception as e:
            logging.error(f"❌ Failed to load DreamShaperXL, falling back to Base SDXL: {e}")
            model_path = snapshot_download(
                repo_id="stabilityai/stable-diffusion-xl-base-1.0",
                cache_dir="./sdxl-model"
            )
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16
            ).to("cuda")

        logging.info(f"🟡 Downloading LoRA weights from {LORA_REPO}...")
        lora_path = hf_hub_download(repo_id=LORA_REPO, filename=LORA_FILENAME)

        logging.info("🟡 Applying LoRA weights with alpha=2.0...")
        self.pipe.load_lora_weights(
            lora_path,
            weight_name="pytorch_lora_weights.safetensors",
            alpha=2.0
        )

        logging.info("🟢 LoRA successfully applied.")

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt for the model",
            default="A low-detail colored ink drawing of a tcg [Card Type] named: [Card Name], description: [User Description]"
        ),
        negative_prompt: str = Input(
            description="Negative prompt",
            default="excessive shading, detailed background, realism, clutter"
        ),
        guidance_scale: float = Input(default=9),
        num_inference_steps: int = Input(default=25),
        seed: int = Input(default=42),
    ) -> Path:
        """Run a prediction"""
        generator = torch.manual_seed(seed)

        # 🔒 Hardcoded portrait SDXL resolution (standard portrait)
        width = 832
        height = 1216

        logging.info(f"🟡 Running inference at {width}x{height}: '{prompt}'")

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            width=width,
            height=height,
        ).images[0]

        output_path = "/tmp/output.png"
        output.save(output_path)

        logging.info(f"🟢 Image saved to {output_path}")
        return Path(output_path)
