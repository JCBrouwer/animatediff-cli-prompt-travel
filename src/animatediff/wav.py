import logging
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import torch
import typer
from diffusers.utils.logging import set_verbosity_error as set_diffusers_verbosity_error
from tqdm import tqdm

from animatediff import get_dir
from animatediff.generate import clear_controlnet_preprocessor, create_pipeline, get_preprocessed_img, run_inference
from animatediff.pipelines import load_text_embeddings
from animatediff.pipelines.animation import AnimationPipeline
from animatediff.schedulers import DiffusionScheduler
from animatediff.settings import InferenceConfig, ModelConfig, get_infer_config
from animatediff.utils.model import get_base_model
from animatediff.utils.pipeline import get_context_params, send_to_device
from animatediff.utils.util import get_resized_image, path_from_cwd, save_video

cli: typer.Typer = typer.Typer(
    context_settings=dict(help_option_names=["-h", "--help"]),
    rich_markup_mode="rich",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
logger = logging.getLogger(__name__)
pipeline: Optional[AnimationPipeline] = None
last_model_path: Optional[Path] = None


def my_controlnet_preprocess(
    cond_img_map: Dict[int, PathLike],
    controlnet_map: Dict[str, Any] = None,
    width: int = 512,
    height: int = 512,
    duration: int = 16,
    out_dir: PathLike = ...,
    device_str: str = None,
):
    if not controlnet_map:
        return None, None

    out_dir = Path(out_dir)  # ensure out_dir is a Path

    # { 0 : { "type_str" : IMAGE, "type_str2" : IMAGE }  }
    controlnet_image_map = {}

    controlnet_type_map = {}

    save_detectmap = controlnet_map["save_detectmap"] if "save_detectmap" in controlnet_map else True

    preprocess_on_gpu = controlnet_map["preprocess_on_gpu"] if "preprocess_on_gpu" in controlnet_map else True
    device_str = device_str if preprocess_on_gpu else None

    for c in controlnet_map:
        item = controlnet_map[c]

        processed = False

        if type(item) is dict:
            if len(cond_img_map) > 0:
                controlnet_type_map[c] = {
                    "controlnet_conditioning_scale": item["controlnet_conditioning_scale"],
                    "control_guidance_start": item["control_guidance_start"],
                    "control_guidance_end": item["control_guidance_end"],
                    "control_scale_list": item["control_scale_list"],
                    "guess_mode": item["guess_mode"] if "guess_mode" in item else False,
                }

                use_preprocessor = item["use_preprocessor"] if "use_preprocessor" in item else True

                for frame_no, img_path in tqdm(cond_img_map.items(), desc=f"Preprocessing images ({c})"):
                    if frame_no < duration:
                        if frame_no not in controlnet_image_map:
                            controlnet_image_map[frame_no] = {}
                        controlnet_image_map[frame_no][c] = get_preprocessed_img(
                            c, get_resized_image(img_path, width, height), use_preprocessor, device_str
                        )
                        processed = True

        if save_detectmap and processed:
            det_dir = out_dir.joinpath(f"{0:02d}_detectmap/{c}")
            det_dir.mkdir(parents=True, exist_ok=True)
            for frame_no in controlnet_image_map:
                save_path = det_dir.joinpath(f"{frame_no:04d}.png")
                if c in controlnet_image_map[frame_no]:
                    controlnet_image_map[frame_no][c].save(save_path)

        clear_controlnet_preprocessor(c)

    clear_controlnet_preprocessor()

    return controlnet_image_map, controlnet_type_map


# fmt:off
@cli.command()
def wav(
    prompts: Annotated[List[str], typer.Option(..., "--prompts", "-p", help="")],
    negative_prompt: Annotated[Optional[str], typer.Option(..., "--negative-prompt", "-n", help="")] = None,

    model_name_or_path: Annotated[Path, typer.Option(..., "--model-path", "-m", path_type=Path, help="Stable diffusion model to use (path or HF repo ID).", rich_help_panel="Model")] = Path("runwayml/stable-diffusion-v1-5"),
    motion_module_path: Annotated[Path, typer.Option(..., "--motion-module-path", "-mm", path_type=Path, help="Path to motion module.", rich_help_panel="Model")] = Path("models/motion-module/mm_sd_v14.ckpt"),

    number: Annotated[int, typer.Option("--number", "-N", min=1, max=99, help="Number of videos to generate (default: 1)", show_default=False, rich_help_panel="Generation")] = 1,
    width: Annotated[int, typer.Option("--width", "-W", min=64, max=3840, help="Width of generated frames", rich_help_panel="Generation")] = 512,
    height: Annotated[int, typer.Option("--height", "-H", min=64, max=2160, help="Height of generated frames", rich_help_panel="Generation")] = 512,
    length: Annotated[int, typer.Option("--length", "-L", min=1, max=9999, help="Number of frames to generate", rich_help_panel="Generation")] = 16,
    context: Annotated[Optional[int], typer.Option("--context", "-C", min=1, max=24, help="Number of frames to condition on (default: max of <length> or 24)", show_default=False, rich_help_panel="Generation")] = None,
    overlap: Annotated[Optional[int], typer.Option("--overlap", "-O", min=1, max=12, help="Number of frames to overlap in context (default: context//4)", show_default=False, rich_help_panel="Generation")] = None,
    stride: Annotated[Optional[int], typer.Option("--stride", "-S", min=0, max=8, help="Max motion stride as a power of 2 (default: 0)", show_default=False, rich_help_panel="Generation")] = None,

    scheduler: Annotated[DiffusionScheduler, typer.Option(..., "--scheduler", "-sc", help="", rich_help_panel="Diffusion")] = "k_dpmpp_sde",
    steps: Annotated[int, typer.Option(..., "--steps", "-st", min=1, max=1000, help="", rich_help_panel="Diffusion")] = 40,
    guidance_scale: Annotated[float, typer.Option(..., "--guidance-scale", "-gs", min=0.0, max=50.0, help="", rich_help_panel="Diffusion")] = 10.0,
    clip_skip: Annotated[int, typer.Option(..., "--clip-skip", "-cs", min=0, max=10, help="", rich_help_panel="Diffusion")] = 0,

    image_prompts: Annotated[Optional[List[Path]], typer.Option(..., "--image-prompts", "-i", help="", rich_help_panel="Image Prompt Adapter")] = None,
    image_prompt_scale: Annotated[float, typer.Option(..., "--image-prompt-scale", "-ips", help="", rich_help_panel="Image Prompt Adapter")] = 0.5,
    image_prompt_plus: Annotated[bool, typer.Option(..., "--image-prompt-plus", "-ipp", is_flag=True, help="", rich_help_panel="Image Prompt Adapter")] = False,

    controlnets: Annotated[Optional[List[str]], typer.Option(..., "--controlnets", "-cn", help="", rich_help_panel="ControlNet")] = None,
    controlnet_prompts: Annotated[Optional[List[Path]], typer.Option(..., "--controlnet-prompts", "-cnp", help="", rich_help_panel="ControlNet")] = None,
    controlnet_scale: Annotated[float, typer.Option(..., "--controlnet-scale", "-cns", help="", rich_help_panel="ControlNet")] = 0.5,
    controlnet_guess_mode: Annotated[bool, typer.Option(..., "--controlnet-guess-mode", "-cng", is_flag=True, help="", rich_help_panel="ControlNet")] = False,
    controlnet_fade_steps: Annotated[int, typer.Option(..., "--controlnet-fade-steps", "-cnf", help="", rich_help_panel="ControlNet")] = 0,

    max_samples_on_vram: Annotated[int, typer.Option(..., "--max-samples-on-vram", help="", rich_help_panel="ControlNet")] = 200,
    max_models_on_vram: Annotated[int, typer.Option(..., "--max-models-on-vram", help="", rich_help_panel="ControlNet")] = 3,
    save_detectmap: Annotated[bool, typer.Option(..., "--save-detectmap", is_flag=True, help="", rich_help_panel="ControlNet")] = False,
    preprocess_on_gpu: Annotated[bool, typer.Option(..., "--preprocess-on-gpu", is_flag=True, help="", rich_help_panel="ControlNet")] = True,
    loop: Annotated[bool, typer.Option(..., "--loop", is_flag=True, help="", rich_help_panel="ControlNet")] = True,

    loras: Annotated[Optional[List[str]], typer.Option(..., "--loras", "-l", help="", rich_help_panel="LoRA")] = None,
    lora_scale: Annotated[float, typer.Option(..., "--lora-scale", "-ls", help="", rich_help_panel="LoRA")] = 0.5,

    device: Annotated[str, typer.Option("--device", help="Device to run on (cpu, cuda, cuda:id)", rich_help_panel="Advanced")] = "cuda",
    compile: Annotated[bool, typer.Option("--compile", is_flag=True, help="EXPERIMENTAL: Use torch.compile to accelerate model inference (compilation can take a while, only useful for larger number of videos).", rich_help_panel="Advanced")] = False,
    use_xformers: Annotated[bool, typer.Option("--xformers", is_flag=True, help="Use XFormers instead of SDP Attention", rich_help_panel="Advanced")] = False,
    force_half_vae: Annotated[bool, typer.Option("--half-vae", is_flag=True, help="Force VAE to use fp16 (not recommended)", rich_help_panel="Advanced")] = False,

    out_dir: Annotated[Path, typer.Option("--out-dir", "-o", path_type=Path, file_okay=False, help="Directory for output folders (frames, gifs, etc)", rich_help_panel="Output")] = Path("output/"),
):
#fmt: on

    # be quiet, diffusers. we care not for your safety checker
    set_diffusers_verbosity_error()

    prompt_map = dict(zip(torch.linspace(0, length, len(prompts) + 1).round().long().tolist(), prompts))
    image_prompt_map = dict(zip(torch.linspace(0, length, len(image_prompts) + 1).round().long().tolist(), image_prompts))
    lora_map = {lora: lora_scale for lora in loras}
    controlnet_map = {
        "max_samples_on_vram": max_samples_on_vram,
        "max_models_on_vram": max_models_on_vram,
        "save_detectmap": save_detectmap,
        "preprocess_on_gpu": preprocess_on_gpu,
        "is_loop": loop,
        **{
            f"controlnet_{controlnet}": {
                "enable": True,
                "use_preprocessor": True,
                "guess_mode": controlnet_guess_mode,
                "controlnet_conditioning_scale": controlnet_scale,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
                "control_scale_list": torch.linspace(controlnet_scale, 0.0, controlnet_fade_steps).tolist(),
            }
            for controlnet in controlnets
        },
    }

    model_config = ModelConfig(
        name='model',
        path=model_name_or_path,
        motion_module=motion_module_path,
        compile=compile,
        scheduler=scheduler,
        steps=steps,
        guidance_scale=guidance_scale,
        clip_skip=clip_skip,
        prompt_map=prompt_map,
        ip_adapter_map={
            "image_prompts": image_prompt_map,
            "scale": image_prompt_scale,
            "plus": image_prompt_plus,
        },
        lora_map=lora_map,
        controlnet_map=controlnet_map,
    )
    infer_config: InferenceConfig = get_infer_config()

    # set sane defaults for context, overlap, and stride if not supplied
    context, overlap, stride = get_context_params(length, context, overlap, stride)

    # turn the device string into a torch.device
    device: torch.device = torch.device(device)

    # Get the base model if we don't have it already
    logger.info(f"Using base model: {model_name_or_path}")
    base_model_path: Path = get_base_model(model_name_or_path, local_dir=get_dir("data/models/huggingface"))

    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir = out_dir.joinpath(f"{time_str}-{Path(model_name_or_path).stem}-{Path(motion_module_path).stem}")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Will save outputs to ./{path_from_cwd(save_dir)}")

    cond_img_map = dict(zip(torch.linspace(0, length, len(prompts) + 1).round().long().tolist(), controlnet_prompts))
    controlnet_image_map, controlnet_type_map = my_controlnet_preprocess(
        cond_img_map=cond_img_map,
        controlnet_map=controlnet_map,
        width=width,
        height=height,
        duration=length,
        out_dir=save_dir,
        device_str=device,
    )

    # beware the pipeline
    global pipeline, last_model_path
    if pipeline is None or last_model_path != model_config.path.resolve():
        pipeline = create_pipeline(
            base_model=base_model_path,
            model_config=model_config,
            infer_config=infer_config,
            use_xformers=use_xformers,
        )
        last_model_path = model_config.path.resolve()
    else:
        logger.info("Pipeline already loaded, skipping initialization")
        # reload TIs; create_pipeline does this for us, but they may have changed
        # since load time if we're being called from another package
        load_text_embeddings(pipeline)

    if pipeline.device == device:
        logger.info("Pipeline already on the correct device, skipping device transfer")
    else:
        pipeline = send_to_device(pipeline, device, freeze=True, force_half=force_half_vae, compile=compile)

    # save config to output directory
    logger.info("Saving prompt config to output directory")
    save_config_path = save_dir.joinpath("prompt.json")
    save_config_path.write_text(model_config.json(indent=4), encoding="utf-8")

    logger.info("Initialization complete!")

    logger.info(f"Generating {number} animations")
    outputs = []
    for i in range(number):
        logger.info(f"Running generation {i + 1} of {number}")
        output = run_inference(
            pipeline=pipeline,
            prompt="this is dummy string",
            n_prompt=negative_prompt,
            seed=torch.seed(),
            steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            duration=length,
            idx=i,
            out_dir=save_dir,
            context_frames=context,
            context_overlap=overlap,
            context_stride=stride,
            clip_skip=clip_skip,
            prompt_map=prompt_map,
            controlnet_map=controlnet_map,
            controlnet_image_map=controlnet_image_map,
            controlnet_type_map=controlnet_type_map,
            image_prompt_map=image_prompt_map,
            ip_adapter_scale=image_prompt_scale,
            ip_adapter_plus=image_prompt_plus,
        )
        outputs.append(output)
        torch.cuda.empty_cache()

    logger.info("Generation complete!")
    logger.info("Done, exiting...")
    return save_dir

if __name__ == "__main__":
    cli()
