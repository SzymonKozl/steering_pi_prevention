from pathlib import Path


import modal


REPO_ROOT = Path(__file__).parent.parent.parent


HF_CACHE_VOL_NAME = "hf-cache"
HF_CACHE_VOL_PATH = "/hf_cache"


def func_wrap(func, app, cfg, **kwargs):
    kwargs_default = {
        "gpu": cfg["modal"]["gpu"],
        "timeout": cfg["modal"]["timeout"], 
        "image": create_image(), 
        "volumes": {HF_CACHE_VOL_PATH: hf_cache_vol()},
        "secrets": [modal.Secret.from_name("huggingface-secret")],
        "cpu": 2.0,
        "memory": 16000
    }
    kwargs_default.update(kwargs)
    return app.function(
        **kwargs_default
    )(func)


def create_image():
    image = modal.Image.debian_slim(python_version="3.12").apt_install("git").uv_pip_install([
        "torch",
        "transformers>=5.0.0",
        "numpy",
        "tqdm",
        "pandas",
        "accelerate==1.12.0",
        "datasets",
        "cuda-toolkit==13.2.1",
        "cupy-cuda13x==14.0.1",
        "cuml-cu13",
        "scikit-learn",
        "kernels==0.17",
        "zstandard",
        "kaleido",
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl"
    ]).workdir("/app").env({"HF_HOME": HF_CACHE_VOL_PATH}).add_local_dir(REPO_ROOT, "/app")
    return image


def get_app(name: str, image):
    app = modal.App(name, image=image)
    return app


def hf_cache_vol():
    return modal.Volume.from_name(HF_CACHE_VOL_NAME)
