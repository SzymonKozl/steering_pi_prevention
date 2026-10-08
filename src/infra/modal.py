import modal


import modal


app = modal.App("dummy-app-role-test", image=image)


@app.function(gpu="A100", timeout=3600, image=image)
def hello_world():
	print("hello world")
	
	
@app.local_entrypoint()
def main():
	hello_world.remote()


def create_image():
    image = modal.Image.debian_slim().apt_install("git").uv_pip_install([
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
        "kernels==0.11.5",
        "zstandard",
        "kaleido",
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl"
    ])
    return image


def get_app(name: str, image):
    app = modal.App(name, image=image)
    return app
