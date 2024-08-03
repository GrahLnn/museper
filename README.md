# museper

Used in the form of a library. Code from [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training). Current suport bs-roformer model only.

## Install

```shell
# If you are not using the uv package manager, you can simply copy the command that starts with pip.
# install pytorch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# install museper
uv pip install https://github.com/GrahLnn/museper.git
```

## Useage

```python
from pathlib import Path

import httpx
from alive_progress import alive_bar
from museper.inference import separate_audio


def download_file(url: str, save_path: Path):
    """
    从指定的 URL 下载文件并保存到指定路径，显示进度条。

    :param url: 要下载的文件的 URL
    :param save_path: 保存下载文件的路径
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with httpx.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(save_path, "wb") as file:
            with alive_bar(total_size, title=f"Downloading {save_path.name}") as bar:
                for chunk in response.iter_bytes(chunk_size=8192):
                    size = file.write(chunk)
                    bar(size)

def check_model_exist() -> tuple[Path, Path]:
    """
    检查模型文件是否存在，如果不存在则下载。

    :return: 本地模型权重文件路径和配置文件路径
    """
    weight_name = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    config_name = "model_bs_roformer_ep_317_sdr_12.9755.yaml"

    weight_url = (
        "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
        + weight_name
    )
    config_url = (
        "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/"
        + config_name
    )

    # 使用用户主目录下的缓存目录
    cache_dir = Path.home() / ".cache" / "delta_context2" / "bs_roformer"
    local_weight_path = cache_dir / weight_name
    local_config_path = cache_dir / config_name

    if not cache_dir.exists():
        download_file(weight_url, local_weight_path)
        download_file(config_url, local_config_path)

    return local_weight_path, local_config_path

def extract_vocal(audio_path: str) -> str:
    weight, config = check_model_exist()
    audio_path: Path = Path(audio_path)
    model_give_name = audio_path.with_name(f"{audio_path.stem}_vocals.wav")
    target_audio_path = audio_path.with_name("vocal.wav")
    if os.path.exists(target_audio_path):
        return str(target_audio_path)

    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        separate_audio(
            config_path=config,
            check_point=weight,
            input_file=audio_path,
            store_dir=None,
            device_id=0,
            extract_instrumental=False,
        )

    shutil.move(model_give_name, target_audio_path)
    os.remove(audio_path)

    return str(target_audio_path)
```