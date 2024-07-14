from pathlib import Path
import os
import httpx
from tqdm import tqdm
from src.museper.inference import separate_audio


def download_file(url: str, save_path: Path, show_progress: bool = True):
    """
    从指定的 URL 下载文件并保存到指定路径，可选显示进度条。

    :param url: 要下载的文件的 URL
    :param save_path: 保存下载文件的路径
    :param show_progress: 是否显示进度条，默认为 True
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with httpx.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(save_path, "wb") as file, tqdm(
            desc=save_path.name,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
            disable=not show_progress,
        ) as progress_bar:
            for chunk in response.iter_bytes(chunk_size=8192):
                size = file.write(chunk)
                progress_bar.update(size)


def check_model_exist(model_dir: Path) -> tuple[Path, Path]:
    """
    检查模型文件是否存在，如果不存在则下载。

    :param model_dir: 模型文件的目录
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

    local_weight_path = model_dir / "bs_roformer" / weight_name
    local_config_path = model_dir / "bs_roformer" / config_name

    if not os.path.exists(model_dir / "bs_roformer"):
        download_file(weight_url, local_weight_path)
        download_file(config_url, local_config_path)
    return local_weight_path, local_config_path


a, b = check_model_exist(Path("models"))
separate_audio(
    config_path=b,
    check_point=a,
    input_file="./t.wav",
    store_dir=None,
    device_id=0,
    extract_instrumental=False,
)
