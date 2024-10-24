import os
import torch
import numpy as np
import soundfile as sf
import librosa
from .utils import demix_track, get_model_from_config


def separate_audio(
    config_path: str,
    check_point: str,
    input_file: str,
    store_dir: str = None,
    device_id: int = 0,
    extract_instrumental: bool = False,
    model_type: str = "bs_roformer",
):
    """
    从单个音频文件中分离出不同的音轨。

    :param model_type: 模型类型，例如 'mdx23c', 'htdemucs' 等
    :param config_path: 配置文件路径
    :param check_point: 模型检查点路径
    :param input_file: 输入音频文件路径
    :param store_dir: 存储结果的目录（默认与输入文件相同）
    :param device_id: GPU 设备 ID（如果可用）
    :param extract_instrumental: 是否提取伴奏
    :return: 分离后的音轨文件路径列表
    """
    # 设置存储目录
    if store_dir is None:
        store_dir = os.path.dirname(input_file)

    # 加载模型和配置
    model, config = get_model_from_config(model_type, config_path)
    if check_point:
        state_dict = torch.load(check_point)
        model.load_state_dict(state_dict)

    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
        model = model.to(device)
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Running inference on CPU. It might be slow...")
        model = model.to(device)

    model.eval()

    # 读取音频文件
    try:
        mix, sr = librosa.load(input_file, sr=44100, mono=False)
    except Exception as e:
        print(f"Cannot read track: {input_file}")
        print(f"Error message: {str(e)}")
        return []

    # 转换单声道到立体声
    if len(mix.shape) == 1:
        mix = np.stack([mix, mix], axis=0)

    mix_orig = mix.copy()
    if "normalize" in config.inference and config.inference["normalize"] is True:
        mono = mix.mean(0)
        mean = mono.mean()
        std = mono.std()
        mix = (mix - mean) / std

    mixture = torch.tensor(mix, dtype=torch.float32)

    # 执行分离

    res = demix_track(config, model, mixture, device)

    # 获取乐器列表
    instruments = config.training.instruments
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]

    # 保存分离后的音轨
    output_files = []
    for instr in instruments:
        estimates = res[instr].T
        if "normalize" in config.inference and config.inference["normalize"] is True:
            estimates = estimates * std + mean
        output_file = os.path.join(
            store_dir,
            f"{os.path.splitext(os.path.basename(input_file))[0]}_{instr}.wav",
        )
        sf.write(output_file, estimates, sr, subtype="FLOAT")
        output_files.append(output_file)

    # 提取伴奏（如果需要）
    if "vocals" in instruments and extract_instrumental:
        instrum_file_name = os.path.join(
            store_dir,
            f"{os.path.splitext(os.path.basename(input_file))[0]}_instrumental.wav",
        )
        estimates = res["vocals"].T
        if "normalize" in config.inference and config.inference["normalize"] is True:
            estimates = estimates * std + mean
        sf.write(instrum_file_name, mix_orig.T - estimates, sr, subtype="FLOAT")
        output_files.append(instrum_file_name)

    return output_files
