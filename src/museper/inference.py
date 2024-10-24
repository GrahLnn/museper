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
    use_tta: bool = False,
    flac_file: bool = False,
    pcm_type: str = "PCM_24",
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
    :param use_tta: 是否使用测试时增强（TTA）
    :param flac_file: 是否输出为FLAC文件
    :param pcm_type: FLAC文件的PCM类型（'PCM_16'或'PCM_24'）
    :return: 分离后的音轨文件路径列表
    """
    # 设置存储目录
    if store_dir is None:
        store_dir = os.path.dirname(input_file)
    os.makedirs(store_dir, exist_ok=True)

    # 加载模型和配置
    model, config = get_model_from_config(model_type, config_path)
    if check_point:
        state_dict = torch.load(check_point, map_location='cpu')
        if model_type in ['htdemucs', 'apollo']:
            if 'state' in state_dict:
                state_dict = state_dict['state']
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)

    # 设置设备
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
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

    # 执行分离（包括TTA）
    if use_tta:
        track_proc_list = [mix.copy(), mix[::-1].copy(), -1. * mix.copy()]
    else:
        track_proc_list = [mix.copy()]

    full_result = []
    for mix_proc in track_proc_list:
        mixture = torch.tensor(mix_proc, dtype=torch.float32)
        res = demix_track(config, model, mixture, device)
        full_result.append(res)

    # 平均TTA结果
    res = full_result[0]
    for i in range(1, len(full_result)):
        for instr in res:
            if i == 2:
                res[instr] += -1.0 * full_result[i][instr]
            elif i == 1:
                res[instr] += full_result[i][instr].flip(dims=[-1])
            else:
                res[instr] += full_result[i][instr]
    for instr in res:
        res[instr] = res[instr] / len(full_result)

    # 获取乐器列表
    instruments = config.training.instruments.copy()
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]

    # 添加伴奏（如果需要）
    if extract_instrumental:
        instr = 'vocals' if 'vocals' in instruments else instruments[0]
        if 'instrumental' not in instruments:
            instruments.append('instrumental')
        res['instrumental'] = torch.tensor(mix_orig) - res[instr]

    # 保存分离后的音轨
    output_files = []
    for instr in instruments:
        estimates = res[instr].T.cpu().numpy()
        if "normalize" in config.inference and config.inference["normalize"] is True:
            estimates = estimates * std + mean
        file_name = os.path.splitext(os.path.basename(input_file))[0]
        if flac_file:
            output_file = os.path.join(store_dir, f"{file_name}_{instr}.flac")
            subtype = 'PCM_16' if pcm_type == 'PCM_16' else 'PCM_24'
            sf.write(output_file, estimates, sr, subtype=subtype)
        else:
            output_file = os.path.join(store_dir, f"{file_name}_{instr}.wav")
            sf.write(output_file, estimates, sr, subtype='FLOAT')
        output_files.append(output_file)

    return output_files
