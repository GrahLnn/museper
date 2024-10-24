# coding: utf-8
__author__ = "Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/"

import numpy as np
import torch
import torch.nn as nn
import yaml
from ml_collections import ConfigDict
from tqdm import tqdm


def get_model_from_config(model_type, config_path):
    with open(config_path) as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    match model_type:
        case "bs_roformer":
            from .models.bs_roformer import BSRoformer

            model = BSRoformer(**dict(config.model))
        case "mel_band_roformer":
            from .models.bs_roformer import MelBandRoformer

            model = MelBandRoformer(**dict(config.model))
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")

    return model, config


def _getWindowingArray(window_size, fade_size):
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window


def demix_track(config, model, mix, device, pbar=False):
    C = config.audio.chunk_size
    N = config.inference.num_overlap
    fade_size = C // 10
    step = int(C // N)
    border = C - step
    batch_size = config.inference.batch_size

    length_init = mix.shape[-1]

    # Do pad from the beginning and end to account floating window results better
    if length_init > 2 * border and (border > 0):
        mix = nn.functional.pad(mix, (border, border), mode="reflect")

    # windowingArray crossfades at segment boundaries to mitigate clicking artifacts
    windowingArray = _getWindowingArray(C, fade_size)

    with torch.cuda.amp.autocast(enabled=config.training.use_amp):
        with torch.inference_mode():
            if config.training.target_instrument is not None:
                req_shape = (1,) + tuple(mix.shape)
            else:
                req_shape = (len(config.training.instruments),) + tuple(mix.shape)

            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)
            i = 0
            batch_data = []
            batch_locations = []
            progress_bar = (
                tqdm(total=mix.shape[1], desc="Processing audio chunks", leave=False)
                if pbar
                else None
            )

            while i < mix.shape[1]:
                # print(i, i + C, mix.shape[1])
                part = mix[:, i : i + C].to(device)
                length = part.shape[-1]
                if length < C:
                    if length > C // 2 + 1:
                        part = nn.functional.pad(
                            input=part, pad=(0, C - length), mode="reflect"
                        )
                    else:
                        part = nn.functional.pad(
                            input=part,
                            pad=(0, C - length, 0, 0),
                            mode="constant",
                            value=0,
                        )
                batch_data.append(part)
                batch_locations.append((i, length))
                i += step

                if len(batch_data) >= batch_size or (i >= mix.shape[1]):
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr)

                    window = windowingArray
                    if i - step == 0:  # First audio chunk, no fadein
                        window[:fade_size] = 1
                    elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                        window[-fade_size:] = 1

                    for j in range(len(batch_locations)):
                        start, l = batch_locations[j]
                        result[..., start : start + l] += (
                            x[j][..., :l].cpu() * window[..., :l]
                        )
                        counter[..., start : start + l] += window[..., :l]

                    batch_data = []
                    batch_locations = []

                if progress_bar:
                    progress_bar.update(step)

            if progress_bar:
                progress_bar.close()

            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

            if length_init > 2 * border and (border > 0):
                # Remove pad
                estimated_sources = estimated_sources[..., border:-border]

    if config.training.target_instrument is None:
        return {k: v for k, v in zip(config.training.instruments, estimated_sources)}
    else:
        return {
            k: v for k, v in zip([config.training.target_instrument], estimated_sources)
        }
