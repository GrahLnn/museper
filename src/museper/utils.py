# coding: utf-8
__author__ = "Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/"

import numpy as np
import torch
import torch.nn as nn
import yaml
from ml_collections import ConfigDict
from .models.bs_roformer import BSRoformer


def get_model_from_config(config_path):
    with open(config_path) as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
    model = BSRoformer(**dict(config.model))
    return model, config


def demix_track(config, model, mix, device):
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

    # Prepare windows arrays (do 1 time for speed up). This trick repairs click problems on the edges of segment
    window_size = C
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window_start = torch.ones(window_size)
    window_middle = torch.ones(window_size)
    window_finish = torch.ones(window_size)
    window_start[-fade_size:] *= fadeout  # First audio chunk, no fadein
    window_finish[:fade_size] *= fadein  # Last audio chunk, no fadeout
    window_middle[-fade_size:] *= fadeout
    window_middle[:fade_size] *= fadein

    with torch.cuda.amp.autocast():
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

                    window = window_middle
                    if i - step == 0:  # First audio chunk, no fadein
                        window = window_start
                    elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                        window = window_finish

                    for j in range(len(batch_locations)):
                        start, l = batch_locations[j]
                        result[..., start : start + l] += (
                            x[j][..., :l].cpu() * window[..., :l]
                        )
                        counter[..., start : start + l] += window[..., :l]

                    batch_data = []
                    batch_locations = []

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
