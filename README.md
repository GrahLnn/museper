# museper

Used in the form of a library. Code from [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training). Current suport bs-roformer model only.

## Install

```shell
# If you are not using the uv package manager, you can simply copy the command that starts with pip.
# install pytorch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# install museper
uv pip install git+https://github.com/GrahLnn/museper.git
```

## Useage

```python
from museper.inference import separate_audio
output_files = separate_audio(
                input_file=audio_path,
                store_dir=None,
                device_id=0,
                extract_instrumental=False,
                model_type="mel_band_roformer",
            )
```
