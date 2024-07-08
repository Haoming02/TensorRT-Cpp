# TensorRT C++
A simple program that implements the **[NVIDIA TensorRT](https://developer.nvidia.com/tensorrt-getting-started)** SDK for high-performance deep learning inference, written in C++

## Features
- **Caption**
    - Generate a caption of the image using Booru tags
- **Upscale**
    - Super resolution the image using a model
- *more coming soon...?*

## Getting Started
*(for Windows)*

#### Requirements
0. Nvidia **RTX** GPU
1. [TensorRT 10.0 SDK](https://developer.nvidia.com/tensorrt/download)
    > An Nvidia Developer account is needed
2. [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)
    > Be sure to download the release specified by your TensorRT version
3. [OpenCV 4.10.0](https://github.com/opencv/opencv/releases/tag/4.10.0)
    > It needs to be exactly this version, unless you're planning to build from source

> Recommended to add the OpenCV `bin` folder to your system **PATH**; otherwise, you have to manually place `opencv_world4100.dll` next to the `.exe`; TensorRT and CUDA Toolkit `bin` folders should be included in **PATH** already during installation

#### Models
> For optional arguments during engine conversion, refer to the [trtexec](#trtexec) section

- **Caption**:
    1. Go to [SmilingWolf](https://huggingface.co/SmilingWolf)'s HuggingFace
    2. Select a tagger model of choice
        > This program was built and tested on [WD SwinV2 Tagger v3](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3)
    3. Download **both** the `.onnx` and the `.csv` files
    4. Convert the `.onnx` model to a `.trt` engine
        - <ins><b>Example</b></ins>
            ```bash
            trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
            ```
    5. Modify the `config.json` file accordingly *(see below)*

- **Upscale**:
    1. Go to [OpenModelDB](https://openmodeldb.info/)
    2. Expand the `Advanced tag selector`, and filter the **Platform** to `ONNX` format
    3. Download a model of choice
        > This program was built and tested on [4x-Nomos8kDAT](https://openmodeldb.info/models/4x-Nomos8kDAT)
    4. Convert the `.onnx` model to a `.trt` engine
        - <ins><b>Example</b></ins>
            ```bash
            trtexec --onnx=4xNomos8kDAT.onnx --saveEngine=4xNomos8kDAT.trt --shapes=input:1x3x128x128 --inputIOFormats=fp32:chw --outputIOFormats=fp32:chw
            ```
    5. Modify the `config.json` file accordingly *(see below)*

#### Configs
> Inside the `config.json` file, you need to have the following fields:

- <ins><b>Required</b></ins>
    - **deviceID:** The ID of the CUDA device
        > Should be `0` if you only have one GPU
    - **mode:** `"caption"` or `"upscale"`
    - **modelPath:** The path to the `.trt` engine
        > Use **absolute** path so it supports drag & drop
    - **inputResolution:** Should be `448` for most tagger models; `64` or `128` for most upscale models
    - **fp16:** Enable to use half precision **I/O**

- <ins><b>Caption</b></ins>
    - **tagsPath:** The path to the `.csv` tags spreadsheet
        > Use **absolute** path so it supports drag & drop
    - **threshold:** The score needed for a tag to be included

- <ins><b>Upscale</b></ins>
    - **overlap:** The overlap between each tile
        > This is to prevent seams
    - **upscaleRatio:** The `Scale` of your upscale model

## Deployment
If you simply want to run the program:

1. Download the built `.exe` from [Releases](https://github.com/Haoming02/TensorRT-Cpp/releases)
2. Place the `config.json` next to the `.exe`
3. Launch the `.exe`

## Development
If you want to build from source:

0. Install [Visual Studio](https://visualstudio.microsoft.com/downloads/) with **C++** module
1. `git` `clone` this repo
2. Open the `.vcxproj` project
3. Modify the `CUDA.props` to point to the correct paths
    - TensorRT
    - CUDA Toolkit
    - OpenCV
4. Download the [Json for C++](https://github.com/nlohmann/json/releases) package, and add the single-file `json.hpp`
5. Download the [CSV for C++](https://github.com/d99kris/rapidcsv/releases) package, and add the single-file `rapidcsv.h`
6. Configure the solution to `Release` *(instead of `Debug`)*
7. Build

> For other OS, you will need to modify `path_util.cpp` to use platform-specific implementation

## Command-Line Arguments
The program can take 2 arguments:

- The first one is the path to an image or a path to a folder of images, which means you can simply drag and drop onto the `.exe` to process. If empty, it will ask for a path instead.

- The second one is the path to the config, allowing you to easily switch between different models and modes. If empty, defaults to `config.json` in the same folder of the `.exe`.

## Benchmark
Running `4xNomos8kDAT` at `fp32`, with input size of `128` and overlap of `16`, on a **RTX 3060**:

- Upscale a `512x512` image:
    - Using [ComfyUI](https://github.com/comfyanonymous/ComfyUI): ~11.6s
    - Using [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge): ~12.8s
    - Using **TensorRT**: ~6.2s

- Upscale a `1024x1024` image:
    - Using [ComfyUI](https://github.com/comfyanonymous/ComfyUI): ~36.5s
    - Using [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge): ~36.9s
    - Using **TensorRT**: ~19.24s

## Roadmap
- [X] Upgrade to TensorRT 10
- [X] Upgrade to OpenCV 4.10.0
- [X] Seamless Tiling
- [X] Support Folder Processing
- [X] Support Half Precision I/O
- [ ] Support Batch Size

<hr>

## trtexec

> Extract the `trtexec.exe` from the downloaded TensorRT `.zip`

<details>
<summary>Parameters</summary>

- **--onnx**: Path to the model to convert
- **--saveEngine**: Path to save the converted engine

<ins>Optional</ins>

- **--shapes**: The shape of the model's input
    > This is only needed for model with dynamic inputs *(**ie.** the upscale models)*
    - The first number is batch size
        > This program currently only supports `1`
    - The second number is the channel count
        > This program currently only supports `3` (RGB)
    - The third and forth numbers are the input dimension of your model
        > Refer to the model page

- **--inputIOFormats:** Specify the precision of the inputs and the channel order

    > **upscale** mode supports `fp32` and `fp16` I/O; **caption** mode only supports `fp32` I/O

    > Most upscale models are `chw`; the tagger models are `hwc`

- **--outputIOFormats:** Same as above

<ins>Precision</ins>

> Specify the precision to store the engine weights in

- **(default):** When omitted, defaults to `fp32` full precision
    > Largest in size; slowest in performance

- **--bf16:** More advanced half precision
    > Second largest in size; similar performance to `fp32`

    > Requires RTX **30** series or newer GPU

- **--fp16:** Half precision
    > Almost half in size; almost double in performance

    > Some models may not work properly *(**eg.** the `DAT` upscale models do not work in `fp16`)*

- **--best:** Let `trtexec` determine the precision to use for each layer, including `fp8`
    > May cause inaccuracy *(**eg.** generate artifacts for upscale models)*

> **I/O** precision and **Weight** precision are independent

</details>
