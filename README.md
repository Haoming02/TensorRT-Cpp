# TensorRT C++
An example program that implements the **[NVIDIA TensorRT](https://developer.nvidia.com/tensorrt-getting-started)** SDK for high-performance deep learning inference, written in C++

## Getting Started
*(for Windows)*

#### Requirements
0. An Nvidia **RTX** GPU
1. Install the [TensorRT 10.0 SDK](https://developer.nvidia.com/tensorrt/download)
    > An Nvidia Developer account is needed
2. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)
    > Be sure to download the release supported by your TensorRT version
3. Install [OpenCV 4.10.0](https://github.com/opencv/opencv/releases)

#### Model
> This example program is built for super-resolution

0. Go to [OpenModelDB](https://openmodeldb.info/)
1. Expand the `Advanced tag selector`, and filter the **Platform** to `ONNX` format
2. Download a model of choice
3. Extract the `trtexec.exe` program included in the SDK
4. Convert the `.onnx` model into a `.trt` engine
    - <ins><b>Example</b></ins>
        ```bash
        trtexec --onnx=4xNomos8kDAT.onnx --saveEngine=4xNomos8kDAT.trt --shapes=input:1x3x128x128 --inputIOFormats=fp32:chw --outputIOFormats=fp32:chw
        ```
    - <ins><b>Parameters</b></ins>
        - **--onnx:** Path to the downloaded model
        - **--saveEngine:** Path to save the converted engine
        - **--shapes:** The shape of the model's input. The first number is batch size *(this currently only supports `1`)*; the second number is the channel count *(this currently only supports `3` for RGB)*; the third and forth numbers are the `Training HR size` of your model. You may also need to change the name of the input layer.
        - **--inputIOFormats:** This example program only supports `fp32:chw`
        - **--outputIOFormats:** Same as above

5. Modify the `config.json` file according to your model
    - **modelPath:** Absolute path to the converted engine
        > Use **absolute path** so it can load the engine regardless of working directory
    - **inputResolution:** The `Training HR size` of your model
    - **overlap:** The overlap between each tile *(If set to `0`, it might cause visible seams)*
    - **upscaleRatio:** The `Scale` of your model
    - **deviceID:** The ID of your CUDA-capable device *(`0` if you only have one GPU)*

#### Deployment
If you simply want to run the program:

1. Download the built `.exe` from [Releases](https://github.com/Haoming02/TensorRT-Cpp/releases)
2. Place the `config.json` next to the `.exe`
3. Place the `opencv_world4100.dll` from **OpenCV** next to the `.exe`
    > `<path to opencv>\build\x64\vc16\bin\opencv_world4100.dll`
4. Double click to launch the `.exe`
5. Enter a path to an image

##### Command Line Arguments
The program can take 2 arguments:

- The first one is the path to an image, which means you can also drag and drop an image onto the `.exe` to process it. If empty, it will ask for a path instead.
- The second one is the path to the config, allowing you to quickly switch between different models using additional tools. If empty, defaults to `config.json` next to the executable.

#### Development
If you want to build from source:

0. Install [Visual Studio](https://visualstudio.microsoft.com/downloads/) with **C++** module
1. `git` `clone` this repo
2. Create a `C++ Console App`
3. Use `Add` -> `Existing items...` to include all the scripts
4. Download the [Json for C++](https://github.com/nlohmann/json/releases) package, and add the single-file `json.hpp`
5. Configure the solution to `Release` *(instead of `Debug`)*
6. `Right Click` the Project -> `Properties`
7. `C/C++` -> `Additional Include Directories` -> `Edit` -> Add **3** Entries:
    - `<path to TensorRT>\lib\include`
    - `<path to CUDA>\<version>\include`
    - `<path to opencv>\build\include`
8. `Linker` -> `Input` -> `Additional Dependencies` -> `Edit` -> Add **3** Entries:
    - `<path to opencv>\build\x64\vc16\lib\*.lib`
    - `<path to TensorRT>\lib\*.lib`
    - `<path to CUDA>\v12.4\lib\x64\*.lib`
9. Place the `opencv_world4100.dll` from **OpenCV** in the project directory
    > `<path to opencv>\build\x64\vc16\bin\opencv_world4100.dll`
10. Build

> For other OS, you will need to modify [`path_util.h`](https://github.com/Haoming02/TensorRT-Cpp/blob/main/src/path_util.h) using platform-specific implementation

## Benchmark
Running `4xNomos8kDAT` with input size of `128` and overlap of `16`* on a RTX 3060:

- Upscale a `512x512` image:
    - Using [ComfyUI](https://github.com/comfyanonymous/ComfyUI): ~11.6s
    - Using [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge): ~12.8s
    - Using **TensorRT**: ~6.2s

- Upscale a `1024x1024` image:
    - Using [ComfyUI](https://github.com/comfyanonymous/ComfyUI): ~36.5s
    - Using [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge): ~36.9s
    - Using **TensorRT**: ~19.24s

> <b>*</b> I don't know the tiling settings of ComfyUI...

## Roadmap
- [X] Upgrade to TensorRT 10
- [X] Upgrade to OpenCV 4.10.0
- [ ] Support Folder Processing
- [ ] Support Batch Size > 1
- [ ] Support Half Precision

<hr>

- Optimization **PR**s are welcomed, as this is my first ever proper C++ program...
