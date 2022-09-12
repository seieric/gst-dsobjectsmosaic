# Gst-dsobjectsmosaic

This plugin blurs objects detected by NVIDIA nvinfer plugin. Fast and smooth since all the blurring processes are done with GPU.

**Note: This plugin is for Jetson only, not works with dGPU.**

![](https://raw.githubusercontent.com/seieric/gst-dsobjectsmosaic/main/gst-dsobjectsmosaic.png "")

## Features
- Blur objects with cuda
- Change size of squares of mosaic
- Specify class ids for which blur should be applied
- Fast and smooth processing

## Gst Properties
| Property | Meaning | Type and Range |
| -------- | ------- | -------------- |
| min-confidence | Minimum confidence of objects to be blurred | Double, 0 to 1
| mosaic-size | Size of each square of mosaic | Integer, 10 to 2147483647 |
| class-ids | Class ids of objects for which blur should be applied | Semicolon delimited integer array |

## Depedencies
- DeepStream 6.1
- OpenCV4 with CUDA support

## Download and Installation
If your environment satisfies the requirements, just run following commands.
```bash
git clone https://github.com/seieric/gst-dsobjectsmosaic.git
cd gst-dsobjectsmosaic
sudo make -j$(nproc) install
```