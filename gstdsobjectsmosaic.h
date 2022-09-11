/**
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __GST_DSEXAMPLE_H__
#define __GST_DSEXAMPLE_H__

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>

/* Open CV headers */
#pragma GCC diagnostic push
#if __GNUC__ >= 8
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
#ifdef WITH_OPENCV
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudafilters.hpp"
#endif
#pragma GCC diagnostic pop

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaEGL.h>
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gst-nvquery.h"
#include "gstnvdsmeta.h"
#include "dsexample_lib/dsexample_lib.h"

/* Package and library details required for plugin_init */
#define PACKAGE "dsobjectsmosaic"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "objects blurring plugin integrated with DeepStream on Jetson (not on dGPU)"
#define BINARY_PACKAGE "blurring objects detected by NVIDIA nvinfer plugin"
#define URL "http://nvidia.com/"


G_BEGIN_DECLS
/* Standard boilerplate stuff */
typedef struct _GstDsExample GstDsExample;
typedef struct _GstDsExampleClass GstDsExampleClass;

/* Standard boilerplate stuff */
#define GST_TYPE_DSEXAMPLE (gst_dsexample_get_type())
#define GST_DSEXAMPLE(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_DSEXAMPLE,GstDsExample))
#define GST_DSEXAMPLE_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_DSEXAMPLE,GstDsExampleClass))
#define GST_DSEXAMPLE_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_DSEXAMPLE, GstDsExampleClass))
#define GST_IS_DSEXAMPLE(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_DSEXAMPLE))
#define GST_IS_DSEXAMPLE_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_DSEXAMPLE))
#define GST_DSEXAMPLE_CAST(obj)  ((GstDsExample *)(obj))

struct _GstDsExample
{
  GstBaseTransform base_trans;

  // Context of the custom algorithm library
  DsExampleCtx *dsexamplelib_ctx;

  // Unique ID of the element. The labels generated by the element will be
  // updated at index `unique_id` of attr_info array in NvDsObjectParams.
  guint unique_id;

  // Frame number of the current input buffer
  guint64 frame_num;

  // CUDA Stream used for allocating the CUDA task
  cudaStream_t cuda_stream;

  // Host buffer to store RGB data for use by algorithm
  void *host_rgb_buf;

  // the intermediate scratch buffer for conversions RGBA
  NvBufSurface *inter_buf;

#ifdef WITH_OPENCV
  // OpenCV mat containing RGB data
  cv::Mat *cvmat;
#endif

  // Input video info (resolution, color format, framerate, etc)
  GstVideoInfo video_info;

  // Resolution at which frames/objects should be processed
  gint processing_width;
  gint processing_height;

  // Flag which defince igpu/dgpu
  guint is_integrated;

  // Amount of objects processed in single call to algorithm
  guint batch_size;

  // GPU ID on which we expect to execute the task
  guint gpu_id;

  // Boolean indicating if entire frame or cropped objects should be processed
  gboolean process_full_frame;
};

// Boiler plate stuff
struct _GstDsExampleClass
{
  GstBaseTransformClass parent_class;
};

GType gst_dsexample_get_type (void);

G_END_DECLS
#endif /* __GST_DSEXAMPLE_H__ */
