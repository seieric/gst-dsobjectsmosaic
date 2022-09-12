/**
 * Copyright (c) 2022, seieric
 * This software is based on DeepStream DsExample Plugin by NVIDIA.
 *
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

#include <string.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <fstream>
#include "gstdsobjectsmosaic.h"
#include <sys/time.h>
GST_DEBUG_CATEGORY_STATIC (gst_dsexample_debug);
#define GST_CAT_DEFAULT gst_dsexample_debug
static GQuark _dsmeta_quark = 0;

/* Enum to identify properties */
enum
{
  PROP_0,
  PROP_UNIQUE_ID,
  PROP_GPU_DEVICE_ID,
  PROP_WIDTH,
  PROP_HEIGHT,
  PROP_MIN_CONFIDENCE,
  PROP_CLASS_IDS
};

#define CHECK_NVDS_MEMORY_AND_GPUID(object, surface)  \
  ({ int _errtype=0;\
   do {  \
    if ((surface->memType == NVBUF_MEM_DEFAULT || surface->memType == NVBUF_MEM_CUDA_DEVICE) && \
        (surface->gpuId != object->gpu_id))  { \
    GST_ELEMENT_ERROR (object, RESOURCE, FAILED, \
        ("Input surface gpu-id doesnt match with configured gpu-id for element," \
         " please allocate input using unified memory, or use same gpu-ids"),\
        ("surface-gpu-id=%d,%s-gpu-id=%d",surface->gpuId,GST_ELEMENT_NAME(object),\
         object->gpu_id)); \
    _errtype = 1;\
    } \
    } while(0); \
    _errtype; \
  })


/* Default values for properties */
#define DEFAULT_UNIQUE_ID 15
#define DEFAULT_GPU_ID 0
#define DEFAULT_MIN_CONFIDENCE 0

#define RGB_BYTES_PER_PIXEL 3
#define RGBA_BYTES_PER_PIXEL 4
#define Y_BYTES_PER_PIXEL 1
#define UV_BYTES_PER_PIXEL 2

#define MIN_INPUT_OBJECT_WIDTH 16
#define MIN_INPUT_OBJECT_HEIGHT 16

#define CHECK_NPP_STATUS(npp_status,error_str) do { \
  if ((npp_status) != NPP_SUCCESS) { \
    g_print ("Error: %s in %s at line %d: NPP Error %d\n", \
        error_str, __FILE__, __LINE__, npp_status); \
    goto error; \
  } \
} while (0)

#define CHECK_CUDA_STATUS(cuda_status,error_str) do { \
  if ((cuda_status) != cudaSuccess) { \
    g_print ("Error: %s in %s at line %d (%s)\n", \
        error_str, __FILE__, __LINE__, cudaGetErrorName(cuda_status)); \
    goto error; \
  } \
} while (0)

/* By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_dsexample_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ RGBA }")));

static GstStaticPadTemplate gst_dsexample_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ RGBA }")));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_dsexample_parent_class parent_class
G_DEFINE_TYPE (GstDsExample, gst_dsexample, GST_TYPE_BASE_TRANSFORM);

static void gst_dsexample_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_dsexample_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_dsexample_set_caps (GstBaseTransform * btrans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_dsexample_start (GstBaseTransform * btrans);
static gboolean gst_dsexample_stop (GstBaseTransform * btrans);

static GstFlowReturn gst_dsexample_transform_ip (GstBaseTransform *
    btrans, GstBuffer * inbuf);

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_dsexample_class_init (GstDsExampleClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class;

  /* Indicates we want to use DS buf api */
  g_setenv ("DS_NEW_BUFAPI", "1", TRUE);

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;
  gstbasetransform_class = (GstBaseTransformClass *) klass;

  /* Overide base class functions */
  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_dsexample_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_dsexample_get_property);

  gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR (gst_dsexample_set_caps);
  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_dsexample_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_dsexample_stop);

  gstbasetransform_class->transform_ip =
      GST_DEBUG_FUNCPTR (gst_dsexample_transform_ip);

  /* Install properties */
  g_object_class_install_property (gobject_class, PROP_UNIQUE_ID,
      g_param_spec_uint ("unique-id",
          "Unique ID",
          "Unique ID for the element. Can be used to identify output of the"
          " element", 0, G_MAXUINT, DEFAULT_UNIQUE_ID, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
      g_param_spec_uint ("gpu-id",
          "Set GPU Device ID",
          "Set GPU Device ID", 0,
          G_MAXUINT, 0,
          GParamFlags
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_WIDTH,
      g_param_spec_uint ("width",
          "video width",
          "video width", 0, 
          G_MAXUINT, 0, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_HEIGHT,
      g_param_spec_uint ("height",
          "video height",
          "video height", 0, 
          G_MAXUINT, 0, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_MIN_CONFIDENCE,
      g_param_spec_double ("min-confidence",
          "minimum confidence of objects to be blurred",
          "minimum confidence of objects to be blurred", 0, 
          G_MAXDOUBLE, DEFAULT_GPU_ID, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  
  g_object_class_install_property (gobject_class, PROP_CLASS_IDS,
      g_param_spec_string ("class-ids",
          "class ids",
          "An array of colon-separated class ids for which blur is applied",
          "", (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  
  /* Set sink and src pad capabilities */
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_dsexample_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_dsexample_sink_template));

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class,
      "DsObjectsMosaic plugin",
      "DsObjectsMosaic Plugin",
      "Blur objects with cuda",
      "seieric");
}

static void
gst_dsexample_init (GstDsExample * dsexample)
{
  GstBaseTransform *btrans = GST_BASE_TRANSFORM (dsexample);

  /* We will not be generating a new buffer. Just adding / updating
   * metadata. */
  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (btrans), TRUE);
  /* We do not want to change the input caps. Set to passthrough. transform_ip
   * is still called. */
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (btrans), TRUE);

  /* Initialize all property variables to default values */
  dsexample->unique_id = DEFAULT_UNIQUE_ID;
  dsexample->gpu_id = DEFAULT_GPU_ID;
  dsexample->class_ids = new std::set<uint>;

  /* This quark is required to identify NvDsMeta when iterating through
   * the buffer metadatas */
  if (!_dsmeta_quark)
    _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_dsexample_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstDsExample *dsexample = GST_DSEXAMPLE (object);
  switch (prop_id) {
    case PROP_UNIQUE_ID:
      dsexample->unique_id = g_value_get_uint (value);
      break;
    case PROP_GPU_DEVICE_ID:
      dsexample->gpu_id = g_value_get_uint (value);
      break;
    case PROP_WIDTH:
      dsexample->width = g_value_get_uint (value);
      break;
    case PROP_HEIGHT:
      dsexample->height = g_value_get_uint (value);
      break;
    case PROP_MIN_CONFIDENCE:
      dsexample->min_confidence = g_value_get_double (value);
      break;
    case PROP_CLASS_IDS:
    {
      std::stringstream str(g_value_get_string(value));
      dsexample->class_ids->clear();
      while(str.peek() != EOF) {
        gint class_id;
        str >> class_id;
        dsexample->class_ids->insert(class_id);
        str.get();
      }
    }
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* Function called when a property of the element is requested. Standard
 * boilerplate.
 */
static void
gst_dsexample_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstDsExample *dsexample = GST_DSEXAMPLE (object);

  switch (prop_id) {
    case PROP_UNIQUE_ID:
      g_value_set_uint (value, dsexample->unique_id);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, dsexample->gpu_id);
      break;
    case PROP_WIDTH:
      g_value_set_uint (value, dsexample->width);
      break;
    case PROP_HEIGHT:
      g_value_set_uint (value, dsexample->height);
      break;
    case PROP_MIN_CONFIDENCE:
      g_value_set_double (value, dsexample->min_confidence);
      break;
    case PROP_CLASS_IDS:
    {
      std::stringstream str;
      for(const auto id : *dsexample->class_ids)
        str << id << ";";
      g_value_set_string (value, str.str ().c_str ());
    }
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * Initialize all resources and start the output thread
 */
static gboolean
gst_dsexample_start (GstBaseTransform * btrans)
{
  GstDsExample *dsexample = GST_DSEXAMPLE (btrans);

  GstQuery *queryparams = NULL;
  guint batch_size = 1;
  int val = -1;

  CHECK_CUDA_STATUS (cudaSetDevice (dsexample->gpu_id),
      "Unable to set cuda device");

  cudaDeviceGetAttribute (&val, cudaDevAttrIntegrated, dsexample->gpu_id);
  dsexample->is_integrated = val;

  dsexample->batch_size = 1;
  queryparams = gst_nvquery_batch_size_new ();
  if (gst_pad_peer_query (GST_BASE_TRANSFORM_SINK_PAD (btrans), queryparams)
      || gst_pad_peer_query (GST_BASE_TRANSFORM_SRC_PAD (btrans), queryparams)) {
    if (gst_nvquery_batch_size_parse (queryparams, &batch_size)) {
      dsexample->batch_size = batch_size;
    }
  }
  GST_DEBUG_OBJECT (dsexample, "Setting batch-size %d \n",
      dsexample->batch_size);
  gst_query_unref (queryparams);

  CHECK_CUDA_STATUS (cudaStreamCreate (&dsexample->cuda_stream),
      "Could not create cuda stream");

  return TRUE;
error:
  if (dsexample->cuda_stream) {
    cudaStreamDestroy (dsexample->cuda_stream);
    dsexample->cuda_stream = NULL;
  }
  return FALSE;
}

/**
 * Stop the output thread and free up all the resources
 */
static gboolean
gst_dsexample_stop (GstBaseTransform * btrans)
{
  GstDsExample *dsexample = GST_DSEXAMPLE (btrans);

  if (dsexample->cuda_stream)
    cudaStreamDestroy (dsexample->cuda_stream);
  dsexample->cuda_stream = NULL;

  delete dsexample->class_ids;

  return TRUE;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_dsexample_set_caps (GstBaseTransform * btrans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstDsExample *dsexample = GST_DSEXAMPLE (btrans);
  /* Save the input video information, since this will be required later. */
  gst_video_info_from_caps (&dsexample->video_info, incaps);

  /* requires RGBA format for blurring the objects in opencv */
   if (dsexample->video_info.finfo->format != GST_VIDEO_FORMAT_RGBA) {
    GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
        ("input format should be RGBA when using blur-objects property"), (NULL));
    goto error;
    }

  return TRUE;

error:
  return FALSE;
}

/*
 * Blur the detected objects
 */
static GstFlowReturn
blur_objects (GstDsExample * dsexample, gint idx,
    NvOSD_RectParams * crop_rect_params, cv::cuda::GpuMat in_mat)
{
  cv::Rect crop_rect;

  if ((crop_rect_params->width == 0) || (crop_rect_params->height == 0)) {
    GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
        ("%s:crop_rect_params dimensions are zero",__func__), (NULL));
    return GST_FLOW_ERROR;
  }

/* rectangle for cropped objects */
  crop_rect = cv::Rect (crop_rect_params->left, crop_rect_params->top,
  crop_rect_params->width, crop_rect_params->height);

/* cuda based mosaic */
  cv::cuda::GpuMat resized_mat;
  cv::cuda::resize(in_mat(crop_rect), resized_mat, cv::Size(0, 0), 0.1, 0.1, cv::INTER_NEAREST);
  cv::cuda::resize(resized_mat, in_mat(crop_rect), cv::Size(crop_rect_params->width, crop_rect_params->height), 0, 0, cv::INTER_NEAREST);

  return GST_FLOW_OK;
}

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
gst_dsexample_transform_ip (GstBaseTransform * btrans, GstBuffer * inbuf)
{
  GstDsExample *dsexample = GST_DSEXAMPLE (btrans);
  GstMapInfo in_map_info;
  GstFlowReturn flow_ret = GST_FLOW_ERROR;
  gdouble scale_ratio = 1.0;

  NvBufSurface *surface = NULL;
  NvDsBatchMeta *batch_meta = NULL;
  NvDsFrameMeta *frame_meta = NULL;
  NvDsMetaList * l_frame = NULL;

  dsexample->frame_num++;
  CHECK_CUDA_STATUS (cudaSetDevice (dsexample->gpu_id),
      "Unable to set cuda device");

  memset (&in_map_info, 0, sizeof (in_map_info));
  if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
    g_print ("Error: Failed to map gst buffer\n");
    goto error;
  }

  nvds_set_input_system_timestamp (inbuf, GST_ELEMENT_NAME (dsexample));
  surface = (NvBufSurface *) in_map_info.data;
  GST_DEBUG_OBJECT (dsexample,
      "Processing Frame %" G_GUINT64_FORMAT " Surface %p\n",
      dsexample->frame_num, surface);

  if (CHECK_NVDS_MEMORY_AND_GPUID (dsexample, surface))
    goto error;

  batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  if (batch_meta == nullptr) {
    GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
        ("NvDsBatchMeta not found for input buffer."), (NULL));
    return GST_FLOW_ERROR;
  }

  if (true) {
    /* Using object crops as input to the algorithm. The objects are detected by
     * the primary detector */
    NvDsMetaList * l_obj = NULL;
    NvDsObjectMeta *obj_meta = NULL;

    if(!dsexample->is_integrated) {
      if (!(surface->memType == NVBUF_MEM_CUDA_UNIFIED || surface->memType == NVBUF_MEM_CUDA_PINNED)){
        GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
            ("%s:need NVBUF_MEM_CUDA_UNIFIED or NVBUF_MEM_CUDA_PINNED memory for opencv blurring",__func__), (NULL));
        return GST_FLOW_ERROR;
      }
    }

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next)
    {
      frame_meta = (NvDsFrameMeta *) (l_frame->data);
      /* Skip all the blurring process when no objects are detected. */
      if (frame_meta->num_obj_meta == 0)
        continue;

      NvBufSurface ip_surf;
      ip_surf = *surface;
      ip_surf.numFilled = ip_surf.batchSize = 1;
      ip_surf.surfaceList = &(surface->surfaceList[frame_meta->batch_id]);
      /* map and modify original buffer directly */
      if (NvBufSurfaceMapEglImage (&ip_surf, 0) != 0) {
        goto error;
      }
      CUresult status;
      CUeglFrame eglFrame;
      CUgraphicsResource pResource = NULL;
      cudaFree(0);
      status = cuGraphicsEGLRegisterImage(&pResource,
		  ip_surf.surfaceList[0].mappedAddr.eglImage,
                  CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
      status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, pResource, 0, 0);
      status = cuCtxSynchronize();

      cv::cuda::GpuMat in_mat(ip_surf.surfaceList[0].planeParams.height[0],
                  ip_surf.surfaceList[0].planeParams.width[0],
                  CV_8UC4, eglFrame.frame.pPitch[0]);

      for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
          l_obj = l_obj->next)
      {
        obj_meta = (NvDsObjectMeta *) (l_obj->data);

        /* Skip too small objects since they cause resizing issues. */
        if (obj_meta->rect_params.width < MIN_INPUT_OBJECT_WIDTH ||
            obj_meta->rect_params.height < MIN_INPUT_OBJECT_HEIGHT ||
            obj_meta->confidence < dsexample->min_confidence )
          continue;

        /* apply blur only for objects with given class ids */
        auto id_itr = dsexample->class_ids->find(obj_meta->class_id);
        if ( id_itr == dsexample->class_ids->end() || *id_itr != obj_meta->class_id)
          continue;

        if (blur_objects (dsexample, frame_meta->batch_id,
          &obj_meta->rect_params, in_mat) != GST_FLOW_OK) {
        /* Error in blurring, skip processing on object. */
          GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
          ("blurring the object failed"), (NULL));
          if (NvBufSurfaceUnMapEglImage (&ip_surf, 0) != 0){
            GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
              ("%s:buffer unmap failed", __func__), (NULL));
          }
          return GST_FLOW_ERROR;
        }
      }

      status = cuCtxSynchronize();
      status = cuGraphicsUnregisterResource(pResource);
      // Destroy the EGLImage
      NvBufSurfaceUnMapEglImage (&ip_surf, 0);
    }
  }
  flow_ret = GST_FLOW_OK;

error:

  nvds_set_output_system_timestamp (inbuf, GST_ELEMENT_NAME (dsexample));
  gst_buffer_unmap (inbuf, &in_map_info);
  return flow_ret;
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
dsexample_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_dsexample_debug, "dsobjectsmosaic", 0,
      "dsobjectsmosaic plugin");

  return gst_element_register (plugin, "dsobjectsmosaic", GST_RANK_PRIMARY,
      GST_TYPE_DSEXAMPLE);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_dsobjectsmosaic,
    DESCRIPTION, dsexample_plugin_init, DS_VERSION, LICENSE, BINARY_PACKAGE, URL)
