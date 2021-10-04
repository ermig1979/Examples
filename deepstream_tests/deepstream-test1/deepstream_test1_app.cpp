/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime_api.h>
#include "gstnvdsmeta.h"

#include "Gst/Pipeline.h"
#include "Gst/Element.h"

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

 /* The muxer output resolution must be set if the input streams will be of
  * different resolution. The muxer will scale all the input frames to this
  * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

  /* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
   * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

gint frame_number = 0;
gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person", "Roadsign" };

/* osd_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */

static GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad* pad, GstPadProbeInfo* info, gpointer u_data)
{
    GstBuffer* buf = (GstBuffer*)info->data;
    guint num_rects = 0;
    NvDsObjectMeta* obj_meta = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    NvDsMetaList* l_frame = NULL;
    NvDsMetaList* l_obj = NULL;
    NvDsDisplayMeta* display_meta = NULL;

    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
        l_frame = l_frame->next) {
        NvDsFrameMeta* frame_meta = (NvDsFrameMeta*)(l_frame->data);
        int offset = 0;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
            l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta*)(l_obj->data);
            if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE) {
                vehicle_count++;
                num_rects++;
            }
            if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
                person_count++;
                num_rects++;
            }
        }
        display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        NvOSD_TextParams* txt_params = &display_meta->text_params[0];
        display_meta->num_labels = 1;
        txt_params->display_text = (char*)g_malloc0(MAX_DISPLAY_LEN);
        offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ", person_count);
        offset = snprintf(txt_params->display_text + offset, MAX_DISPLAY_LEN, "Vehicle = %d ", vehicle_count);

        /* Now set the offsets where the string should appear */
        txt_params->x_offset = 10;
        txt_params->y_offset = 12;

        /* Font , font-color and font-size */
        txt_params->font_params.font_name = "Serif";
        txt_params->font_params.font_size = 10;
        txt_params->font_params.font_color.red = 1.0;
        txt_params->font_params.font_color.green = 1.0;
        txt_params->font_params.font_color.blue = 1.0;
        txt_params->font_params.font_color.alpha = 1.0;

        /* Text background color */
        txt_params->set_bg_clr = 1;
        txt_params->text_bg_clr.red = 0.0;
        txt_params->text_bg_clr.green = 0.0;
        txt_params->text_bg_clr.blue = 0.0;
        txt_params->text_bg_clr.alpha = 1.0;

        nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }

    g_print("Frame Number = %d Number of objects = %d "
        "Vehicle Count = %d Person Count = %d\n",
        frame_number, num_rects, vehicle_count, person_count);
    frame_number++;
    return GST_PAD_PROBE_OK;
}

static gboolean bus_call(GstBus* bus, GstMessage* msg, gpointer data)
{
    GMainLoop* loop = (GMainLoop*)data;
    switch (GST_MESSAGE_TYPE(msg))
    {
    case GST_MESSAGE_EOS:
        g_print("End of stream\n");
        g_main_loop_quit(loop);
        break;
    case GST_MESSAGE_ERROR:
    {
        gchar* debug;
        GError* error;
        gst_message_parse_error(msg, &error, &debug);
        g_printerr("ERROR from element %s: %s\n",
            GST_OBJECT_NAME(msg->src), error->message);
        if (debug)
            g_printerr("Error details: %s\n", debug);
        g_free(debug);
        g_error_free(error);
        g_main_loop_quit(loop);
        break;
    }
    default:
        break;
    }
    return TRUE;
}

struct KeyboardData
{
    GstElement* pipeline;
    GMainLoop* loop;
};

static gboolean handle_keyboard(GIOChannel* source, GIOCondition cond, KeyboardData* data)
{
    gchar* str = NULL;

    if (g_io_channel_read_line(source, &str, NULL, NULL, NULL) != G_IO_STATUS_NORMAL)
    {
        return TRUE;
    }

    switch (g_ascii_tolower(str[0]))
    {
    case 'q':
        g_printerr("Key 'q' is pressed. Try to stop pipeline.\n");
        gst_element_set_state(data->pipeline, GST_STATE_NULL);
        g_main_loop_quit(data->loop);
        break;
    default:
        break;
    }

    g_free(str);

    return TRUE;
}

int main(int argc, char* argv[])
{
    Gst::Pipeline pipeline;
    GMainLoop* loop = NULL;
    Gst::Element source, h264parser, decoder, streammux;
    GstElement * sink = NULL, * pgie = NULL, * nvvidconv = NULL, * nvosd = NULL;

    GstElement* transform = NULL;
    GstBus* bus = NULL;
    guint bus_watch_id;
    GstPad* osd_sink_pad = NULL;

    int current_device = -1;
    cudaGetDevice(&current_device);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);
    /* Check input arguments */
    if (argc != 2)
    {
        g_printerr("Usage: %s <H264 filename>\n", argv[0]);
        return -1;
    }

    /* Standard GStreamer initialization */
    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    /* Create gstreamer elements */
    /* Create Pipeline element that will form a connection of other elements */
    if (!pipeline.InitNew("dstest1-pipeline"))
        return -1;

    /* Source element for reading from the file */
    if (!source.FactoryMake("filesrc", "file-source"))
        return -1;

    /* Since the data format in the input file is elementary h264 stream,
     * we need a h264parser */
    if (!h264parser.FactoryMake("h264parse", "h264-parser"))
        return -1;

    /* Use nvdec_h264 for hardware accelerated decode on GPU */
    if (!decoder.FactoryMake("nvv4l2decoder", "nvv4l2-decoder"))
        return -1;

    /* Create nvstreammux instance to form batches from one or more sources. */
    if (!streammux.FactoryMake("nvstreammux", "stream-muxer"))
        return -1;

    /* Use nvinfer to run inferencing on decoder's output,
     * behaviour of inferencing is set through config file */
    pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");

    /* Use convertor to convert from NV12 to RGBA as required by nvosd */
    nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");

    /* Create OSD to draw on the converted RGBA buffer */
    nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");

    /* Finally render the osd output */
    if (prop.integrated)
    {
        transform = gst_element_factory_make("nvegltransform", "nvegl-transform");
    }
    sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");

    if (!pgie || !nvvidconv || !nvosd || !sink) {
        g_printerr("One element could not be created. Exiting.\n");
        return -1;
    }

    if (!transform && prop.integrated) {
        g_printerr("One tegra element could not be created. Exiting.\n");
        return -1;
    }

    /* we set the input filename to the source element */
    source.Set("location", argv[1]);

    streammux.Set("batch-size", 1);
    streammux.Set("width", MUXER_OUTPUT_WIDTH);
    streammux.Set("height", MUXER_OUTPUT_HEIGHT);
    streammux.Set("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC);

    /* Set all the necessary properties of the nvinfer element,
     * the necessary ones are : */
    const char* pgie_config = "dstest1_pgie_config.txt";
    if (access(pgie_config, F_OK) == -1)
    {
        system("pwd");
        g_printerr("File '%s' is not exist!\n", pgie_config);
        return -1;
    }
    g_object_set(G_OBJECT(pgie), "config-file-path", pgie_config, NULL);

    /* we add a message handler */
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline.Handle()));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    KeyboardData keyboardData;
    keyboardData.pipeline = pipeline.Handle();
    keyboardData.loop = loop;
    GIOChannel* io_stdin;
    io_stdin = g_io_channel_unix_new(fileno(stdin));
    g_io_add_watch(io_stdin, G_IO_IN, (GIOFunc)handle_keyboard, &keyboardData);

    /* Set up the pipeline */
    /* we add all elements into the pipeline */
    if (prop.integrated)
    {
        gst_bin_add_many(GST_BIN(pipeline.Handle()),
            source.Handle(), h264parser.Handle(), decoder.Handle(), streammux.Handle(), pgie,
            nvvidconv, nvosd, transform, sink, NULL);
    }
    else
    {
        gst_bin_add_many(GST_BIN(pipeline.Handle()),
            source.Handle(), h264parser.Handle(), decoder.Handle(), streammux.Handle(), pgie,
            nvvidconv, nvosd, sink, NULL);
    }

    {
        GstPad* sinkpad, * srcpad;
        gchar pad_name_sink[16] = "sink_0";
        gchar pad_name_src[16] = "src";

        sinkpad = gst_element_get_request_pad(streammux.Handle(), pad_name_sink);
        if (!sinkpad) {
            g_printerr("Streammux request sink pad failed. Exiting.\n");
            return -1;
        }

        srcpad = gst_element_get_static_pad(decoder.Handle(), pad_name_src);
        if (!srcpad) {
            g_printerr("Decoder request src pad failed. Exiting.\n");
            return -1;
        }

        if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
            g_printerr("Failed to link decoder to stream muxer. Exiting.\n");
            return -1;
        }

        gst_object_unref(sinkpad);
        gst_object_unref(srcpad);
    }

    /* we link the elements together */
    /* file-source -> h264-parser -> nvh264-decoder ->
     * nvinfer -> nvvidconv -> nvosd -> video-renderer */

    if (!gst_element_link_many(source.Handle(), h264parser.Handle(), decoder.Handle(), NULL))
    {
        g_printerr("Elements could not be linked: 1. Exiting.\n");
        return -1;
    }

    if (prop.integrated) 
    {
        if (!gst_element_link_many(streammux.Handle(), pgie,
            nvvidconv, nvosd, transform, sink, NULL)) 
        {
            g_printerr("Elements could not be linked: 2. Exiting.\n");
            return -1;
        }
    }
    else 
    {
        if (!gst_element_link_many(streammux.Handle(), pgie,
            nvvidconv, nvosd, sink, NULL)) 
        {
            g_printerr("Elements could not be linked: 2. Exiting.\n");
            return -1;
        }
    }

    /* Lets add probe to get informed of the meta data generated, we add probe to
     * the sink pad of the osd element, since by that time, the buffer would have
     * had got all the metadata. */
    osd_sink_pad = gst_element_get_static_pad(nvosd, "sink");
    if (!osd_sink_pad)
        g_print("Unable to get sink pad\n");
    else
        gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
            osd_sink_pad_buffer_probe, NULL, NULL);
    gst_object_unref(osd_sink_pad);

    /* Set the pipeline to "playing" state */
    g_print("Now playing: %s\n", argv[1]);
    if (!pipeline.Play())
        return -1;

    /* Wait till pipeline encounters an error or EOS */
    g_print("Running...\n");
    g_main_loop_run(loop);

    /* Out of the main loop, clean up nicely */
    pipeline.Release(true);
    g_source_remove(bus_watch_id);
    g_io_channel_unref(io_stdin);
    g_main_loop_unref(loop);
    return 0;
}
