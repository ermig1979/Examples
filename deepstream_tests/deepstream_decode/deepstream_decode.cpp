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
#include "Gst/Options.h"
#include "Gst/Utils.h"

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
        g_main_loop_quit(data->loop);
        //gst_element_set_state(data->pipeline, GST_STATE_NULL);
        break;
    default:
        break;
    }

    g_free(str);

    return TRUE;
}

int main(int argc, char* argv[])
{
#if 0
    Gst::Pipeline pipeline;
    GMainLoop* loop = NULL;
    Gst::Element source, qtdemux, h264parser, decoder, streammux, sink, pgie, nvvidconv, nvosd;

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

    if (!qtdemux.FactoryMake("qtdemux", "qtdemux"))
        return -1;
    
    /* Since the data format in the input file is elementary h264 stream,
     * we need a h264parser */
    if (!h264parser.FactoryMake("h264parse", "h264-parser"))
        return -1;

    /* Use nvdec_h264 for hardware accelerated decode on GPU */
    if (!decoder.FactoryMake("nvv4l2decoder", "nvv4l2-decoder"))
        return -1;
    //if (!decoder.FactoryMake("avdec_h264", "avdec_h264-decoder"))
    //    return -1;

    ///* Create nvstreammux instance to form batches from one or more sources. */
    //if (!streammux.FactoryMake("nvstreammux", "stream-muxer"))
    //    return -1;

    ///* Use nvinfer to run inferencing on decoder's output,
    // * behaviour of inferencing is set through config file */
    //if (!pgie.FactoryMake("nvinfer", "primary-nvinference-engine"))
    //    return -1;

    ///* Use convertor to convert from NV12 to RGBA as required by nvosd */
    //if (!nvvidconv.FactoryMake("nvvideoconvert", "nvvideo-converter"))
    //    return -1;

    ///* Create OSD to draw on the converted RGBA buffer */
    //if (!nvosd.FactoryMake("nvdsosd", "nv-onscreendisplay"))
    //    return -1;

    ///* Finally render the osd output */
    //if (!sink.FactoryMake("fakesink", "fake-sink"))//"nveglglessink", "nvvideo-renderer"))
    //    return -1;

    /* we set the input filename to the source element */
    source.Set("location", argv[1]);

    //streammux.Set("batch-size", 1);
    //streammux.Set("width", MUXER_OUTPUT_WIDTH);
    //streammux.Set("height", MUXER_OUTPUT_HEIGHT);
    //streammux.Set("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC);

    /* Set all the necessary properties of the nvinfer element,
     * the necessary ones are : */
    //const char* pgie_config = "dstest1_pgie_config.txt";
    //if (access(pgie_config, F_OK) == -1)
    //{
    //    system("pwd");
    //    g_printerr("File '%s' is not exist!\n", pgie_config);
    //    return -1;
    //}
    //pgie.Set("config-file-path", pgie_config);

    /* we add a message handler */
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline.Handle()));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    //KeyboardData keyboardData;
    //keyboardData.pipeline = pipeline.Handle();
    //keyboardData.loop = loop;
    //GIOChannel* io_stdin;
    //io_stdin = g_io_channel_unix_new(fileno(stdin));
    //g_io_add_watch(io_stdin, G_IO_IN, (GIOFunc)handle_keyboard, &keyboardData);

    /* Set up the pipeline */
    /* we add all elements into the pipeline */
    gst_bin_add_many(GST_BIN(pipeline.Handle()),
        source.Handle(), qtdemux.Handle(), h264parser.Handle(), decoder.Handle(), /*streammux.Handle(), */sink.Handle(), NULL);
        //pgie.Handle(), nvvidconv.Handle(), nvosd.Handle(), NULL);

    //GstPad* sinkpad, * srcpad;
    //gchar pad_name_sink[16] = "sink_0";
    //gchar pad_name_src[16] = "src";

    //sinkpad = gst_element_get_request_pad(streammux.Handle(), pad_name_sink);
    //if (!sinkpad) {
    //    g_printerr("Streammux request sink pad failed. Exiting.\n");
    //    return -1;
    //}

    //srcpad = gst_element_get_static_pad(decoder.Handle(), pad_name_src);
    //if (!srcpad) {
    //    g_printerr("Decoder request src pad failed. Exiting.\n");
    //    return -1;
    //}

    //if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
    //    g_printerr("Failed to link decoder to stream muxer. Exiting.\n");
    //    return -1;
    //}

    //gst_object_unref(sinkpad);
    //gst_object_unref(srcpad);

    /* we link the elements together */
    /* file-source -> h264-parser -> nvh264-decoder ->
     * nvinfer -> nvvidconv -> nvosd -> video-renderer */

    if (!gst_element_link_many(source.Handle(), qtdemux.Handle(), NULL))
    {
        g_printerr("Elements could not be linked: 1. Exiting.\n");
        return -1;
    }

    if (!gst_element_link_many(h264parser.Handle(), decoder.Handle(), sink.Handle(), NULL))
    {
        g_printerr("Elements could not be linked: 1. Exiting.\n");
        return -1;
    }

    g_signal_connect_data(qtdemux.Handle(), "pad-added", G_CALLBACK(linkElements), h264parser.Handle(), NULL, GConnectFlags(0));

    //{
    //    GstPad* sinkpad, * srcpad;
    //    gchar pad_name_sink[16] = "sink_0";
    //    gchar pad_name_src[16] = "src";

    //    sinkpad = gst_element_get_request_pad(qtdemux.Handle(), pad_name_sink);
    //    if (!sinkpad) {
    //        g_printerr("qtdemux request sink pad failed. Exiting.\n");
    //        return -1;
    //    }

    //    srcpad = gst_element_get_static_pad(h264parser.Handle(), pad_name_src);
    //    if (!srcpad) {
    //        g_printerr("h264parser request src pad failed. Exiting.\n");
    //        return -1;
    //    }

    //    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
    //        g_printerr("Failed to link qtdemux to h264parser. Exiting.\n");
    //        return -1;
    //    }
    //    gst_object_unref(sinkpad);
    //    gst_object_unref(srcpad);
    //}
    //if (!gst_element_link_many(qtdemux.Handle(), h264parser.Handle(), NULL))
    //{
    //    g_printerr("Elements could not be linked: 1. Exiting.\n");
    //    return -1;
    //}

    //if (!gst_element_link_many(streammux.Handle(), sink.Handle(), NULL))
    //    //pgie.Handle(), nvvidconv.Handle(), nvosd.Handle(), NULL))
    //{
    //    g_printerr("Elements could not be linked: 2. Exiting.\n");
    //    return -1;
    //}

    /* Lets add probe to get informed of the meta data generated, we add probe to
     * the sink pad of the osd element, since by that time, the buffer would have
     * had got all the metadata. */
    //osd_sink_pad = gst_element_get_static_pad(nvosd.Handle(), "sink");
    //if (!osd_sink_pad)
    //    g_print("Unable to get sink pad\n");
    //else
    //    gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
    //        osd_sink_pad_buffer_probe, NULL, NULL);
    //gst_object_unref(osd_sink_pad);

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
    //g_io_channel_unref(io_stdin);
    g_main_loop_unref(loop);
    return 0;
#else
    Gst::Options options(argc, argv);

    GMainLoop* loop;

    GstElement* pipeline, * source, * demuxer, * parser, * decoder, * sink;
    GstBus* bus;
    guint bus_watch_id;

    /* Initialisation */
    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    /* Check input arguments */
    //if (argc < 2) {
    //    g_printerr("Usage: %s <mpegts filename>\n", argv[0]);
    //    return -1;
    //}

    /* Create gstreamer elements */
    pipeline = gst_pipeline_new("video-player");
    if (!pipeline) {
        g_printerr("Pipeline could not be created: [pipeline]. Exiting.\n");
        return EXIT_FAILURE;
    }

    source = gst_element_factory_make("filesrc", "file-source");
    demuxer = gst_element_factory_make("qtdemux", "qt-demuxer");
    parser = gst_element_factory_make("h264parse", "h264parse-decoder");

    //decoder = gst_element_factory_make("avdec_h264", "avdec_h264-decoder");
    decoder = gst_element_factory_make("nvv4l2decoder", "nvv4l2-decoder");
    
    //sink = gst_element_factory_make("nveglglessink", "video-output");
    sink = gst_element_factory_make("fakesink", "video-output");
    

    if (!source || !demuxer || !parser || !decoder || !sink) {
        g_printerr("An element could not be created. Exiting.\n");
        return EXIT_FAILURE;
    }

    /* Set up the pipeline */

    /* set the input filename to the source element */
    g_object_set(G_OBJECT(source), "location", options.source.c_str(), NULL);

    /* add a message handler */
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, Gst::BusCallback, loop);
    gst_object_unref(bus);

    /* add all elements into the pipeline: file-source | ts-demuxer | h264parse | decoder | video-output */
    gst_bin_add_many(GST_BIN(pipeline), source, demuxer, parser, decoder, sink, NULL);

    /* note that the demuxer will be linked to the decoder dynamically.
     * The source pad(s) will be created at run time, by the demuxer when it detects the amount and nature of streams.
     * Therefore we connect a callback function which will be executed when the "pad-added" is emitted.
    */
    gst_element_link(source, demuxer);
    gst_element_link_many(parser, decoder, sink, NULL);
    //g_signal_connect(demuxer, "pad-added", G_CALLBACK(LinkElements), parser); // link dynamic pad
    g_signal_connect_data(demuxer, "pad-added", G_CALLBACK(Gst::LinkElements), parser, NULL, GConnectFlags(0));

    /* Set the pipeline to "playing" state*/
    g_print("Now playing: %s\n", options.source.c_str());
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    /* Iterate */
    g_print("Running...\n");
    g_main_loop_run(loop);

    /* Out of the main loop, clean up nicely */
    g_print("Returned, stopping playback\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);

    g_print("Deleting pipeline\n");
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);

    return 0;
#endif
}
