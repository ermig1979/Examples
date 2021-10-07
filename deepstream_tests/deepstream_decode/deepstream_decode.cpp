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
#include "Gst/MainLoop.h"

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
    Gst::Options options(argc, argv);

    std::cout << "Deepstream try to play '" << options.source << "' :" << std::endl;

    gst_init(&argc, &argv);
    {
        Gst::MainLoop loop;

        Gst::Pipeline pipeline;
        if (!pipeline.InitNew("video-player"))
            return 1;

        Gst::Element source, demuxer, parser, decoder, sink;
        if (!source.FactoryMake("filesrc", "file-source"))
            return 1;
        if (!demuxer.FactoryMake("qtdemux", "qt-demuxer"))
            return 1;
        if (!parser.FactoryMake("h264parse", "h264parse-decoder"))
            return 1;
        if (!decoder.FactoryMake("nvv4l2decoder", "nvv4l2-decoder"))//("avdec_h264", "avdec_h264-decoder")
            return 1;
        if (!sink.FactoryMake("fakesink", "video-output"))
            return 1;

        source.Set("location", options.source);

        if (!loop.AddWatch(pipeline))
            return 1;

        /* add all elements into the pipeline: file-source | ts-demuxer | h264parse | decoder | video-output */
        gst_bin_add_many(GST_BIN(pipeline.Handle()), source.Handle(), demuxer.Handle(), parser.Handle(), decoder.Handle(), sink.Handle(), NULL);

        /* note that the demuxer will be linked to the decoder dynamically.
         * The source pad(s) will be created at run time, by the demuxer when it detects the amount and nature of streams.
         * Therefore we connect a callback function which will be executed when the "pad-added" is emitted.
        */
        gst_element_link(source.Handle(), demuxer.Handle());
        gst_element_link_many(parser.Handle(), decoder.Handle(), sink.Handle(), NULL);
        g_signal_connect_data(demuxer.Handle(), "pad-added", G_CALLBACK(Gst::LinkElements), parser.Handle(), NULL, GConnectFlags(0));

        if (!pipeline.Play())
            return 1;

        loop.Run();

        pipeline.Stop();
        pipeline.Release();
    }
    return 0;
}
