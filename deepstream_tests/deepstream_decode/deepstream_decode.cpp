#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime_api.h>
#include "gstnvdsmeta.h"

#include "Gst/Element.h"
#include "Gst/Options.h"
#include "Gst/Utils.h"
#include "Gst/MainLoop.h"

int main(int argc, char* argv[])
{
    Gst::Options options(argc, argv);

    std::cout << "Deepstream try to play file '" << options.source << "' :" << std::endl;

    gst_init(&argc, &argv);

    {
        Gst::MainLoop loop;

        Gst::Element pipeline, source, demuxer, parser, decoder, sink;
        if (!pipeline.PipelineNew("video-player"))
            return 1;

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

        if (!(loop.BusAddWatch(pipeline) && loop.IoAddWatch()))
            return 1;

        if(!(pipeline.Add(source) && pipeline.Add(demuxer) && pipeline.Add(parser) && pipeline.Add(decoder) && pipeline.Add(sink)))
            return 1;

        if (!Gst::StaticLink(source, demuxer))
            return 1;
        if (!(Gst::StaticLink(parser, decoder) && Gst::StaticLink(decoder, sink)))
            return 1;
        if (!Gst::DynamicLink(demuxer, parser, "pad-added"))
            return 1;

        if (!pipeline.SetState(GST_STATE_PLAYING))
            return 1;

        loop.Run();

        pipeline.Release();
    }

    return 0;
}
