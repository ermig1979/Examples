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

struct Options : Gst::Options
{
    Gst::String source;
    Gst::String decoderType;

    Options(int argc, char* argv[])
        : Gst::Options(argc, argv)
    {
        source = GetArg2("-s", "--source");
        decoderType = GetArg2("-dt", "--decoderType", "hard", false, {"hard", "soft"});
    }

    ~Options()
    {
    }

    bool Rtsp() const
    {
        return source.substr(0, 7) == "rtsp://";
    }
};

bool InitFileDecoder(const Options& options, Gst::Element & pipeline)
{
    Gst::Element source, demuxer, parser, decoder, sink;

    if (!source.FactoryMake("filesrc", "file-source"))
        return false;
    source.Set("location", options.source);

    if (!demuxer.FactoryMake("qtdemux", "qt-demuxer"))
        return false;

    if (!parser.FactoryMake("h264parse", "h264parse-decoder"))
        return false;

    if (options.decoderType == "hard")
    {
        if (!decoder.FactoryMake("nvv4l2decoder", "nvv4l2-decoder"))
            return false;
    }
    else if (options.decoderType == "soft")
    {
        if (!decoder.FactoryMake("avdec_h264", "avdec_h264-decoder"))
            return false;
    }
    else
        return false;

    if (!sink.FactoryMake("fakesink", "video-output"))
        return false;

    if (!pipeline.Add(source, demuxer, parser, decoder, sink))
        return false;

    if (!Gst::StaticLink(source, demuxer))
        return false;

    if (!Gst::DynamicLink(demuxer, parser))
        return false;

    if (!(Gst::StaticLink(parser, decoder, sink)))
        return false;

    return true;
}

bool InitRtspDecoder(const Options& options, Gst::Element& pipeline)
{
    Gst::Element source, depay, parser, decoder, sink;

    if (!source.FactoryMake("rtspsrc", "rtsp-source"))
        return false;
    source.Set("location", options.source);
    source.Set("latency", 100);

    if (!depay.FactoryMake("rtph264depay", "h264depay-loader"))
        return false;

    if (!parser.FactoryMake("h264parse", "h264parse-decoder"))
        return false;

    if (options.decoderType == "hard")
    {
        if (!decoder.FactoryMake("nvv4l2decoder", "nvv4l2-decoder"))
            return false;
    }
    else if (options.decoderType == "soft")
    {
        if (!decoder.FactoryMake("avdec_h264", "avdec_h264-decoder"))
            return false;
    }
    else
        return false;

    if (!sink.FactoryMake("fakesink", "video-output"))
        return false;

    if (!pipeline.Add(source, depay, parser, decoder, sink))
        return false;

    if (!Gst::DynamicLink(source, depay))
        return false;

    if (!(Gst::StaticLink(depay, parser, decoder, sink)))
        return false;

    return true;
}

int main(int argc, char* argv[])
{
    Options options(argc, argv);

    gst_init(&argc, &argv);

    std::cout << "Deepstream decode test :" << std::endl;

    Gst::MainLoop loop;

    Gst::Element pipeline;
    if (!pipeline.PipelineNew("video-player"))
        return 1;

    if (!(loop.BusAddWatch(pipeline) && loop.IoAddWatch()))
        return 1;

    if (options.Rtsp())
    {
        if (!InitRtspDecoder(options, pipeline))
            return 1;
    }
    else
    {
        if (!InitFileDecoder(options, pipeline))
            return 1;
    }

    if (!pipeline.SetState(GST_STATE_PLAYING))
        return 1;

    loop.Run();

    pipeline.Release();

    return 0;
}
