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
#include "Gst/Caps.h"

struct Options : Gst::Options
{
    Gst::String source;
    Gst::String decoderType;
    Gst::String output;
    Gst::String encoderType;

    Options(int argc, char* argv[])
        : Gst::Options(argc, argv)
    {
        source = GetArg2("-s", "--source");
        decoderType = GetArg2("-dt", "--decoderType", "hard", false, {"hard", "soft"});
        output = GetArg2("-o", "--output");
        encoderType = GetArg2("-et", "--encoderType", "hard", false, { "hard", "soft" });
    }

    ~Options()
    {
    }

    bool Rtsp() const
    {
        return source.substr(0, 7) == "rtsp://";
    }
};

bool InitFileTranscoder(const Options& options, Gst::Element & pipeline)
{
    Gst::Element source, demuxer, parser, decoder, filter, converter, encoder, muxer, sink;

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

    if (options.decoderType == "soft" && options.encoderType == "soft")
    {
        if (!converter.FactoryMake("videoconvert", "video-converter"))
            return false;
    }
    else
    {
        if (!converter.FactoryMake("nvvideoconvert", "nv-video-converter"))
            return false;
    }

    if (!filter.FactoryMake("capsfilter", "caps-filter"))
        return false;
    if (options.encoderType == "soft")
    {
        Gst::Caps caps("video/x-raw");
        caps.SetString("format", "NV12");
        filter.Set("caps", caps);
    }
    else
    {
        Gst::Caps caps("video/x-raw");
        caps.SetString("format", "I420");
        caps.SetString("memory", "NVMM");
        filter.Set("caps", caps);
    }
    
    if (options.encoderType == "soft")
    {
        if (!encoder.FactoryMake("x264enc", "x264enc-encoder"))
            return false;
    }
    else
    {
        if (!encoder.FactoryMake("nvv4l2h264enc", "nvv4l2h264enc-encoder"))
            return false;
    }

    if (!muxer.FactoryMake("qtmux", "qt-muxer"))
        return false;

    if (!sink.FactoryMake("filesink", "video-output"))
        return false;
    sink.Set("location", options.output);

    if (!pipeline.Add(source, demuxer, parser, decoder))
        return false;
    if (!pipeline.Add(filter, converter, encoder, muxer, sink))
        return false;

    if (!Gst::StaticLink(source, demuxer))
        return false;

    if (!Gst::DynamicLink(demuxer, parser))
        return false;

    if (!Gst::StaticLink(parser, decoder, converter))
        return false;

    if (!Gst::StaticLink(converter, filter, encoder))
        return false;

    if (!Gst::StaticLink(encoder, muxer, sink))
        return false;


    return true;
}

bool InitRtspTranscoder(const Options& options, Gst::Element& pipeline)
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

    std::cout << "Deepstream transcode test :" << std::endl;

    Gst::MainLoop loop;

    Gst::Element pipeline;
    if (!pipeline.PipelineNew("video-transcoder"))
        return 1;

    if (!(loop.BusAddWatch(pipeline) && loop.IoAddWatch()))
        return 1;

    if (options.Rtsp())
    {
        if (!InitRtspTranscoder(options, pipeline))
            return 1;
    }
    else
    {
        if (!InitFileTranscoder(options, pipeline))
            return 1;
    }

    if (!pipeline.SetState(GST_STATE_PLAYING))
        return 1;

    loop.Run();

    pipeline.Release();

    return 0;
}
