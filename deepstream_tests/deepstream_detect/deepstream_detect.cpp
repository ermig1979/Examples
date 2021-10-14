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
    Gst::String output;
    Gst::String encoderType;
    int bitrate;

    Options(int argc, char* argv[])
        : Gst::Options(argc, argv)
    {
        source = GetArg2("-s", "--source");
        decoderType = GetArg2("-dt", "--decoderType", "hard", false, { "hard", "soft" });
        output = GetArg2("-o", "--output");
        encoderType = GetArg2("-et", "--encoderType", "hard", false, { "hard", "soft" });
        bitrate = Gst::FromString<int>(GetArg2("-b", "--bitrate", "5000", false))*1000;
    }

    ~Options()
    {
    }

    bool Rtsp() const
    {
        return source.substr(0, 7) == "rtsp://";
    }
};

bool InitFileDetector(const Options& options, Gst::Element & pipeline)
{
    Gst::Element source, demuxer, decParser, decoder, streamMuxer, converter, filter, encoder, encParser, muxer, sink;

    if (!source.FactoryMake("filesrc", "file-source"))
        return false;
    source.Set("location", options.source);

    if (!demuxer.FactoryMake("qtdemux", "qt-demuxer"))
        return false;

    if (!decParser.FactoryMake("h264parse", "h264parse-decoder"))
        return false;

    if (options.decoderType == "soft")
    {
        if (!decoder.FactoryMake("avdec_h264", "avdec_h264-decoder"))
            return false;
    }
    else
    {
        if (!decoder.FactoryMake("nvv4l2decoder", "nvv4l2-decoder"))
            return false;
    }

    if (!streamMuxer.FactoryMake("nvstreammux", "stream-muxer"))
        return false;
    streamMuxer.Set("batch-size", 1);
    streamMuxer.Set("height", 1080);
    streamMuxer.Set("width", 1920);

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
        if (!filter.SetCapsFromString("video/x-raw, format=I420"))
            return false;
        if (!encoder.FactoryMake("x264enc", "x264enc-encoder"))
            return false;
    }
    else
    {
        if (!filter.SetCapsFromString("video/x-raw(memory:NVMM), format=I420"))
            return false;
        if (!encoder.FactoryMake("nvv4l2h264enc", "nvv4l2h264enc-encoder"))
            return false;
    }
    encoder.Set("bitrate", options.bitrate);

    if (!encParser.FactoryMake("h264parse", "h264parse-encoder"))
        return false;

    if (!muxer.FactoryMake("qtmux", "qt-muxer"))
        return false;

    if (!sink.FactoryMake("filesink", "video-output"))
        return false;
    sink.Set("location", options.output);

    if (!pipeline.BinAdd(source, demuxer, decParser, decoder, streamMuxer))
        return false;
    if (!pipeline.BinAdd(filter, converter, encoder, encParser, muxer, sink))
        return false;

    if (!Gst::StaticLink(source, demuxer))
        return false;

    if (!Gst::DynamicLink(demuxer, decParser))
        return false;

    if (!Gst::StaticLink(decParser, decoder))
        return false;

    if (!Gst::PadLink(decoder, "src", streamMuxer, "sink_0"))
        return false;

    if (!Gst::StaticLink(streamMuxer, converter, filter, encoder))
        return false;

    if (!Gst::StaticLink(encoder, encParser, muxer, sink))
        return false;

    return true;
}

bool InitRtspDetector(const Options& options, Gst::Element& pipeline)
{
    Gst::Element source, depay, decParser, decoder, converter, filter, encoder, encParser, muxer, sink;

    if (!source.FactoryMake("rtspsrc", "rtsp-source"))
        return false;
    source.Set("location", options.source);
    source.Set("latency", 100);

    if (!depay.FactoryMake("rtph264depay", "h264depay-loader"))
        return false;

    if (!decParser.FactoryMake("h264parse", "h264parse-decoder"))
        return false;

    if (options.decoderType == "soft")
    {
        if (!decoder.FactoryMake("avdec_h264", "avdec_h264-decoder"))
            return false;
    }
    else
    {
        if (!decoder.FactoryMake("nvv4l2decoder", "nvv4l2-decoder"))
            return false;
    }

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
        if (!filter.SetCapsFromString("video/x-raw, format=I420"))
            return false;
        if (!encoder.FactoryMake("x264enc", "x264enc-encoder"))
            return false;
    }
    else
    {
        if (!filter.SetCapsFromString("video/x-raw(memory:NVMM), format=I420"))
            return false;
        if (!encoder.FactoryMake("nvv4l2h264enc", "nvv4l2h264enc-encoder"))
            return false;
    }

    if (!encParser.FactoryMake("h264parse", "h264parse-encoder"))
        return false;

    if (!muxer.FactoryMake("qtmux", "qt-muxer"))
        return false;

    if (!sink.FactoryMake("filesink", "video-output"))
        return false;
    sink.Set("location", options.output);
    sink.Set("async", FALSE);

    if (!pipeline.BinAdd(source, depay, decParser, decoder, filter))
        return false;
    if (!pipeline.BinAdd(converter, encoder, encParser, muxer, sink))
        return false;

    if (!Gst::DynamicLink(source, depay))
        return false;

    if (!Gst::StaticLink(depay, decParser, decoder, converter))
        return false;

    if (!Gst::StaticLink(converter, filter, encoder, encParser))
        return false;

    if (!Gst::StaticLink(encParser, muxer, sink))
        return false;

    return true;
}

int main(int argc, char* argv[])
{
    Options options(argc, argv);

    gst_init(&argc, &argv);

    std::cout << "Deepstream detect test :" << std::endl;

    Gst::MainLoop loop;

    Gst::Element pipeline;
    if (!pipeline.PipelineNew("video-detector"))
        return 1;

    if (!(loop.BusAddWatch(pipeline) && loop.IoAddWatch()))
        return 1;

    if (options.Rtsp())
    {
        if (!InitRtspDetector(options, pipeline))
            return 1;
    }
    else
    {
        if (!InitFileDetector(options, pipeline))
            return 1;
    }

    if (!pipeline.SetState(GST_STATE_PLAYING))
        return 1;

    loop.Run();

    pipeline.Release();

    return 0;
}
