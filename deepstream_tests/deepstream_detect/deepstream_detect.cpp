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
    Gst::String detectorConfig;
    Gst::String output;
    Gst::String encoderType;
    int bitrate;

    Options(int argc, char* argv[])
        : Gst::Options(argc, argv)
    {
        source = GetArg2("-s", "--source");
        detectorConfig = GetArg2("-dc", "--detectorConfig", "./data/detect_0/config.txt", false);
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

bool InitPipeline(const Options& options, Gst::Element & pipeline)
{
    Gst::Element source, demuxOrDepay, decParser, decoder, streamMuxer, detector, annConverter, encConverter, encFilter, encoder, encParser, muxer, sink;

    if (options.Rtsp())
    {
        if (!source.FactoryMake("rtspsrc", "rtsp-source"))
            return false;
        source.Set("latency", 100);
        source.Set("location", options.source);

        if (!demuxOrDepay.FactoryMake("rtph264depay", "h264depay-loader"))
            return false;
    }
    else
    {
        if (!source.FactoryMake("filesrc", "file-source"))
            return false;
        source.Set("location", options.source);

        if (!demuxOrDepay.FactoryMake("qtdemux", "qt-demuxer"))
            return false;
    }

    if (!decParser.FactoryMake("h264parse", "h264parse-decoder"))
        return false;

    if (!decoder.FactoryMake("nvv4l2decoder", "nvv4l2-decoder"))
        return false;

    if (!streamMuxer.FactoryMake("nvstreammux", "stream-muxer"))
        return false;
    streamMuxer.Set("batch-size", 1);
    streamMuxer.Set("height", 1080);
    streamMuxer.Set("width", 1920);

    if (!detector.FactoryMake("nvinfer", "nvinference-engine"))
        return false;
    if (!(Gst::IsFileExist(options.detectorConfig) && detector.Set("config-file-path", options.detectorConfig)))
        return false;

    if (!annConverter.FactoryMake("nvvideoconvert", "annotator-nv-video-converter"))
        return false;

    if (!encFilter.FactoryMake("capsfilter", "enc-caps-filter"))
        return false;
    if (options.encoderType == "soft")
    {
        if (!encConverter.FactoryMake("videoconvert", "encoder-video-converter"))
            return false;
        if (!encFilter.SetCapsFromString("video/x-raw, format=I420"))
            return false;
        if (!encoder.FactoryMake("x264enc", "x264enc-encoder"))
            return false;
    }
    else
    {
        if (!encConverter.FactoryMake("nvvideoconvert", "encoder-nv-video-converter"))
            return false;
        if (!encFilter.SetCapsFromString("video/x-raw(memory:NVMM), format=I420"))
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

    if (!pipeline.BinAdd(source, demuxOrDepay, decParser, decoder))
        return false;
    if (!pipeline.BinAdd(streamMuxer, detector, annConverter))
        return false;
    if (!pipeline.BinAdd(encConverter, encFilter, encoder, encParser, muxer, sink))
        return false;

    if (options.Rtsp())
    {
        if (!Gst::DynamicLink(source, demuxOrDepay))
            return false;
        if (!Gst::StaticLink(demuxOrDepay, decParser))
            return false;
    }
    else
    {
        if (!Gst::StaticLink(source, demuxOrDepay))
            return false;
        if (!Gst::DynamicLink(demuxOrDepay, decParser))
            return false;
    }

    if (!Gst::StaticLink(decParser, decoder))
        return false;

    if (!Gst::PadLink(decoder, "src", streamMuxer, "sink_0"))
        return false;

    if (!Gst::StaticLink(streamMuxer, detector, annConverter))
        return false;

    if (!Gst::StaticLink(annConverter, encConverter, encFilter, encoder))
        return false;

    if (!Gst::StaticLink(encoder, encParser, muxer, sink))
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

    if (!InitPipeline(options, pipeline))
        return 1;

    if (!pipeline.SetState(GST_STATE_PLAYING))
        return 1;

    loop.Run();

    pipeline.Release();

    return 0;
}
