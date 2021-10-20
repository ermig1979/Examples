#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include "gstnvdsmeta.h"
#include "gst-nvmessage.h"

#include "Gst/Element.h"
#include "Gst/Options.h"
#include "Gst/Utils.h"
#include "Gst/MainLoop.h"
#include "Gst/SourceBin.h"

struct Options : Gst::Options
{
    Gst::Strings sources;
    Gst::String decoderType;
    Gst::String detectorConfig;
    int logFrameRate;
    Gst::String output;
    Gst::String encoderType;
    int bitrate;

    Options(int argc, char* argv[])
        : Gst::Options(argc, argv)
    {
        for (size_t i = 0;; ++i)
        {
            Gst::String shortName = Gst::String("-s") + std::to_string(i);
            Gst::String longName = Gst::String("--source") + std::to_string(i);
            if (HasArg(shortName, longName))
                sources.push_back(GetArg2(shortName, longName));
            else
                break;
        }
        detectorConfig = GetArg2("-dc", "--detectorConfig", "./data/detect_0/config.txt", false);
        logFrameRate = Gst::FromString<int>(GetArg2("-lfr", "--logFrameRate", "30", false));
        output = GetArg2("-o", "--output");
        encoderType = GetArg2("-et", "--encoderType", "hard", false, { "hard", "soft" });
        bitrate = Gst::FromString<int>(GetArg2("-b", "--bitrate", "5000", false))*1000;
    }

    ~Options()
    {
    }

private:
};

static GstPadProbeReturn TilerSrcPadBufferProbe(GstPad* pad, GstPadProbeInfo* info, gpointer u_data)
{
    const Options& options = *(const Options*)u_data;
    size_t i = 0, n = options.sources.size();
    NvDsBatchMeta* batchMeta = gst_buffer_get_nvds_batch_meta((GstBuffer*)info->data);
    for (NvDsMetaList* frame = batchMeta->frame_meta_list; frame != NULL; frame = frame->next, i++)
    {
        NvDsFrameMeta* frameMeta = (NvDsFrameMeta*)(frame->data);
        guint total = 0, vehicles = 0, persons = 0;
        for (NvDsMetaList* obj = frameMeta->obj_meta_list; obj != NULL; obj = obj->next)
        {
            NvDsObjectMeta* meta = (NvDsObjectMeta*)(obj->data);
            if (meta->class_id == 0)
            {
                vehicles++;
                total++;
            }
            if (meta->class_id == 2)
            {
                persons++;
                total++;
            }
        }
        if (Gst::logLevel >= Gst::LogInfo && frameMeta->frame_num % options.logFrameRate == 0)
        {
            if(i == 0)
                std::cout << "Frame[" << frameMeta->frame_num << "]:";
            std::cout << " Src-" << i;
            std::cout << " (T: " << total;
            std::cout << ", V: " << vehicles;
            std::cout << ", P: " << persons;
            if(i == n - 1)
                std::cout << ")." << std::endl;
            else
                std::cout << "),  ";
        }    
    }
    return GST_PAD_PROBE_OK;
}

bool InitPipeline(const Options& options, Gst::Element & pipeline)
{
    Gst::Element streamMuxer, detector, tiler, osdConverter, osdDrawer, encConverter, encFilter, encoder, encParser, muxer, sink, queue[5];

    if (!streamMuxer.FactoryMake("nvstreammux", "stream-muxer"))
        return false;
    streamMuxer.Set("batch-size", (int)options.sources.size());
    streamMuxer.Set("height", 1080);
    streamMuxer.Set("width", 1920);
    if (!pipeline.BinAdd(streamMuxer))
        return false;

    for (size_t i = 0; i < 5; ++i)
    {
        if (!queue[i].FactoryMake("queue", Gst::String("queue-") + std::to_string(i)))
            return false;
        if (!pipeline.BinAdd(queue[i]))
            return false;
    }

    for (size_t i = 0; i < options.sources.size(); ++i)
    {
        Gst::SourceBin sourceBin;
        if (!sourceBin.CreateSourceBin(options.sources[i], i))
            return false;
        if (!pipeline.BinAdd(sourceBin))
            return false;
        if (!Gst::PadLink(sourceBin, "src", streamMuxer, Gst::String("sink_") + std::to_string(i)))
            return false;
    }

    if (!detector.FactoryMake("nvinfer", "nvinference-engine"))
        return false;
    if (!(Gst::IsFileExist(options.detectorConfig) && detector.Set("config-file-path", options.detectorConfig)))
        return false;
    int batchSize;
    if (!detector.Get("batch-size", batchSize))
        return false;
    if (batchSize != options.sources.size())
    {
        if (Gst::logLevel >= Gst::LogWarning)
            std::cout << "Warning: Override batch size " << batchSize << " from config to " << options.sources.size() << " !" << std::endl;
        detector.Set("batch-size", options.sources.size());
    }
    if (!pipeline.BinAdd(detector))
        return false;

    if (!tiler.FactoryMake("nvmultistreamtiler", "nv-tiler"))
        return false;
    guint tilerRows = (guint)::sqrt((double)options.sources.size());
    guint tilerCols = (guint)::ceil(1.0 * options.sources.size() / tilerRows);
    tiler.Set("rows", tilerRows);
    tiler.Set("columns", tilerCols);
    tiler.Set("height", 1080);
    tiler.Set("width", 1920);
    if (!pipeline.BinAdd(tiler))
        return false;

    if (!osdConverter.FactoryMake("nvvideoconvert", "osd-nv-video-converter"))
        return false;
    if (!pipeline.BinAdd(osdConverter))
        return false;

    if (!osdDrawer.FactoryMake("nvdsosd", "nv-onscreendisplay"))
        return false;
    osdDrawer.Set("process-mode", 0);
    osdDrawer.Set("display-text", 0);
    if (!pipeline.BinAdd(osdDrawer))
        return false;
     
    if (!encFilter.FactoryMake("capsfilter", "enc-caps-filter"))
        return false;
    if (options.encoderType == "soft")
    {
        if (!encConverter.FactoryMake("videoconvert", "enc-video-converter"))
            return false;
        if (!encFilter.SetCapsFromString("video/x-raw, format=I420"))
            return false;
        if (!encoder.FactoryMake("x264enc", "x264enc-encoder"))
            return false;
    }
    else
    {
        if (!encConverter.FactoryMake("nvvideoconvert", "enc-nv-video-converter"))
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
    sink.Set("qos", 0); // ?
    sink.Set("location", options.output);
    if (!pipeline.BinAdd(encConverter, encFilter, encoder, encParser, muxer, sink))
        return false;

    if (!Gst::StaticLink(streamMuxer, queue[0], detector, queue[1]))
        return false;

    if (!Gst::StaticLink(queue[1], tiler, queue[2], osdConverter))
        return false;

    if (!Gst::StaticLink(osdConverter, queue[3], osdDrawer, queue[4]))
        return false;

    if (!Gst::StaticLink(queue[4], encConverter, encFilter, encoder))
        return false;

    if (!Gst::StaticLink(encoder, encParser, muxer, sink))
        return false;

    if (!detector.AddPadProb("src", TilerSrcPadBufferProbe, (void*)&options))
        return false;

    return true;
}

int main(int argc, char* argv[])
{
    Options options(argc, argv);

    gst_init(&argc, &argv);

    std::cout << "Deepstream multi detect test :" << std::endl;

    Gst::MainLoop loop;

    Gst::Element pipeline;
    if (!pipeline.PipelineNew("video-multi-detector"))
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
