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

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

int frame_number = 0;

GstPadProbeReturn OsdDrawerCallback(GstPad* pad, GstPadProbeInfo* info, gpointer u_data)
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
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
    {
        NvDsFrameMeta* frame_meta = (NvDsFrameMeta*)(l_frame->data);
        int offset = 0;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) 
        {
            obj_meta = (NvDsObjectMeta*)(l_obj->data);
            if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE) 
            {
                vehicle_count++;
                num_rects++;
            }
            if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) 
            {
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

bool InitPipeline(const Options& options, Gst::Element & pipeline)
{
    Gst::Element source, demuxOrDepay, decParser, decoder, streamMuxer, detector, osdConverter, osdDrawer, encConverter, encFilter, encoder, encParser, muxer, sink;

    if (options.Rtsp())
    {
        if (!source.FactoryMake("rtspsrc", "rtsp-source"))
            return false;
        source.Set("latency", 100);

        if (!demuxOrDepay.FactoryMake("rtph264depay", "h264depay-loader"))
            return false;
    }
    else
    {
        if (!source.FactoryMake("filesrc", "file-source"))
            return false;

        if (!demuxOrDepay.FactoryMake("qtdemux", "qt-demuxer"))
            return false;
    }
    source.Set("location", options.source);

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

    if (!osdConverter.FactoryMake("nvvideoconvert", "osd-nv-video-converter"))
        return false;
    if (!osdDrawer.FactoryMake("nvdsosd", "nv-onscreendisplay"))
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
    sink.Set("location", options.output);

    if (!pipeline.BinAdd(source, demuxOrDepay, decParser, decoder))
        return false;
    if (!pipeline.BinAdd(streamMuxer, detector, osdConverter, osdDrawer))
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

    if (!Gst::StaticLink(streamMuxer, detector, osdConverter, osdDrawer))
        return false;

    if (!Gst::StaticLink(osdDrawer, encConverter, encFilter, encoder))
        return false;

    if (!Gst::StaticLink(encoder, encParser, muxer, sink))
        return false;

    if (!osdDrawer.AddPadProb(OsdDrawerCallback, "sink"))
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
