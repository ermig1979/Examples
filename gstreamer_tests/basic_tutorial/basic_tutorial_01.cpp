#include "basic_tutorial.h"

#include <gst/gst.h>

int basic_tutorial_01(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Error : Too few arguments!" << std::endl;
        return 1;
    }
    std::string pipeline_description = path_to_pipeline_description(argv[1]);

    GstElement* pipeline;
    GstBus* bus;
    GstMessage* msg;

    /* Initialize GStreamer */
    gst_init(&argc, &argv);

    /* Build the pipeline */
    pipeline = gst_parse_launch(pipeline_description.c_str(), NULL);

    /* Start playing */
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    /* Wait until error or EOS */
    bus = gst_element_get_bus(pipeline);
    msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE,
            GstMessageType(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

    /* Free resources */
    if (msg != NULL)
        gst_message_unref(msg);
    gst_object_unref(bus);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    return 0;
}