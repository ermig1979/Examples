#include "options.h"
#include "strings.h"

#include <gst/gst.h>

namespace Test
{
    int BasicTutorial01(Options options)
    {
        if (options.source.empty())
        {
            std::cout << "Source file is not setted!" << std::endl;
            return 1;
        }
        std::string pipeline_description = PathToPipelineDescription(options.source);

        /* Initialize GStreamer */
        gst_init(options.ArgcPtr(), options.ArgvPtr());

        /* Build the pipeline */
        GstElement* pipeline = gst_parse_launch(pipeline_description.c_str(), NULL);
        if (pipeline == NULL)
        {
            std::cout << "Function gst_parse_launch return NULL!" << std::endl;
            return 1;
        }

        /* Start playing */
        GstStateChangeReturn state = gst_element_set_state(pipeline, GST_STATE_PLAYING);
        if (state == GST_STATE_CHANGE_FAILURE)
        {
            std::cout << "Function gst_element_set_state is failed!" << std::endl;
            return 1;
        }

        /* Wait until error or EOS */
        GstBus* bus = gst_element_get_bus(pipeline);
        if (bus == NULL)
        {
            std::cout << "Function gst_element_get_bus return NULL!" << std::endl;
            return 1;
        }
        GstMessage* msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE, GstMessageType(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

        /* Free resources */
        if (msg != NULL)
            gst_message_unref(msg);
        gst_object_unref(bus);
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);

        return 0;
    }
}