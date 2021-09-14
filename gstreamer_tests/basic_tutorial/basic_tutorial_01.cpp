#include "options.h"

#include "Gst/Element.h"

namespace Test
{
    int BasicTutorial01(Options options)
    {
        if (options.source.empty())
        {
            std::cout << "Source file is not setted!" << std::endl;
            return 1;
        }

        /* Initialize GStreamer */
        gst_init(options.ArgcPtr(), options.ArgvPtr());

        Gst::Element pipeline;
        if (!pipeline.PipelineFromFile(options.source))
            return 1;

        if (!pipeline.SetState(GST_STATE_PLAYING))
            return 1;

        /* Wait until error or EOS */
        GstBus* bus = gst_element_get_bus(pipeline.Handler());
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

        return 0;
    }
}