#include "options.h"

#include <gst/gst.h>

#include "Gst/Pipeline.h"

namespace Test
{
    int BasicTutorial02(Options options)
    {
        GstBus* bus;
        GstMessage* msg;
        GstStateChangeReturn ret;

        /* Initialize GStreamer */
        gst_init(options.ArgcPtr(), options.ArgvPtr());

        Gst::Pipeline pipeline;
        if (!pipeline.InitNew("test-pipeline"))
            return 1;

        /* Create the elements */
        GstElement* source = gst_element_factory_make("videotestsrc", "source");
        GstElement* sink = gst_element_factory_make("autovideosink", "sink");

        if (!source || !sink) 
        {
            g_printerr("Not all elements could be created.\n");
            return -1;
        }

        /* Build the pipeline */
        gst_bin_add_many(GST_BIN(pipeline.Handler()), source, sink, NULL);
        if (gst_element_link(source, sink) == FALSE)
        {
            g_printerr("Elements could not be linked.\n");
            return -1;
        }

        /* Modify the source's properties */
        g_object_set(source, "pattern", 0, NULL);

        if (!pipeline.Play())
            return 1;

        /* Wait until error or EOS */
        bus = gst_element_get_bus(pipeline.Handler());
        msg =
            gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE,
                GstMessageType(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

        /* Parse message */
        if (msg != NULL) 
        {
            GError* err;
            gchar* debug_info;

            switch (GST_MESSAGE_TYPE(msg)) 
            {
            case GST_MESSAGE_ERROR:
                gst_message_parse_error(msg, &err, &debug_info);
                g_printerr("Error received from element %s: %s\n",
                    GST_OBJECT_NAME(msg->src), err->message);
                g_printerr("Debugging information: %s\n",
                    debug_info ? debug_info : "none");
                g_clear_error(&err);
                g_free(debug_info);
                break;
            case GST_MESSAGE_EOS:
                g_print("End-Of-Stream reached.\n");
                break;
            default:
                /* We should not reach here because we only asked for ERRORs and EOS */
                g_printerr("Unexpected message received.\n");
                break;
            }
            gst_message_unref(msg);
        }

        /* Free resources */
        gst_object_unref(bus);

        return 0;
    }
}