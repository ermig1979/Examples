#include "Utils.h"
#include "Options.h"

namespace Gst
{
    gboolean BusCallback(GstBus* bus, GstMessage* msg, gpointer data)
    {
        GMainLoop* loop = (GMainLoop*)data;
        String type;
        switch (GST_MESSAGE_TYPE(msg))
        {
        case GST_MESSAGE_EOS:
            g_print("End of stream\n");
            g_main_loop_quit(loop);
            break;
        case GST_MESSAGE_ERROR:
        {
            gchar* debug;
            GError* error;
            gst_message_parse_error(msg, &error, &debug);
            g_printerr("ERROR from element %s: %s\n",
                GST_OBJECT_NAME(msg->src), error->message);
            if (debug)
                g_printerr("Error details: %s\n", debug);
            g_free(debug);
            g_error_free(error);
            g_main_loop_quit(loop);
            break;
        }
        case GST_MESSAGE_STATE_CHANGED: type = "StateChanged"; break;
        default:
            break;
        }
        if (Gst::logLevel >= Gst::LogDebug)
        {
            std::cout << "Message: ";
            if (type.empty())
            {
                std::cout << " src = " << msg->src->name;
                std::cout << " type = " << msg->type;
            }
            else
            {
                std::cout << msg->src->name << " :  \t" << type;
            }
            std::cout << std::endl << std::flush;
        }
        return TRUE;
    }

    void LinkElements(GstElement* element, GstPad* sourcePad, gpointer data)
    {
        GstElement* sinkElement = (GstElement*)data;
        if (Gst::logLevel >= Gst::LogDebug)
            g_print("Link: %s and %s: ", element->object.name, sinkElement->object.name);
        GstPad* sinkPad = gst_element_get_static_pad(sinkElement, "sink");
        if (sinkPad == NULL)
        {
            if (Gst::logLevel >= Gst::LogError)
                g_print("Error: function gst_element_get_static_pad return NULL!\n");
            return;
        }
        GstPadLinkReturn padLinkReturn = gst_pad_link(sourcePad, sinkPad);
        if (padLinkReturn != GST_PAD_LINK_OK)
        {
            if (padLinkReturn != GST_PAD_LINK_WAS_LINKED)
            {
                if (Gst::logLevel >= Gst::LogError)
                    g_print("Error: Can't link %s and %s (error = %d) !\n", element->object.name, sinkElement->object.name, padLinkReturn);
                return;
            }
            else
            {
                if (Gst::logLevel >= Gst::LogWarning)
                    g_print("Warning: %s and %s already was linked! ", element->object.name, sinkElement->object.name);
            }
        }
        gst_object_unref(sinkPad);
        if (Gst::logLevel >= Gst::LogDebug)
            g_print("OK. \n", element->object.name);
    }
}
