#include "Utils.h"
#include "Options.h"
#include "Element.h"

namespace Gst
{
    bool StaticLink(Element& a, Element& b)
    {
        if (gst_element_link(a.Handle(), b.Handle()) == FALSE)
        {
            if (Gst::logLevel >= Gst::LogError)
                std::cout << "Can't link '" << a.Name() << "' and '" << b.Name() << "'!" << std::endl;
            return false;
        }
        else
        {
            if (Gst::logLevel >= Gst::LogDebug)
                std::cout << "Link '" << a.Name() << "' and '" << b.Name() << "'." << std::endl;
            return true;
        }
    }

    bool StaticLink(Element& a, Element& b, Element& c)
    {
        return StaticLink(a, b) && StaticLink(b, c);
    }

    bool StaticLink(Element& a, Element& b, Element& c, Element& d)
    {
        return StaticLink(a, b) && StaticLink(b, c) && StaticLink(c, d);
    }

    static void DynamicLinkCallback(GstElement* element, GstPad* sourcePad, gpointer data)
    {
        GstElement* sinkElement = (GstElement*)data;
        if (Gst::logLevel >= Gst::LogDebug)
            g_print("Dynamic link: %s and %s: ", element->object.name, sinkElement->object.name);
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
            g_print("OK. \n");
    }

    bool DynamicLink(Element & a, Element & b)
    {
        gulong id = g_signal_connect_data(a.Handle(), "pad-added", G_CALLBACK(DynamicLinkCallback), b.Handle(), NULL, GConnectFlags(0));
        if (id == 0 && Gst::logLevel >= Gst::LogError)
            std::cout << "Can't set dynamic link between '" << a.Name() << "' and '" << b.Name() << "'!" << std::endl;
        else
        {
            if (Gst::logLevel >= Gst::LogDebug)
                std::cout << "Set dynamic link between '" << a.Name() << "' and '" << b.Name() << "'." << std::endl;
        }
        return id != 0;
    }

    String StateToString(GstState state)
    {
        String names[6] = { "Undefined", "Null", "Ready", "Paused", "Playing", "Unknown"};
        return state >= GST_STATE_VOID_PENDING && state <= GST_STATE_PLAYING ? names[(int)state] : names[5];
    }
}
