#include <stdio.h>
#include <unistd.h>

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

    bool PadLink(Element& a, const String& aSrcName, Element& b, const String& bSinkName)
    {
        bool result = false;
        GstPad* srcPad = gst_element_get_static_pad(a.Handle(), aSrcName.c_str());
        if (srcPad == NULL) 
        {
            if (Gst::logLevel >= Gst::LogError)
                std::cout << "Can't get static pad '" << aSrcName << "' from element '" << a.Name() << "'!" << std::endl;
        }
        else
        {
            GstPad * sinkPad = gst_element_get_request_pad(b.Handle(), bSinkName.c_str());
            if (sinkPad == NULL) 
            {
                if (Gst::logLevel >= Gst::LogError)
                    std::cout << "Can't get request pad '" << bSinkName << "' from element '" << b.Name() << "'!" << std::endl;
            }
            else
            {
                if (gst_pad_link(srcPad, sinkPad) != GST_PAD_LINK_OK)
                {
                    if (Gst::logLevel >= Gst::LogError)
                        std::cout << "Can't make pad link between '" << a.Name() << "'->" << aSrcName << " and '" << b.Name() << "'->" << bSinkName << " !" << std::endl;
                }
                else
                {
                    if (Gst::logLevel >= Gst::LogDebug)
                        std::cout << "Make pad link between '" << a.Name() << "'->" << aSrcName << " and '" << b.Name() << "'->" << bSinkName << " ." << std::endl;
                    result = true;
                }
                gst_object_unref(sinkPad);
            }
            gst_object_unref(srcPad);
        }
        return result;
    }

    String StateToString(GstState state)
    {
        String names[6] = { "Undefined", "Null", "Ready", "Paused", "Playing", "Unknown"};
        return state >= GST_STATE_VOID_PENDING && state <= GST_STATE_PLAYING ? names[(int)state] : names[5];
    }

    bool IsFileExist(const String& name)
    {
        if (access(name.c_str(), F_OK) == -1)
        {
            if (Gst::logLevel >= Gst::LogError)
            {
                std::cout << "File '" << name << "' is not exist!" << std::endl;
                std::cout << "Current directory: " << std::flush;
                system("pwd");
            }
            return false;
        }
        return true;
    }
}
