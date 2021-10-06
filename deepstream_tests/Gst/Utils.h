#pragma once

#include "Common.h"

namespace Gst
{
    gboolean BusCallback(GstBus* bus, GstMessage* msg, gpointer data);

    void LinkElements(GstElement* element, GstPad* sourcePad, gpointer data);
}
