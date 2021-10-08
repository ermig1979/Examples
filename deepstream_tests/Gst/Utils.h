#pragma once

#include "Common.h"

namespace Gst
{
    struct Element;

    bool StaticLink(Element& a, Element& b);

    bool StaticLink(Element& a, Element& b, Element& c);

    bool DynamicLink(Element & a, Element & b, const String & desc);

    String StateToString(GstState state);
}
