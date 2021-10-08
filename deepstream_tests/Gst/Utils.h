#pragma once

#include "Common.h"

namespace Gst
{
    struct Element;

    bool StaticLink(Element& a, Element& b);

    bool StaticLink(Element& a, Element& b, Element& c);

    bool StaticLink(Element& a, Element& b, Element& c, Element& d);

    bool DynamicLink(Element & a, Element & b);

    String StateToString(GstState state);
}
