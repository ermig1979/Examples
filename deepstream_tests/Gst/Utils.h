#pragma once

#include "Common.h"

namespace Gst
{
    struct Element;

    bool StaticLink(Element& a, Element& b);

    bool StaticLink(Element& a, Element& b, Element& c);

    bool StaticLink(Element& a, Element& b, Element& c, Element& d);

    bool DynamicLink(Element & a, Element & b);

    bool PadLink(Element& a, const String & aSrcName, Element& b, const String & bSinkName);

    String StateToString(GstState state);

    String ToString(int value, int width);

    bool IsFileExist(const String& name);
}
