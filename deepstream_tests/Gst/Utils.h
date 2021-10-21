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

    String ToString(GstState state);

    String ToString(GstMessageType type);

    String ToString(int value, int width);

    String ExpandToLeft(const String& value, size_t count);

    String ExpandToRight(const String& value, size_t count);

    bool IsFileExist(const String& name);
}
