#pragma once

#include "common.h"

namespace Test
{
    inline String PathToPipelineDescription(const String& path)
    {
        return "playbin uri=file:///" + path;
    }
}
