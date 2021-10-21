#pragma once

#include <iostream>
#include <sstream>
#include <vector>

#include <assert.h>

#include <gst/gst.h>

namespace Gst
{
    typedef std::string String;
    typedef std::vector<String> Strings;
    typedef std::vector<int> Ints;

    enum LogLevel
    {
        LogNone,
        LogError,
        LogWarning,
        LogInfo,
        LogDebug,
    };

    extern LogLevel logLevel;

    template <class T> inline T FromString(const String& str)
    {
        std::stringstream ss(str);
        T t;
        ss >> t;
        return t;
    }
}
