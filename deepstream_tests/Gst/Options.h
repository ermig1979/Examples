#pragma once

#include "Args.h"

namespace Gst
{
    struct Options : public ArgsParser
    {
        String source;

        Options(int argc, char* argv[])
            : ArgsParser(argc, argv, true)
        {
            source = GetArg2("-s", "--source");
            logLevel = (LogLevel)FromString<int>(GetArg2("-l", "--logLevel", "0", false));
        }
    };
}
