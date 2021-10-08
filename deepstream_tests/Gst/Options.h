#pragma once

#include "Args.h"

namespace Gst
{
    struct Options : public ArgsParser
    {
        Options(int argc, char* argv[])
            : ArgsParser(argc, argv, true)
        {
            logLevel = (LogLevel)FromString<int>(GetArg2("-ll", "--logLevel", "0", false));
        }
    };
}
