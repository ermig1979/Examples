#pragma once

#include "args.h"

namespace Test
{
    struct Options : public ArgsParser
    {
        Strings includeFilter;
        Strings excludeFilter;
        String source;

        Options(int argc, char* argv[])
            : ArgsParser(argc, argv)
        {
            includeFilter = GetArgs("-if", Strings(), false);
            excludeFilter = GetArgs("-ef", Strings(), false);
            source = GetArg("-s", "", false);
        }
    };
}