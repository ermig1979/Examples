#pragma once

#include "common.h"

namespace Test
{
    struct ArgsParser
    {
    public:
        ArgsParser(int argc, char* argv[])
            : _argc(argc)
            , _argv(argv)
        {
        }

        int* ArgcPtr() { return &_argc; }
        char*** ArgvPtr() { return &_argv; }

    protected:
        String GetArg(const String& name, const String& default_ = String(), bool exit = true)
        {
            return GetArgs({ name }, { default_ }, exit)[0];
        }

        String GetArg2(const String& name1, const String& name2, const String& default_ = String(), bool exit = true)
        {
            return GetArgs({ name1, name2 }, { default_ }, exit)[0];
        }

        Strings GetArgs(const String& name, const Strings& defaults = Strings(), bool exit = true)
        {
            return GetArgs(Strings({ name }), defaults, exit);
        }

        Strings GetArgs(const Strings& names, const Strings& defaults = Strings(), bool exit = true)
        {
            Strings values;
            for (int a = 1; a < _argc; ++a)
            {
                String arg = _argv[a];
                for (size_t n = 0; n < names.size(); ++n)
                {
                    const String& name = names[n];
                    if (arg.substr(0, name.size()) == name && arg.substr(name.size(), 1) == "=")
                        values.push_back(arg.substr(name.size() + 1));
                }
            }
            if (values.empty())
            {
                if (defaults.empty() && exit)
                {
                    std::cout << "Argument '";
                    for (size_t n = 0; n < names.size(); ++n)
                        std::cout << (n ? " | " : "") << names[n];
                    std::cout << "' is absent!" << std::endl;
                    ::exit(1);
                }
                else
                    return defaults;
            }
            return values;
        }

        String AppName() const
        {
            return _argv[0];
        }

    private:
        int _argc;
        char** _argv;
    };
}