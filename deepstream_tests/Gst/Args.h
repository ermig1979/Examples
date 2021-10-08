#pragma once

#include "Common.h"

namespace Gst
{
    struct ArgsParser
    {
    public:
        ArgsParser(int argc, char* argv[], bool alt = false)
            : _argc(argc)
            , _argv(argv)
            , _alt(alt)
        {
        }

        int* ArgcPtr() { return &_argc; }
        char*** ArgvPtr() { return &_argv; }

    protected:
        String GetArg(const String& name, const String& default_ = String(), bool exit = true, const Strings& valids = Strings())
        {
            return GetArgs({ name }, { default_ }, exit, valids)[0];
        }

        String GetArg2(const String& name1, const String& name2, const String& default_ = String(), bool exit = true, const Strings& valids = Strings())
        {
            return GetArgs({ name1, name2 }, { default_ }, exit, valids)[0];
        }

        Strings GetArgs(const String& name, const Strings& defaults = Strings(), bool exit = true, const Strings& valids = Strings())
        {
            return GetArgs(Strings({ name }), defaults, exit, valids);
        }

        Strings GetArgs(const Strings& names, const Strings& defaults = Strings(), bool exit = true, const Strings & valids = Strings())
        {
            Strings values;
            for (int a = 1; a < _argc; ++a)
            {
                String arg = _argv[a];
                for (size_t n = 0; n < names.size(); ++n)
                {
                    const String& name = names[n];
                    if (arg.substr(0, name.size()) == name)
                    {
                        String value;
                        if (_alt)
                        {
                            if(arg.substr(name.size(), 1) == "=")
                                value = arg.substr(name.size() + 1);
                        }
                        else
                        {
                            a++;
                            if (a < _argc)
                                value = _argv[a];
                        }
                        if (valids.size())
                        {
                            bool found = false;
                            for (size_t v = 0; v < valids.size() && !found; ++v)
                                if (valids[v] == value)
                                    found = true;
                            if (!found)
                            {
                                std::cout << "Argument '";
                                for (size_t i = 0; i < names.size(); ++i)
                                    std::cout << (i ? " | " : "") << names[i];
                                std::cout << "' is equal to " << value << " ! Its valid values : { ";
                                for (size_t i = 0; i < names.size(); ++i)
                                    std::cout << (i ? " | " : "") << values[i];
                                std::cout << " }." << std::endl;
                                ::exit(1);
                            }
                        }                        
                        values.push_back(value);
                    }
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
        bool _alt;
    };
}