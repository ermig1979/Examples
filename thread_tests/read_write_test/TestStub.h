#pragma once

#include "TestBase.h"

namespace Test
{
    class TestStub : public TestBase
    {
    public:
        TestStub(const Options& options)
            : TestBase("stub", options)
        {
            Run();
        }

    protected:
        virtual void Read(Data& data)
        {
        };

        virtual void Write(const Data& data)
        {
        };
    };

    //---------------------------------------------------------------------------------------------

    class TestNoSync : public TestBase
    {
    public:
        TestNoSync(const Options& options)
            : TestBase("no sync", options)
            , _data(options.size)
        {
            Run();
        }

    protected:
        virtual void Read(Data& data)
        {
            data.Assign(_data);
        };

        virtual void Write(const Data& data)
        {
            _data.Assign(data);
        };

    private:
        Data _data;
    };
}
