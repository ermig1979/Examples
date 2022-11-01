#pragma once

#include <shared_mutex>
#include <mutex>

#include "TestBase.h"

namespace Test
{
    class TestMutex : public TestBase
    {
    public:
        TestMutex(const Options& options)
            : TestBase("std::mutex", options)
            , _data(options.size)
        {
            Run();
        }

    protected:
        virtual void Read(Data& data)
        {
            std::lock_guard<std::mutex> lock(_mutex);
            data.Assign(_data);
        };

        virtual void Write(const Data& data)
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _data.Assign(data);
        };

    private:
        Data _data;
        std::mutex _mutex;
    };

    //---------------------------------------------------------------------------------------------

    class TestSharedMutex : public TestBase
    {
    public:
        TestSharedMutex(const Options& options)
            : TestBase("std::shared_mutex", options)
            , _data(options.size)
        {
            Run();
        }

    protected:
        virtual void Read(Data& data)
        {
            std::shared_lock<std::shared_mutex> lock(_mutex);
            data.Assign(_data);
        };

        virtual void Write(const Data& data)
        {
            std::unique_lock<std::shared_mutex> lock(_mutex);
            _data.Assign(data);
        };

    private:
        Data _data;
        std::shared_mutex _mutex;
    };
}
