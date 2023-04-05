#pragma once

#include <atomic>

#include "TestBase.h"

namespace Test
{
    class TestAtomicV1 : public TestBase
    {
    public:
        TestAtomicV1(const Options& options)
            : TestBase("atomic: simple spin lock", options)
            , _data(options.size)
            , _sync(0)
        {
            Run();
        }

    protected:
        virtual void Read(Data& data)
        {
            while (true)
            {
                bool expected = 0;
                if (_sync.compare_exchange_weak(expected, 1, std::memory_order_acquire))
                    break;
            }
            data.Assign(_data);
            _sync.store(0, std::memory_order_release);
        };

        virtual void Write(const Data& data)
        {
            while (true)
            {
                bool expected = 0;
                if (_sync.compare_exchange_weak(expected, 1, std::memory_order_acquire))
                    break;
            }
            _data.Assign(data);
            _sync.store(0, std::memory_order_release);
        };

    private:
        Data _data;
        std::atomic<bool> _sync;
    };

    //---------------------------------------------------------------------------------------------

    class TestAtomicV2 : public TestBase
    {
    public:
        TestAtomicV2(const Options& options)
            : TestBase("atomic: spin lock + read/write counters", options)
            , _data(options.size)
            , _sync(0)
            , _writers(0)
            , _readers(0)
        {
            Run();
        }

    protected:
        virtual void Read(Data& data)
        {
            while (true)
            {
                bool expected = 0;
                if (_sync.compare_exchange_weak(expected, 1, std::memory_order_acquire))
                    break;
            }              
            while (_writers.load());
            _readers++;
            _sync.store(0, std::memory_order_release);
            data.Assign(_data);
            _readers--;
        };

        virtual void Write(const Data& data)
        {
            while (true)
            {
                bool expected = 0;
                if (_sync.compare_exchange_weak(expected, 1, std::memory_order_acquire))
                    break;
            }
            _writers.store(1);
            while (_readers.load());
            _sync.store(0, std::memory_order_release);
            _data.Assign(data);
            _writers.store(0);
        };

    private:
        Data _data;
        std::atomic<bool> _sync;
        std::atomic<int> _readers, _writers;
    };

    //---------------------------------------------------------------------------------------------

    class TestAtomicV3 : public TestBase
    {
        static const int WRITE_BIT = 0x00010000;
    public:
        TestAtomicV3(const Options& options)
            : TestBase("atomic: spin lock + read/write mask", options)
            , _data(options.size)
            , _sync(0)
        {
            Run();
        }

    protected:
        virtual void Read(Data& data)
        {
            while (true)
            {
                int readers = _sync.load(std::memory_order_acquire) & (~WRITE_BIT);
                if (_sync.compare_exchange_weak(readers, readers + 1, std::memory_order_acquire))
                    break;
                //Sleep(0);
            }
            data.Assign(_data);
            _sync.fetch_sub(1, std::memory_order_release);
        };

        virtual void Write(const Data& data)
        {
            _sync.fetch_or(WRITE_BIT, std::memory_order_acquire);
            while (_sync.load(std::memory_order_acquire) & (~WRITE_BIT))
                Sleep(0);
            _data.Assign(data);
            _sync.store(0, std::memory_order_release);
        };

    private:
        Data _data;
        std::atomic<int> _sync;
    };
}
