#pragma once

#include "Utils.h"
#include "Data.h"

namespace Test
{
    struct Options
    {
        int size, count;
        double time;
        Options(int s = 1024, int c = 10, double t = 1.0)
            : size(s)
            , count(c)
            , time(t)
        {
        }
    }; 

    class BaseTest
    {
    public:
        typedef int Int;
        typedef std::vector<Int> Ints;
        typedef std::thread Thread;
        typedef std::vector<Thread> Threads;

    public:

        BaseTest(const String& name, const Options& options)
            : _name(name)
            , _options(options)
            , _finish(0)
            , _writeCount(0)
        {
        }

        void Run()
        {
            std::cout << "Start test '" << _name << "' read " << _options.size << " B for " << _options.time * 1000;
            std::cout << " ms in " << _options.count << " threads: " << std::endl;

            _writeCount = 0;
            _readCounts.resize(_options.count, 0);
            _finish = Time() + _options.time;
            _writeThread = Thread(&BaseTest::WriteTask, this);
            _readThreads.reserve(_options.count);
            for (int i = 0; i < _options.count; ++i)
                _readThreads.emplace_back(Thread(&BaseTest::ReadTask, this, i));

            if (_writeThread.joinable())
                _writeThread.join();
            for (int i = 0; i < _options.count; ++i)
            {
                if (_readThreads[i].joinable())
                    _readThreads[i].join();
            }

            Int readCount = 0;
            for (int i = 0; i < _options.count; ++i)
                readCount += _readCounts[i];

            std::cout << " Write count: " << PrettyString(_writeCount) << std::endl;
            std::cout << " Read count: " << PrettyString(readCount) << std::endl;
            std::cout << "End test '" << _name << "' " << std::endl << std::endl;
        }

        virtual ~BaseTest()
        {
        }

    protected:

        virtual void Read(Data& data) = 0;

        virtual void Write(const Data& data) = 0;

        void WriteTask()
        {
            Data data(_options.size);
            double current = Time();
            while (current < _finish)
            {
                data.Set(rand());
                Write(data);
                Sleep(0);
                current = Time();
                _writeCount++;
            }
        };

        void ReadTask(int id)
        {
            Data data(_options.size);
            double current = Time();
            while (current < _finish)
            {
                Read(data);
                if (!data.Valid())
                {
                    std::cout << " Read error in thread " << id << "!" << std::endl;
                    return;
                }
                _readCounts[id]++;
                Sleep(0);
                current = Time();
            }
        };

        String _name;
        Options _options;
        double _finish;
        Int _writeCount;
        Ints _readCounts;
        Thread _writeThread;
        Threads _readThreads;
    };
}
