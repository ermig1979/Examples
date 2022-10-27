#pragma once

#include <assert.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <array>
#include <thread>
#include <vector>
#include <chrono>

inline double Time()
{
    using namespace std::chrono;
    const static time_point<high_resolution_clock> start = high_resolution_clock::now();
    time_point<high_resolution_clock> time = high_resolution_clock::now();
    return duration<double>(time - start).count();
}

inline void Sleep(unsigned int miliseconds)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(miliseconds));
}

struct Data
{
    std::vector< uint8_t> data;

    inline Data(int size, uint8_t val = 0)
        : data(size, val)
    {
    }

    inline bool Valid() const
    {
        uint8_t val0 = data[0];
        for (size_t i = 0; i < data.size(); ++i)
        {
            if (data[i] != val0)
                return false;
        }
        return true;
    }

    inline void Set(uint8_t val)
    {
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = val;
    }

    inline void Assign(const Data& src)
    {
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = src.data[i];
    }
};

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

class Test
{
public:
    typedef int Int;
    typedef std::vector<Int> Ints;
    typedef std::string String;
    typedef std::thread Thread;
    typedef std::vector<Thread> Threads;

public:

    Test(const String & name, const Options & options)
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
        _writeThread = Thread(&Test::WriteTask, this);
        _readThreads.reserve(_options.count);
        for (int i = 0; i < _options.count; ++i)
            _readThreads.emplace_back(Thread(&Test::ReadTask, this, i));

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

        std::cout << " Write count: " << _writeCount << std::endl;
        std::cout << " Read count: " << readCount << std::endl;
        std::cout << "End test '" << _name << "' " << std::endl << std::endl;
    }

    virtual ~Test()
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
            Sleep(1);
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
