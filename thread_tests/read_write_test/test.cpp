#include "defs.h"

#include <shared_mutex>
#include <mutex>
#include <atomic>

class StubTest : public Test
{
public:
    StubTest(const Options& options)
        : Test("stub", options)
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

class ErrorTest : public Test
{
public:
    ErrorTest(const Options& options)
        : Test("error", options)
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

class StdMutexTest : public Test
{
public:
    StdMutexTest(const Options& options)
        : Test("std::mutex", options)
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

class StdSharedMutexTest : public Test
{
public:
    StdSharedMutexTest(const Options& options)
        : Test("std::shared_mutex", options)
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

class CondVarAndAtomicTest : public Test
{
public:
    CondVarAndAtomicTest(const Options& options)
        : Test("std::conditional_variable + std::atomic", options)
        , _data(options.size)
        , _writers(0)
        , _readers(0)
    {
        Run();
    }

protected:
    virtual void Read(Data& data)
    {
        if (_writers > 0)
        {
            std::unique_lock<std::mutex> lock(_mutex);
            _start.wait(lock, [this] {return _writers <= 0; });
        }
        _readers++;
        data.Assign(_data);
        _readers--;
        _start.notify_all();
    };

    virtual void Write(const Data& data)
    {
        _writers++;
        if (_readers > 0)
        {
            std::unique_lock<std::mutex> lock(_mutex);
            _start.wait(lock, [this] {return _readers <= 0; });
        }
        _data.Assign(data);
        _writers--;
        _start.notify_all();
    };

private:
    Data _data;
    std::atomic<int> _writers, _readers;
    std::mutex _mutex;
    std::condition_variable _start;
};

int main(int argc, char* argv[])
{
    Options options(1024, 16, 5.0);

    //StubTest stub(options);

    //ErrorTest error(options);

    StdMutexTest stdMutex(options);

    StdSharedMutexTest stdSharedMutex(options);

    CondVarAndAtomicTest condVarAndAtomic(options);

    return 0;
}