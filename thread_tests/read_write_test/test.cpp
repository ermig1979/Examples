#include "Test.h"

#include <shared_mutex>
#include <mutex>
#include <atomic>

namespace Test
{
    class StubTest : public BaseTest
    {
    public:
        StubTest(const Options& options)
            : BaseTest("stub", options)
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

    class ErrorTest : public BaseTest
    {
    public:
        ErrorTest(const Options& options)
            : BaseTest("error", options)
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

    //---------------------------------------------------------------------------------------------

    class StdMutexTest : public BaseTest
    {
    public:
        StdMutexTest(const Options& options)
            : BaseTest("std::mutex", options)
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

    class StdSharedMutexTest : public BaseTest
    {
    public:
        StdSharedMutexTest(const Options& options)
            : BaseTest("std::shared_mutex", options)
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

    //---------------------------------------------------------------------------------------------

    class CondVarAndAtomicTest : public BaseTest
    {
    public:
        CondVarAndAtomicTest(const Options& options)
            : BaseTest("std::conditional_variable + std::atomic", options)
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
            if (_readers <= 0)
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
            Sleep(1);
        };

    private:
        Data _data;
        std::atomic<int> _writers, _readers;
        std::mutex _mutex;
        std::condition_variable _start;
    };

    //---------------------------------------------------------------------------------------------

    class MultipleBufferTest : public BaseTest
    {
    public:
        MultipleBufferTest(const Options& options)
            : BaseTest("multiple buffer", options)
        {
            _buffers.reserve(32);
            _buffers.push_back(new Buffer(_options.size));
            _current = 0;
            Run();
        }

        virtual ~MultipleBufferTest()
        {
            std::cout << " Buffer count: " << _buffers.size() << std::endl;
            for (size_t i = 0; i < _buffers.size(); ++i)
                delete _buffers[i];
            _buffers.clear();
        }

    protected:
        virtual void Read(Data& data)
        {
            Buffer* buffer = _buffers[_current];
            buffer->readers++;
            data.Assign(buffer->data);
            buffer->readers--;
        };

        virtual void Write(const Data& data)
        {
            size_t next = 0, curr = _current;
            for (; next < _buffers.size(); ++next)
            {
                if (next == curr)
                    continue;
                Buffer* buffer = _buffers[next];
                if (buffer->readers <= 0)
                    break;
            }
            if(next == _buffers.size())
                _buffers.push_back(new Buffer(_options.size));
            Buffer* buffer = _buffers[next];
            int readers = buffer->readers;
            if(readers != 0)
                std::cout << " MultipleBufferTest::Write(): _buffers[" << next
                << "]->readers = " << readers << " _current = " << curr << std::endl;
            buffer->data.Assign(data);
            _current = (int)next;
        };

    private:
        struct Buffer
        {
            Data data;
            std::atomic<int> readers;

            Buffer(int size)
                : data(size)
            {
                readers = 0;
            }
        };
        typedef std::vector<Buffer*> BufferPtrs;

        BufferPtrs _buffers;
        std::atomic<int> _current;
    };
}

//-------------------------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    Test::Options options(256, 16, 1.0);

    //Test::StubTest stub(options);

    //Test::ErrorTest error(options);

    Test::StdMutexTest stdMutex(options);

    Test::StdSharedMutexTest stdSharedMutex(options);

    Test::CondVarAndAtomicTest condVarAndAtomic(options);

    Test::MultipleBufferTest multipleBuffer(options);

    return 0;
}