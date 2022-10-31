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
            const size_t SIZE = 4;
            _buffers.reserve(SIZE);
            for(size_t i = 0; i < SIZE; ++i)
                _buffers.push_back(new Buffer(_options.size));
            _current.store(0);
            _write.store(0);
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
            //while (true)
            //{
            //    bool expected = 0;
            //    if (_write.compare_exchange_weak(expected, 1, std::memory_order_acquire))
            //        break;
            //}
            Buffer* current = _buffers[_current.load()];
            current->count++;
            //_write.store(0, std::memory_order_release);
            data.Assign(current->data);
            current->count--;
        };

        virtual void Write(const Data& data)
        {
#if 0
            size_t next = _buffers.size(), curr = _current;
            do
            {
                next = (curr + 1) % _buffers.size();
                Buffer* buffer = _buffers[next];
                while (buffer->count.load());
            } while (next == _buffers.size());
            Buffer *buffer = _buffers[next];
            buffer->data.Assign(data);
            _current.store(next);
#else
            size_t curr = _current;
            for (size_t next = (curr + 1) % _buffers.size();; next = (next + 1) % _buffers.size())
            {
                if (next == curr)
                    continue;
                Buffer* buffer = _buffers[next];
                if (buffer->count.load())
                    continue;
                buffer->data.Assign(data);
                //while (true)
                //{
                //    bool expected = 0;
                //    if (_write.compare_exchange_weak(expected, 1, std::memory_order_acquire))
                //        break;
                //}
                _current.store(next);
                //_write.store(0, std::memory_order_release);
                break;
            }
#endif
        };

    private:
        struct Buffer
        {
            Data data;
            std::atomic<int> count;

            Buffer(int size)
                : data(size)
            {
                count.store(0);
            }
        };
        typedef std::vector<Buffer*> BufferPtrs;

        BufferPtrs _buffers;
        std::atomic<size_t> _current;
        std::atomic<bool> _write;
    };
}

//-------------------------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    Test::Options options(256, 16, 10.0);

    //Test::StubTest stub(options);

    //Test::ErrorTest error(options);

    //Test::StdMutexTest stdMutex(options);

    //Test::StdSharedMutexTest stdSharedMutex(options);

    //Test::CondVarAndAtomicTest condVarAndAtomic(options);

    Test::MultipleBufferTest multipleBuffer(options);

    return 0;
}