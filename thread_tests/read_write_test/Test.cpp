#include "TestStub.h"
#include "TestMutex.h"
#include "TestAtomic.h"

namespace Test
{
    class CondVarV1Test : public TestBase
    {
    public:
        CondVarV1Test(const Options& options)
            : TestBase("std::conditional_variable + std::atomic", options)
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
            //Sleep(1);
        };

    private:
        Data _data;
        std::atomic<int> _writers, _readers;
        std::mutex _mutex;
        std::condition_variable _start;
    };

    //---------------------------------------------------------------------------------------------

    class CondVarV2Test : public TestBase
    {
        static const int WRITE_BIT = 0x00010000;
    public:
        CondVarV2Test(const Options& options)
            : TestBase("std::conditional_variable + std::atomic", options)
            , _data(options.size)
            , _atomic(0)
        {
            Run();
        }

    protected:
        virtual void Read(Data& data)
        {
            if (_atomic & WRITE_BIT)
            {
                std::unique_lock<std::mutex> lock(_mutex);
                _start.wait(lock, [this] {return (_atomic & WRITE_BIT) == 0; });
            }
            _start.notify_all();
            _atomic++;
            data.Assign(_data);
            _atomic--;
            if (_atomic == 0)
                _start.notify_all();
        };

        virtual void Write(const Data& data)
        {
            _atomic.fetch_or(WRITE_BIT);
            if (_atomic & (~WRITE_BIT))
            {
                std::unique_lock<std::mutex> lock(_mutex);
                _start.wait(lock, [this] {return  (_atomic & (~WRITE_BIT)) == 0; });
            }
            //_start.notify_all();
            _data.Assign(data);
            _atomic.store(0);
            _start.notify_all();
            //Sleep(1);
        };

    private:
        Data _data;
        std::atomic<int> _atomic;
        std::mutex _mutex;
        std::condition_variable _start;
    };

    //---------------------------------------------------------------------------------------------

    class MultipleBufferTest : public TestBase
    {
    public:
        MultipleBufferTest(const Options& options)
            : TestBase("multiple buffer", options)
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

namespace Test
{
    void TestWithErrors()
    {
        Options options(256, 16, 1.0);

        TestStub stub(options);

        TestNoSync noSync(options);
    }

    void TestValidated()
    {
        Options options(256, 16, 3.0);

        TestMutex stdMutex(options);

        TestSharedMutex stdSharedMutex(options);

        //CondVarV1Test condVarV1(options);

        TestAtomicV1 atomicV1(options);

        TestAtomicV2 atomicV2(options);

        TestAtomicV3 atomicV3(options);
    }

    void TestExperiments()
    {
        Options options(256, 2, 3.0);

        CondVarV2Test condVarV2(options);
    }
}

//-------------------------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    //Test::TestWithErrors();

    //Test::TestValidated();

    Test::TestExperiments();

    return 0;
}