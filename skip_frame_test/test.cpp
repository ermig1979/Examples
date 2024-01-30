#include <immintrin.h>
#include <assert.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <thread>
#include <mutex>
#include <deque>

#ifdef _MSC_VER
#define NOMINMAX
#include <windows.h>

__forceinline double Time()
{
    LARGE_INTEGER counter, frequency;
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&frequency);
    return double(counter.QuadPart) / double(frequency.QuadPart);
}
#else
#include <sys/time.h>

inline __attribute__((always_inline)) double Time()
{
    timeval t1;
    gettimeofday(&t1, NULL);
    return t1.tv_sec + t1.tv_usec * 0.000001;
}
#endif

#ifdef _stat
#undef _stat
#endif

inline void Sleep(double miliseconds)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(int(miliseconds)));
}

inline void Work(double seconds)
{
    double current = Time(), finish = current + seconds, sum = 1;
    while (current < finish && sum > 0)
    {
        for (int i = 0; i < 1000; ++i)
            sum += i;
        current = Time();
    }
}

inline double Random(double min, double max)
{
    return min + double(rand()) / double(RAND_MAX) * (max - min);
}

struct Frame
{
    enum State
    {
        Empty,
        Sent,
        Received,
        Decoded,
        Processed,
        Missed,
        Skipped
    } state;

    int id;
    double time;

    Frame()
        : state(Empty)
        , time(0)
        , id(-1)
    {
    }
};

class FrameQueue
{
    std::deque<Frame> _queue;
    mutable std::mutex _mutex;
public:
    FrameQueue()
    {
    }

    void Push(const Frame& frame)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _queue.push_back(frame);
    }

    size_t Size() const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _queue.size();
    }

    Frame Get() const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _queue.front();
    }

    void Pop()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _queue.pop_front();
    }

};

enum LogLevel
{
    LogNone = 0,
    LogError,
    LogWarning,
    LogInfo,
    LogVerbose,
    LogDebug
};

struct Options
{
    double sourceFrameInterval;
    double testTime;
    double receiveDelayMin, receiveDelayMax;
    double decodeTimeMin, decodeTimeMax;
    double processTimeMin, processTimeMax;
    int decoderQueueSize;
    LogLevel logLevel;

    Options()
    {
        sourceFrameInterval = 0.040;
        testTime = 10.0;
        receiveDelayMin = 0.100;
        receiveDelayMax = 0.200;
        decodeTimeMin = 0.010;
        decodeTimeMax = 0.020;
        processTimeMin = 0.150;
        processTimeMax = 0.250;
        logLevel = LogDebug;
        decoderQueueSize = 25;
    }
};

struct Stat
{
    int missed, skipped, processed;
    Stat()
    {
        missed = 0;
        skipped = 0;
        processed = 0;
    }
};

struct Scene
{
    bool inited;
    double totalSourceTime, totalDecodeTime, totalProcessTime;
    double lastSourceTime, startProcessTime, finishProcessTime;

    Scene()
        : inited(false)
    {
        totalSourceTime = 0.0;
        totalDecodeTime = 0.0; 
        totalProcessTime = 0.0;
    }

    void Start(double sourceTime)
    {
        startProcessTime = Time();
        if (inited)
        {
            totalSourceTime += sourceTime - lastSourceTime;
            totalDecodeTime += startProcessTime - finishProcessTime;
        }
        lastSourceTime = sourceTime;
        inited = true;
    }

    void Finish()
    {
        finishProcessTime = Time();
        totalProcessTime += finishProcessTime - startProcessTime;
    }

    bool NeedSkip()
    {
        return totalProcessTime + totalDecodeTime > totalSourceTime;
    }
};

class Engine
{
    const Options _options;
    FrameQueue _sourceFrames, _decoderFrames;
    double _startTime;
    std::thread _sourceThread, _receiverThread, _decoderThread;
    std::mutex _logMutex;
    bool _sourceStop;
    Scene _scene;
    Stat _stat;
public:
    Engine(const Options& options = Options())
        : _options(options)
        , _sourceStop(false)
    {
    }

    void Run()
    {
        _sourceThread = std::thread(&Engine::SourceThread, this);
        _receiverThread = std::thread(&Engine::ReceiverThread, this);
        _decoderThread = std::thread(&Engine::DecoderThread, this);

        if (_sourceThread.joinable())
            _sourceThread.join();
        if (_receiverThread.joinable())
            _receiverThread.join();
        if (_decoderThread.joinable())
            _decoderThread.join();
    }

private:

    void SourceThread()
    {
        if (_options.logLevel >= LogInfo)
        {
            std::lock_guard<std::mutex> lock(_logMutex);
            std::cout << "Source start. " << std::endl << std::flush;
        }
        _startTime = Time();
        double timeFinish = _startTime + _options.testTime;
        double nextTime = _startTime;

        int frameId = 0;
        _sourceStop = false;
        while (nextTime <= timeFinish)
        {
            double currentTime = Time();
            if (currentTime >= nextTime)
            {
                Frame frame;
                frame.time = nextTime;
                frame.id = frameId;
                frame.state = Frame::Sent;
                _sourceFrames.Push(frame);
                nextTime += _options.sourceFrameInterval;
                frameId++;
                if (_options.logLevel >= LogDebug && 0)
                {
                    std::lock_guard<std::mutex> lock(_logMutex);
                    std::cout << "Source send frame " << frame.id << " at " << int((frame.time - _startTime) * 1000.0) << std::endl << std::flush;
                }
            }
            Sleep(1.0);
        }
        _sourceStop = true;
        if (_options.logLevel >= LogInfo)
        {
            std::lock_guard<std::mutex> lock(_logMutex);
            std::cout << "Source finished. " << std::endl << std::flush;
        }
    }

    void ReceiverThread()
    {
        if (_options.logLevel >= LogInfo)
        {
            std::lock_guard<std::mutex> lock(_logMutex);
            std::cout << "Receiver start. " << std::endl << std::flush;
        }
        while (true)
        {
            Sleep(1.0);
            if (_sourceFrames.Size() == 0)
            {
                if (_sourceStop)
                    break;
                continue;
            }
            double delayTime = Random(_options.receiveDelayMin, _options.receiveDelayMax);
            double currentTime = Time();
            Frame frame = _sourceFrames.Get();
            if (currentTime >= frame.time + delayTime)
            {
                _sourceFrames.Pop();
                frame.state = Frame::Received;
                _decoderFrames.Push(frame);
                if (_options.logLevel >= LogDebug)
                {
                    std::lock_guard<std::mutex> lock(_logMutex);
                    std::cout << "Receiver received frame " << frame.id << " at " << int((currentTime - _startTime) * 1000.0) << std::endl << std::flush;
                }
            }
        }
        if (_options.logLevel >= LogInfo)
        {
            std::lock_guard<std::mutex> lock(_logMutex);
            std::cout << "Receiver finished. " << std::endl << std::flush;
        }
    }

    void DecoderThread()
    {
        if (_options.logLevel >= LogInfo)
        {
            std::lock_guard<std::mutex> lock(_logMutex);
            std::cout << "Decoder start. " << std::endl << std::flush;
        }
        while (true)
        {
            Sleep(1.0);
            if (_decoderFrames.Size() == 0)
            {
                if (_sourceStop)
                    break;
                continue;
            }
            while (_decoderFrames.Size() > _options.decoderQueueSize)
            {
                Frame frame = _decoderFrames.Get();
                _decoderFrames.Pop();
                frame.state = Frame::Missed;
                _stat.missed++;
                if (_options.logLevel >= LogWarning)
                {
                    std::lock_guard<std::mutex> lock(_logMutex);
                    std::cout << "Warning: Decoder missed frame " << frame.id << " at " << int((Time() - _startTime) * 1000.0) << std::endl << std::flush;
                }
            }
            Frame frame = _decoderFrames.Get();
            _decoderFrames.Pop();
            if (_options.logLevel >= LogDebug)
            {
                std::lock_guard<std::mutex> lock(_logMutex);
                std::cout << "Decoder start frame " << frame.id << " at " << int((Time() - _startTime) * 1000.0) << std::endl << std::flush;
            }
            Work(Random(_options.decodeTimeMin, _options.decodeTimeMax));
            if (_options.logLevel >= LogDebug)
            {
                std::lock_guard<std::mutex> lock(_logMutex);
                std::cout << "Decoder finish frame " << frame.id << " at " << int((Time() - _startTime) * 1000.0) << std::endl << std::flush;
            }
            Process(frame);
        }
        if (_options.logLevel >= LogInfo)
        {
            std::lock_guard<std::mutex> lock(_logMutex);
            std::cout << "Decoder finished. " << std::endl << std::flush;
            std::cout << "Missed frames: " << _stat.missed << std::endl << std::flush;
            std::cout << "Skipped frames: " << _stat.skipped << std::endl << std::flush;
            std::cout << "Processed frames: " << _stat.processed << std::endl << std::flush;
        }
    }

    void Process(Frame& frame)
    {
        _scene.Start(frame.time);
        if (_options.logLevel >= LogDebug)
        {
            std::lock_guard<std::mutex> lock(_logMutex);
            std::cout << "Processing start frame " << frame.id << " at " << int((Time() - _startTime) * 1000.0) << std::endl << std::flush;
        }
        if (_scene.NeedSkip())
        {
            frame.state = Frame::Skipped;
            _stat.skipped++;
            if (_options.logLevel >= LogVerbose)
            {
                std::lock_guard<std::mutex> lock(_logMutex);
                std::cout << "Skip frame " << frame.id << " at " << int((Time() - _startTime) * 1000.0) << std::endl << std::flush;
            }
        }
        else
        {
            Work(Random(_options.processTimeMin, _options.processTimeMax));
            _stat.processed++;
            frame.state = Frame::Processed;
            if (_options.logLevel >= LogVerbose)
            {
                std::lock_guard<std::mutex> lock(_logMutex);
                std::cout << "Process frame " << frame.id << " at " << int((Time() - _startTime) * 1000.0) << std::endl << std::flush;
            }
        }
        _scene.Finish();
    }
};

int main(int argc, char* argv[])
{
    Options options;
    options.testTime = 10.0;
    options.sourceFrameInterval = 0.040;
    options.receiveDelayMin = 0.100;
    options.receiveDelayMax = 0.200;
    options.decodeTimeMin = 0.010;
    options.decodeTimeMax = 0.020;
    options.processTimeMin = 0.150;
    options.processTimeMax = 0.250;
    options.decoderQueueSize = 25;
    options.logLevel = LogVerbose;

    Engine engine(options);

    engine.Run();

    return 0;
}