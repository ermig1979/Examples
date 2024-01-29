#pragma once

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
    double sourceInterval;
    double testTime;
    double receiveDelayMin, receiveDelayMax;
    double decodeTimeMin, decodeTimeMax;
    double processTimeMin, processTimeMax;
    int decoderQueueSize;
    LogLevel logLevel;

    Options()
    {
        sourceInterval = 0.040;
        testTime = 10.0;
        receiveDelayMin = 0.100;
        receiveDelayMax = 0.200;
        decodeTimeMin = 0.015;
        decodeTimeMax = 0.025;
        processTimeMin = 0.150;
        processTimeMax = 0.250;
        logLevel = LogVerbose;
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
    bool started, finished, inited;
    double startTime, lastSourceTime, lastStartTime, lastProcessingTime, lastFinishTime, sourceTime, decodeTime, processTime;

    Scene()
        : inited(false)
        , started(false)
        , finished(false)
    {
        sourceTime = 0.0;
        decodeTime = 0.0;
        processTime = 0.0;
    }

    void Start(double time)
    {
        startTime = Time();
        if (inited)
        {
            sourceTime += time - lastSourceTime;
            decodeTime += startTime - lastFinishTime;
        }
        lastSourceTime = time;
        lastStartTime = startTime;
        inited = true;
    }

    void Finish()
    {
         double finishTime = Time();       
         processTime += finishTime - startTime;
         lastFinishTime = finishTime;
    }

    void Skip()
    {
        lastFinishTime = startTime;
    }

    bool NeedSkip() 
    {
        return processTime > sourceTime * 0.5;
    }
};

class Engine
{
    const Options _options;
    std::deque<Frame> _sourceFrames, _decoderFrames;
    double _startTime;
    std::thread _sourceThread, _receiverThread, _decoderThread;
    std::mutex _sourceMutex, _logMutex, _decoderMutex;
    bool _sourceStop;
    Scene _scene;
    Stat _stat;
public:
    Engine(const Options & options = Options())
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
                frame.time = currentTime;
                frame.id = frameId;
                frame.state = Frame::Sent;
                std::lock_guard<std::mutex> lock(_sourceMutex);
                _sourceFrames.push_back(frame);
                nextTime += _options.sourceInterval;
                frameId++;
                if (_options.logLevel >= LogDebug)
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
            Frame frame;
            {
                std::lock_guard<std::mutex> lock(_sourceMutex);
                if (_sourceFrames.empty())
                {
                    if (_sourceStop)
                        break;
                    continue;
                }
                double delayTime = Random(_options.receiveDelayMin, _options.receiveDelayMax);
                double currentTime = Time();
                frame = _sourceFrames.front();
                if (currentTime >= frame.time + delayTime)
                {
                    _sourceFrames.pop_front();
                    frame.state = Frame::Received;
                    {
                        std::lock_guard<std::mutex> lock(_decoderMutex);
                        _decoderFrames.push_back(frame);
                    }
                    if (_options.logLevel >= LogDebug)
                    {
                        std::lock_guard<std::mutex> lock(_logMutex);
                        std::cout << "Receiver received frame " << frame.id << " at " << int((currentTime - _startTime) * 1000.0) << std::endl << std::flush;
                    }
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
            Frame frame;
            {
                std::lock_guard<std::mutex> lock(_decoderMutex);
                if (_decoderFrames.empty())
                {
                    if (_sourceStop)
                        break;
                    continue;
                }
                while (_decoderFrames.size() > _options.decoderQueueSize)
                {
                    frame = _decoderFrames.front();
                    _decoderFrames.pop_front();
                    frame.state = Frame::Missed;
                    _stat.missed++;
                    if (_options.logLevel >= LogWarning)
                    {
                        std::lock_guard<std::mutex> lock(_logMutex);
                        std::cout << "Warning: Decoder missed frame " << frame.id << " at " << int((Time() - _startTime) * 1000.0) << std::endl << std::flush;
                    }
                }
                frame = _decoderFrames.front();
                _decoderFrames.pop_front();
            }
            if (frame.state)
            {
                double decodeTime = Random(_options.decodeTimeMin, _options.decodeTimeMax);
                Sleep(decodeTime * 1000.0);
                Process(frame);
            }
        }
        if (_options.logLevel >= LogInfo)
        {
            std::lock_guard<std::mutex> lock(_logMutex);
            std::cout << "Decoder finished. " << std::endl << std::flush;
            std::cout << "Missed frames: " << _stat.missed << std::endl << std::flush;
            std::cout << "Skipped frames: " << _stat.skipped<< std::endl << std::flush;
            std::cout << "Processed frames: " << _stat.processed << std::endl << std::flush;
        }
    }

    void Process(Frame &frame)
    {
        _scene.Start(frame.time);
        if (_options.logLevel >= LogDebug)
        {
            std::lock_guard<std::mutex> lock(_logMutex);
            std::cout << "Processing start frame " << frame.id << " at " << int((_scene.startTime - _startTime) * 1000.0) << std::endl << std::flush;
        }
        if (_scene.NeedSkip())
        {
            frame.state = Frame::Skipped;
            _stat.skipped++;
            _scene.Skip();
            if (_options.logLevel >= LogVerbose)
            {
                std::lock_guard<std::mutex> lock(_logMutex);
                std::cout << "Skip frame " << frame.id << " at " << int((_scene.startTime - _startTime) * 1000.0) << std::endl << std::flush;
            }
        }
        else
        {
            double processTime = Random(_options.processTimeMin, _options.processTimeMax);
            Sleep(processTime * 1000.0);
            _stat.processed++;
            _scene.Finish();
            frame.state = Frame::Processed;
            if (_options.logLevel >= LogVerbose)
            {
                std::lock_guard<std::mutex> lock(_logMutex);
                std::cout << "Process frame " << frame.id << " at " << int((_scene.startTime - _startTime) * 1000.0) << std::endl << std::flush;
            }
        }
    }
};


