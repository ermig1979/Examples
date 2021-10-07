#pragma once

#include "Pipeline.h"
#include "Utils.h"

namespace Gst
{
    class MainLoop
    {
    public:
        MainLoop()
            : _loop(NULL)
            , _busWatchId(0)
        {
            _loop = g_main_loop_new(NULL, FALSE);
            if (_loop == NULL)
            {
                if (Gst::logLevel >= Gst::LogError)
                    std::cout << "Can't create main loop!" << std::endl;
            }
        }

        ~MainLoop()
        {
            if (_busWatchId)
            {
                g_source_remove(_busWatchId);
                _busWatchId = 0;
            }
            if (_loop)
            {
                g_main_loop_unref(_loop);
                _loop = NULL;
            }
        }

        bool AddWatch(Pipeline & pipeline)
        {
            GstBus * bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline.Handle()));
            if (bus)
            {
                _busWatchId = gst_bus_add_watch(bus, Gst::BusCallback, _loop);
                gst_object_unref(bus);
                return true;
            }
            else
                return false;
        }

        bool Run()
        {
            if (Gst::logLevel >= Gst::LogInfo)
                std::cout << "Run main loop:" << std::endl;
            g_main_loop_run(_loop);
            return true;
        }

        GMainLoop* Handle()
        {
            return _loop;
        }

    private:
        GMainLoop* _loop;
        guint _busWatchId;
    };
}
