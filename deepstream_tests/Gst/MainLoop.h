#pragma once

#include "Element.h"
#include "Utils.h"

namespace Gst
{
    class MainLoop
    {
    public:
        MainLoop()
            : _loop(NULL)
            , _busWatchId(0)
            , _ioStdIn(NULL)
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
                if (Gst::logLevel >= Gst::LogDebug)
                    std::cout << "MainLoop : remove Bus watch." << std::endl;
            }
            if (_ioStdIn)
            {
                g_io_channel_unref(_ioStdIn);
                _ioStdIn = NULL;
                if (Gst::logLevel >= Gst::LogDebug)
                    std::cout << "MainLoop : remove IO watch." << std::endl;
            }
            if (_loop)
            {
                g_main_loop_unref(_loop);
                _loop = NULL;
                if (Gst::logLevel >= Gst::LogDebug)
                    std::cout << "MainLoop : deleted." << std::endl;
            }
        }

        bool BusAddWatch(Element & pipeline)
        {
            GstBus * bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline.Handle()));
            if (bus)
            {
                _busWatchId = gst_bus_add_watch(bus, BusCallback, _loop);
                gst_object_unref(bus);
                if (_busWatchId)
                {
                    if (Gst::logLevel >= Gst::LogDebug)
                        std::cout << "MainLoop : add Bus watch." << std::endl;
                    return true;
                }
                else
                {
                    if (Gst::logLevel >= Gst::LogError)
                        std::cout << "MainLoop : Can't add Bus watch!" << std::endl;
                    return false;
                }
            }
            else
            {
                if (Gst::logLevel >= Gst::LogError)
                    std::cout << "MainLoop: gst_pipeline_get_bus(" << pipeline.Name() << ") return null!" << std::endl;
                return false;
            }
        }

        bool IoAddWatch()
        {
            _ioStdIn = g_io_channel_unix_new(fileno(stdin));
            if (_ioStdIn == NULL)
            {
                if (Gst::logLevel >= Gst::LogError)
                    std::cout << "MainLoop: g_io_channel_unix_new() return null!" << std::endl;
                return false;
            }
            if (g_io_add_watch(_ioStdIn, G_IO_IN, (GIOFunc)IoCallback, _loop) == 0)
            {
                if (Gst::logLevel >= Gst::LogError)
                    std::cout << "MainLoop : Can't add IO watch!" << std::endl;
                return false;
            }
            if (Gst::logLevel >= Gst::LogDebug)
                std::cout << "MainLoop : add IO watch." << std::endl;
            return true;
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
        GIOChannel* _ioStdIn;

        static gboolean BusCallback(GstBus* bus, GstMessage* msg, gpointer data)
        {
            GMainLoop* loop = (GMainLoop*)data;
            String type;
            LogLevel level = LogDebug;
            switch (GST_MESSAGE_TYPE(msg))
            {
            case GST_MESSAGE_EOS:
                g_main_loop_quit(loop);
                type = "End Of Stream";
                level = LogInfo;
                break;
            case GST_MESSAGE_ERROR:
                if (Gst::logLevel >= Gst::LogError)
                {
                    gchar* debug;
                    GError* error;
                    gst_message_parse_error(msg, &error, &debug);
                    g_printerr("ERROR from element %s: %s\n",
                        GST_OBJECT_NAME(msg->src), error->message);
                    if (debug)
                        g_printerr("Error details: %s\n", debug);
                    g_free(debug);
                    g_error_free(error);
                }
                g_main_loop_quit(loop);
                return TRUE;
            case GST_MESSAGE_TAG: type = "Tag"; break;
            case GST_MESSAGE_STATE_CHANGED: type = "State Changed"; break;
            case GST_MESSAGE_NEW_CLOCK: type = "New Clock"; break;
            default:
                break;
            }
            if (Gst::logLevel >= Gst::LogDebug)
            {
                std::cout << "Message: ";
                if (type.empty())
                {
                    std::cout << " src = " << msg->src->name;
                    std::cout << " type = " << msg->type;
                }
                else
                {
                    std::cout << msg->src->name << " :  \t" << type;
                }
                std::cout << std::endl << std::flush;
            }
            return TRUE;
        }

        static gboolean IoCallback(GIOChannel* source, GIOCondition cond, gpointer* data)
        {
            GMainLoop* loop = (GMainLoop*)data;

            gchar* str = NULL;

            if (g_io_channel_read_line(source, &str, NULL, NULL, NULL) != G_IO_STATUS_NORMAL)
                return TRUE;

            switch (g_ascii_tolower(str[0]))
            {
            case 'q':
                if (Gst::logLevel >= Gst::LogInfo)
                    std::cout << "Key 'q' is pressed. Try to stop pipeline." << std::endl;
                g_main_loop_quit(loop);
                break;
            default:
                break;
            }

            if(str)
                g_free(str);

            return TRUE;
        }
    };
}
