#pragma once

#include "Element.h"

namespace Gst
{
    class Pipeline
    {
    public:
        Pipeline()
            : _pipeline(NULL)
            , _playing(false)
        {
        }

        ~Pipeline()
        {
            Release();
        }        
        
        bool InitFromFile(const String& path)
        {
            String description = "playbin uri=file:///" + path;
            _pipeline = gst_parse_launch(description.c_str(), NULL);
            if (_pipeline == NULL)
            {
                std::cout << "Can't init pipline by file '" << path << "' !" << std::endl;
                return false;
            }
            return true;
        }

        bool InitNew(const String& name)
        {
            _pipeline = gst_pipeline_new(name.c_str());
            if (_pipeline == NULL)
            {
                if (Gst::logLevel >= Gst::LogError)
                    std::cout << "Can't init new pipeline '" << name << "' !" << std::endl;
                return false;
            }
            else
            {
                if (Gst::logLevel >= Gst::LogDebug)
                    std::cout << "Pipeline '" << name << "' was inited." << std::endl;
            }
            return true;
        }

        bool Release()
        {
            if (_pipeline)
            {
                Stop();
                if (Gst::logLevel >= Gst::LogDebug)
                    std::cout << "Delete pipeline." << std::endl << std::flush;
                gst_object_unref(_pipeline);
                _pipeline = NULL;
            }
            return _pipeline == NULL;
        }

        bool Play()
        {
            if (Gst::logLevel >= Gst::LogDebug)
                std::cout << "Start playing: " << std::flush;
            GstStateChangeReturn state = gst_element_set_state(_pipeline, GST_STATE_PLAYING);
            if (state == GST_STATE_CHANGE_FAILURE)
            {
                if (Gst::logLevel >= Gst::LogError)
                    std::cout << "Can't play pipeline!" << std::endl;
                _playing = false;
            }
            else
            {
                if (Gst::logLevel >= Gst::LogDebug)
                    std::cout << " OK. " << std::endl << std::flush;
                _playing = true;
            }
            return _playing;
        }

        bool Stop()
        {
            if (_playing)
            {
                if (Gst::logLevel >= Gst::LogDebug)
                    std::cout << "Stop playback: " << std::flush;
                GstStateChangeReturn state = gst_element_set_state(_pipeline, GST_STATE_NULL);
                if (state == GST_STATE_CHANGE_FAILURE && Gst::logLevel >= Gst::LogError)
                    std::cout << "Can't stop pipeline!" << std::endl << std::flush;
                else if (Gst::logLevel >= Gst::LogDebug)
                    std::cout << " OK. " << std::endl << std::flush;
                _playing = false;
            }
            return _playing == false;
        }

        GstElement* Handle()
        {
            return _pipeline;
        }

    private:
        GstElement* _pipeline;
        bool _playing;
    };
}
