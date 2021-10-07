#pragma once

#include "Utils.h"

namespace Gst
{
    class Element
    {
    public:
        Element()
            : _element(NULL)
            , _owner(false)
        {
        }

        bool FileParseLaunch(const String& path)
        {
            String description = "playbin uri=file:///" + path;
            _element = gst_parse_launch(description.c_str(), NULL);
            if (_element == NULL)
            {
                std::cout << "Can't init pipeline by file '" << path << "' !" << std::endl;
                return false;
            }
            _owner = true;
            return true;
        }

        bool PipelineNew(const String& name)
        {
            _element = gst_pipeline_new(name.c_str());
            if (_element == NULL)
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
            _owner = true;
            return true;
        }

        bool FactoryMake(const String& factory, const String& element)
        {
            _element = gst_element_factory_make(factory.c_str(), element.c_str());
            if (_element == NULL)
            {
                if (Gst::logLevel >= Gst::LogError)
                    std::cout << "Factory '" << factory << "' can't make element '" << element << "' !" << std::endl;
                return false;
            }
            if (Gst::logLevel >= Gst::LogDebug)
                std::cout << "Factory '" << factory << "' make element '" << element << "'." << std::endl;
            _owner = true;
            return true;
        }        
        
        virtual ~Element()
        {
            Release();
        }

        void Release()
        {
            if (_element)
            {
                if (_owner)
                {
                    if (Gst::logLevel >= Gst::LogDebug)
                        std::cout << "Delete element '" << Name() << "':" << std::endl;
                    GstState state;
                    gst_element_get_state(_element, &state, NULL, GST_CLOCK_TIME_NONE);
                    if (state != GST_STATE_NULL)
                        SetState(GST_STATE_NULL);
                    gst_object_unref(_element);
                }
                _element = NULL;
            }
        }

        bool SetState(GstState state)
        {
            GstStateChangeReturn result = gst_element_set_state(_element, state);
            if (result == GST_STATE_CHANGE_FAILURE)
            {
                if (Gst::logLevel >= Gst::LogError)
                    std::cout << Name() << ": can't set state " << StateToString(state) << " !" << std::endl;
                return false;
            }
            if (Gst::logLevel >= Gst::LogDebug)
                std::cout << Name() << ": set state to " << StateToString(state) << " ." << std::endl;
            return true;
        }

        bool Set(const String & name, const String& value)
        {
            if (Gst::logLevel >= Gst::LogDebug)
                std::cout << "Element '" << Name() << "' set property '" << name << "' to '" << value << "'." << std::endl;
            g_object_set(G_OBJECT(_element), name.c_str(), value.c_str(), NULL);
            return true;
        }

        bool Set(const String& name, int value)
        {
            if (Gst::logLevel >= Gst::LogDebug)
                std::cout << "Element '" << Name() << "' set property '" << name << "' to '" << value << "'." << std::endl;
            g_object_set(G_OBJECT(_element), name.c_str(), value, NULL);
            return true;
        }

        bool Add(Element & element)
        {
            if (gst_bin_add(GST_BIN(_element), element._element) == FALSE)
            {
                if (Gst::logLevel >= Gst::LogError)
                    std::cout << Name() << ": can't add " << element.Name() << " !" << std::endl;
                return false;
            }
            else
            {
                if (Gst::logLevel >= Gst::LogDebug)
                    std::cout << Name() << ": add " << element.Name() << "." << std::endl;
                element._owner = false;
                return true;
            }
        }

        GstElement* Handle()
        {
            return _element;
        }

        String Name() const
        {
            return _element ? _element->object.name : "";
        }

    protected:
        GstElement* _element;
        bool _owner;
    };
}
