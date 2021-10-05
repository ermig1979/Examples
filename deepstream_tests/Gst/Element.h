#pragma once

#include "Common.h"

namespace Gst
{
    class Element
    {
    public:
        Element()
            : _element(NULL)
        {
        }

        ~Element()
        {
            if (_element)
            {
                GstState state;
                gst_element_get_state(_element, &state, NULL, GST_CLOCK_TIME_NONE);
                if (state != GST_STATE_NULL)
                {
                    GstStateChangeReturn result = gst_element_set_state(_element, GST_STATE_NULL);
                    if (result == GST_STATE_CHANGE_FAILURE)
                        std::cout << "Can't set element state!" << std::endl;
                }
                gst_object_unref(_element);
                _element = NULL;
            }
        }

        bool PipelineFromFile(const String& path)
        {
            String description = "playbin uri=file:///" + path;
            _element = gst_parse_launch(description.c_str(), NULL);
            if (_element == NULL)
            {
                std::cout << "Can't init pipline by file '" << path << "' !" << std::endl;
                return false;
            }
            return true;
        }

        bool PipelineNew(const String& name)
        {
            _element = gst_pipeline_new(name.c_str());
            if (_element == NULL)
            {
                std::cout << "Can't init new pipeline '" << name << "' !" << std::endl;
                return false;
            }
            return true;
        }

        bool FactoryMake(const String& factory, const String& element)
        {
            _element = gst_element_factory_make(factory.c_str(), element.c_str());
            if (_element == NULL)
            {
                std::cout << "Factory '" << factory << "' can't make element '" << element << "' !" << std::endl;
                return false;
            }
            return true;
        }

        bool SetState(GstState state)
        {
            GstStateChangeReturn result = gst_element_set_state(_element, state);
            if (result == GST_STATE_CHANGE_FAILURE)
            {
                std::cout << "Can't set state " << state << " !" << std::endl;
                return false;
            }
            return true;
        }

        bool Set(const String & name, const String& value)
        {
            g_object_set(G_OBJECT(_element), name.c_str(), value.c_str(), NULL);

            return true;
        }

        bool Set(const String& name, int value)
        {
            g_object_set(G_OBJECT(_element), name.c_str(), value, NULL);

            return true;
        }

        GstElement* Handle()
        {
            return _element;
        }

    private:
        GstElement* _element;
    };
}
