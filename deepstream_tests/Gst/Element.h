#pragma once

#include "Common.h"

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

        ~Element()
        {
            if (_element && 0)
            {
                if (Gst::logLevel >= Gst::LogDebug)
                    std::cout << "Delete element '" << Name() << "'." << std::endl;
                if (_owner)
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
                }
                _element = NULL;
            }
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
            else
                _owner = true;
            return true;
        }

        bool SetState(GstState state)
        {
            GstStateChangeReturn result = gst_element_set_state(_element, state);
            if (result == GST_STATE_CHANGE_FAILURE)
            {
                if (Gst::logLevel >= Gst::LogError)
                    std::cout << "Can't set state " << state << " !" << std::endl;
                return false;
            }
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
