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
                GstStateChangeReturn result = gst_element_get_state(_element, &state, NULL, 0);
                if (result == GST_STATE_CHANGE_FAILURE)
                    std::cout << "Can't get element state!" << std::endl;
                else
                {
                    if (result != GST_STATE_NULL)
                    {
                        GstStateChangeReturn result = gst_element_set_state(_element, GST_STATE_NULL);
                        if (result == GST_STATE_CHANGE_FAILURE)
                            std::cout << "Can't set element state!" << std::endl;
                    }
                }
                gst_object_unref(_element);
            }
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

        GstElement* Handler()
        {
            return _element;
        }

    private:
        GstElement* _element;
    };
}
