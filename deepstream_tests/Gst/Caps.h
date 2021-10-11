#pragma once

#include <iostream>
#include <sstream>
#include <vector>

#include <gst/gst.h>

namespace Gst
{
    struct Caps
    {
        Caps(const String & type)
            : _type(type)
            , _caps(gst_caps_new_empty_simple(type.c_str()))
        {
            if (_caps == NULL)
            {
                if(Gst::logLevel >= Gst::LogError)
                    std::cout << "Can't create '" << _type << "' caps !" << std::endl;
                exit(1);
            }
            if (Gst::logLevel >= Gst::LogDebug)
                std::cout << "Create '" << _type << "' caps." << std::endl;
        }

        ~Caps()
        {
            if (_caps)
            {
                gst_caps_unref(_caps);
                _caps = NULL;
            }
        }

        const String& Type() const
        {
            return _type;
        }

        bool SetString(const String& field, const String & value)
        {
            if (Gst::logLevel >= Gst::LogDebug)
                std::cout << "Caps '" << _type << " set '" << field << "' to '" << value << "'." << std::endl;
            gst_caps_set_simple(_caps, field.c_str(), G_TYPE_STRING, value.c_str(), NULL);
            return true;
        }

        bool SetInt(const String& field, int value)
        {
            if (Gst::logLevel >= Gst::LogDebug)
                std::cout << "Caps '" << _type << " set '" << field << "' to '" << value << "'." << std::endl;
            gst_caps_set_simple(_caps, field.c_str(), G_TYPE_INT, value, NULL);
            return true;
        }

        bool SetFraction(const String& field, int denominator, int numerator)
        {
            if (Gst::logLevel >= Gst::LogDebug)
                std::cout << "Caps '" << _type << "' set '" << field << "' to '" << numerator <<"//" << denominator << "'." << std::endl;
            gst_caps_set_simple(_caps, field.c_str(), GST_TYPE_FRACTION, denominator, numerator, NULL);
            return true;
        }

        GstCaps* Handle()
        {
            return _caps;
        }

    protected:
        String _type;
        GstCaps * _caps;
    };
}
