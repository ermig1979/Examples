#pragma once

#include "Common.h"

namespace Gst
{
    class Bus
    {
    public:
        Bus()
            : _bus(NULL)
        {
        }

        ~Bus()
        {
            if (_bus)
            {
                gst_object_unref(_bus);
                _bus = NULL;
            }
        }

        GstBus* Handle()
        {
            return _bus;
        }

    private:
        GstBus* _bus;
    };
}
