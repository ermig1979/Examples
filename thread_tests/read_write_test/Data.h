#pragma once

#include "Defs.h"

namespace Test
{
    struct Data
    {
        Bytes data;

        inline Data(int size = 0, Byte val = 0)
            : data(size, val)
        {
        }

        inline void Resize(int size = 0, Byte val = 0)
        {
            data.resize(size, val);
        }

        inline bool Valid() const
        {
            Byte val0 = data[0];
            for (size_t i = 0; i < data.size(); ++i)
            {
                if (data[i] != val0)
                    return false;
            }
            return true;
        }

        inline void Set(Byte val)
        {
            for (size_t i = 0; i < data.size(); ++i)
                data[i] = val;
        }

        inline void Assign(const Data& src)
        {
            for (size_t i = 0; i < data.size(); ++i)
                data[i] = src.data[i];
        }
    };

    typedef std::vector<Data> Datas;
}
