#pragma once

#include "defs.h"
#include "float16.h"

namespace cs
{
    class Descriptor
    {
    public:
        virtual ~Descriptor()
        {
        }

        virtual String Name() const = 0;

        virtual size_t Size() const = 0;

        virtual float CosineDistance(const Descriptor& other) const  = 0;
    };
    typedef std::shared_ptr<Descriptor> DescriptorPtr;
    typedef std::vector<DescriptorPtr> DescriptorPtrs;

    //-------------------------------------------------------------------------------------------------

    class DescriptorFloat32 : public Descriptor
    {
    public:
        DescriptorFloat32()
        {
        }

        void Init(size_t size, float lo = -20.0f, float hi = 20.0f)
        {
            _data.resize(size);
            for (size_t i = 0; i < _data.size(); ++i)
            {
                float val = float(rand()) / float(RAND_MAX);
                _data[i] = lo + val * (hi - lo);
            }
        }

        bool Load(std::istream & is, const String& delimeter = " ")
        {
            String line;
            std::getline(is, line);
            if (line.empty())
                return false;
            _data.clear();            
            size_t current = 0;
            while (current != String::npos)
            {
                size_t next = line.find(delimeter, current);
                String value = line.substr(current, next - current);
                if (!value.empty())
                    _data.push_back(std::stof(value));
                current = next;
                if (current != String::npos)
                    current += delimeter.size();
            }
            return !_data.empty();
        }

        const float * Data() const
        {
            return _data.data();
        }

        virtual String Name() const
        {
            return "Float32";
        }

        virtual size_t Size() const
        {
            return _data.size();
        }

        virtual float CosineDistance(const Descriptor& other) const
        {
            return CosineDistanceFloat(dynamic_cast<const DescriptorFloat32&>(other));
        }

    protected:
        Buffer32f _data;

        float CosineDistanceFloat(const DescriptorFloat32& other) const
        {
            assert(this->Size() == other.Size());
            float tt = 0, to = 0, oo = 0;
            for (size_t i = 0; i < _data.size(); ++i)
            {
                float _t = this->_data[i];
                float _o = other._data[i];
                tt += _t * _t;
                to += _t * _o;
                oo += _o * _o;
            }
            return 1.0f - to / ::sqrt(oo * tt);
        }
    };
    typedef std::shared_ptr<DescriptorFloat32> DescriptorFloat32Ptr;
    typedef std::vector<DescriptorFloat32Ptr> DescriptorFloat32Ptrs;

    //-------------------------------------------------------------------------------------------------

    class DescriptorFloat16 : public Descriptor
    {
    public:
        DescriptorFloat16(const DescriptorFloat32 & orig)
        {
            _data.resize(orig.Size());
            for (size_t i = 0; i < _data.size(); ++i)
                _data[i] = Float32ToFloat16(orig.Data()[i]);
        }

        virtual String Name() const
        {
            return "Float16";
        }

        virtual size_t Size() const
        {
            return _data.size();
        }

        virtual float CosineDistance(const Descriptor& other) const
        {
            return CosineDistanceFloat16(dynamic_cast<const DescriptorFloat16&>(other));
        }

    protected:
        Buffer16u _data;

        float CosineDistanceFloat16(const DescriptorFloat16& other) const
        {
            assert(this->Size() == other.Size());
            float tt = 0, to = 0, oo = 0;
            for (size_t i = 0; i < _data.size(); ++i)
            {
                float _t = Float16ToFloat32(this->_data[i]);
                float _o = Float16ToFloat32(other._data[i]);
                tt += _t * _t;
                to += _t * _o;
                oo += _o * _o;
            }
            return 1.0f - to / ::sqrt(oo * tt);
        }
    };
    typedef std::shared_ptr<DescriptorFloat16> DescriptorFloat16Ptr;
    typedef std::vector<DescriptorFloat16Ptr> DescriptorFloat16Ptrs;
}


