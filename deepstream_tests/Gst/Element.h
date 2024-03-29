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

        bool FactoryMake(const String& factory, const String& element = String())
        {
            const char* name = element.empty() ? NULL : element.c_str();
            _element = gst_element_factory_make(factory.c_str(), name);
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
                    gst_element_get_state(_element, &state, NULL, GST_MSECOND*1000);
                    if (state == GST_STATE_PLAYING)
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
                    std::cout << Name() << ": can't set state " << ToString(state) << " !" << std::endl;
                return false;
            }
            if (Gst::logLevel >= Gst::LogDebug)
                std::cout << Name() << ": set state to " << ToString(state) << " ." << std::endl;
            return true;
        }

        bool Set(const String & name, const String& value)
        {
            if (Gst::logLevel >= Gst::LogDebug)
            {
                String old;
                gchar* buf = NULL;
                g_object_get(G_OBJECT(_element), name.c_str(), &buf, NULL);
                if (buf)
                {
                    old = buf;
                    g_free(buf);
                }
                std::cout << "Element '" << Name() << "' change property '" << name << "' from '" << old << "' to '" << value << "'." << std::endl;
            }
            g_object_set(G_OBJECT(_element), name.c_str(), value.c_str(), NULL);
            return true;
        }

        bool Set(const String& name, int value)
        {
            if (Gst::logLevel >= Gst::LogDebug)
            {
                int old;
                g_object_get(G_OBJECT(_element), name.c_str(), &old, NULL);
                std::cout << "Element '" << Name() << "' change property '" << name << "' from '" << old << "' to '" << value << "'." << std::endl;
            }
            g_object_set(G_OBJECT(_element), name.c_str(), value, NULL);
            return true;
        }

        bool Get(const String& name, String & value)
        {
            gchar* buffer = NULL;
            g_object_get(G_OBJECT(_element), name.c_str(), &buffer, NULL);
            if(buffer)
            {
                value = buffer;
                if (Gst::logLevel >= Gst::LogDebug)
                    std::cout << "Element '" << Name() << "' property '" << name << "' is equel to '" << value << "'." << std::endl;
                g_free(buffer);
                return true;
            }
            else
            {
                if (Gst::logLevel >= Gst::LogError)
                    std::cout << "Can' get element '" << Name() << "' property '" << name << "' !" << std::endl;
                return false;
            }
        }

        bool Get(const String& name, int & value)
        {
            g_object_get(G_OBJECT(_element), name.c_str(), &value, NULL);
            if (Gst::logLevel >= Gst::LogDebug)
                std::cout << "Element '" << Name() << "' property '" << name << "' is equel to '" << value << "'." << std::endl;
            return true;
        }

        bool SetCapsFromString(const String& string)
        {
            GstCaps * caps = gst_caps_from_string(string.c_str());
            if (caps == NULL)
            {
                if (Gst::logLevel >= Gst::LogError)
                    std::cout << "Element '" << Name() << "' : Can't create caps from string: '" << string << "' !" << std::endl;
                return false;
            }
            if (Gst::logLevel >= Gst::LogDebug)
                std::cout << "Element '" << Name() << "' : set caps from string '" << string << "." << std::endl;
            g_object_set(G_OBJECT(_element), "caps", caps, NULL);
            gst_caps_unref(caps);
            return true;
        }

        bool BinAdd(Element & element)
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

        bool AddPadProb(const String & name, GstPadProbeCallback callback, void * data = NULL)
        {
            bool result = false;
            GstPad * pad = gst_element_get_static_pad(_element, name.c_str());
            if (pad)
            {
                if (gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, callback, data, NULL))
                {
                    if (Gst::logLevel >= Gst::LogDebug)
                        std::cout << Name() << ": set prob to pad '" << name << "'." << std::endl;
                    result = true;
                }
                else
                {
                    if (Gst::logLevel >= Gst::LogError)
                        std::cout << Name() << ": Can't set prob to pad '" << name << "'!" << std::endl;
                }
                gst_object_unref(pad);
            }
            else
            {
                if (Gst::logLevel >= Gst::LogError)
                    std::cout << Name() << ": Can't get pad '" << name << "'!" << std::endl;
            }
            return result;
        }

        bool BinAdd(Element& elem0, Element& elem1)
        {
            return BinAdd(elem0) && BinAdd(elem1);
        }

        bool BinAdd(Element& elem0, Element& elem1, Element& elem2)
        {
            return BinAdd(elem0, elem1) && BinAdd(elem2);
        }

        bool BinAdd(Element& elem0, Element& elem1, Element& elem2, Element& elem3)
        {
            return BinAdd(elem0, elem1) && BinAdd(elem2, elem3);
        }

        bool BinAdd(Element& elem0, Element& elem1, Element& elem2, Element& elem3, Element& elem4)
        {
            return BinAdd(elem0, elem1, elem2, elem3) && BinAdd(elem4);
        }

        bool BinAdd(Element& elem0, Element& elem1, Element& elem2, Element& elem3, Element& elem4, Element& elem5)
        {
            return BinAdd(elem0, elem1, elem2, elem3) && BinAdd(elem4, elem5);
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
