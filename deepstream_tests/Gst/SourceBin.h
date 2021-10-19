#pragma once

#include "Element.h"

namespace Gst
{
    class SourceBin : public Element
    {
    public:
        SourceBin()
            : Element()
        {
        }

        bool CreateSourceBin(const String& uri, size_t index)
        {
            String binName = String("source-bin-") + ToString(index, 2);
            _element = gst_bin_new(binName.c_str());
            if (_element == NULL)
            {
                if (Gst::logLevel >= Gst::LogError)
                    std::cout << "Can't create '" << binName << "' !" << std::endl;
                return false;
            }
            if (Gst::logLevel >= Gst::LogDebug)
                std::cout << "Create '" << binName << "' ." << std::endl;

            if (!_uriDecoder.FactoryMake("uridecodebin", "uri-decode-bin"))
                return false;
            _uriDecoder.Set("uri", uri);

            if (g_signal_connect(G_OBJECT(_uriDecoder.Handle()), "pad-added", G_CALLBACK(NewPadCallback), _element) == 0)
            {
                if (Gst::logLevel >= Gst::LogError)
                    std::cout << "Can't connect signal 'pad-added' from '" << _uriDecoder.Name() << "' to '" << Name() << "' !" << std::endl;
                return false;
            }
            if (Gst::logLevel >= Gst::LogDebug)
                std::cout << "Connect signal 'pad_added' from '" << _uriDecoder.Name() << "' to '" << Name() << "' !" << std::endl;

            if (g_signal_connect(G_OBJECT(_uriDecoder.Handle()), "child-added", G_CALLBACK(DecodebinChildAddedCallback), _element) == 0)
            {
                if (Gst::logLevel >= Gst::LogError)
                    std::cout << "Can't connect signal 'child-added' from '" << _uriDecoder.Name() << "' to '" << Name() << "' !" << std::endl;
                return false;
            }
            if (Gst::logLevel >= Gst::LogDebug)
                std::cout << "Connect signal 'child-added' from '" << _uriDecoder.Name() << "' to '" << Name() << "' !" << std::endl;

            if (!BinAdd(_uriDecoder))
                return false;

            if (!gst_element_add_pad(_element, gst_ghost_pad_new_no_target("src", GST_PAD_SRC))) 
            {
                if (Gst::logLevel >= Gst::LogError)
                    std::cout << "Failed to add ghost pad in '" << Name() << "' !" << std::endl;
                return false;
            }
            if (Gst::logLevel >= Gst::LogDebug)
                std::cout << "Add ghost pad in '" << Name() << "' ." << std::endl;

            return true;
        }

    protected:
        Element _uriDecoder;

        static void NewPadCallback(GstElement* decodebin, GstPad* decoder_src_pad, gpointer data)
        {
            g_print("In NewPadCallback\n");
            GstCaps* caps = gst_pad_get_current_caps(decoder_src_pad);
            const GstStructure* str = gst_caps_get_structure(caps, 0);
            const gchar* name = gst_structure_get_name(str);
            GstElement* source_bin = (GstElement*)data;
            GstCapsFeatures* features = gst_caps_get_features(caps, 0);

            /* Need to check if the pad created by the decodebin is for video and not
             * audio. */
            if (!strncmp(name, "video", 5)) {
                /* Link the decodebin pad only if decodebin has picked nvidia
                 * decoder plugin nvdec_*. We do this by checking if the pad caps contain
                 * NVMM memory features. */
                if (gst_caps_features_contains(features, "memory:NVMM")) {
                    /* Get the source bin ghost pad */
                    GstPad* bin_ghost_pad = gst_element_get_static_pad(source_bin, "src");
                    if (!gst_ghost_pad_set_target(GST_GHOST_PAD(bin_ghost_pad),
                        decoder_src_pad)) {
                        g_printerr("Failed to link decoder src pad to source bin ghost pad\n");
                    }
                    gst_object_unref(bin_ghost_pad);
                }
                else {
                    g_printerr("Error: Decodebin did not pick nvidia decoder plugin.\n");
                }
            }
        }

        static void DecodebinChildAddedCallback(GstChildProxy* child_proxy, GObject* object, gchar* name, gpointer user_data)
        {
            g_print("Decodebin child added: %s\n", name);
            if (g_strrstr(name, "decodebin") == name) 
            {
                g_signal_connect(G_OBJECT(object), "child-added", G_CALLBACK(DecodebinChildAddedCallback), user_data);
            }
        }
    };
}
