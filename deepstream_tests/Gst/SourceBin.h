#pragma once

#include "Element.h"

namespace Gst
{
    class VideoSourceBin : public Element
    {
    public:
        VideoSourceBin()
            : Element()
        {
        }

        bool Create(String uri, size_t index)
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

            Element uriDecoder;
            if (!uriDecoder.FactoryMake("uridecodebin", "uri-decode-bin"))
                return false;
            if (uri.substr(0, 7) != "rtsp://" && uri.substr(0, 7) != "file://")
                uri = String("file://") + uri;
            uriDecoder.Set("uri", uri);

            if (g_signal_connect(G_OBJECT(uriDecoder.Handle()), "pad-added", G_CALLBACK(NewPadCallback), _element) == 0)
            {
                if (Gst::logLevel >= Gst::LogError)
                    std::cout << "Can't connect signal 'pad-added' from '" << uriDecoder.Name() << "' to '" << Name() << "' !" << std::endl;
                return false;
            }
            if (Gst::logLevel >= Gst::LogDebug)
                std::cout << "Connect signal 'pad_added' from '" << uriDecoder.Name() << "' to '" << Name() << "' !" << std::endl;

            if (g_signal_connect(G_OBJECT(uriDecoder.Handle()), "child-added", G_CALLBACK(DecodebinChildAddedCallback), _element) == 0)
            {
                if (Gst::logLevel >= Gst::LogError)
                    std::cout << "Can't connect signal 'child-added' from '" << uriDecoder.Name() << "' to '" << Name() << "' !" << std::endl;
                return false;
            }
            if (Gst::logLevel >= Gst::LogDebug)
                std::cout << "Connect signal 'child-added' from '" << uriDecoder.Name() << "' to '" << Name() << "' !" << std::endl;

            if (!BinAdd(uriDecoder))
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

    //---------------------------------------------------------------------------------------------

    class ImageSourceBin : public Element
    {
    public:
        ImageSourceBin()
            : Element()
        {
        }

        bool Create(String uri, size_t index)
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

            bool multiFileSrc = false;
            Element source, parser, decoder;

            if (uri.find("%d", 0) != String::npos)
            {
                if (!source.FactoryMake("multifilesrc", "multi-image-source"))
                    return false;
                multiFileSrc = true;
            }
            else
            {
                if (!source.FactoryMake("filesrc", "image-source"))
                    return false;
            }
            source.Set("location", uri);

            if (!parser.FactoryMake("jpegparse", "jpeg-parser"))
                return false;

            if (!decoder.FactoryMake("nvv4l2decoder", "nvv4l2-decoder"))
                return false;

            if (!BinAdd(source, parser, decoder))
                return false;

            if (!StaticLink(source, parser, decoder))
                return true;

            if (!gst_element_add_pad(_element, gst_ghost_pad_new_no_target("src", GST_PAD_SRC)))
            {
                if (Gst::logLevel >= Gst::LogError)
                    std::cout << "Failed to add ghost pad in '" << Name() << "' !" << std::endl;
                return false;
            }
            if (Gst::logLevel >= Gst::LogDebug)
                std::cout << "Add ghost pad in '" << Name() << "' ." << std::endl;


            GstPad* srcpad = gst_element_get_static_pad(decoder.Handle(), "src");
            if (!srcpad) 
            {
                g_printerr("Failed to get src pad of source bin. Exiting.\n");
                return false;
            }

            GstPad* bin_ghost_pad = gst_element_get_static_pad(_element, "src");
            if (!gst_ghost_pad_set_target(GST_GHOST_PAD(bin_ghost_pad), srcpad)) 
            {
                g_printerr("Failed to link decoder src pad to source bin ghost pad\n");
                return false;
            }

            return true;
        }

    protected:

    };
}
