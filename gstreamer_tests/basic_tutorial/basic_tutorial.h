#pragma once

#include <iostream>

typedef int (*basic_tutorial_ptr)(int argc, char* argv[]);

int basic_tutorial_01(int argc, char* argv[]);

inline std::string path_to_pipeline_description(const std::string& path)
{
    return "playbin uri=file:///" + path;
}