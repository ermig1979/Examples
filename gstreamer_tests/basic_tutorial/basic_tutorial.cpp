#include "basic_tutorial.h"

static int run_test(basic_tutorial_ptr test, const std::string& name, int argc, char* argv[])
{
    std::cout << name << " start:" << std::endl;
    if (test(argc, argv)) 
    { 
        std::cout << name << " is failed!" << std::endl; 
        return 1; 
    } 
    else
    {
        std::cout << name << " is OK." << std::endl;
        return 0;
    }
}

#define RUN_TEST(test, argc, argv) if(run_test(test, #test, argc, argv)) { return 1; }

int main(int argc, char* argv[])
{
    RUN_TEST(basic_tutorial_01, argc, argv);

    return 0;
}