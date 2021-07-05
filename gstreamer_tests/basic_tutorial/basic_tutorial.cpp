#include "options.h"

namespace Test
{
    struct Test
    {
        typedef int (*TestPtr)(Options options);

        String name;
        TestPtr test;

        Test(const String& n, TestPtr t)
            : name(n)
            , test(t)
        {
        }

        bool Required(const Options& options) const
        {
            bool required = options.includeFilter.empty();
            for (size_t i = 0; i < options.includeFilter.size() && !required; ++i)
                if (name.find(options.includeFilter[i]) != std::string::npos)
                    required = true;
            for (size_t i = 0; i < options.excludeFilter.size() && required; ++i)
                if (name.find(options.excludeFilter[i]) != std::string::npos)
                    required = false;
            return required;
        }
    };
    typedef std::vector<Test> Tests;
    Tests g_tests;

#define TEST_ADD(test) \
    int test(Options options); \
    bool test##Add() { g_tests.push_back(Test(#test, test)); return true; } \
    bool test##Added = test##Add();

    TEST_ADD(BasicTutorial01);
    TEST_ADD(BasicTutorial02);
    TEST_ADD(BasicTutorial03);
    TEST_ADD(BasicTutorial04);

    int RunTests(const Options& options)
    {
        for (const Test & test : g_tests)
        {
            if (test.Required(options))
            {
                std::cout << test.name << " start:" << std::endl;
                if (test.test(options))
                {
                    std::cout << test.name << " is failed!" << std::endl;
                    return 1;
                }
                else
                {
                    std::cout << test.name << " is OK." << std::endl;
                    return 0;
                }
            }
        }
        return 0;
    }
}

int main(int argc, char* argv[])
{
    Test::Options options(argc, argv);

    return RunTests(options);
}