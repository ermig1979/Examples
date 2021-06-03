#include "utils.h"

void print_device_info()
{
    int deviceCount;
    CHECK(cudaGetDeviceCount(&deviceCount));
    for (int device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        CHECK(cudaGetDeviceProperties(&deviceProp, device));
        std::cout << "Device " << device << ": '" << deviceProp.name << "'." << std::endl;
        std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << "." << std::endl;
        std::cout << "Device global memory: " << int(deviceProp.totalGlobalMem / 1024 / 1024) << " MB." << std::endl;
        std::cout << "Shared memory per block: " << int(deviceProp.sharedMemPerBlock / 1024) << " kB." << std::endl;
        std::cout << "Registers per block: " << int(deviceProp.regsPerBlock / 1024) << " kB." << std::endl;
        std::cout << std::endl;
    }
}

//-----------------------------------------------------------------------------

void print_graph_info(cudaGraph_t graph)
{
    typedef  std::vector<cudaGraphNode_t> Nodes;
    size_t numNodes = 0;
    CHECK(cudaGraphGetNodes(graph, NULL, &numNodes));
    std::cout << "Number of graph nodes: " << numNodes << std::endl;
    Nodes nodes(numNodes);
    CHECK(cudaGraphGetNodes(graph, nodes.data(), &numNodes));
    for (size_t i = 0; i < numNodes; ++i)
    {
        cudaGraphNodeType type;
        CHECK(cudaGraphNodeGetType(nodes[i], &type));
        std::cout << "Node[" << i << "] type: " << type << std::endl;
    }
    size_t numEdges = 0;
    CHECK(cudaGraphGetEdges(graph, NULL, NULL, &numEdges));
    std::cout << "Number of graph edges: " << numEdges << std::endl;
    Nodes from(numEdges), to(numEdges);
    CHECK(cudaGraphGetEdges(graph, from.data(), to.data(), &numEdges));
    for (size_t i = 0; i < numEdges; ++i)
    {
        //cudaGraphNodeType type;
        //CHECK(cudaGraphNodeGetType(nodes[i], &type));
        std::cout << "Edge[" << i << "]" << std::endl;
    }
}

//-----------------------------------------------------------------------------

inline float square(float x)
{
    return x * x;
}

inline float invalid(float a, float b, float e2)
{
    float d2 = square(a - b);
    return d2 > e2 && d2 > e2 * (a * a + b * b);
}

bool check(const cpu_tensor_t& control, const cpu_tensor_t& current, const std::string& desc, float eps, int count)
{
    assert(control.size == current.size);
    float e2 = square(eps);
    int errors = 0;
    for (int i = 0; i < control.size; ++i)
    {
        if (invalid(control.data[i], current.data[i], e2))
        {
            std::cout << desc << " : check error at " << i << ": ";
            std::cout << std::setprecision(4) << std::fixed << control.data[i] << " != " << current.data[i] << std::endl;
            errors++;
            if (errors >= count)
                return false;
        }
    }
    return errors == 0;
}

