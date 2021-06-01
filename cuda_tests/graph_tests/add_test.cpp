#include "tests.h"
#include "nodes.h"
#include "tensor.h"
#include "utils.h"

void add_test()
{
    std::cout << "Start add_test:" << std::endl;

    const size_t N = 1, C = 1, H = 128, W = 128;
    shape_t shape = to_shape(N, C, H, W);

    cpu_tensor_t ca(shape), cb(shape), cc(shape), cd(shape);

    init_rand(ca, 0.0f, 1.0f);
    init_rand(cb, 0.0f, 1.0f);

    add_cpu((int)ca.size, ca.data, cb.data, cd.data);

    gpu_tensor_t ga(shape), gb(shape), gc(shape);

    cudaGraph_t graph;
    CHECK(cudaGraphCreate(&graph, 0));

    cudaMemcpy3DParms aCopyParams;
    aCopyParams.srcArray = NULL;
    aCopyParams.srcPos = make_cudaPos(0, 0, 0);
    aCopyParams.srcPtr = make_cudaPitchedPtr(ca.data, ca.size * sizeof(float), ca.size, 1);
    aCopyParams.dstArray = NULL;
    aCopyParams.dstPos = make_cudaPos(0, 0, 0);
    aCopyParams.dstPtr = make_cudaPitchedPtr(ga.data, ga.size * sizeof(float), ga.size, 1);
    aCopyParams.extent = make_cudaExtent(ga.size * sizeof(float), 1, 1);
    aCopyParams.kind = cudaMemcpyHostToDevice;
    cudaGraphNode_t aCopyNode;
    CHECK(cudaGraphAddMemcpyNode(&aCopyNode, graph, NULL, 0, &aCopyParams));

    cudaMemcpy3DParms bCopyParams;
    bCopyParams.srcArray = NULL;
    bCopyParams.srcPos = make_cudaPos(0, 0, 0);
    bCopyParams.srcPtr = make_cudaPitchedPtr(cb.data, cb.size * sizeof(float), cb.size, 1);
    bCopyParams.dstArray = NULL;
    bCopyParams.dstPos = make_cudaPos(0, 0, 0);
    bCopyParams.dstPtr = make_cudaPitchedPtr(gb.data, gb.size * sizeof(float), gb.size, 1);
    bCopyParams.extent = make_cudaExtent(gb.size * sizeof(float), 1, 1);
    bCopyParams.kind = cudaMemcpyHostToDevice;
    cudaGraphNode_t bCopyNode;
    CHECK(cudaGraphAddMemcpyNode(&bCopyNode, graph, NULL, 0, &bCopyParams));

    cudaGraphNode_t addNode;
    cudaKernelNodeParams addParams;
    addParams.func = (void*)add_kernel;
    int threads = 16;
    int blocks = ((int)ga.size + threads - 1) / threads;
    addParams.gridDim = dim3(blocks, 1, 1);
    addParams.blockDim = dim3(threads, 1, 1);
    addParams.sharedMemBytes = 0;
    int size = (int)ga.size;
    void* addArgs[4] = { (void*)&size, (void*)&ga.data, (void*)&gb.data, (void*)&gc.data };
    addParams.kernelParams = addArgs;
    addParams.extra = NULL;
    std::vector<cudaGraphNode_t> addDependencies;
    addDependencies.push_back(aCopyNode);
    addDependencies.push_back(bCopyNode);
    CHECK(cudaGraphAddKernelNode(&addNode, graph, addDependencies.data(), addDependencies.size(), &addParams));

    cudaMemcpy3DParms cCopyParams;
    cCopyParams.srcArray = NULL;
    cCopyParams.srcPos = make_cudaPos(0, 0, 0);
    cCopyParams.srcPtr = make_cudaPitchedPtr(gc.data, gc.size * sizeof(float), gc.size, 1);
    cCopyParams.dstArray = NULL;
    cCopyParams.dstPos = make_cudaPos(0, 0, 0);
    cCopyParams.dstPtr = make_cudaPitchedPtr(cc.data, cc.size * sizeof(float), cc.size, 1);
    cCopyParams.extent = make_cudaExtent(gc.size * sizeof(float), 1, 1);
    cCopyParams.kind = cudaMemcpyDeviceToHost;
    cudaGraphNode_t cCopyNode;
    std::vector<cudaGraphNode_t> cCopyDependencies;
    cCopyDependencies.push_back(addNode);
    CHECK(cudaGraphAddMemcpyNode(&cCopyNode, graph, cCopyDependencies.data(), cCopyDependencies.size(), &cCopyParams));

    cudaGraphNode_t* nodes = NULL;
    size_t numNodes = 0;
    CHECK(cudaGraphGetNodes(graph, nodes, &numNodes));
    printf("Num of nodes in the graph created manually = %zu\n", numNodes);

    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cudaGraphExec_t graphExec;
    CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    CHECK(cudaGraphLaunch(graphExec, stream));

    CHECK(cudaStreamSynchronize(stream));

    CHECK(cudaGraphExecDestroy(graphExec));

    CHECK(cudaGraphDestroy(graph));

    CHECK(cudaStreamDestroy(stream));

    if (!check(cd, cc, "add_test"))
        return;

    std::cout << "add_test is finished." << std::endl;
}

