#include "tests.h"
#include "nodes.h"
#include "tensor.h"
#include "utils.h"

struct add_args_t
{
    cudaStream_t stream;
    cublasHandle_t handle;
    int size;
    const float* a;
    const float* b;
    float* c;
};

void add_host_cublas(void* user_data)
{
    const add_args_t* args = (add_args_t*)user_data;
    add_cublas(args->handle, args->size, args->a, args->b, args->c);
}

void cublas_test()
{
    std::cout << "Start cublas_test:" << std::endl;

    const size_t N = 1, C = 1, H = 32, W = 32;
    shape_t shape = to_shape(N, C, H, W);

    cpu_tensor_t ca(shape), cb(shape), cc(shape), cd(shape);

    init_rand(ca, 0.0f, 1.0f);
    init_rand(cb, 0.0f, 1.0f);
    init_rand(cc, -2.0f, -1.0f);

    add_cpu((int)ca.size, ca.data, cb.data, cd.data);

    gpu_tensor_t ga(shape), gb(shape), gc(shape);

    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

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

    cublasHandle_t cublasHandle;
    cublasCreate_v2(&cublasHandle);
    cublasSetStream_v2(cublasHandle, stream);

    cudaGraphNode_t addNode;
    cudaHostNodeParams addParams;
    add_args_t addArgs;
    addArgs.stream = stream;
    addArgs.handle = cublasHandle;
    addArgs.size = (int)ga.size;
    addArgs.a = ga.data;
    addArgs.b = gb.data;
    addArgs.c = gc.data;
    addParams.fn= add_host_cublas;
    addParams.userData = &addArgs;
    std::vector<cudaGraphNode_t> addDependencies;
    addDependencies.push_back(aCopyNode);
    addDependencies.push_back(bCopyNode);
    CHECK(cudaGraphAddHostNode(&addNode, graph, addDependencies.data(), addDependencies.size(), &addParams));

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

    cudaGraphExec_t graphExec;
    CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    CHECK(cudaGraphLaunch(graphExec, stream));

    CHECK(cudaStreamSynchronize(stream));

    cublasDestroy_v2(cublasHandle);

    CHECK(cudaGraphExecDestroy(graphExec));

    CHECK(cudaGraphDestroy(graph));

    CHECK(cudaStreamDestroy(stream));


    if (!check(cd, cc, "cublas_test"))
        return;

    std::cout << "cublas_test is finished." << std::endl;
}

