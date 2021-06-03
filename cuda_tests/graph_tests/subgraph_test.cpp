#include "tests.h"
#include "nodes.h"
#include "tensor.h"
#include "utils.h"

void subgraph_manual_test()
{
    std::cout << "Start subgraph_manual_test:" << std::endl;

    const size_t N = 1, C = 1, H = 32, W = 32;
    shape_t shape = to_shape(N, C, H, W);

    cpu_tensor_t ca(shape), cb(shape), cc(shape), cd(shape);

    init_rand(ca, 0.0f, 1.0f);
    init_rand(cb, 0.0f, 1.0f);
    init_rand(cc, -2.0f, -1.0f);

    add_cpu((int)ca.size, ca.data, cb.data, cd.data);
    scale_cpu((int)cd.size, 0.5f, cd.data);

    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    gpu_tensor_t ga(shape, stream), gb(shape, stream), gc(shape, stream);

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

    cudaGraph_t subgraph;
    CHECK(cudaGraphCreate(&subgraph, 0));

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
    CHECK(cudaGraphAddKernelNode(&addNode, subgraph, NULL, 0, &addParams));

    cudaGraphNode_t scaleNode;
    cudaKernelNodeParams scaleParams;
    scaleParams.func = (void*)scale_kernel;
    scaleParams.gridDim = dim3(blocks, 1, 1);
    scaleParams.blockDim = dim3(threads, 1, 1);
    scaleParams.sharedMemBytes = 0;
    float scale = 0.5f;
    void* scaleArgs[3] = { (void*)&size, (void*)&scale, (void*)&gc.data };
    scaleParams.kernelParams = scaleArgs;
    scaleParams.extra = NULL;
    std::vector<cudaGraphNode_t> scaleDependencies;
    scaleDependencies.push_back(addNode);
    CHECK(cudaGraphAddKernelNode(&scaleNode, subgraph, scaleDependencies.data(), scaleDependencies.size(), &scaleParams));

    cudaGraphNode_t subgraphNode;
    std::vector<cudaGraphNode_t> subgraphDependencies;
    subgraphDependencies.push_back(aCopyNode);
    subgraphDependencies.push_back(bCopyNode);
    CHECK(cudaGraphAddChildGraphNode(&subgraphNode, graph, subgraphDependencies.data(), subgraphDependencies.size(), subgraph));

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
    cCopyDependencies.push_back(subgraphNode);
    CHECK(cudaGraphAddMemcpyNode(&cCopyNode, graph, cCopyDependencies.data(), cCopyDependencies.size(), &cCopyParams));

    print_graph_info(graph);

    cudaGraphExec_t graphExec;
    CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    CHECK(cudaGraphLaunch(graphExec, stream));

    CHECK(cudaStreamSynchronize(stream));

    CHECK(cudaGraphExecDestroy(graphExec));

    CHECK(cudaGraphDestroy(graph));

    CHECK(cudaGraphDestroy(subgraph));

    CHECK(cudaStreamDestroy(stream));

    if (!check(cd, cc, "cublas_test"))
        return;

    std::cout << "subgraph_manual_test is finished." << std::endl;
}

void subgraph_capture_test()
{
    std::cout << "Start subgraph_capture_test:" << std::endl;

    const size_t N = 1, C = 1, H = 32, W = 32;
    shape_t shape = to_shape(N, C, H, W);

    cpu_tensor_t ca(shape), cb(shape), cc(shape), cd(shape);

    init_rand(ca, 0.0f, 1.0f);
    init_rand(cb, 0.0f, 1.0f);
    init_rand(cc, -2.0f, -1.0f);

    add_cpu((int)ca.size, ca.data, cb.data, cd.data);
    scale_cpu((int)cd.size, 0.5f, cd.data);

    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cublasHandle_t cublasHandle;
    CHECK(cublasCreate(&cublasHandle));
    CHECK(cublasSetStream(cublasHandle, stream));

    gpu_tensor_t ga(shape, stream), gb(shape, stream), gc(shape, stream);

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

    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));

    add_cublas(cublasHandle, (int)ga.size, ga.data, gb.data, gc.data);
    scale_cublas(cublasHandle, (int)ga.size, 0.5f, gc.data);

    cudaGraph_t subgraph;
    CHECK(cudaStreamEndCapture(stream, &subgraph));

    cudaGraphNode_t subgraphNode;
    std::vector<cudaGraphNode_t> subgraphDependencies;
    subgraphDependencies.push_back(aCopyNode);
    subgraphDependencies.push_back(bCopyNode);
    CHECK(cudaGraphAddChildGraphNode(&subgraphNode, graph, subgraphDependencies.data(), subgraphDependencies.size(), subgraph));

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
    cCopyDependencies.push_back(subgraphNode);
    CHECK(cudaGraphAddMemcpyNode(&cCopyNode, graph, cCopyDependencies.data(), cCopyDependencies.size(), &cCopyParams));

    print_graph_info(graph);

    cudaGraphExec_t graphExec;
    CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    CHECK(cudaGraphLaunch(graphExec, stream));

    CHECK(cudaStreamSynchronize(stream));

    CHECK(cudaGraphExecDestroy(graphExec));

    CHECK(cudaGraphDestroy(graph));

    CHECK(cudaGraphDestroy(subgraph));

    cublasDestroy(cublasHandle);

    CHECK(cudaStreamDestroy(stream));

    if (!check(cd, cc, "cublas_test"))
        return;

    std::cout << "subgraph_capture_test is finished." << std::endl;
}

