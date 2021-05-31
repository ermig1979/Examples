#include "defs.h"
#include "utils.h"
#include "nodes.h"

int main(int argc, char* argv[])
{
    print_device_info();

    const size_t N = 1, C = 1, H = 128, W = 128;

    cpu_tensor_t ca(to_shape(N, C, H, W)), cb(to_shape(N, C, H, W)), cc(to_shape(N, C, H, W));

    init_rand(ca, 0.0f, 1.0f);
    init_rand(cb, 0.0f, 1.0f);

    gpu_tensor_t ga(to_shape(N, C, H, W)), gb(to_shape(N, C, H, W)), gc(to_shape(N, C, H, W));

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
    CHECK(cudaGraphAddKernelNode(&addNode, graph, NULL, 0, &addParams));

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
    CHECK(cudaGraphAddMemcpyNode(&cCopyNode, graph, NULL, 0, &cCopyParams));

    CHECK(cudaGraphDestroy(graph));
}