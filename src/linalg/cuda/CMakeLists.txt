add_library(linalg_cudaops 
cuda_util.cuh
linalg_cudaops_t.cuh
linalg_cudaops.cuh linalg_cudaops.cu
)

add_dependencies(linalg_cudaops linalg)

install(TARGETS linalg_cudaops DESTINATION lib)
install(FILES cuda_util.cuh linalg_cudaops_t.cuh linalg_cudaops.cuh DESTINATION include)
