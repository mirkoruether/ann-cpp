add_library(annlib_cudaops 
activation_function_cudaops.cuh activation_function_cudaops.cu
sgd_trainer_cudaops.cuh sgd_trainer_cudaops.cu
)

add_dependencies(annlib_cudaops linalg linalg_cudaops)

install(TARGETS annlib_cudaops DESTINATION lib)
install(FILES 
activation_function_cudaops.cuh
sgd_trainer_cudaops.cuh
DESTINATION include)