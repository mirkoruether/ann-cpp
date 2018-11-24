add_library(annlib 
activation_function.h       activation_function.cpp
cost_function.h             cost_function.cpp
gradient_based_optimizer.h  gradient_based_optimizer.cpp
net_init.h                  net_init.cpp
neural_network.h			neural_network.cpp
sgd_trainer.h               sgd_trainer.cpp
training_data.h
weight_norm_penalty.h       weight_norm_penalty.cpp
)

add_dependencies(annlib linalg)

install(TARGETS annlib DESTINATION lib)
install(FILES 
activation_function.h     
cost_function.h            
gradient_based_optimizer.h  
net_init.h
neural_network.h
sgd_trainer.h               
training_data.h
weight_norm_penalty.h 
DESTINATION include)