add_library(annlib 
trainingdata.h 
networklayer.h			networklayer.cpp 
neuralnetwork.h			neuralnetwork.cpp 
sgdtrainer.h			sgdtrainer.cpp
activation_function.h	activation_function.cpp
costfunction.h			costfunction.cpp
costfunctionreg.h		costfunctionreg.cpp 
netinit.h				netinit.cpp)

add_dependencies(annlib linalg)

install(TARGETS annlib DESTINATION lib)
install(FILES 
trainingdata.h 
networklayer.h 
neuralnetwork.h 
sgdtrainer.h 
activationfunction.h 
costfunction.h 
costfunctionreg.h 
netinit.h
DESTINATION include)