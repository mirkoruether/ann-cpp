add_library(annlib NetworkLayer.h NetworkLayer.cpp NeuralNetwork.h NeuralNetwork.cpp SGDTrainer.cpp SGDTrainer.h ActivationFunction.cpp ActivationFunction.h CostFunction.cpp CostFunction.h TrainingData.h ActivationFunction.cpp CostFunctionRegularization.cpp CostFunctionRegularization.h)

install(TARGETS annlib DESTINATION lib)
install(FILES NetworkLayer.h NeuralNetwork.h SGDTrainer.h ActivationFunction.h CostFunction.h TrainingData.h CostFunctionRegularization.h DESTINATION include)