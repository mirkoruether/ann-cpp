add_library(annlib_tasks
        unambiguous_classification.h unambiguous_classification.cpp
        )

add_dependencies(annlib_tasks annlib linalg)

install(TARGETS annlib_tasks DESTINATION lib)
install(FILES
        unambiguous_classification.h
        DESTINATION include)