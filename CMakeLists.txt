cmake_minimum_required(VERSION 3.27)
project(cluprop LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package (Eigen3 REQUIRED NO_MODULE)
# find_package(Boost 1.71 REQUIRED NO_MODULE)
include(CTest)

add_executable(cluprop
        src/main.cpp # This is to run with main, compile with Release
        src/utilities.cpp
        src/csr_graph.cpp
        src/cluprop.cpp
)


# This is only for CluProp library
#add_library(cluprop
#        src/cluprop.cpp
#        src/csr_graph.cpp
#        src/utilities.cpp
#)
#add_executable(cluprop_run src/main.cpp)

target_link_libraries (cluprop PRIVATE Eigen3::Eigen)
# target_include_directories(cluprop PRIVATE ${Boost_INCLUDE_DIRS})
find_package(OpenMP)

if(OpenMP_CXX_FOUND) # Ubuntu
    target_link_libraries(cluprop PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(cluprop PRIVATE -march=native)
elseif(APPLE) # Apple
    target_include_directories(cluprop PRIVATE /opt/homebrew/opt/libomp/include)
    target_link_directories(cluprop PRIVATE /opt/homebrew/opt/libomp/lib)
    target_compile_options(cluprop PRIVATE -Xpreprocessor -fopenmp)
    target_link_libraries(cluprop PRIVATE omp)
else()
    message(FATAL_ERROR
            "CluProp requires OpenMP. Install an OpenMP-capable compiler/runtime.")
endif()

#if(BUILD_TESTING)
#    add_executable(csr_graph_test
#            test/test_csr_graph.cpp
#            src/csr_graph.cpp
#    )
#    target_include_directories(csr_graph_test PRIVATE src)
#
#    if(OpenMP_CXX_FOUND)
#        target_link_libraries(csr_graph_test PRIVATE OpenMP::OpenMP_CXX)
#    elseif(APPLE)
#        target_include_directories(csr_graph_test PRIVATE /opt/homebrew/opt/libomp/include)
#        target_link_directories(csr_graph_test PRIVATE /opt/homebrew/opt/libomp/lib)
#        target_compile_options(csr_graph_test PRIVATE -Xpreprocessor -fopenmp)
#        target_link_libraries(csr_graph_test PRIVATE omp)
#    endif()
#
#    add_test(NAME csr_graph_test COMMAND csr_graph_test)
#endif()
