cmake_minimum_required(VERSION 3.27)
project(cluprop LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package (Eigen3 REQUIRED NO_MODULE)
find_package(Boost 1.71 REQUIRED NO_MODULE)
find_package(OpenMP)

add_executable(cluprop
#        src/main.cpp
        src/utilities.cpp
        src/cluprop.cpp
)

target_link_libraries (cluprop PRIVATE Eigen3::Eigen)
target_include_directories(cluprop PRIVATE ${Boost_INCLUDE_DIRS})

#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()

if(OpenMP_CXX_FOUND) # Ubuntu
    target_link_libraries(cluprop PRIVATE OpenMP::OpenMP_CXX)
elseif(APPLE) # Apple
    target_include_directories(cluprop PRIVATE /opt/homebrew/opt/libomp/include)
    target_link_directories(cluprop PRIVATE /opt/homebrew/opt/libomp/lib)
    target_link_libraries(cluprop PRIVATE omp)
endif()