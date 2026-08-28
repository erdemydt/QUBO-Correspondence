cmake_minimum_required(VERSION 3.16)
project(qubo_playground CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(qubo_maxcut src/main.cpp)
target_include_directories(qubo_maxcut PRIVATE include)
