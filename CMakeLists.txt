cmake_minimum_required(VERSION 3.5...3.22)

project(InVesalius)

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)

add_subdirectory(invesalius_cy)
