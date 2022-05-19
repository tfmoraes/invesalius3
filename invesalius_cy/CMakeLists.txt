cmake_minimum_required(VERSION 3.5...3.22)
project(InVesalius)

find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)
find_package(PythonExtensions REQUIRED)
find_package(OpenMP REQUIRED)

set(cython_modules cy_mesh floodfill interpolation mips transforms)

foreach(cython_module ${cython_modules})
  set(module_name "${cython_module}")
  add_cython_target(${module_name} ${cython_module}.pyx PY3 CXX)
  add_library(${module_name} MODULE ${${module_name}})
  target_link_libraries(${module_name} OpenMP::OpenMP_CXX)
  target_include_directories(${module_name} PRIVATE "." ${PYTHON_INCLUDE_DIRS} ${NumPy_INCLUDE_DIRS})
  python_extension_module(${module_name})
  install(TARGETS ${module_name} DESTINATION invesalius_cy)
endforeach()
