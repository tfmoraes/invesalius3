
find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)
# find_package(NumPy REQUIRED)
find_package(OpenMP REQUIRED)

set(cython_modules cy_mesh floodfill interpolation mips transforms)

foreach(cython_module ${cython_modules})
add_custom_command(
  OUTPUT ${cython_module}.cxx
  COMMENT
    "Making ${CMAKE_CURRENT_BINARY_DIR}/${cython_module}.cxx from ${CMAKE_CURRENT_SOURCE_DIR}/${cython_module}.pyx"
  COMMAND Python::Interpreter -m cython --cplus "${CMAKE_CURRENT_SOURCE_DIR}/${cython_module}.pyx" --output-file ${cython_module}.cxx
  VERBATIM)
  python_add_library(${cython_module} MODULE ${cython_module}.cxx WITH_SOABI)
  target_include_directories(${cython_module} PRIVATE "." ${Python_NumPy_INCLUDE_DIRS})
  target_link_libraries(${cython_module} PRIVATE OpenMP::OpenMP_CXX)
  install(TARGETS ${cython_module} DESTINATION invesalius_cy)
endforeach()
