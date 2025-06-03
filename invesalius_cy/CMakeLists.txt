find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRED)

include(UseCython)
find_package(OpenMP REQUIRED)

set(cython_modules mips interpolation transforms)
set(cython_modules_cpp cy_mesh floodfill)

# Ensure modules are built in the correct directory on Windows
if(WIN32)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

foreach(cython_module ${cython_modules})
  cython_transpile(${cython_module}.pyx LANGUAGE C OUTPUT ${cython_module}.c)
  python_add_library(${cython_module} MODULE ${cython_module}.c WITH_SOABI)
  target_include_directories(${cython_module}
                             PRIVATE "." ${Python_NumPy_INCLUDE_DIRS})
  target_link_libraries(${cython_module} PRIVATE OpenMP::OpenMP_C)
  install(TARGETS ${cython_module} DESTINATION .)
endforeach()

foreach(cython_module ${cython_modules_cpp})
  cython_transpile(${cython_module}.pyx LANGUAGE CXX OUTPUT ${cython_module}.cxx)
  python_add_library(${cython_module} MODULE ${cython_module}.cxx WITH_SOABI)
  target_include_directories(${cython_module}
                             PRIVATE "." ${Python_NumPy_INCLUDE_DIRS})
  target_link_libraries(${cython_module} PRIVATE OpenMP::OpenMP_CXX)
  install(TARGETS ${cython_module} DESTINATION .)
endforeach()
