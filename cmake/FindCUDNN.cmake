# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#.rst:
# FindCUDNN
# -------
#
# Find CUDNN library
#
# Valiables that affect result:
# <VERSION>, <REQUIRED>, <QUIETLY>: as usual
#
# Result variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project:
#
# ``CUDNN_INCLUDE``
#   where to find cudnn.h.
# ``CUDNN_LIBRARY``
#   the libraries to link against to use CUDNN.
# ``CUDNN_FOUND``
#   If false, do not try to use CUDNN.
# ``CUDNN_VERSION``
#   Version of the CUDNN library we looked for 
#

# Unlike some other packages, we'll look for exact match (major) only.

function(CUDNN_INSTALL version dest_dir)

  if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
      set(__url_arch_name linux-x64 )
    elseif("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "ppc")
      set(__url_arch_name linux-ppc64le ) 
      #  TX1 has to be installed via JetPack
    endif()
  elseif  (APPLE)
    set(__url_arch_name osx-x64)
  elseif(WIN32)
    if(CMAKE_SYSTEM_VERSION MATCHES "10")
      set(__url_arch_name windows10)
    else()
      set(__url_arch_name windows7)
    endif()
  endif()
  
  # Download and install CUDNN locally if not found on the system
  if(__url_arch_name) 
    set(__download_dir ${CMAKE_CURRENT_BINARY_DIR}/downloads)
    file(MAKE_DIRECTORY ${__download_dir})
    set(__cudnn_filename cudnn-${CUDA_VERSION}-${__url_arch_name}-v${version}.tgz)
    set(__base_url http://developer.download.nvidia.com/compute/redist/cudnn)
    set(__cudnn_url ${__base_url}/v${version}/${__cudnn_filename})
    set(__cudnn_tgz ${__download_dir}/${__cudnn_filename})
    
    if(NOT EXISTS ${__cudnn_tgz})
      message("Downloading CUDNN library from NVIDIA...")
      file(DOWNLOAD ${__cudnn_url} ${__cudnn_tgz}
	SHOW_PROGRESS STATUS CUDNN_STATUS
	)
      if("${CUDNN_STATUS}" MATCHES "0")
	execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${__cudnn_tgz}" WORKING_DIRECTORY "${__download_dir}")
      else()
	message("Was not able to download CUDNN. Please install CuDNN manually from https://developer.nvidia.com/cuDNN")
      endif()
    endif()
    
    if(WIN32)
      file(GLOB __cudnn_binfiles ${__download_dir}/cuda/bin*/*)
      install(FILES ${__cudnn_binfiles} 
	DESTINATION  "${dest_dir}/bin")
    endif()
    
    file(GLOB __cudnn_incfiles ${__download_dir}/cuda/include/*)
    install(FILES ${__cudnn_incfiles} 
      DESTINATION  "${dest_dir}/include")

    file(GLOB __cudnn_libfiles ${__download_dir}/cuda/lib*/*)
    install(FILES ${__cudnn_libfiles} 
      DESTINATION  "${dest_dir}/lib")

  endif(__url_arch_name)
endfunction()

#####################################################

get_filename_component(__libpath_cudart ${CUDA_CUDART_LIBRARY} PATH)
unset(CUDNN_LIBRARY CACHE)

find_path(CUDNN_INCLUDE cudnn.h
  PATHS ${CUDNN_PATH} $ENV{CUDNN_PATH} ${CUDA_TOOLKIT_INCLUDE}
  DOC "Path to CUDNN include directory." )

find_library(CUDNN_LIBRARY NAMES libcudnn.so.${CUDNN_FIND_VERSION_MAJOR} libcudnn.${CUDNN_FIND_VERSION_MAJOR}.dylib cudnn64_${CUDNN_FIND_VERSION_MAJOR}.dll
  PATH_SUFFIXES so.${CUDNN_FIND_VERSION_MAJOR}
  PATHS $ENV{CUDNN_PATH} $ENV{LD_LIBRARY_PATH} ${__libpath_cudart}
  DOC "Path to CUDNN library directory." )

# if(CUDNN_LIBRARY)
  # Note: we do not require cudnn.h - check for CUDNN_INCLUDE if needed
#  set(CUDNN_FOUND TRUE)
#  message(STATUS "Found CUDNN library: ${CUDNN_LIBRARY}")
#endif()

# include(${CMAKE_CURRENT_LIST_DIR}/FindPackageHandleStandardArgs.cmake)

find_package_handle_standard_args(CUDNN
                                  REQUIRED_VARS CUDNN_LIBRARY 
                                  VERSION_VAR   CUDNN_VERSION)

mark_as_advanced(CUDNN_INCLUDE CUDNN_LIBRARY )
