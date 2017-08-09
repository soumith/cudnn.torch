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
# <EXACT> : as usual, plus we do find '5.1' version if you wanted '5' 
#           (not if you wanted '5.0', as usual)   
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
# Exported functions
# ^^^^^^^^^^^^^^^^
# function(CUDNN_INSTALL version __dest_libdir [__dest_incdir])
#  This function will try to download and install CUDNN.
#  CUDNN5 and CUDNN6 are supported.
#
#

function(CUDNN_INSTALL version dest_libdir dest_incdir dest_bindir)
  message(STATUS "CUDNN_INSTALL: Installing CUDNN ${version}, lib:${dest_libdir}, inc:${dest_incdir}, bin:${dest_bindir}")
  string(REGEX REPLACE "-rc$" "" version_base "${version}")
  set(tar_libdir cuda/lib64)
  set(tar_incdir cuda/include)

  if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(url_extension tgz)
    if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
      set(url_arch_name linux-x64 )
    elseif("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "ppc")
      set(url_arch_name linux-ppc64le ) 
      #  TX1 has to be installed via JetPack
    endif()
  elseif  (APPLE)
    set(url_extension tgz)
    set(tar_libdir cuda/lib)
    set(url_arch_name osx-x64)
  elseif(WIN32)
    set(url_extension zip)
    set(tar_bindir cuda/bin)
    set(tar_libdir cuda/lib/x64)
    if(CMAKE_SYSTEM_VERSION MATCHES "10")
      set(url_arch_name windows10-x64)
    else()
      set(url_arch_name windows7-x64)
    endif()
  endif()
  
  # Download and install CUDNN locally if not found on the system
  if(url_arch_name) 
    set(download_dir ${CMAKE_CURRENT_BINARY_DIR}/downloads/cudnn${version})
    file(MAKE_DIRECTORY ${download_dir})
    set(cudnn_filename cudnn-${CUDA_VERSION}-${url_arch_name}-v${version}.${url_extension})
    set(base_url http://developer.download.nvidia.com/compute/redist/cudnn)
    set(cudnn_url ${base_url}/v${version_base}/${cudnn_filename})
    set(cudnn_file ${download_dir}/${cudnn_filename})
    
    if(NOT EXISTS ${cudnn_file})
      message(STATUS "Downloading CUDNN library from NVIDIA...")
      file(DOWNLOAD ${cudnn_url} ${cudnn_file}
	SHOW_PROGRESS STATUS cudnn_status
	)
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzvf ${cudnn_file} WORKING_DIRECTORY ${download_dir} RESULT_VARIABLE cudnn_status)

      if(NOT "${cudnn_status}" MATCHES "0")
	message(STATUS "Was not able to download CUDNN from ${cudnn_url}. Please install CuDNN manually from https://developer.nvidia.com/cuDNN")
      endif()
    endif()
    
    if(dest_bindir AND tar_bindir)
      file(COPY ${download_dir}/${tar_bindir}/ DESTINATION ${dest_bindir})
    endif()

    if(dest_incdir)
      file(COPY ${download_dir}/${tar_incdir}/ DESTINATION  ${dest_incdir})
    endif()

    file(COPY ${download_dir}/${tar_libdir}/ DESTINATION  ${dest_libdir} )

    execute_process(COMMAND ldconfig)
    
    get_filename_component(dest_dir ${dest_libdir} DIRECTORY)
    set(CUDNN_ROOT_DIR ${dest_dir} PARENT_SCOPE)
    unset(CUDNN_LIBRARY CACHE)
    unset(CUDNN_INCLUDE_DIR CACHE)

  endif(url_arch_name)
endfunction()

#####################################################
# FIXME
    unset(CUDNN_LIBRARY CACHE)
    unset(CUDNN_INCLUDE_DIR CACHE)


macro(__cudnn_find_library _var _names)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(_cuda_64bit_lib_dir lib/x64 lib6 libx64 bin64)
  endif()
  find_library(${_var}
    NAMES ${_names}
    PATHS ${CMAKE_INSTALL_PREFIX} ${CUDNN_ROOT_DIR} ${__libpath_cudart}
    ENV CUDA_LIB_PATH
    ENV LD_LIBRARY_PATH
    PATH_SUFFIXES ${_cuda_64bit_lib_dir} lib bin
    DOC "CUDNN Library")
endmacro()

get_filename_component(__libpath_cudart "${CUDA_CUDART_LIBRARY}" PATH)

# Make a separate find_libabry call with major version.
# That will find 6.0 even if 5.x is first in the path.
if(CUDNN_FIND_VERSION_EXACT)
  set(__cudnn_ver_suffix ".${CUDNN_FIND_VERSION_MAJOR}")
  if(CUDNN_FIND_VERSION_PATCH)
    set(__cudnn_ver_suffix
      ".${CUDNN_FIND_VERSION_MAJOR}.${CUDNN_FIND_VERSION_MINOR}.${CUDNN_FIND_VERSION_PATCH}")
  else()
    if (CUDNN_FIND_VERSION_MINOR)
      message(STATUS "Warning: Ignoring VERSION_MINOR for CUDNN version if VERSION_PATCH not set")
    endif()
  endif()
  __cudnn_find_library(CUDNN_LIBRARY 
    libcudnn.so${__cudnn_ver_suffix} libcudnn${__cudnn_ver_suffix}.dylib ${__cudnn_lib_win_name}
    )
endif()

#Now, try no version on shared object
if(NOT CUDNN_LIBRARY)
  __cudnn_find_library(CUDNN_LIBRARY 
    cudnn
    )
endif()

# verify exact version by the header
if(CUDNN_LIBRARY)
  get_filename_component(__found_cudnn_root ${CUDNN_LIBRARY} PATH)
  find_path(CUDNN_INCLUDE_DIR 
    NAMES cudnn.h
    HINTS ${__found_cudnn_root} ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_INCLUDE} 
    PATH_SUFFIXES include 
    DOC "Path to CUDNN include directory." )
  if(NOT CUDNN_INCLUDE_DIR)
    # no header found: assume everything we could check, matches
    set(CUDNN_VERSION_MAJOR ${CUDNN_FIND_VERSION_MAJOR})
    set(CUDNN_VERSION_MINOR ${CUDNN_FIND_VERSION_MINOR})
    set(CUDNN_VERSION_PATCH ${CUDNN_FIND_VERSION_PATCH})
  else()
    file(READ ${CUDNN_INCLUDE_DIR}/cudnn.h CUDNN_VERSION_FILE_CONTENTS)
    string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
      CUDNN_MAJOR "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
      CUDNN_VERSION_MAJOR "${CUDNN_MAJOR}")
    string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
      CUDNN_MINOR "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
      CUDNN_VERSION_MINOR "${CUDNN_MINOR}")
    string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
      CUDNN_PATCH "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
      CUDNN_VERSION_PATCH "${CUDNN_PATCH}")  
  endif()

  if (CUDNN_FIND_VERSION_PATCH)
    set(CUDNN_VERSION ${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH})
  else()
    if (CUDNN_FIND_VERSION_MINOR)
      set(CUDNN_VERSION ${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR})
    endif()
  endif()

endif()
  
find_package_handle_standard_args(
  CUDNN 
  FOUND_VAR CUDNN_FOUND
  REQUIRED_VARS CUDNN_LIBRARY 
  VERSION_VAR   CUDNN_VERSION
  )

if(CUDNN_FOUND)
  set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})
  set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})
endif()  
