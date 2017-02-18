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
# function(CUDNN_INSTALL version dest_dir)
#  This function will try to download and install CUDNN.
#  CUDNN5 and CUDNN6 are supported.
#
#

function(CUDNN_INSTALL version dest_dir)
  string(REGEX REPLACE "-rc$" "" version_base "${version}")

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
    set(__cudnn_url ${__base_url}/v${version_base}/${__cudnn_filename})
    set(__cudnn_tgz ${__download_dir}/${__cudnn_filename})
    
    if(NOT EXISTS ${__cudnn_tgz})
      message("Downloading CUDNN library from NVIDIA...")
      file(DOWNLOAD ${__cudnn_url} ${__cudnn_tgz}
	SHOW_PROGRESS STATUS CUDNN_STATUS
	)
      if("${CUDNN_STATUS}" MATCHES "0")
	execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${__cudnn_tgz}" WORKING_DIRECTORY "${__download_dir}")
      else()
	message("Was not able to download CUDNN from ${__cudnn_url}. Please install CuDNN manually from https://developer.nvidia.com/cuDNN")
	file(REMOVE ${__cudnn_tgz})
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
    unset(CUDNN_LIBRARY CACHE)
    unset(CUDNN_INCLUDE_DIR CACHE)
  endif(__url_arch_name)
endfunction()

#####################################################

find_package(PkgConfig)
pkg_check_modules(PC_CUDNN QUIET CUDNN)

get_filename_component(__libpath_cudart ${CUDA_CUDART_LIBRARY} PATH)

find_path(CUDNN_INCLUDE_DIR 
  NAMES cudnn.h
  PATHS ${PC_CUDNN_INCLUDE_DIRS} ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_INCLUDE}
  DOC "Path to CUDNN include directory." )

# We use major only in library search as major/minor is not entirely consistent among platforms.
# Also, looking for exact minor version of .so is in general not a good idea.
# More strict enforcement of minor/patch version is done if/when the header file is examined.
if(CUDNN_FIND_VERSION_EXACT)
  SET(__cudnn_ver_suffix ".${CUDNN_FIND_VERSION_MAJOR}")
  SET(__cudnn_lib_win_name cudnn64_${CUDNN_FIND_VERSION_MAJOR}.dll)
  SET(CUDNN_MAJOR_VERSION ${CUDNN_FIND_MAJOR_VERSION})
else()
  SET(__cudnn_lib_win_name cudnn64.dll)
endif()

find_library(CUDNN_LIBRARY 
  NAMES libcudnn.so${__cudnn_ver_suffix} libcudnn${__cudnn_ver_suffix}.dylib ${__cudnn_lib_win_name}
  PATHS $ENV{LD_LIBRARY_PATH} ${__libpath_cudart} ${PC_CUDNN_LIBRARY_DIRS}
  DOC "CUDNN library." )

mark_as_advanced(CUDNN_INCLUDE_DIR CUDNN_LIBRARY)

if(CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
  file(READ ${CUDNN_INCLUDE_DIR}/cudnn.h CUDNN_VERSION_FILE_CONTENTS)
  string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
    CUDNN_MAJOR_VERSION "${CUDNN_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
    CUDNN_MAJOR_VERSION "${CUDNN_MAJOR_VERSION}")
  string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
    CUDNN_MINOR_VERSION "${CUDNN_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
    CUDNN_MINOR_VERSION "${CUDNN_MINOR_VERSION}")
  string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
    CUDNN_PATCH_VERSION "${CUDNN_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
    CUDNN_PATCH_VERSION "${CUDNN_PATCH_VERSION}")  
endif()


if(CUDNN_MAJOR_VERSION)
## Fixing the case where 5.1 does not fit 'exact' 5.
  set(CUDNN_VERSION ${CUDNN_MAJOR_VERSION}.${CUDNN_MINOR_VERSION})
  if(CUDNN_FIND_VERSION_EXACT AND "x${CUDNN_FIND_VERSION_MINOR}" STREQUAL "x")
    if(CUDNN_MAJOR_VERSION EQUAL CUDNN_FIND_VERSION_MAJOR)
      set(CUDNN_VERSION ${CUDNN_FIND_VERSION})
    endif()
  endif()
    math(EXPR CUDNN_VERSION_NUM "${CUDNN_MAJOR_VERSION} * 1000 + ${CUDNN_MINOR_VERSION} * 100 + ${CUDNN_PATCH_VERSION}")
else()
# Try to set CUDNN version from config file
  set(CUDNN_VERSION ${PC_CUDNN_CFLAGS_OTHER})
endif()
  
find_package_handle_standard_args(CUDNN
  FOUND_VAR CUDNN_FOUND
  REQUIRED_VARS CUDNN_LIBRARY 
  VERSION_VAR   CUDNN_VERSION
  )

if(CUDNN_FOUND)
  set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})
  set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})
  set(CUDNN_DEFINITIONS ${PC_CUDNN_CFLAGS_OTHER})
endif()
