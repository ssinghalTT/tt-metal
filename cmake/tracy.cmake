# Built as outlined in Tracy documentation (pg.12)
set(TRACY_HOME ${PROJECT_SOURCE_DIR}/tt_metal/third_party/tracy)

if(NOT ENABLE_TRACY)
    # Stub Tracy::TracyClient to provide the headers which themselves provide stubs
    add_library(TracyClient INTERFACE)
    add_library(Tracy::TracyClient ALIAS TracyClient)
    target_include_directories(TracyClient SYSTEM INTERFACE "$<BUILD_INTERFACE:${TRACY_HOME}/public>")

    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
        target_sources(
            TracyClient
            INTERFACE
                FILE_SET api
                TYPE HEADERS
                BASE_DIRS ${TRACY_HOME}/public
                FILES
                    ${TRACY_HOME}/public/tracy/Tracy.hpp
                    ${TRACY_HOME}/public/common/TracyApi.h
                    ${TRACY_HOME}/public/common/TracyColor.hpp
                    ${TRACY_HOME}/public/common/TracySystem.hpp
        )
    endif()
    return()
endif()

add_subdirectory(${TRACY_HOME})

set_target_properties(
    TracyClient
    PROPERTIES
        EXCLUDE_FROM_ALL
            TRUE
        LIBRARY_OUTPUT_DIRECTORY
            "${PROJECT_BINARY_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY
            "${PROJECT_BINARY_DIR}/lib"
        OUTPUT_NAME
            "tracy"
)

target_compile_definitions(TracyClient PUBLIC TRACY_ENABLE)
target_compile_options(TracyClient PUBLIC -fno-omit-frame-pointer)
target_link_options(TracyClient PUBLIC -rdynamic)

# Our current fork of tracy does not have CMake support for these subdirectories
# Once we update, we can change this
include(ExternalProject)
ExternalProject_Add(
    tracy_csv_tools
    PREFIX ${TRACY_HOME}/csvexport/build/unix
    SOURCE_DIR ${TRACY_HOME}/csvexport/build/unix
    BINARY_DIR ${PROJECT_BINARY_DIR}/tools/profiler/bin
    INSTALL_DIR ${PROJECT_BINARY_DIR}/tools/profiler/bin
    STAMP_DIR "${PROJECT_BINARY_DIR}/tmp/tracy_stamp"
    TMP_DIR "${PROJECT_BINARY_DIR}/tmp/tracy_tmp"
    DOWNLOAD_COMMAND
        ""
    CONFIGURE_COMMAND
        ""
    INSTALL_COMMAND
        cp ${TRACY_HOME}/csvexport/build/unix/csvexport-release .
    BUILD_COMMAND
        cd ${TRACY_HOME}/csvexport/build/unix && CXX=g++ TRACY_NO_LTO=1 make -f
        ${TRACY_HOME}/csvexport/build/unix/Makefile
)
ExternalProject_Add(
    tracy_capture_tools
    PREFIX ${TRACY_HOME}/capture/build/unix
    SOURCE_DIR ${TRACY_HOME}/capture/build/unix
    BINARY_DIR ${PROJECT_BINARY_DIR}/tools/profiler/bin
    INSTALL_DIR ${PROJECT_BINARY_DIR}/tools/profiler/bin
    STAMP_DIR "${PROJECT_BINARY_DIR}/tmp/tracy_stamp"
    TMP_DIR "${PROJECT_BINARY_DIR}/tmp/tracy_tmp"
    DOWNLOAD_COMMAND
        ""
    CONFIGURE_COMMAND
        ""
    INSTALL_COMMAND
        cp ${TRACY_HOME}/capture/build/unix/capture-release .
    BUILD_COMMAND
        cd ${TRACY_HOME}/capture/build/unix && CXX=g++ TRACY_NO_LTO=1 make -f ${TRACY_HOME}/capture/build/unix/Makefile
)
add_custom_target(
    tracy_tools
    ALL
    DEPENDS
        tracy_csv_tools
        tracy_capture_tools
)
