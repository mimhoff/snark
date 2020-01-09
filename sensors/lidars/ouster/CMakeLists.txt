set( PROJECT "ouster" )
set( TARGET_NAME snark_${PROJECT} )

find_package( Threads )
find_package( Eigen3 REQUIRED )
find_package( VTK REQUIRED COMPONENTS vtkCommonCore NO_MODULE )

# Load additional vtk components; prefer newer vtk opengl module
# https://www.visitusers.org/index.php?title=VTK_7.1_Upgrade
# https://www.vtk.org/Wiki/VTK/Build_System_Migration
set( VIZ_VTK_COMPONENTS vtkCommonCore vtkCommonDataModel vtkCommonMath
                        vtkCommonTransforms vtkFiltersSources vtkImagingCore
                        vtkInteractionStyle vtkRenderingCore )
if( TARGET vtkRenderingOpenGL2 )
  list( APPEND VIZ_VTK_COMPONENTS vtkRenderingOpenGL2 )
else()
  list( APPEND VIZ_VTK_COMPONENTS vtkRenderingOpenGL )
endif()
find_package( VTK REQUIRED COMPONENTS ${VIZ_VTK_COMPONENTS} NO_MODULE )

include(${VTK_USE_FILE})

include_directories( SYSTEM ${EIGEN3_INCLUDE_DIR} ${VTK_INCLUDE_DIRS} )

file( GLOB includes *.h)
file( GLOB sources *.cpp )

source_group( ${TARGET_NAME} FILES ${sources} ${includes} )
add_library( ${TARGET_NAME} ${sources} ${includes} )
set_target_properties( ${TARGET_NAME} PROPERTIES ${snark_LIBRARY_PROPERTIES} )
target_link_libraries( ${TARGET_NAME} ${snark_ALL_LIBRARIES} ${VTK_LIBRARIES} )
install(
    TARGETS ${TARGET_NAME}
    RUNTIME DESTINATION ${snark_INSTALL_BIN_DIR} COMPONENT Runtime     # .exe, .dll
    LIBRARY DESTINATION ${snark_INSTALL_LIB_DIR} COMPONENT Runtime     # .so, mod.dll
    ARCHIVE DESTINATION ${snark_INSTALL_LIB_DIR} COMPONENT Development # .a, .lib
)

add_subdirectory( applications )
