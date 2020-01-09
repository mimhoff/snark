install( PROGRAMS ouster-cat DESTINATION ${CMAKE_INSTALL_PREFIX}/${snark_INSTALL_BIN_DIR} )

set( ouster_view_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/ouster_view )
file( GLOB ouster_view_sources ${ouster_view_src_dir}/*.cpp )
file( GLOB ouster_view_includes ${ouster_view_src_dir}/*.h )

source_group( ouster-view FILES ouster-view.cpp ${ouster_view_sources} ${ouster_view_includes} )
add_executable( ouster-view ouster-view.cpp ${ouster_view_sources} ${ouster_view_includes} )
target_link_libraries( ouster-view snark_ouster )
install( TARGETS ouster-view RUNTIME DESTINATION ${snark_INSTALL_BIN_DIR} COMPONENT Runtime )
