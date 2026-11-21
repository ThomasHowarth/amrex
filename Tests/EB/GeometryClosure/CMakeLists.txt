foreach(D IN LISTS AMReX_SPACEDIM)
    if (D EQUAL 1)
       continue()
    endif ()

    set(_sources main.cpp)
    set(_input_files inputs)

    # Sphere crossing the domain boundary
    setup_test(${D} _sources _input_files)

    # Tilted plane: the volume moments are exact
    setup_test(${D} _sources _input_files
       BASE_NAME EB_GeometryClosure_plane
       RUNTIME_SUBDIR plane
       CMDLINE_PARAMS_AFTER_INPUTS eb2.geom_type=plane "eb2.plane_point=0.5 0.5 0.5"
                                   "eb2.plane_normal=0.3 -0.8 0.5" max_vol_err=1.e-12
                                   max_coarse_area_err=1.e-12)

    # Wavy wall, periodic along it, on an anisotropic grid
    setup_test(${D} _sources _input_files
       BASE_NAME EB_GeometryClosure_periodic
       RUNTIME_SUBDIR periodic
       CMDLINE_PARAMS_AFTER_INPUTS eb2.geom_type=parser
                                   "eb2.parser_function=0.4+0.15*sin(6.283185307179586*y)*cos(6.283185307179586*z)-x"
                                   "n_cell=32 48 16" "geometry.is_periodic=0 1 1"
       NTASKS 2)

    unset(_sources)
    unset(_input_files)
endforeach()
