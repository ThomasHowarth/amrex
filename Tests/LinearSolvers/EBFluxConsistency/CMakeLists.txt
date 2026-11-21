# The checks run on the host, and need double precision.
if (NOT AMReX_EB OR NOT AMReX_GPU_BACKEND STREQUAL NONE
    OR AMReX_PRECISION STREQUAL "SINGLE")
   return()
endif ()

foreach(D IN LISTS AMReX_SPACEDIM)
    if (D EQUAL 1)
       continue()
    endif ()

    set(_sources main.cpp)
    set(_input_files inputs)

    # Dirichlet walls and Dirichlet EB
    setup_test(${D} _sources _input_files)

    # Homogeneous Neumann EB, mixed and periodic walls, beta on face
    # centroids, two components and anisotropic cells
    setup_test(${D} _sources _input_files
       BASE_NAME LinearSolvers_EBFluxConsistency_mixed
       RUNTIME_SUBDIR mixed
       CMDLINE_PARAMS_AFTER_INPUTS eb_bc=neumann beta_on_centroid=1 ncomp=2 n_cell=32 16 24
                                   lo_bc=periodic neumann inhomog_neumann
                                   hi_bc=periodic inhomog_neumann dirichlet)

    # Robin walls (one component only) with homogeneous Dirichlet EB and a
    # scalar EB beta
    setup_test(${D} _sources _input_files
       BASE_NAME LinearSolvers_EBFluxConsistency_robin
       RUNTIME_SUBDIR robin
       CMDLINE_PARAMS_AFTER_INPUTS eb_bc=homog_dirichlet eb_beta_scalar=1
                                   lo_bc=robin robin robin hi_bc=robin robin robin)

    unset(_sources)
    unset(_input_files)
endforeach()
