# The tolerance needs double precision.
if (AMReX_PRECISION STREQUAL "SINGLE")
   return()
endif ()

foreach(D IN LISTS AMReX_SPACEDIM)
    if (D EQUAL 1)
       continue()
    endif ()

    set(_sources main.cpp)
    set(_input_files inputs)

    set(_ops abeclap tensor)
    if (AMReX_EB)
       list(APPEND _ops ebabeclap ebtensor)
    endif ()

    foreach(_op IN LISTS _ops)
       setup_test(${D} _sources _input_files
          BASE_NAME LinearSolvers_RefluxConsistency_${_op}
          RUNTIME_SUBDIR ${_op}
          CMDLINE_PARAMS_AFTER_INPUTS op=${_op})
    endforeach()

    # Tensor operators with the fine level at the domain boundary and
    # periodic in x
    foreach(_op IN LISTS _ops)
       if (NOT _op MATCHES "tensor")
          continue()
       endif ()
       setup_test(${D} _sources _input_files
          BASE_NAME LinearSolvers_RefluxConsistency_${_op}_boundary
          RUNTIME_SUBDIR ${_op}_boundary
          CMDLINE_PARAMS_AFTER_INPUTS op=${_op} "fine_lo=0 0 0.25" "fine_hi=0.5 0.75 0.75"
                                      "is_periodic=1 0 0" bc=neumann)
    endforeach()

    unset(_ops)
    unset(_sources)
    unset(_input_files)
endforeach()
