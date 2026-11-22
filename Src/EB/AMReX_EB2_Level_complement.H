#ifndef AMREX_EB2_LEVEL_COMPLEMENT_H_
#define AMREX_EB2_LEVEL_COMPLEMENT_H_
#include <AMReX_Config.H>

#include <AMReX_EB2_Level.H>

/**
 * \file AMReX_EB2_Level_complement.H
 *
 * EB2 level holding the complement of another level's geometry.
 */

namespace amrex::EB2 {

//! Tag type for the GShopLevel base of ComplementLevel. It builds no geometry itself.
struct ComplementTag {};

/**
 * \brief EB level of the region that another level's geometry covers.
 *
 * The finest level is built from the data of an existing level, cell by cell, instead of
 * from an implicit function, so that it is the exact complement of that level: in every
 * cell the volume fractions add up to one, and in a cut cell the boundary area and centroid
 * are the same and the boundary normal is reversed. The area fractions add up to one on
 * every face that the boundary does not lie on; a face it lies on is closed on both sides,
 * and its boundary moves to the cell on the complement's side. Coarser levels are coarsened
 * from it in the same way as for any other geometry.
 */
class ComplementLevel
    : public GShopLevel<ComplementTag>
{
public:

    //! Build the complement of \p level, which must have been built on \p geom.
    ComplementLevel (IndexSpace const* is, Level const& level, Geometry const& geom);

    //! Coarsening constructor used to build the levels below the finest one.
    ComplementLevel (IndexSpace const* is, int ilev, int max_grid_size, int ngrow,
                     Geometry const& geom, ComplementLevel& fineLevel);

    //! Whether the coarse levels are extruded at the non-periodic domain faces.
    [[nodiscard]] bool extrudeCoarseLevels () const noexcept { return m_extrude_coarse_levels; }

    // public for cuda
    void define_complement (Level const& level);
};

}

#endif
