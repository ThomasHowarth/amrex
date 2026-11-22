#ifndef AMREX_EB2_INDEXSPACE_COMPLEMENT_H_
#define AMREX_EB2_INDEXSPACE_COMPLEMENT_H_
#include <AMReX_Config.H>

#include <AMReX_EB2.H>
#include <AMReX_EB2_Level_complement.H>

/**
 * \file AMReX_EB2_IndexSpace_complement.H
 *
 * EB index space holding the complement of another index space's geometry.
 */

namespace amrex::EB2 {

/**
 * \brief Index space of the region that another index space's geometry covers.
 *
 * See EB2::BuildComplement.
 */
class IndexSpaceComplement
    : public IndexSpace
{
public:

    /**
     * \brief Build the complement of the geometry of \p index_space.
     *
     * \param index_space Index space to complement. It must have a level on \p geom.
     * \param geom        Finest-level Geometry.
     * \param required_coarsening_level Number of coarse levels required by the application.
     * \param max_coarsening_level      Upper bound on the number of coarse levels built.
     * \param ngrow       Number of EB grow cells on the coarse levels.
     */
    IndexSpaceComplement (IndexSpace const& index_space, const Geometry& geom,
                          int required_coarsening_level, int max_coarsening_level, int ngrow);

    IndexSpaceComplement (IndexSpaceComplement const&) = delete;
    IndexSpaceComplement (IndexSpaceComplement &&) = delete;
    void operator= (IndexSpaceComplement const&) = delete;
    void operator= (IndexSpaceComplement &&) = delete;

    ~IndexSpaceComplement () override = default;

    //! Return the EB level matching \p geom.Domain().
    [[nodiscard]] const Level& getLevel (const Geometry& geom) const final;
    //! Return the Geometry whose domain equals \p dom.
    [[nodiscard]] const Geometry& getGeometry (const Box& dom) const final;
    //! Domain of the coarsest level.
    [[nodiscard]] const Box& coarsestDomain () const final {
        return m_geom.back().Domain();
    }
    //! Not supported: a complement has no implicit function to build finer levels from.
    void addFineLevels (int num_new_fine_levels) final;
    //! Not supported: a regular coarse level would be completely covered here.
    void addRegularCoarseLevels (int num_new_coarse_levels) final;

private:

    Vector<ComplementLevel> m_level;
    Vector<Geometry> m_geom;
    Vector<Box> m_domain;
    Vector<int> m_ngrow;
};

}

#endif
