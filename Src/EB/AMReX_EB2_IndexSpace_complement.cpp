#include <AMReX_EB2_IndexSpace_complement.H>

#include <algorithm>
#include <string>

namespace amrex::EB2 {

IndexSpaceComplement::IndexSpaceComplement (IndexSpace const& index_space, const Geometry& geom,
                                            int required_coarsening_level,
                                            int max_coarsening_level, int ngrow)
{
    Gpu::LaunchSafeGuard lsg(true); // Always use GPU

    AMREX_ALWAYS_ASSERT(required_coarsening_level >= 0 && required_coarsening_level <= 30);
    max_coarsening_level = std::max(required_coarsening_level,max_coarsening_level);
    max_coarsening_level = std::min(30,max_coarsening_level);

    m_level.reserve(max_coarsening_level+1);
    m_level.emplace_back(this, index_space.getLevel(geom), geom);

    // The grow cells of the coarse levels follow from the finest one. ngrow is only used to
    // extend them at the domain faces when the geometry extends its coarse levels there.
    int const ng_extrude = m_level.back().extrudeCoarseLevels() ? std::max(ngrow,0) : 0;

    m_geom.push_back(geom);
    m_domain.push_back(geom.Domain());

    for (int ilev = 1; ilev <= max_coarsening_level; ++ilev)
    {
        bool coarsenable = m_geom.back().Domain().coarsenable(2,2);
        if (!coarsenable) {
            if (ilev <= required_coarsening_level) {
                amrex::Abort("IndexSpaceComplement: domain is not coarsenable at level "
                             +std::to_string(ilev));
            } else {
                break;
            }
        }

        Box cdomain = amrex::coarsen(m_geom.back().Domain(),2);
        Geometry cgeom = amrex::coarsen(m_geom.back(),2);
        m_level.emplace_back(this, ilev, EB2::max_grid_size, ng_extrude, cgeom, m_level[ilev-1]);
        if (!m_level.back().isOK()) {
            m_level.pop_back();
            if (ilev <= required_coarsening_level) {
                amrex::Abort("IndexSpaceComplement: failed to build required coarse EB level "
                             +std::to_string(ilev));
            } else {
                break;
            }
        }
        m_geom.push_back(cgeom);
        m_domain.push_back(cdomain);
    }

    // Code that sizes its hierarchy from the geometry, e.g. with EB2::maxCoarseningLevel(geom),
    // would ask the complement for levels it does not have.
    if (index_space.coarsestDomain().numPts() < coarsestDomain().numPts()
        && m_level.size() < static_cast<std::size_t>(max_coarsening_level+1))
    {
        amrex::Warning("IndexSpaceComplement: the complement has fewer coarse levels than the "
                       "geometry; use EB2::maxCoarseningLevel with the complement's IndexSpace");
    }
}

const Level&
IndexSpaceComplement::getLevel (const Geometry& geom) const
{
    auto it = std::ranges::find(m_domain, geom.Domain());
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(it != std::end(m_domain),
                                     "IndexSpaceComplement::getLevel: Geometry not found");
    auto i = std::distance(m_domain.begin(), it);
    return m_level[i];
}

const Geometry&
IndexSpaceComplement::getGeometry (const Box& dom) const
{
    auto it = std::ranges::find(m_domain, dom);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(it != std::end(m_domain),
                                     "IndexSpaceComplement::getGeometry: domain not found");
    auto i = std::distance(m_domain.begin(), it);
    return m_geom[i];
}

void
IndexSpaceComplement::addFineLevels (int num_new_fine_levels)
{
    if (num_new_fine_levels > 0) {
        amrex::Abort("IndexSpaceComplement::addFineLevels: not supported");
    }
}

void
IndexSpaceComplement::addRegularCoarseLevels (int num_new_coarse_levels)
{
    if (num_new_coarse_levels > 0) {
        amrex::Abort("IndexSpaceComplement::addRegularCoarseLevels: not supported");
    }
}

}
