#include <AMReX_EB2_Level_complement.H>
#include <AMReX_EB2_C.H>
#include <AMReX_EB2.H>
#include <AMReX_GpuBuffer.H>
#include <AMReX_ParReduce.H>

#include <limits>

namespace amrex::EB2 {

ComplementLevel::ComplementLevel (IndexSpace const* is, Level const& level, Geometry const& geom)
    : GShopLevel<ComplementTag>(is, geom)
{
    BL_PROFILE("EB2::ComplementLevel()-fine");

    define_complement(level);
}

ComplementLevel::ComplementLevel (IndexSpace const* is, int ilev, int max_grid_size, int ngrow,
                                  Geometry const& geom, ComplementLevel& fineLevel)
    : GShopLevel<ComplementTag>(is, ilev, max_grid_size, ngrow, geom, fineLevel)
{}

namespace {

// A cut cell that the level fills completely. Its boundary lies on its faces.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool is_full_cut (EBCellFlag flag, Real vfrac) noexcept
{
    return flag.isSingleValued() && vfrac >= 1.0_rt;
}

}

void
ComplementLevel::define_complement (Level const& f)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(f.isOK(), "EB2::ComplementLevel: the level is not built");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(f.m_shift == 0,
        "EB2::ComplementLevel: shifted levels are not supported");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(f.m_geom.Domain() == m_geom.Domain(),
        "EB2::ComplementLevel: the level was built on a different domain");

    // Without cut cells, the level is either all regular, and its complement completely
    // covered, or it was never built because it is completely covered itself.
    if (f.isAllRegular() || !f.hasEBInfo()) {
        amrex::Abort("EB2::ComplementLevel: the geometry has no cut cells, so its complement is "
                     "completely covered");
    }

    m_ngrow = f.m_ngrow;
    m_extrude_coarse_levels = f.m_extrude_coarse_levels;
    m_grids = f.m_grids;
    m_dmap = f.m_dmap;

    // The complement is covered where the level is regular: everywhere in the domain (grown
    // by m_ngrow outside the non-periodic faces) that is neither in a cut box nor covered.
    {
        BoxList bl = f.m_grids.boxList();
        if (!f.m_covered_grids.empty()) {
            bl.join(f.m_covered_grids.boxList());
        }
        BoxList covered = amrex::complementIn(amrex::grow(m_geom.Domain(), m_ngrow), bl);
        if (!covered.isEmpty()) {
            covered.maxSize(EB2::max_grid_size);
            m_covered_grids = BoxArray(std::move(covered));
        }
    }

    // Everything is defined like the level's data, ghost cells included, so that the ghost
    // cells the level built, which are not all reachable by FillBoundary, carry over.
    MFInfo mf_info;
    mf_info.SetTag("EB2::Level");
    auto define_like = [&] (MultiFab& mf, MultiFab const& src)
    {
        mf.define(src.boxArray(), src.DistributionMap(), src.nComp(), src.nGrowVect(), mf_info);
    };
    m_cellflag.define(f.m_cellflag.boxArray(), f.m_cellflag.DistributionMap(), 1,
                      f.m_cellflag.nGrowVect(), mf_info);
    define_like(m_volfrac, f.m_volfrac);
    define_like(m_centroid, f.m_centroid);
    define_like(m_bndryarea, f.m_bndryarea);
    define_like(m_bndrycent, f.m_bndrycent);
    define_like(m_bndrynorm, f.m_bndrynorm);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        define_like(m_areafrac[idim], f.m_areafrac[idim]);
        define_like(m_facecent[idim], f.m_facecent[idim]);
        define_like(m_edgecent[idim], f.m_edgecent[idim]);
    }
    define_like(m_levelset, f.m_levelset);

    // The level's data is defined on its boxes grown by one cell (face, edge), its cell flags
    // and level set on all its ghost cells. The complement is computed where the data is
    // defined, and the rest is copied, so that it stays as undefined as it was.
    IntVect const ngdef(1);

    // Multi-valued cells have no complement here.
    {
        auto const& fflag = f.m_cellflag.const_arrays();
        Long nmulti = ParReduce(TypeList<ReduceOpSum>{}, TypeList<Long>{}, f.m_cellflag, ngdef,
        [=] AMREX_GPU_DEVICE (int b, int i, int j, int k) noexcept -> GpuTuple<Long>
        {
            return {Long(fflag[b](i,j,k).isMultiValued())};
        });
        ParallelAllReduce::Sum(nmulti, ParallelContext::CommunicatorSub());
        if (nmulti > 0) {
            amrex::Abort("EB2::ComplementLevel: multi-valued cells are not supported");
        }
    }

    // The level set is negative in the region the geometry keeps and nonnegative where it
    // is covered, so a node with level set 0 belongs to the covered region. In the
    // complement it is in the region kept.
    constexpr Real tiny = std::numeric_limits<Real>::min();
    constexpr Real eps = std::numeric_limits<Real>::epsilon();

    // Physical face areas
    auto const dx = m_geom.CellSizeArray();
    GpuArray<Real,AMREX_SPACEDIM> fa;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        fa[idim] = AMREX_D_TERM(Real(1.0), *dx[(idim+1)%AMREX_SPACEDIM],
                                           *dx[(idim+2)%AMREX_SPACEDIM]);
    }

    Long nopposite = 0;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion()) reduction(+:nopposite)
#endif
    for (MFIter mfi(m_volfrac); mfi.isValid(); ++mfi)
    {
        Box const& vbx = mfi.validbox();
        Box const& dbx = amrex::grow(vbx, ngdef);
        auto const& fflag = f.m_cellflag.const_array(mfi);
        auto const& fvol = f.m_volfrac.const_array(mfi);
        auto const& cflag = m_cellflag.array(mfi);
        auto const& vol = m_volfrac.array(mfi);
        auto const& cent = m_centroid.array(mfi);
        auto const& ba = m_bndryarea.array(mfi);
        auto const& bc = m_bndrycent.array(mfi);
        auto const& bn = m_bndrynorm.array(mfi);

        // Cells
        {
            auto const& fcent = f.m_centroid.const_array(mfi);
            auto const& fba = f.m_bndryarea.const_array(mfi);
            auto const& fbc = f.m_bndrycent.const_array(mfi);
            auto const& fbn = f.m_bndrynorm.const_array(mfi);
            ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                auto flg = fflag(i,j,k);
                if (flg.isSingleValued() && !dbx.contains(i,j,k)) {
                    vol(i,j,k) = fvol(i,j,k);
                    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                        cent(i,j,k,n) = fcent(i,j,k,n);
                        bc(i,j,k,n) = fbc(i,j,k,n);
                        bn(i,j,k,n) = fbn(i,j,k,n);
                    }
                    ba(i,j,k) = fba(i,j,k);
                    flg.setConnected();
                } else if (flg.isSingleValued() && !is_full_cut(flg, fvol(i,j,k))) {
                    Real const v = fvol(i,j,k);
                    Real const vc = 1.0_rt - v;
                    vol(i,j,k) = vc;
                    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                        // The first moments of the two parts add up to that of the cell, 0.
                        cent(i,j,k,n) = amrex::Clamp(-v*fcent(i,j,k,n)/vc, -0.5_rt, 0.5_rt);
                        bc(i,j,k,n) = fbc(i,j,k,n);
                        bn(i,j,k,n) = -fbn(i,j,k,n);
                    }
                    ba(i,j,k) = fba(i,j,k);
                    flg.setSingleValued();
                    flg.setConnected();
                } else {
                    // Regular in the complement if covered here. Covered if regular here, or
                    // cut but filled completely, in which case its boundary moves to the
                    // neighbors across its faces below.
                    bool const regular = flg.isCovered();
                    vol(i,j,k) = regular ? 1.0_rt : 0.0_rt;
                    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                        cent(i,j,k,n) = 0.0_rt;
                        bc(i,j,k,n) = -1.0_rt;
                        bn(i,j,k,n) = 0.0_rt;
                    }
                    ba(i,j,k) = 0.0_rt;
                    if (regular) {
                        flg.setRegular();
                        flg.setConnected();
                    } else {
                        flg.setCovered();
                    }
                }
                cflag(i,j,k) = flg;
            });
        }

        // Faces. The part of a face that is open neither here nor in the complement is
        // boundary lying on the face. Here it belongs to the cell on the side this geometry
        // keeps, which is cut but filled completely. That cell has no volume in the
        // complement, so all its faces are closed there.
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            Box const& fbx = m_areafrac[idim][mfi].box();
            Box const& dfbx = amrex::grow(amrex::surroundingNodes(vbx,idim), ngdef);
            IntVect const e = IntVect::TheDimensionVector(idim);
            auto const& fap = f.m_areafrac[idim].const_array(mfi);
            auto const& ffc = f.m_facecent[idim].const_array(mfi);
            auto const& ap = m_areafrac[idim].array(mfi);
            auto const& fc = m_facecent[idim].array(mfi);
            ParallelFor(fbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                IntVect const iv(AMREX_D_DECL(i,j,k));
                if (!dfbx.contains(iv)) {
                    ap(iv) = fap(iv);
                    for (int n = 0; n < AMREX_SPACEDIM-1; ++n) {
                        fc(iv,n) = ffc(iv,n);
                    }
                    return;
                }
                IntVect const lo = iv-e;
                bool const closed =
                    (dbx.contains(lo) && is_full_cut(fflag(lo), fvol(lo))) ||
                    (dbx.contains(iv) && is_full_cut(fflag(iv), fvol(iv)));
                Real const a = fap(iv);
                Real const ac = closed ? 0.0_rt : 1.0_rt - a;
                ap(iv) = ac;
                for (int n = 0; n < AMREX_SPACEDIM-1; ++n) {
                    Real const c = (ac > 0.0_rt && a > 0.0_rt) ? -a*ffc(iv,n)/ac : 0.0_rt;
                    fc(iv,n) = amrex::Clamp(c, -0.5_rt, 0.5_rt);
                }
            });
        }

        // A cell next to a completely filled cut cell gets the closed part of their common
        // face as boundary, with its normal pointing out of the cell. It is combined with the
        // cell's own boundary as EB2 combines boundary pieces: the boundary area and normal
        // come from the sum of the area vectors, and the centroid is such that the
        // divergence theorem for x_d holds in each direction d.
        {
            GpuArray<Array4<Real const>,AMREX_SPACEDIM> fapa, ffca, apa;
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                fapa[idim] = f.m_areafrac[idim].const_array(mfi);
                ffca[idim] = f.m_facecent[idim].const_array(mfi);
                apa[idim] = m_areafrac[idim].const_array(mfi);
            }
            Gpu::Buffer<Long> nopp_d({Long(0)});
            Long* nopp = nopp_d.data();
            ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                IntVect const iv(AMREX_D_DECL(i,j,k));
                if (cflag(iv).isCovered()) { return; }

                // Area, area vector, and first moments of the pieces
                Real A = 0.0_rt;
                Real V[AMREX_SPACEDIM], M[AMREX_SPACEDIM], Mc[AMREX_SPACEDIM];
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    V[d] = 0.0_rt; M[d] = 0.0_rt; Mc[d] = 0.0_rt;
                }
                bool found = false;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    IntVect const e = IntVect::TheDimensionVector(d);
                    for (int side = 0; side < 2; ++side) {
                        IntVect const nb = side ? iv+e : iv-e;
                        IntVect const face = side ? iv+e : iv;
                        if (!is_full_cut(fflag(nb), fvol(nb))) { continue; }
                        Real const w = 1.0_rt - fapa[d](face) - apa[d](face);
                        if (w <= 0.0_rt) { continue; }
                        found = true;
                        Real const sgn = side ? 1.0_rt : -1.0_rt;
                        Real const area = w * fa[d];
                        A += area;
                        V[d] += sgn * area;
                        M[d] += 0.5_rt * area;
                        Mc[d] += 0.5_rt * sgn * area;
                        // Centroid of the closed part of the face in its tangential directions
                        Real const a = fapa[d](face);
                        int m = 0;
                        for (int t = 0; t < AMREX_SPACEDIM; ++t) {
                            if (t == d) { continue; }
                            Real const c = (a > 0.0_rt) ? -a*ffca[d](face,m)/w : 0.0_rt;
                            Mc[t] += area * c;
                            ++m;
                        }
                    }
                }
                if (!found) { return; }

                if (ba(iv) > 0.0_rt) {
                    Real s = 0.0_rt;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        s += (bn(iv,d)*fa[d]) * (bn(iv,d)*fa[d]);
                    }
                    Real const aeb = ba(iv) * std::sqrt(s);
                    A += aeb;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        V[d] += aeb * bn(iv,d);
                        M[d] += aeb * bn(iv,d) * bc(iv,d);
                        Mc[d] += aeb * bc(iv,d);
                    }
                }

                Real vn = 0.0_rt;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) { vn += V[d]*V[d]; }
                vn = std::sqrt(vn);
                if (vn <= eps * A) {
                    // Boundary on opposite faces of the cell, as for a body one cell thick
                    Gpu::Atomic::Add(nopp, Long(1));
                    return;
                }
                Real s = 0.0_rt;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    Real const n = V[d] / vn;
                    bn(iv,d) = n;
                    s += (n*fa[d]) * (n*fa[d]);
                    Real const b = (std::abs(V[d]) > eps * vn) ? M[d] / V[d] : Mc[d] / A;
                    bc(iv,d) = amrex::Clamp(b, -0.5_rt, 0.5_rt);
                }
                ba(iv) = vn / std::sqrt(s);
                auto flg = cflag(iv);
                if (flg.isRegular()) {
                    flg.setSingleValued();
                    flg.setConnected();
                    cflag(iv) = flg;
                }
            });
            nopposite += *(nopp_d.copyToHost());
        }

        // Edges: 1 for a regular edge, -1 for a covered one, otherwise the centroid of the
        // part kept, in [-1/2,1/2]. Which end of a cut edge is kept follows from the level
        // set, as when the level was built.
        auto const& fls = f.m_levelset.const_array(mfi);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            Box const& ebx = m_edgecent[idim][mfi].box();
            Box const& debx = amrex::grow(amrex::convert(vbx, ebx.ixType()), ngdef);
            IntVect const e = IntVect::TheDimensionVector(idim);
            auto const& fec = f.m_edgecent[idim].const_array(mfi);
            auto const& ec = m_edgecent[idim].array(mfi);
            ParallelFor(ebx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                IntVect const iv(AMREX_D_DECL(i,j,k));
                if (!debx.contains(iv)) {
                    ec(iv) = fec(iv);
                    return;
                }
                Real const c = fec(iv);
                if (c == 1.0_rt) {
                    ec(iv) = -1.0_rt;
                } else if (c == -1.0_rt) {
                    ec(iv) = 1.0_rt;
                } else {
                    // Length of the part the level keeps
                    bool const lo_kept = fls(iv) < fls(iv+e);
                    Real const len = lo_kept ? 2.0_rt*c + 1.0_rt : 1.0_rt - 2.0_rt*c;
                    if (len <= 0.0_rt) {
                        ec(iv) = 1.0_rt;
                    } else if (len >= 1.0_rt) {
                        ec(iv) = -1.0_rt;
                    } else {
                        ec(iv) = amrex::Clamp(lo_kept ? c + 0.5_rt : c - 0.5_rt, -0.5_rt, 0.5_rt);
                    }
                }
            });
        }

        // Level set
        {
            auto const& ls = m_levelset.array(mfi);
            ParallelFor(m_levelset[mfi].box(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real const l = fls(i,j,k);
                ls(i,j,k) = (l == 0.0_rt) ? -tiny : -l;
            });
        }
    }

    ParallelAllReduce::Sum(nopposite, ParallelContext::CommunicatorSub());
    if (nopposite > 0) {
        amrex::Abort("EB2::ComplementLevel: the complement has cells with boundary on opposite "
                     "faces, such as a body one cell thick");
    }

    // The boundary that moved across faces was computed on the valid cells only.
    auto const& period = m_geom.periodicity();
    m_cellflag.FillBoundary(period);
    m_bndryarea.FillBoundary(period);
    m_bndrycent.FillBoundary(period);
    m_bndrynorm.FillBoundary(period);

    // Connections between cells, from the area fractions, as Level::buildCellFlag does.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(m_cellflag); mfi.isValid(); ++mfi)
    {
        auto const& cflag = m_cellflag.array(mfi);
        AMREX_D_TERM(auto const& apx = m_areafrac[0].const_array(mfi);,
                     auto const& apy = m_areafrac[1].const_array(mfi);,
                     auto const& apz = m_areafrac[2].const_array(mfi););
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            if (!cflag(i,j,k).isCovered()) {
#if (AMREX_SPACEDIM == 3)
                build_cellflag_from_ap(i, j, k, cflag, apx, apy, apz);
#else
                amrex::ignore_unused(k);
                build_cellflag_from_ap(i, j, cflag, apx, apy);
#endif
            }
        });
    }
    m_cellflag.FillBoundary(period);

    m_ok = true;
}

}
