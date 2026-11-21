// Checks that the EB2 geometry data is consistent with itself, on the level
// it is built on and on every coarsened level:
//
//  flags      Regular cells have volume fraction 1 and all face area
//             fractions 1; covered cells have 0 and 0; cut cells have a
//             volume fraction in (0,1].
//  bounds     Centroids (volume, face, boundary) in [-1/2,1/2], unit
//             boundary normals in cut cells.
//  area       The divergence theorem for a constant field in each direction
//             d: (a_lo - a_hi) A_d = n_d A_eb, with A_d the face area, a the
//             area fractions, n the boundary normal and A_eb the boundary
//             area. It holds for any geometry on the level EB2 builds. On
//             the levels EB2 coarsens from it, the boundary area is the sum
//             of the fine areas but the normal is the normalized sum of the
//             fine area vectors, so it holds there only for a planar
//             boundary; those levels are checked against max_coarse_area_err
//             (not checked by default).
//  volume     The divergence theorem for x_d e_d in each direction d:
//             kappa = (a_lo + a_hi)/2 + n_d b_d A_eb / A_d, with b the
//             boundary centroid. It holds exactly only where the boundary is
//             planar in the cell, so it is checked only when max_vol_err is
//             set (for example for eb2.geom_type = plane).
//  ghost      The data in a factory's ghost cells equals the data of the
//             neighboring box (FillBoundary).
//  decomp     The data does not depend on the box decomposition
//             (max_grid_size against max_grid_size_2).
//
// The relative errors are printed per level and check, and the run aborts if
// one exceeds its bound.

#include <AMReX.H>
#include <AMReX_EB2.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <memory>
#include <string>

using namespace amrex;

namespace {

enum Check : int { flags = 0, bounds, area, volume, ghost, decomp, nchecks };
const char* check_name[] = {"flags", "bounds", "area", "volume", "ghost", "decomp"};

// All of a factory's geometry data as plain MultiFabs (regular and covered
// boxes filled with the corresponding values), with the factory's ghost cells
struct GeomData
{
    MultiFab vfrac, vcent, barea, bcent, bnorm;
    Array<MultiFab,AMREX_SPACEDIM> apert, fcent;
    iMultiFab flag; // 0: regular, 1: single valued, 2: covered, 3: other

    explicit GeomData (EBFArrayBoxFactory const& fact)
    {
        auto const& vf = fact.getVolFrac();
        vfrac.define(vf.boxArray(), vf.DistributionMap(), 1, vf.nGrow());
        MultiFab::Copy(vfrac, vf, 0, 0, 1, vf.nGrow());
        vcent = fact.getCentroid().ToMultiFab(0.0_rt, 0.0_rt);
        barea = fact.getBndryArea().ToMultiFab(0.0_rt, 0.0_rt);
        bcent = fact.getBndryCent().ToMultiFab(-1.0_rt, -1.0_rt);
        bnorm = fact.getBndryNormal().ToMultiFab(0.0_rt, 0.0_rt);
        auto const& ap = fact.getAreaFrac();
        auto const& fc = fact.getFaceCent();
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            apert[d] = ap[d]->ToMultiFab(1.0_rt, 0.0_rt);
            fcent[d] = fc[d]->ToMultiFab(0.0_rt, 0.0_rt);
        }
        auto const& fl = fact.getMultiEBCellFlagFab();
        flag.define(fl.boxArray(), fl.DistributionMap(), 1, fl.nGrow());
        auto const& fa = fl.const_arrays();
        auto const& ia = flag.arrays();
        ParallelFor(flag, IntVect(fl.nGrow()), [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
        {
            auto f = fa[b](i,j,k);
            ia[b](i,j,k) = f.isRegular() ? 0 : (f.isSingleValued() ? 1 : (f.isCovered() ? 2 : 3));
        });
        Gpu::streamSynchronize();
    }

    // The cell-centered quantities, for comparisons
    [[nodiscard]] Vector<MultiFab const*> cellData () const {
        return {&vfrac, &vcent, &barea, &bcent, &bnorm};
    }
    [[nodiscard]] Vector<MultiFab const*> faceData () const {
        Vector<MultiFab const*> r;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { r.push_back(&apert[d]); r.push_back(&fcent[d]); }
        return r;
    }
};

// Per-cell flags, bounds, area and volume errors (valid cells) in err
void cell_checks (GeomData const& g, Geometry const& geom, MultiFab& err)
{
    auto const dx = geom.CellSizeArray();
    Real const dv = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
    GpuArray<Real,AMREX_SPACEDIM> fa;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { fa[d] = dv / dx[d]; }
    Real const amax = *std::max_element(fa.begin(), fa.end());

    auto const& flag = g.flag.const_arrays();
    auto const& vf = g.vfrac.const_arrays();
    auto const& vc = g.vcent.const_arrays();
    auto const& ba = g.barea.const_arrays();
    auto const& bc = g.bcent.const_arrays();
    auto const& bn = g.bnorm.const_arrays();
    GpuArray<MultiArray4<Real const>,AMREX_SPACEDIM> ap, fc;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        ap[d] = g.apert[d].const_arrays();
        fc[d] = g.fcent[d].const_arrays();
    }
    auto const& e = err.arrays();

    ParallelFor(err, [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
    {
        for (int n = 0; n < nchecks; ++n) { e[b](i,j,k,n) = 0.0_rt; }
        IntVect const iv(AMREX_D_DECL(i,j,k));
        Real alo[AMREX_SPACEDIM], ahi[AMREX_SPACEDIM];
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            alo[d] = ap[d][b](iv);
            ahi[d] = ap[d][b](iv + IntVect::TheDimensionVector(d));
        }
        int const f = flag[b](i,j,k);
        Real const kappa = vf[b](i,j,k);

        Real ferr = 0.0_rt;
        if (f == 0 || f == 2) {
            Real const v = (f == 0) ? 1.0_rt : 0.0_rt;
            ferr = std::abs(kappa - v);
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                ferr = amrex::max(ferr, std::abs(alo[d] - v), std::abs(ahi[d] - v));
            }
        } else if (f == 1) {
            if (!(kappa > 0.0_rt && kappa <= 1.0_rt)) { ferr = 1.0_rt; }
        } else {
            ferr = 1.0_rt; // multi-valued is not expected
        }
        e[b](i,j,k,flags) = ferr;
        if (f != 1) { return; }

        // Cut cell
        Real berr = 0.0_rt;
        Real nn = 0.0_rt;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            berr = amrex::max(berr, std::abs(vc[b](i,j,k,d)) - 0.5_rt,
                              std::abs(bc[b](i,j,k,d)) - 0.5_rt);
            nn += bn[b](i,j,k,d) * bn[b](i,j,k,d);
            for (int m = 0; m < AMREX_SPACEDIM-1; ++m) {
                if (alo[d] > 0.0_rt) {
                    berr = amrex::max(berr, std::abs(fc[d][b](iv,m)) - 0.5_rt);
                }
                if (ahi[d] > 0.0_rt) {
                    berr = amrex::max(berr, std::abs(fc[d][b](iv + IntVect::TheDimensionVector(d),m))
                                      - 0.5_rt);
                }
            }
        }
        e[b](i,j,k,bounds) = amrex::max(berr, std::abs(std::sqrt(nn) - 1.0_rt));

        // Physical boundary area from the stored (normalized) one
        Real s = 0.0_rt;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            s += (bn[b](i,j,k,d) * fa[d]) * (bn[b](i,j,k,d) * fa[d]);
        }
        Real const aeb = ba[b](i,j,k) * std::sqrt(s);

        Real aerr = 0.0_rt, verr = 0.0_rt;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            Real const nd = bn[b](i,j,k,d);
            aerr = amrex::max(aerr, std::abs((alo[d] - ahi[d]) * fa[d] - nd * aeb) / amax);
            verr = amrex::max(verr, std::abs(kappa - 0.5_rt * (alo[d] + ahi[d])
                                             - nd * bc[b](i,j,k,d) * aeb / fa[d]));
        }
        e[b](i,j,k,area) = aerr;
        e[b](i,j,k,volume) = verr;
    });
}

// Largest difference between a's ghost cells and those filled from the
// neighboring boxes
Real ghost_diff (MultiFab const& a, Geometry const& geom)
{
    MultiFab b(a.boxArray(), a.DistributionMap(), a.nComp(), a.nGrow());
    MultiFab::Copy(b, a, 0, 0, a.nComp(), a.nGrow());
    b.FillBoundary(geom.periodicity());
    MultiFab::Subtract(b, a, 0, 0, a.nComp(), a.nGrow());
    Real r = 0.0_rt;
    for (int n = 0; n < a.nComp(); ++n) { r = amrex::max(r, b.norminf(n, a.nGrow())); }
    return r;
}

// Largest difference between a and b (same data, different boxes) on a's
// valid region
Real decomp_diff (MultiFab const& a, MultiFab const& b)
{
    MultiFab c(a.boxArray(), a.DistributionMap(), a.nComp(), 0);
    c.ParallelCopy(b, 0, 0, a.nComp());
    MultiFab::Subtract(c, a, 0, 0, a.nComp(), 0);
    Real r = 0.0_rt;
    for (int n = 0; n < a.nComp(); ++n) { r = amrex::max(r, c.norminf(n)); }
    return r;
}

} // namespace

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        Vector<int> n_cell_in{32};
        int max_grid_size = 16;
        int max_grid_size_2 = 8;
        int max_coarsening_level = 3;
        int ngrow = 2;
        Real max_err = 1.e-12;     // flags, bounds, area, ghost, decomp
        Real max_vol_err = -1.0;   // volume (negative: not checked)
        Real max_coarse_area_err = -1.0; // area on coarsened levels
        {
            ParmParse pp;
            pp.queryarr("n_cell", n_cell_in);
            pp.query("max_grid_size", max_grid_size);
            pp.query("max_grid_size_2", max_grid_size_2);
            pp.query("max_coarsening_level", max_coarsening_level);
            pp.query("ngrow", ngrow);
            pp.query("max_err", max_err);
            pp.query("max_vol_err", max_vol_err);
            pp.query("max_coarse_area_err", max_coarse_area_err);
        }
        IntVect n_cell;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            n_cell[d] = n_cell_in[std::min(d, int(n_cell_in.size())-1)];
        }

        RealBox rb;
        Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0,0,0)};
        {
            ParmParse pp("geometry");
            Vector<Real> lo(AMREX_SPACEDIM, 0.0_rt), hi(AMREX_SPACEDIM, 1.0_rt);
            pp.queryarr("prob_lo", lo);
            pp.queryarr("prob_hi", hi);
            rb = RealBox(lo.data(), hi.data());
            Vector<int> per(AMREX_SPACEDIM, 0);
            pp.queryarr("is_periodic", per);
            for (int d = 0; d < AMREX_SPACEDIM; ++d) { is_periodic[d] = per[d]; }
        }
        Geometry geom(Box(IntVect(0), n_cell-1), rb, CoordSys::cartesian, is_periodic);

        EB2::Build(geom, 0, max_coarsening_level);

        Real worst[nchecks] = {};
        Real worst_coarse_area = 0.0_rt;
        for (int lev = 0; lev <= max_coarsening_level; ++lev) {
            IntVect const ratio(1 << lev);
            if (!geom.Domain().coarsenable(ratio)) { break; }
            Geometry const cgeom = amrex::coarsen(geom, ratio);

            BoxArray ba(cgeom.Domain());
            ba.maxSize(std::max(max_grid_size >> lev, 1));
            DistributionMapping dm(ba);
            auto fact = makeEBFabFactory(cgeom, ba, dm, {ngrow,ngrow,ngrow}, EBSupport::full);
            GeomData g(*fact);

            Real errs[nchecks] = {};
            IntVect where[nchecks];

            MultiFab err(ba, dm, nchecks, 0);
            cell_checks(g, cgeom, err);
            for (int n = flags; n <= volume; ++n) {
                errs[n] = err.max(n);
                where[n] = err.maxIndex(n);
            }

            for (auto const* mf : g.cellData()) {
                errs[ghost] = amrex::max(errs[ghost], ghost_diff(*mf, cgeom));
            }
            for (auto const* mf : g.faceData()) {
                errs[ghost] = amrex::max(errs[ghost], ghost_diff(*mf, cgeom));
            }

            BoxArray ba2(cgeom.Domain());
            ba2.maxSize(std::max(max_grid_size_2 >> lev, 1));
            DistributionMapping dm2(ba2);
            auto fact2 = makeEBFabFactory(cgeom, ba2, dm2, {ngrow,ngrow,ngrow}, EBSupport::full);
            GeomData g2(*fact2);
            auto c1 = g.cellData(), c2 = g2.cellData();
            for (int m = 0; m < int(c1.size()); ++m) {
                errs[decomp] = amrex::max(errs[decomp], decomp_diff(*c1[m], *c2[m]));
            }
            auto f1 = g.faceData(), f2 = g2.faceData();
            for (int m = 0; m < int(f1.size()); ++m) {
                errs[decomp] = amrex::max(errs[decomp], decomp_diff(*f1[m], *f2[m]));
            }

            amrex::Print() << "Level " << lev << " (" << cgeom.Domain().size() << " cells, "
                           << ba.size() << " boxes):\n";
            for (int n = 0; n < nchecks; ++n) {
                amrex::Print() << "  " << check_name[n] << ": " << errs[n];
                if (n <= volume && errs[n] > 0.0_rt) { amrex::Print() << " at " << where[n]; }
                amrex::Print() << "\n";
                if (n == area && lev > 0) {
                    worst_coarse_area = amrex::max(worst_coarse_area, errs[n]);
                } else {
                    worst[n] = amrex::max(worst[n], errs[n]);
                }
            }
        }

        bool failed = false;
        for (int n = 0; n < nchecks; ++n) {
            Real const bound = (n == volume) ? max_vol_err : max_err;
            if (bound >= 0.0_rt && worst[n] > bound) {
                amrex::Print() << "FAILED: " << check_name[n] << " error " << worst[n]
                               << " > " << bound << "\n";
                failed = true;
            }
        }
        if (max_coarse_area_err >= 0.0_rt && worst_coarse_area > max_coarse_area_err) {
            amrex::Print() << "FAILED: area error on the coarsened levels " << worst_coarse_area
                           << " > " << max_coarse_area_err << "\n";
            failed = true;
        }
        if (failed) { amrex::Abort("EB2 geometry data is inconsistent"); }
        amrex::Print() << "PASSED\n";
    }
    amrex::Finalize();
}
