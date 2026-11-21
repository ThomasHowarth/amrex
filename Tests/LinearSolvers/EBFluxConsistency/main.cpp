// Consistency of MLEBABecLap's operator with its own fluxes.
//
// For any phi (not a solution), each uncovered cell must satisfy
//
//   kappa*(L(phi) - alpha*a*phi) = beta*( sum_d dxinv_d*(ap_hi*F_hi - ap_lo*F_lo)
//                                        + fac*ebarea*F_eb )
//
// where L comes from MLMG::apply, F from MLMG::getFluxes at face centroids
// (-b grad phi, without beta), and F_eb from MLMG::getEBFluxes.  The test
// reports the largest mismatch, relative to max |kappa*(L - alpha*a*phi)| on
// the level, for each kind of cell, and aborts if it exceeds max_rel_err.
// It checks a single level.

#include <AMReX.H>
#include <AMReX_EB2.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_MLEBABecLap.H>
#include <AMReX_MLLinOp_K.H>
#include <AMReX_MLMG.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <array>
#include <cstdint>
#include <string>

using namespace amrex;

namespace {

LinOpBCType parse_bc (std::string const& s)
{
    if (s == "dirichlet")       { return LinOpBCType::Dirichlet; }
    if (s == "neumann")         { return LinOpBCType::Neumann; }
    if (s == "inhomog_neumann") { return LinOpBCType::inhomogNeumann; }
    if (s == "robin")           { return LinOpBCType::Robin; }
    if (s == "periodic")        { return LinOpBCType::Periodic; }
    amrex::Abort("Unknown bc " + s);
    return LinOpBCType::bogus;
}

// A number in [0,1) from the cell, component and field, so that the data do
// not depend on the number of MPI ranks.
Real hash01 (IntVect const& iv, int n, int field)
{
    std::uint64_t h = 0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(field + 1);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        h ^= static_cast<std::uint64_t>(iv[d] + 1024) + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
    }
    h ^= static_cast<std::uint64_t>(n) + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
    // splitmix64 finalizer
    h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ULL;
    h = (h ^ (h >> 27)) * 0x94D049BB133111EBULL;
    h ^= h >> 31;
    return static_cast<Real>(h >> 11) * Real(0x1.0p-53);
}

// Fill mf (valid and ghost cells) with lo + (hi-lo)*U(0,1) plus, if smooth
// is true, a smooth function of position.  Each field has its own id.
void fill (MultiFab& mf, Geometry const& geom, Real lo, Real hi, bool smooth, int field)
{
    auto const problo = geom.ProbLoArray();
    auto const dx = geom.CellSizeArray();
    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        Box const& bx = mfi.fabbox();
        auto const& a = mf.array(mfi);
        IndexType const typ = mf.ixType();
        amrex::LoopOnCpu(bx, mf.nComp(), [&] (int i, int j, int k, int n)
        {
            IntVect const iv(AMREX_D_DECL(i,j,k));
            Real v = lo + (hi-lo)*hash01(iv, n, field);
            if (smooth) {
                Real s = Real(1.0);
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    Real x = problo[d] + (iv[d] + (typ.cellCentered(d) ? 0.5_rt : 0.0_rt))*dx[d];
                    s *= std::cos(Real(2.0+n+d)*x);
                }
                v += s;
            }
            a(i,j,k,n) = v;
        });
    }
}

// Kinds of cells reported separately.
enum Kind { RegInterior, RegDomain, CutInterior, CutDomain, NKinds };
const char* kind_name[NKinds] = {"regular", "regular at domain bdry", "cut", "cut at domain bdry"};

}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        ParmParse pp;
        Vector<int> n_cell_v{32}; // one value, or one per direction
        pp.queryarr("n_cell", n_cell_v);
        IntVect n_cell(n_cell_v[0]);
        if (n_cell_v.size() >= AMREX_SPACEDIM) {
            for (int d = 0; d < AMREX_SPACEDIM; ++d) { n_cell[d] = n_cell_v[d]; }
        }
        int max_grid_size = 32;
        pp.query("max_grid_size", max_grid_size);
        int ncomp = 1;
        pp.query("ncomp", ncomp);
        Vector<Real> prob_lo(AMREX_SPACEDIM, 0.0), prob_hi(AMREX_SPACEDIM, 1.0);
        pp.queryarr("prob_lo", prob_lo, 0, AMREX_SPACEDIM);
        pp.queryarr("prob_hi", prob_hi, 0, AMREX_SPACEDIM);
        Vector<std::string> lo_bc_s(AMREX_SPACEDIM, "dirichlet"), hi_bc_s(AMREX_SPACEDIM, "dirichlet");
        pp.queryarr("lo_bc", lo_bc_s, 0, AMREX_SPACEDIM);
        pp.queryarr("hi_bc", hi_bc_s, 0, AMREX_SPACEDIM);
        std::string eb_bc = "neumann"; // neumann, dirichlet, homog_dirichlet
        pp.query("eb_bc", eb_bc);
        bool eb_beta_scalar = false;
        pp.query("eb_beta_scalar", eb_beta_scalar);
        bool beta_on_centroid = false;
        pp.query("beta_on_centroid", beta_on_centroid);
        Real alpha = 0.7, beta = 1.3;
        pp.query("alpha", alpha);
        pp.query("beta", beta);
        Real max_rel_err = 1.e-10;
        pp.query("max_rel_err", max_rel_err);
        int seed = 0; // changes the random data
        pp.query("seed", seed);

        Array<LinOpBCType,AMREX_SPACEDIM> lobc, hibc;
        Array<int,AMREX_SPACEDIM> is_periodic{};
        bool has_robin = false;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            lobc[d] = parse_bc(lo_bc_s[d]);
            hibc[d] = parse_bc(hi_bc_s[d]);
            AMREX_ALWAYS_ASSERT((lobc[d] == LinOpBCType::Periodic) == (hibc[d] == LinOpBCType::Periodic));
            is_periodic[d] = (lobc[d] == LinOpBCType::Periodic);
            has_robin = has_robin || lobc[d] == LinOpBCType::Robin || hibc[d] == LinOpBCType::Robin;
        }

        RealBox rb(prob_lo.data(), prob_hi.data());
        Geometry const geom(Box(IntVect(0), n_cell - 1), rb, CoordSys::cartesian, is_periodic);
        BoxArray grids(geom.Domain());
        grids.maxSize(max_grid_size);
        DistributionMapping const dmap(grids);

        EB2::Build(geom, 0, 0);
        auto const factory = makeEBFabFactory(geom, grids, dmap, {2,2,2}, EBSupport::full);
        auto const& fact = *factory;

        MultiFab phi(grids, dmap, ncomp, 1, MFInfo(), fact);
        MultiFab Lphi(grids, dmap, ncomp, 0, MFInfo(), fact);
        MultiFab acoef(grids, dmap, 1, 0, MFInfo(), fact);
        MultiFab ebflux(grids, dmap, ncomp, 0, MFInfo(), fact);
        MultiFab robin_a, robin_b, robin_f, phieb, beb;
        Array<MultiFab,AMREX_SPACEDIM> bcoef, flux;
        fill(phi, geom, -0.1, 0.1, true, seed+0);
        fill(acoef, geom, 0.5, 1.5, false, seed+1);
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            BoxArray const fba = amrex::convert(grids, IntVect::TheDimensionVector(d));
            bcoef[d].define(fba, dmap, ncomp, 0, MFInfo(), fact);
            flux[d].define(fba, dmap, ncomp, 0, MFInfo(), fact);
            fill(bcoef[d], geom, 0.5, 1.5, false, seed+2+d);
        }
        if (has_robin) {
            robin_a.define(grids, dmap, ncomp, 1, MFInfo(), fact);
            robin_b.define(grids, dmap, ncomp, 1, MFInfo(), fact);
            robin_f.define(grids, dmap, ncomp, 1, MFInfo(), fact);
            fill(robin_a, geom, 0.0, 1.0, false, seed+5);
            fill(robin_b, geom, 1.0, 2.0, false, seed+6);
            fill(robin_f, geom, -1.0, 1.0, false, seed+7);
        }
        if (eb_bc != "neumann") {
            phieb.define(grids, dmap, ncomp, 0, MFInfo(), fact);
            beb.define(grids, dmap, ncomp, 0, MFInfo(), fact);
            fill(phieb, geom, -1.0, 1.0, false, seed+8);
            fill(beb, geom, 0.5, 1.5, false, seed+9);
        }

        LPInfo info;
        info.setMaxCoarseningLevel(0);
        MLEBABecLap mleb({geom}, {grids}, {dmap}, info, {factory.get()}, ncomp);
        mleb.setDomainBC(lobc, hibc);
        mleb.setScalars(alpha, beta);
        if (has_robin) {
            mleb.setLevelBC(0, &phi, &robin_a, &robin_b, &robin_f);
        } else {
            mleb.setLevelBC(0, &phi);
        }
        mleb.setACoeffs(0, acoef);
        mleb.setBCoeffs(0, amrex::GetArrOfConstPtrs(bcoef),
                        beta_on_centroid ? MLLinOp::Location::FaceCentroid
                                         : MLLinOp::Location::FaceCenter);
        if (eb_bc == "dirichlet") {
            if (eb_beta_scalar) {
                mleb.setEBDirichlet(0, phieb, 1.1);
            } else {
                mleb.setEBDirichlet(0, phieb, beb);
            }
        } else if (eb_bc == "homog_dirichlet") {
            if (eb_beta_scalar) {
                mleb.setEBHomogDirichlet(0, 1.1);
            } else {
                mleb.setEBHomogDirichlet(0, beb);
            }
        } else {
            AMREX_ALWAYS_ASSERT(eb_bc == "neumann");
        }

        MLMG mlmg(mleb);
        mlmg.apply({&Lphi}, {&phi});
        mlmg.getFluxes({amrex::GetArrOfPtrs(flux)}, {&phi}, MLMG::Location::FaceCentroid);
        mlmg.getEBFluxes({&ebflux}, {&phi});

        auto const& flags = fact.getMultiEBCellFlagFab();
        auto const& vfrac = fact.getVolFrac();
        auto const area = fact.getAreaFrac();
        auto const& barea = fact.getBndryArea();
        auto const dxinv = geom.InvCellSizeArray();
        Box const& dom = geom.Domain();

        Real lmax = 0.0;
        std::array<Real,NKinds> emax{};
        std::array<IntVect,NKinds> eloc{};
        std::array<int,NKinds> ecomp{};
        std::array<Long,NKinds> ncells{};
        for (MFIter mfi(Lphi); mfi.isValid(); ++mfi) {
            Box const& vbx = mfi.validbox();
            FabType const typ = flags[mfi].getType(vbx);
            if (typ == FabType::covered) { continue; }
            bool const cut = (typ != FabType::regular);
            auto const& fl = flags.const_array(mfi);
            auto const& L = Lphi.const_array(mfi);
            auto const& x = phi.const_array(mfi);
            auto const& a = acoef.const_array(mfi);
            auto const& feb = ebflux.const_array(mfi);
            Array<Array4<Real const>,AMREX_SPACEDIM> fd, apd;
            Array4<Real const> kap, ba;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                fd[d] = flux[d].const_array(mfi);
                if (cut) { apd[d] = area[d]->const_array(mfi); }
            }
            if (cut) {
                kap = vfrac.const_array(mfi);
                ba = barea.const_array(mfi);
            }
            amrex::LoopOnCpu(vbx, [&] (int i, int j, int k)
            {
                if (fl(i,j,k).isCovered()) { return; }
                IntVect const iv(AMREX_D_DECL(i,j,k));
                bool const cell_cut = !fl(i,j,k).isRegular();
                bool at_domain = false;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    if (!is_periodic[d] && (iv[d] == dom.smallEnd(d) || iv[d] == dom.bigEnd(d))) {
                        at_domain = true;
                    }
                }
                int const kind = cell_cut ? (at_domain ? CutDomain : CutInterior)
                                          : (at_domain ? RegDomain : RegInterior);
                ++ncells[kind];

                // kappa, and the EB area times dxinv (as in the operator)
                Real kappa = 1.0;
                Real ebfac = 0.0;
                if (cell_cut) {
                    kappa = kap(i,j,k);
                    if (fl(i,j,k).isSingleValued()) {
                        AMREX_D_TERM(auto const& apx = apd[0];, auto const& apy = apd[1];,
                                     auto const& apz = apd[2];)
#if (AMREX_SPACEDIM == 2)
                        Real dapx = (apx(i,j,k)-apx(i+1,j,k))*(dxinv[0]/dxinv[1]);
                        Real dapy = (apy(i,j,k)-apy(i,j+1,k));
                        Real an = std::sqrt(dapx*dapx+dapy*dapy);
                        Real ebarea = get_eb_area(ba(i,j,k), dapx/an, dapy/an, dxinv[0]/dxinv[1]);
                        ebfac = dxinv[1]*ebarea;
#else
                        Real dapx = (apx(i,j,k)-apx(i+1,j,k))*(dxinv[0]/dxinv[1])*(dxinv[0]/dxinv[2]);
                        Real dapy = (apy(i,j,k)-apy(i,j+1,k))*(dxinv[0]/dxinv[2]);
                        Real dapz = (apz(i,j,k)-apz(i,j,k+1))*(dxinv[0]/dxinv[1]);
                        Real an = std::sqrt(dapx*dapx+dapy*dapy+dapz*dapz);
                        Real ebarea = get_eb_area(ba(i,j,k), dapx/an, dapy/an, dapz/an,
                                                  dxinv[0]/dxinv[1], dxinv[0]/dxinv[2]);
                        ebfac = dxinv[1]*(dxinv[2]/dxinv[0])*ebarea;
#endif
                    }
                }

                for (int n = 0; n < ncomp; ++n) {
                    Real div = 0.0;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        IntVect const hi = iv + IntVect::TheDimensionVector(d);
                        Real const aphi = cut ? apd[d](hi) : Real(1.0);
                        Real const aplo = cut ? apd[d](iv) : Real(1.0);
                        div += dxinv[d]*(aphi*fd[d](hi,n) - aplo*fd[d](iv,n));
                    }
                    Real const lhs = kappa*(L(i,j,k,n) - alpha*a(i,j,k)*x(i,j,k,n));
                    Real const r = std::abs(lhs - beta*(div + ebfac*feb(i,j,k,n)));
                    lmax = std::max(lmax, std::abs(lhs));
                    if (r > emax[kind]) { emax[kind] = r; eloc[kind] = iv; ecomp[kind] = n; }
                }
            });
        }

        ParallelDescriptor::ReduceRealMax(lmax);
        amrex::Print() << "max |kappa*(L - alpha*a*phi)| = " << lmax << "\n";
        bool failed = false;
        for (int kind = 0; kind < NKinds; ++kind) {
            Real e = emax[kind];
            Long nc = ncells[kind];
            ParallelDescriptor::ReduceRealMax(e);
            ParallelDescriptor::ReduceLongSum(nc);
            if (nc == 0) { continue; }
            Real const rel = e / lmax;
            if (!(rel <= max_rel_err)) { failed = true; }
            amrex::Print() << "  " << kind_name[kind] << " (" << nc << " cells): max rel mismatch = "
                           << rel;
            if (ParallelDescriptor::NProcs() == 1 && e > 0.0) {
                amrex::Print() << " at " << eloc[kind] << " comp " << ecomp[kind];
            }
            amrex::Print() << "\n";
        }

        if (failed) {
            amrex::Abort("MLEBABecLap operator and fluxes are inconsistent");
        }
        amrex::Print() << "PASS\n";
    }
    amrex::Finalize();
}
