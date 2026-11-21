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
// On a coarse face covered by a finer level, ap*F is the average of the fine
// ap*F, as reflux puts in the coarse operator.

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
#include <limits>
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
                    Real x = problo[d] + (iv[d] + (typ.cellCentered(d) ? 0.5 : 0.0))*dx[d];
                    s *= std::cos(Real(2.0+n+d)*x);
                }
                v += s;
            }
            a(i,j,k,n) = v;
        });
    }
}

// Kinds of cells reported separately.
enum Kind { RegInterior, RegDomain, CutInterior, CutDomain, RegNextToFine, CutNextToFine,
            NextToCrse, NKinds };
const char* kind_name[NKinds] = {"regular", "regular at domain bdry", "cut", "cut at domain bdry",
                                 "regular coarse next to fine", "cut coarse next to fine",
                                 "fine next to coarse"};

}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        ParmParse pp;
        Vector<int> n_cell_v{64}; // one value, or one per direction
        pp.queryarr("n_cell", n_cell_v);
        IntVect n_cell(n_cell_v[0]);
        if (n_cell_v.size() >= AMREX_SPACEDIM) {
            for (int d = 0; d < AMREX_SPACEDIM; ++d) { n_cell[d] = n_cell_v[d]; }
        }
        int max_grid_size = 32;
        pp.query("max_grid_size", max_grid_size);
        int max_level = 0;
        pp.query("max_level", max_level);
        int ncomp = 1;
        pp.query("ncomp", ncomp);
        Vector<Real> prob_lo(AMREX_SPACEDIM, 0.0), prob_hi(AMREX_SPACEDIM, 1.0);
        pp.queryarr("prob_lo", prob_lo);
        pp.queryarr("prob_hi", prob_hi);
        Vector<Real> fine_lo(AMREX_SPACEDIM, 0.25), fine_hi(AMREX_SPACEDIM, 0.75);
        pp.queryarr("fine_lo", fine_lo); // fraction of the domain
        pp.queryarr("fine_hi", fine_hi);
        Vector<std::string> lo_bc_s(AMREX_SPACEDIM, "dirichlet"), hi_bc_s(AMREX_SPACEDIM, "dirichlet");
        pp.queryarr("lo_bc", lo_bc_s);
        pp.queryarr("hi_bc", hi_bc_s);
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

        int const nlevels = max_level + 1;
        Vector<Geometry> geom(nlevels);
        Vector<BoxArray> grids(nlevels);
        Vector<DistributionMapping> dmap(nlevels);
        RealBox rb(prob_lo.data(), prob_hi.data());
        Box const domain0(IntVect(0), n_cell - 1);
        for (int lev = 0; lev < nlevels; ++lev) {
            Box const domain = amrex::refine(domain0, 1 << lev);
            geom[lev].define(domain, rb, CoordSys::cartesian, is_periodic);
            Box region = domain;
            if (lev > 0) { // the same fraction of the domain on every finer level
                IntVect lo, hi;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    lo[d] = static_cast<int>(fine_lo[d]*domain0.length(d));
                    hi[d] = static_cast<int>(fine_hi[d]*domain0.length(d)) - 1;
                }
                region = amrex::refine(Box(lo,hi), 1 << lev);
            }
            grids[lev] = BoxArray(region);
            grids[lev].maxSize(max_grid_size);
            dmap[lev].define(grids[lev]);
        }

        EB2::Build(geom[max_level], max_level, max_level);

        Vector<std::unique_ptr<EBFArrayBoxFactory>> factory(nlevels);
        for (int lev = 0; lev < nlevels; ++lev) {
            factory[lev] = makeEBFabFactory(geom[lev], grids[lev], dmap[lev], {2,2,2},
                                            EBSupport::full);
        }

        Vector<MultiFab> phi(nlevels), Lphi(nlevels), acoef(nlevels), ebflux(nlevels);
        Vector<MultiFab> robin_a(nlevels), robin_b(nlevels), robin_f(nlevels);
        Vector<MultiFab> phieb(nlevels), beb(nlevels);
        Vector<Array<MultiFab,AMREX_SPACEDIM>> bcoef(nlevels), flux(nlevels);
        for (int lev = 0; lev < nlevels; ++lev) {
            auto const& fact = *factory[lev];
            phi[lev].define(grids[lev], dmap[lev], ncomp, 1, MFInfo(), fact);
            Lphi[lev].define(grids[lev], dmap[lev], ncomp, 0, MFInfo(), fact);
            acoef[lev].define(grids[lev], dmap[lev], 1, 0, MFInfo(), fact);
            ebflux[lev].define(grids[lev], dmap[lev], ncomp, 0, MFInfo(), fact);
            fill(phi[lev], geom[lev], -0.1, 0.1, true, seed+0);
            fill(acoef[lev], geom[lev], 0.5, 1.5, false, seed+1);
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                BoxArray const fba = amrex::convert(grids[lev], IntVect::TheDimensionVector(d));
                bcoef[lev][d].define(fba, dmap[lev], ncomp, 0, MFInfo(), fact);
                flux[lev][d].define(fba, dmap[lev], ncomp, 0, MFInfo(), fact);
                fill(bcoef[lev][d], geom[lev], 0.5, 1.5, false, seed+2+d);
            }
            if (has_robin) {
                robin_a[lev].define(grids[lev], dmap[lev], ncomp, 1, MFInfo(), fact);
                robin_b[lev].define(grids[lev], dmap[lev], ncomp, 1, MFInfo(), fact);
                robin_f[lev].define(grids[lev], dmap[lev], ncomp, 1, MFInfo(), fact);
                fill(robin_a[lev], geom[lev], 0.0, 1.0, false, seed+5);
                fill(robin_b[lev], geom[lev], 1.0, 2.0, false, seed+6);
                fill(robin_f[lev], geom[lev], -1.0, 1.0, false, seed+7);
            }
            if (eb_bc != "neumann") {
                phieb[lev].define(grids[lev], dmap[lev], ncomp, 0, MFInfo(), fact);
                beb[lev].define(grids[lev], dmap[lev], ncomp, 0, MFInfo(), fact);
                fill(phieb[lev], geom[lev], -1.0, 1.0, false, seed+8);
                fill(beb[lev], geom[lev], 0.5, 1.5, false, seed+9);
            }
        }

        LPInfo info;
        info.setMaxCoarseningLevel(0);
        MLEBABecLap mleb(geom, grids, dmap, info, amrex::GetVecOfConstPtrs(factory), ncomp);
        mleb.setDomainBC(lobc, hibc);
        mleb.setScalars(alpha, beta);
        auto const beta_loc = beta_on_centroid ? MLLinOp::Location::FaceCentroid
                                               : MLLinOp::Location::FaceCenter;
        for (int lev = 0; lev < nlevels; ++lev) {
            if (has_robin) {
                mleb.setLevelBC(lev, &phi[lev], &robin_a[lev], &robin_b[lev], &robin_f[lev]);
            } else {
                mleb.setLevelBC(lev, &phi[lev]);
            }
            mleb.setACoeffs(lev, acoef[lev]);
            mleb.setBCoeffs(lev, amrex::GetArrOfConstPtrs(bcoef[lev]), beta_loc);
            if (eb_bc == "dirichlet") {
                if (eb_beta_scalar) {
                    mleb.setEBDirichlet(lev, phieb[lev], 1.1);
                } else {
                    mleb.setEBDirichlet(lev, phieb[lev], beb[lev]);
                }
            } else if (eb_bc == "homog_dirichlet") {
                if (eb_beta_scalar) {
                    mleb.setEBHomogDirichlet(lev, 1.1);
                } else {
                    mleb.setEBHomogDirichlet(lev, beb[lev]);
                }
            } else {
                AMREX_ALWAYS_ASSERT(eb_bc == "neumann");
            }
        }

        MLMG mlmg(mleb);
        mlmg.apply(amrex::GetVecOfPtrs(Lphi), amrex::GetVecOfPtrs(phi));
        mlmg.getFluxes(amrex::GetVecOfArrOfPtrs(flux), amrex::GetVecOfPtrs(phi),
                       MLMG::Location::FaceCentroid);
        mlmg.getEBFluxes(amrex::GetVecOfPtrs(ebflux), amrex::GetVecOfPtrs(phi));

        // On each coarse face covered by fine faces, the average of the fine
        // area-weighted fluxes (ap*F), which reflux puts in the coarse
        // operator (components [0,ncomp)), and of the fine fluxes without
        // area fractions (components [ncomp,2*ncomp)).  Other faces hold a
        // marker.
        Real const marker = std::numeric_limits<Real>::max();
        Vector<Array<MultiFab,AMREX_SPACEDIM>> fine_apf(nlevels);
        for (int lev = 0; lev < max_level; ++lev) {
            auto const fine_area = factory[lev+1]->getAreaFrac();
            BoxArray const cba = amrex::coarsen(grids[lev+1], 2);
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                IntVect const nodal = IntVect::TheDimensionVector(d);
                fine_apf[lev][d].define(amrex::convert(grids[lev], nodal), dmap[lev], 2*ncomp, 0);
                fine_apf[lev][d].setVal(marker);
                MultiFab avg(amrex::convert(cba, nodal), dmap[lev+1], 2*ncomp, 0);
                for (MFIter mfi(avg); mfi.isValid(); ++mfi) {
                    auto const& c = avg.array(mfi);
                    auto const& f = flux[lev+1][d].const_array(mfi);
                    bool const has_ap = fine_area[d]->ok(mfi);
                    auto const& ap = has_ap ? fine_area[d]->const_array(mfi) : Array4<Real const>{};
                    amrex::LoopOnCpu(mfi.validbox(), ncomp, [&] (int i, int j, int k, int n)
                    {
                        IntVect const civ(AMREX_D_DECL(i,j,k));
                        Real sum = 0.0, sum_noap = 0.0;
                        int nf = 0;
                        // the fine faces on this coarse face: 2 per transverse direction
                        Box fb(2*civ, 2*civ + (IntVect(1) - nodal));
                        amrex::LoopOnCpu(fb, [&] (int ii, int jj, int kk)
                        {
                            Real const a = has_ap ? ap(ii,jj,kk) : Real(1.0);
                            sum += a * f(ii,jj,kk,n);
                            sum_noap += f(ii,jj,kk,n);
                            ++nf;
                        });
                        c(i,j,k,n) = sum / nf;
                        c(i,j,k,n+ncomp) = sum_noap / nf;
                    });
                }
                fine_apf[lev][d].ParallelCopy(avg);
            }
        }

        bool failed = false;
        for (int lev = 0; lev < nlevels; ++lev) {
            auto const& fact = *factory[lev];
            auto const& flags = fact.getMultiEBCellFlagFab();
            auto const& vfrac = fact.getVolFrac();
            auto const area = fact.getAreaFrac();
            auto const& barea = fact.getBndryArea();
            auto const dxinv = geom[lev].InvCellSizeArray();
            Box const& dom = geom[lev].Domain();

            // Mask: 1 where a finer level covers the cell, 2 where the cell
            // touches the finer level's region, 0 elsewhere
            iMultiFab fine_mask(grids[lev], dmap[lev], 1, 0);
            fine_mask.setVal(0);
            if (lev < max_level) {
                BoxArray cba = amrex::coarsen(grids[lev+1], 2);
                for (MFIter mfi(fine_mask); mfi.isValid(); ++mfi) {
                    auto const& m = fine_mask.array(mfi);
                    Box const& vbx = mfi.validbox();
                    for (auto const& isect : cba.intersections(amrex::grow(vbx,1))) {
                        Box const b = amrex::grow(isect.second,1) & vbx;
                        amrex::LoopOnCpu(b, [&] (int i, int j, int k) { m(i,j,k) = 2; });
                    }
                    for (auto const& isect : cba.intersections(vbx)) {
                        amrex::LoopOnCpu(isect.second, [&] (int i, int j, int k) { m(i,j,k) = 1; });
                    }
                }
            }
            BoxArray const crse_region = (lev > 0) ? grids[lev] : BoxArray();

            Real lmax = 0.0;
            std::array<Real,NKinds> emax{};
            std::array<IntVect,NKinds> eloc{};
            std::array<int,NKinds> ecomp{};
            std::array<Long,NKinds> ncells{};
            Real emax_ya = 0.0; // see below
            for (MFIter mfi(Lphi[lev]); mfi.isValid(); ++mfi) {
                Box const& vbx = mfi.validbox();
                FabType const typ = flags[mfi].getType(vbx);
                if (typ == FabType::covered) { continue; }
                bool const cut = (typ != FabType::regular);
                auto const& fl = flags.const_array(mfi);
                auto const& L = Lphi[lev].const_array(mfi);
                auto const& x = phi[lev].const_array(mfi);
                auto const& a = acoef[lev].const_array(mfi);
                auto const& feb = ebflux[lev].const_array(mfi);
                auto const& m = fine_mask.const_array(mfi);
                AMREX_D_TERM(auto const& fx = flux[lev][0].const_array(mfi);,
                             auto const& fy = flux[lev][1].const_array(mfi);,
                             auto const& fz = flux[lev][2].const_array(mfi););
                Array4<Real const> kap, ba, AMREX_D_DECL(apx, apy, apz);
                if (cut) {
                    kap = vfrac.const_array(mfi);
                    ba = barea.const_array(mfi);
                    AMREX_D_TERM(apx = area[0]->const_array(mfi);,
                                 apy = area[1]->const_array(mfi);,
                                 apz = area[2]->const_array(mfi););
                }
                Array<Array4<Real const>,AMREX_SPACEDIM> fapf;
                if (lev < max_level) {
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        fapf[d] = fine_apf[lev][d].const_array(mfi);
                    }
                }
                Array<Array4<Real const>,AMREX_SPACEDIM> const fd{AMREX_D_DECL(fx,fy,fz)};
                Array<Array4<Real const>,AMREX_SPACEDIM> const apd{AMREX_D_DECL(apx,apy,apz)};
                // ap*F on a face, with the fine average on coarse/fine faces
                auto apF = [&] (int d, IntVect const& f, int n) -> Real
                {
                    if (fapf[d] && fapf[d](f,n) != marker) { return fapf[d](f,n); }
                    Real const ap = cut ? apd[d](f) : Real(1.0);
                    return ap * fd[d](f,n);
                };
                amrex::LoopOnCpu(vbx, ncomp, [&] (int i, int j, int k, int n)
                {
                    if (fl(i,j,k).isCovered() || m(i,j,k) == 1) { return; }
                    IntVect const iv(AMREX_D_DECL(i,j,k));
                    Real kappa = 1.0;
                    Real div = 0.0;
                    Real ebterm = 0.0;
                    Real div_crse = 0.0; // coarse ap*F only
                    Real corr = 0.0;     // reflux correction without area or volume fractions
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        IntVect const hi = iv + IntVect::TheDimensionVector(d);
                        div += dxinv[d]*(apF(d,hi,n) - apF(d,iv,n));
                        for (int s : {1, -1}) {
                            IntVect const f = (s == 1) ? hi : iv;
                            Real const ap = cut ? apd[d](f) : Real(1.0);
                            div_crse += s*dxinv[d]*ap*fd[d](f,n);
                            if (fapf[d] && fapf[d](f,n) != marker) {
                                corr += s*dxinv[d]*(fapf[d](f,n+ncomp) - fd[d](f,n));
                            }
                        }
                    }
                    if (cut && !fl(i,j,k).isRegular()) {
                        kappa = kap(i,j,k);
                        if (fl(i,j,k).isSingleValued()) {
#if (AMREX_SPACEDIM == 2)
                            Real dapx = (apx(i,j,k)-apx(i+1,j,k))*(dxinv[0]/dxinv[1]);
                            Real dapy = (apy(i,j,k)-apy(i,j+1,k));
                            Real an = std::sqrt(dapx*dapx+dapy*dapy);
                            Real ebarea = get_eb_area(ba(i,j,k), dapx/an, dapy/an, dxinv[0]/dxinv[1]);
                            ebterm = dxinv[1]*ebarea*feb(i,j,k,n);
#else
                            Real dapx = (apx(i,j,k)-apx(i+1,j,k))*(dxinv[0]/dxinv[1])*(dxinv[0]/dxinv[2]);
                            Real dapy = (apy(i,j,k)-apy(i,j+1,k))*(dxinv[0]/dxinv[2]);
                            Real dapz = (apz(i,j,k)-apz(i,j,k+1))*(dxinv[0]/dxinv[1]);
                            Real an = std::sqrt(dapx*dapx+dapy*dapy+dapz*dapz);
                            Real ebarea = get_eb_area(ba(i,j,k), dapx/an, dapy/an, dapz/an,
                                                      dxinv[0]/dxinv[1], dxinv[0]/dxinv[2]);
                            ebterm = dxinv[1]*(dxinv[2]/dxinv[0])*ebarea*feb(i,j,k,n);
#endif
                        }
                    }
                    Real const lhs = kappa*(L(i,j,k,n) - alpha*a(i,j,k)*x(i,j,k,n));
                    Real const r = std::abs(lhs - beta*(div + ebterm));
                    lmax = std::max(lmax, std::abs(lhs));

                    bool at_domain = false;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        if (!is_periodic[d] && (iv[d] == dom.smallEnd(d) || iv[d] == dom.bigEnd(d))) {
                            at_domain = true;
                        }
                    }
                    bool next_to_crse = false;
                    if (lev > 0) {
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            for (int s : {-1, 1}) {
                                IntVect nb = iv; nb[d] += s;
                                if (dom.contains(nb) && !crse_region.contains(nb)) { next_to_crse = true; }
                            }
                        }
                    }
                    bool const cell_cut = !fl(i,j,k).isRegular();
                    int kind = (m(i,j,k) == 2) ? (cell_cut ? CutNextToFine : RegNextToFine)
                        : next_to_crse ? NextToCrse
                        : cell_cut ? (at_domain ? CutDomain : CutInterior)
                                   : (at_domain ? RegDomain : RegInterior);
                    ++ncells[kind];
                    if (r > emax[kind]) { emax[kind] = r; eloc[kind] = iv; ecomp[kind] = n; }
                    if (kind == CutNextToFine) {
                        // What the operator gives if reflux adds the flux
                        // difference without area and volume fractions
                        Real const r_ya = std::abs(lhs - beta*(div_crse + ebterm) - kappa*beta*corr);
                        emax_ya = std::max(emax_ya, r_ya);
                    }
                });
            }

            ParallelDescriptor::ReduceRealMax(lmax);
            amrex::Print() << "Level " << lev << ": max |kappa*(L - alpha*a*phi)| = " << lmax << "\n";
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
                if (kind == CutNextToFine) {
                    ParallelDescriptor::ReduceRealMax(emax_ya);
                    amrex::Print() << "    (against reflux without area and volume fractions: "
                                   << emax_ya / lmax << ")\n";
                }
            }
        }

        if (failed) {
            amrex::Abort("MLEBABecLap operator and fluxes are inconsistent");
        }
        amrex::Print() << "PASS\n";
    }
    amrex::Finalize();
}
