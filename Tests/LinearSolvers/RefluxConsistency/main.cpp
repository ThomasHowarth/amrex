// Checks that a two-level MLMG::apply refluxes the operator's own fluxes.
//
// For a coarse cell next to the fine level, the two-level operator replaces
// the coarse flux on each coarse/fine face by the average of the fine fluxes
// on it. So, with kappa the volume fraction, ap the area fractions, F the
// fluxes from MLMG::getFluxes (without beta) and L1 the operator of the
// coarse level alone (same solution, coefficients and boundary conditions),
//
//   kappa (L - L1) = beta sum_{c/f faces} s dxinv_d (avg_fine(ap F) - ap_c F_c),
//
// with s = +1 on the high face and -1 on the low face. This holds for any
// fields, so the solution and coefficients are smooth functions; the shear
// viscosity varies in space so that the tensor operators' cross terms matter.
//
// op = abeclap (MLABecLaplacian), tensor (MLTensorOp), ebabeclap
// (MLEBABecLap, EB Dirichlet) or ebtensor (MLEBTensorOp, EB no-slip wall).
// The EB is set with eb2.* (default: a sphere of radius 0.3 at the center,
// fluid outside). The run aborts if the relative error exceeds max_rel_err.

#include <AMReX.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLTensorOp.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#ifdef AMREX_USE_EB
#include <AMReX_EB2.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_EBMultiFabUtil.H>
#include <AMReX_MLEBABecLap.H>
#include <AMReX_MLEBTensorOp.H>
#endif

#include <limits>
#include <memory>
#include <string>

using namespace amrex;

namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real field (Real x, Real y, Real z, int n)
{
    amrex::ignore_unused(z);
    Real const s = Real(0.37)*Real(n+1);
    return std::sin(Real(2.1)*x + s) * std::cos(Real(1.7)*y - s)
#if (AMREX_SPACEDIM == 3)
        * std::cos(Real(1.3)*z + Real(0.5)*s)
#endif
        + Real(0.2)*x*y;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real viscosity (Real x, Real y, Real z)
{
    return Real(1.0) + Real(0.5)*std::sin(Real(3.0)*x)*std::cos(Real(2.0)*y + z);
}

// f at the cell centers (including the ghost cells) or face centers of mf
template <typename F>
void fill (MultiFab& mf, Geometry const& geom, F const& f)
{
    auto const dx = geom.CellSizeArray();
    auto const plo = geom.ProbLoArray();
    IntVect const typ = mf.ixType().toIntVect();
    auto const& a = mf.arrays();
    ParallelFor(mf, mf.nGrowVect(), mf.nComp(), [=] AMREX_GPU_DEVICE (int b, int i, int j, int k, int n)
    {
        IntVect const iv(AMREX_D_DECL(i,j,k));
        Real x[3] = {0.0_rt, 0.0_rt, 0.0_rt};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            x[d] = plo[d] + (Real(iv[d]) + (typ[d] ? 0.0_rt : 0.5_rt)) * dx[d];
        }
        a[b](i,j,k,n) = f(x[0], x[1], x[2], n);
    });
}

struct Problem
{
    std::string op;
    Vector<Geometry> geom;
    Vector<BoxArray> grids;
    Vector<DistributionMapping> dmap;
    Real alpha = 0.7, beta = 1.3, bulk = 0.3;
    LinOpBCType bc = LinOpBCType::Dirichlet;
    int ncomp = 1;
    bool is_eb = false;
    bool is_tensor = false;
#ifdef AMREX_USE_EB
    Vector<std::unique_ptr<EBFArrayBoxFactory>> factory;
#endif

    // A MultiFab on level lev (with the EB factory for the EB operators),
    // cell-centered or nodal in the directions where typ is 1
    [[nodiscard]] MultiFab make (int lev, int nc, int ng, IntVect const& typ = IntVect(0)) const
    {
        BoxArray const ba = amrex::convert(grids[lev], typ);
#ifdef AMREX_USE_EB
        if (is_eb) { return MultiFab(ba, dmap[lev], nc, ng, MFInfo(), *factory[lev]); }
#endif
        return MultiFab(ba, dmap[lev], nc, ng);
    }

    // The operator on levels [0, nlevs), with sol giving the domain boundary
    // values, applied to sol; the fluxes (without beta) go to flux
    void apply (int nlevs, Vector<MultiFab>& sol, Vector<MultiFab>& out,
                Vector<Array<MultiFab,AMREX_SPACEDIM>>* flux) const
    {
        Vector<Geometry> g(geom.begin(), geom.begin()+nlevs);
        Vector<BoxArray> ba(grids.begin(), grids.begin()+nlevs);
        Vector<DistributionMapping> dm(dmap.begin(), dmap.begin()+nlevs);
        LPInfo info;
        info.setMaxCoarseningLevel(0);

        std::unique_ptr<MLLinOp> linop;

        // The coefficients on both levels. The two-level operator replaces
        // the coarse ones under the fine level (the coarse/fine faces
        // included) by averages of the fine ones, so the coarse level gets
        // those averages for the single-level operator too.
        Vector<MultiFab> acoef(2), bcc(2);
        Vector<Array<MultiFab,AMREX_SPACEDIM>> bcoef(2);
        for (int lev = 0; lev < 2; ++lev) {
            acoef[lev] = make(lev, 1, 0);
            acoef[lev].setVal(1.0);
            bcc[lev] = make(lev, 1, 0);
            fill(bcc[lev], geom[lev], [] AMREX_GPU_DEVICE (Real x, Real y, Real z, int)
                 { return viscosity(x,y,z); });
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                bcoef[lev][d] = make(lev, 1, 0, IntVect::TheDimensionVector(d));
                fill(bcoef[lev][d], geom[lev], [] AMREX_GPU_DEVICE (Real x, Real y, Real z, int)
                     { return viscosity(x,y,z); });
            }
        }
#ifdef AMREX_USE_EB
        if (is_eb) {
            amrex::EB_average_down_faces(GetArrOfConstPtrs(bcoef[1]), GetArrOfPtrs(bcoef[0]),
                                         IntVect(2), geom[0]);
            amrex::EB_average_down_boundaries(bcc[1], bcc[0], 2, 0);
        } else
#endif
        {
            amrex::average_down_faces(GetArrOfConstPtrs(bcoef[1]), GetArrOfPtrs(bcoef[0]),
                                      IntVect(2), geom[0]);
        }

        Array<LinOpBCType,AMREX_SPACEDIM> lobc, hibc;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            lobc[d] = g[0].isPeriodic(d) ? LinOpBCType::Periodic : bc;
            hibc[d] = lobc[d];
        }

        auto setup = [&] (auto& lo) {
            if (is_tensor) {
                lo.setDomainBC(Vector<Array<LinOpBCType,AMREX_SPACEDIM>>(ncomp, lobc),
                               Vector<Array<LinOpBCType,AMREX_SPACEDIM>>(ncomp, hibc));
            } else {
                lo.setDomainBC(lobc, hibc);
            }
            for (int lev = 0; lev < nlevs; ++lev) { lo.setLevelBC(lev, &sol[lev]); }
            lo.setScalars(alpha, beta);
            for (int lev = 0; lev < nlevs; ++lev) { lo.setACoeffs(lev, acoef[lev]); }
        };

        if (op == "abeclap") {
            auto p = std::make_unique<MLABecLaplacian>(g, ba, dm, info);
            setup(*p);
            for (int lev = 0; lev < nlevs; ++lev) { p->setBCoeffs(lev, GetArrOfConstPtrs(bcoef[lev])); }
            linop = std::move(p);
        } else if (op == "tensor") {
            auto p = std::make_unique<MLTensorOp>(g, ba, dm, info);
            setup(*p);
            for (int lev = 0; lev < nlevs; ++lev) {
                p->setShearViscosity(lev, GetArrOfConstPtrs(bcoef[lev]));
                p->setBulkViscosity(lev, bulk);
            }
            linop = std::move(p);
        }
#ifdef AMREX_USE_EB
        else {
            Vector<EBFArrayBoxFactory const*> f;
            for (int lev = 0; lev < nlevs; ++lev) { f.push_back(factory[lev].get()); }
            if (op == "ebabeclap") {
                auto p = std::make_unique<MLEBABecLap>(g, ba, dm, info, f, ncomp);
                setup(*p);
                for (int lev = 0; lev < nlevs; ++lev) {
                    p->setBCoeffs(lev, GetArrOfConstPtrs(bcoef[lev]));
                    p->setEBHomogDirichlet(lev, bcc[lev]);
                }
                linop = std::move(p);
            } else if (op == "ebtensor") {
                auto p = std::make_unique<MLEBTensorOp>(g, ba, dm, info, f);
                setup(*p);
                for (int lev = 0; lev < nlevs; ++lev) {
                    p->setShearViscosity(lev, GetArrOfConstPtrs(bcoef[lev]), MLLinOp::Location::FaceCenter);
                    p->setBulkViscosity(lev, bulk);
                    p->setEBShearViscosity(lev, bcc[lev]);
                    p->setEBBulkViscosity(lev, bulk);
                }
                linop = std::move(p);
            }
        }
#endif
        if (!linop) { amrex::Abort("Unknown op " + op); }

        MLMG mlmg(*linop);
        mlmg.apply(GetVecOfPtrs(out), GetVecOfPtrs(sol));
        if (flux) {
            Vector<Array<MultiFab*,AMREX_SPACEDIM>> fp;
            for (int lev = 0; lev < nlevs; ++lev) { fp.push_back(GetArrOfPtrs((*flux)[lev])); }
            mlmg.getFluxes(fp, GetVecOfPtrs(sol), is_eb ? MLLinOp::Location::FaceCentroid : MLLinOp::Location::FaceCenter);
        }
    }
};

} // namespace

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        Problem p;
        Vector<int> n_cell_in{32};
        int max_grid_size = 16;
        Vector<Real> fine_lo(AMREX_SPACEDIM, 0.25_rt), fine_hi(AMREX_SPACEDIM, 0.75_rt);
        std::string bc{"dirichlet"};
        Vector<int> is_periodic(AMREX_SPACEDIM, 0);
        Real max_rel_err = 1.e-10;
        p.op = "tensor";
        {
            ParmParse pp;
            pp.query("op", p.op);
            pp.queryarr("n_cell", n_cell_in);
            pp.query("max_grid_size", max_grid_size);
            pp.queryarr("fine_lo", fine_lo, 0, AMREX_SPACEDIM);
            pp.queryarr("fine_hi", fine_hi, 0, AMREX_SPACEDIM);
            pp.query("bc", bc);
            pp.queryarr("is_periodic", is_periodic, 0, AMREX_SPACEDIM);
            pp.query("alpha", p.alpha);
            pp.query("beta", p.beta);
            pp.query("bulk_viscosity", p.bulk);
            pp.query("max_rel_err", max_rel_err);
        }
        p.is_eb = (p.op == "ebabeclap" || p.op == "ebtensor");
        p.is_tensor = (p.op == "tensor" || p.op == "ebtensor");
        p.ncomp = p.is_tensor ? AMREX_SPACEDIM : 1;
        p.bc = (bc == "neumann") ? LinOpBCType::Neumann : LinOpBCType::Dirichlet;
#ifndef AMREX_USE_EB
        if (p.is_eb) { amrex::Abort("op " + p.op + " needs AMReX_EB"); }
#endif

        IntVect n_cell;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            n_cell[d] = n_cell_in[std::min(d, int(n_cell_in.size())-1)];
        }
        RealBox rb({AMREX_D_DECL(0.0_rt,0.0_rt,0.0_rt)}, {AMREX_D_DECL(1.0_rt,1.0_rt,1.0_rt)});
        Array<int,AMREX_SPACEDIM> per{AMREX_D_DECL(is_periodic[0],is_periodic[1],is_periodic[2])};
        Box const cdomain(IntVect(0), n_cell-1);
        p.geom.emplace_back(cdomain, rb, CoordSys::cartesian, per);
        p.geom.emplace_back(amrex::refine(cdomain,2), rb, CoordSys::cartesian, per);

        IntVect flo, fhi;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            flo[d] = int(fine_lo[d]*n_cell[d]);
            fhi[d] = int(fine_hi[d]*n_cell[d]) - 1;
        }
        p.grids.emplace_back(cdomain);
        p.grids[0].maxSize(max_grid_size);
        p.grids.emplace_back(amrex::refine(Box(flo,fhi),2));
        p.grids[1].maxSize(max_grid_size);
        for (int lev = 0; lev < 2; ++lev) { p.dmap.emplace_back(p.grids[lev]); }

#ifdef AMREX_USE_EB
        if (p.is_eb) {
            ParmParse pp("eb2");
            if (!pp.contains("geom_type")) {
                pp.add("geom_type", std::string("sphere"));
                pp.addarr("sphere_center", std::vector<Real>{AMREX_D_DECL(0.5_rt,0.5_rt,0.5_rt)});
                pp.add("sphere_radius", 0.3_rt);
                pp.add("sphere_has_fluid_inside", 0);
            }
            EB2::Build(p.geom[1], 1, 1);
            for (int lev = 0; lev < 2; ++lev) {
                p.factory.push_back(makeEBFabFactory(p.geom[lev], p.grids[lev], p.dmap[lev],
                                                     {2,2,2}, EBSupport::full));
            }
        }
#endif

        Vector<MultiFab> sol(2), out(2);
        Vector<Array<MultiFab,AMREX_SPACEDIM>> flux(2);
        for (int lev = 0; lev < 2; ++lev) {
            sol[lev] = p.make(lev, p.ncomp, 1);
            out[lev] = p.make(lev, p.ncomp, 0);
            fill(sol[lev], p.geom[lev], [] AMREX_GPU_DEVICE (Real x, Real y, Real z, int n)
                 { return field(x,y,z,n); });
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                flux[lev][d].define(amrex::convert(p.grids[lev], IntVect::TheDimensionVector(d)),
                                    p.dmap[lev], p.ncomp, 0);
            }
        }
        p.apply(2, sol, out, &flux);

        Vector<MultiFab> sol1(1), out1(1);
        // Filled again: the two-level apply has changed the ghost cells of
        // sol, which also give the Dirichlet values
        sol1[0] = p.make(0, p.ncomp, 1);
        fill(sol1[0], p.geom[0], [] AMREX_GPU_DEVICE (Real x, Real y, Real z, int n)
             { return field(x,y,z,n); });
        out1[0] = p.make(0, p.ncomp, 0);
        p.apply(1, sol1, out1, nullptr);

        // Area and volume fractions
        MultiFab vfrac(p.grids[0], p.dmap[0], 1, 0);
        Array<MultiFab,AMREX_SPACEDIM> capf, fapf;
        vfrac.setVal(1.0);
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            capf[d].define(flux[0][d].boxArray(), p.dmap[0], 1, 0);
            fapf[d].define(flux[1][d].boxArray(), p.dmap[1], 1, 0);
            capf[d].setVal(1.0);
            fapf[d].setVal(1.0);
        }
#ifdef AMREX_USE_EB
        if (p.is_eb) {
            MultiFab::Copy(vfrac, p.factory[0]->getVolFrac(), 0, 0, 1, 0);
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                MultiFab::Copy(capf[d], p.factory[0]->getAreaFrac()[d]->ToMultiFab(1.0_rt, 0.0_rt),
                               0, 0, 1, 0);
                MultiFab::Copy(fapf[d], p.factory[1]->getAreaFrac()[d]->ToMultiFab(1.0_rt, 0.0_rt),
                               0, 0, 1, 0);
            }
        }
#endif

        // avg_fine(ap F) on the coarse faces under the fine level
        Array<MultiFab,AMREX_SPACEDIM> favg;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            MultiFab apF(flux[1][d].boxArray(), p.dmap[1], p.ncomp, 0);
            MultiFab::Copy(apF, flux[1][d], 0, 0, p.ncomp, 0);
            for (int n = 0; n < p.ncomp; ++n) { MultiFab::Multiply(apF, fapf[d], 0, n, 1, 0); }
            favg[d].define(flux[0][d].boxArray(), p.dmap[0], p.ncomp, 0);
            favg[d].setVal(0.0);
            amrex::average_down_faces(apF, favg[d], IntVect(2), p.geom[0]);
        }

        // Coarse cells covered by the fine level, with a ghost cell that
        // includes the periodic images of the fine level
        iMultiFab const covered = amrex::makeFineMask(p.grids[0], p.dmap[0], IntVect(1),
                                                      p.grids[1], IntVect(2),
                                                      p.geom[0].periodicity(), 0, 1);

        // err: |lhs - rhs| in comps [0,ncomp) for cut cells and
        // [ncomp,2 ncomp) for regular cells; corr: |rhs|; scale: |kappa L|
        MultiFab err(p.grids[0], p.dmap[0], 2*p.ncomp, 0);
        MultiFab corr(p.grids[0], p.dmap[0], p.ncomp, 0);
        MultiFab scale(p.grids[0], p.dmap[0], p.ncomp, 0);
        {
            auto const dxinv = p.geom[0].InvCellSizeArray();
            Real const beta = p.beta;
            int const ncomp = p.ncomp;
            auto const& cm = covered.const_arrays();
            auto const& kap = vfrac.const_arrays();
            auto const& L2 = out[0].const_arrays();
            auto const& L1 = out1[0].const_arrays();
            GpuArray<MultiArray4<Real const>,AMREX_SPACEDIM> F, ap, avg;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                F[d] = flux[0][d].const_arrays();
                ap[d] = capf[d].const_arrays();
                avg[d] = favg[d].const_arrays();
            }
            auto const& e = err.arrays();
            auto const& c = corr.arrays();
            auto const& s = scale.arrays();
            ParallelFor(err, [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
            {
                IntVect const iv(AMREX_D_DECL(i,j,k));
                Real const kappa = kap[b](i,j,k);
                bool const cut = kappa > 0.0_rt && kappa < 1.0_rt;
                for (int n = 0; n < 2*ncomp; ++n) { e[b](i,j,k,n) = 0.0_rt; }
                for (int n = 0; n < ncomp; ++n) {
                    c[b](i,j,k,n) = 0.0_rt;
                    s[b](i,j,k,n) = cm[b](i,j,k) ? 0.0_rt : std::abs(kappa*L2[b](i,j,k,n));
                }
                if (cm[b](i,j,k) || kappa == 0.0_rt) { return; }
                for (int n = 0; n < ncomp; ++n) {
                    Real rhs = 0.0_rt;
                    bool cf = false;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        IntVect const e_d = IntVect::TheDimensionVector(d);
                        if (cm[b](iv+e_d)) {
                            rhs += dxinv[d] * (avg[d][b](iv+e_d,n) - ap[d][b](iv+e_d)*F[d][b](iv+e_d,n));
                            cf = true;
                        }
                        if (cm[b](iv-e_d)) {
                            rhs -= dxinv[d] * (avg[d][b](iv,n) - ap[d][b](iv)*F[d][b](iv,n));
                            cf = true;
                        }
                    }
                    if (!cf) { continue; }
                    rhs *= beta;
                    Real const lhs = kappa * (L2[b](i,j,k,n) - L1[b](i,j,k,n));
                    e[b](i,j,k,n + (cut ? 0 : ncomp)) = std::abs(lhs - rhs);
                    c[b](i,j,k,n) = std::abs(rhs);
                }
            });
        }

        amrex::Print() << "op = " << p.op << ", " << p.grids[0].size() << " coarse and "
                       << p.grids[1].size() << " fine boxes\n";
        Real worst = 0.0_rt;
        for (int n = 0; n < p.ncomp; ++n) {
            Real const sc = amrex::max(scale.norminf(n), std::numeric_limits<Real>::min());
            Real const ecut = err.norminf(n) / sc;
            Real const ereg = err.norminf(n + p.ncomp) / sc;
            amrex::Print() << "  comp " << n << ": max |kappa L| = " << sc
                           << ", max reflux correction / that = " << corr.norminf(n)/sc
                           << "\n    rel error: cut coarse cells next to fine " << ecut
                           << ", regular " << ereg << "\n";
            worst = amrex::max(worst, ecut, ereg);
        }
        if (worst > max_rel_err) {
            amrex::Abort("Two-level operator does not reflux its fluxes: "
                         + std::to_string(worst) + " > " + std::to_string(max_rel_err));
        }
        amrex::Print() << "PASSED\n";
    }
    amrex::Finalize();
}
