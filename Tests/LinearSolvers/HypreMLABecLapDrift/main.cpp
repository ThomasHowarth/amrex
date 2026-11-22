// HypreMLABecLap with the upwind drift term of setDriftCoeffs:
//
//   a_scalar a phi - b_scalar div(b grad phi) - c_scalar div(c phi) = rhs
//
// on one level, with Dirichlet and Neumann domain boundaries and face
// velocities that change sign (transport by c: c_scalar has the sign opposite
// to b_scalar's). The right-hand side is the operator, written
// out here independently of the matrix assembly, applied to a known phi; the
// solve must give phi back to the solver tolerance.

#include <AMReX.H>
#include <AMReX_HypreMLABecLap.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

#include <cmath>

using namespace amrex;

namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real exact_phi (RealVect const& x)
{
    return std::sin(Real(2.0)*x[0] + Real(0.5)) * std::cos(Real(1.5)*x[1])
#if (AMREX_SPACEDIM == 3)
        * (Real(1.0) + Real(0.3)*x[2]*x[2])
#endif
        + Real(0.2);
}

// Face velocity in direction idim, changing sign inside the domain
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real velocity (int idim, RealVect const& x)
{
    constexpr Real pi = Real(3.141592653589793);
    if (idim == 0) { return std::cos(Real(2.0)*pi*x[1]) + Real(0.3); }
    if (idim == 1) { return std::sin(Real(2.0)*pi*x[0]); }
    return Real(0.5) - x[0];
}

// Value of the ghost cell across a Dirichlet face from the boundary value and
// the two cells inside (quadratic through the face and the cell centers)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real dirichlet_ghost (Real bc, Real v0, Real v1)
{
    return Real(8.0/3.0)*bc - Real(2.0)*v0 + Real(1.0/3.0)*v1;
}

}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        int n_cell = 32;
        int max_grid_size = 16;
        Real ascalar = 1.0;
        Real bscalar = 1.0;
        Real cscalar = -20.0;
        Real tol = 1.e-11;
        Real max_error = 1.e-8;
        int verbose = 1;
        {
            ParmParse pp;
            pp.query("n_cell", n_cell);
            pp.query("max_grid_size", max_grid_size);
            pp.query("ascalar", ascalar);
            pp.query("bscalar", bscalar);
            pp.query("cscalar", cscalar);
            pp.query("tol", tol);
            pp.query("max_error", max_error);
            pp.query("verbose", verbose);
        }

        // Neumann at the low y face, Dirichlet elsewhere
        Array<LinOpBCType,AMREX_SPACEDIM> lobc{AMREX_D_DECL(LinOpBCType::Dirichlet,
                                                            LinOpBCType::Neumann,
                                                            LinOpBCType::Dirichlet)};
        Array<LinOpBCType,AMREX_SPACEDIM> hibc{AMREX_D_DECL(LinOpBCType::Dirichlet,
                                                            LinOpBCType::Dirichlet,
                                                            LinOpBCType::Dirichlet)};

        Box const domain(IntVect(0), IntVect(n_cell-1));
        Geometry const geom(domain, RealBox(AMREX_D_DECL(0.,0.,0.), AMREX_D_DECL(1.,1.,1.)),
                            CoordSys::cartesian, {AMREX_D_DECL(0,0,0)});
        BoxArray ba(domain);
        ba.maxSize(max_grid_size);
        DistributionMapping const dm(ba);

        auto const problo = geom.ProbLoArray();
        auto const probhi = geom.ProbHiArray();
        auto const dx = geom.CellSizeArray();
        auto const dxinv = geom.InvCellSizeArray();

        MultiFab acoef(ba, dm, 1, 0);
        Array<MultiFab,AMREX_SPACEDIM> bcoef, ccoef;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            BoxArray const fba = amrex::convert(ba, IntVect::TheDimensionVector(idim));
            bcoef[idim].define(fba, dm, 1, 0);
            ccoef[idim].define(fba, dm, 1, 0);
        }

        // phi with its exact values, and the boundary values (at the domain
        // faces) in the ghost cells outside the domain
        MultiFab phi_exact(ba, dm, 1, 1);
        MultiFab bcdata(ba, dm, 1, 1);
        for (MFIter mfi(phi_exact); mfi.isValid(); ++mfi) {
            Box const& gbx = mfi.fabbox();
            auto const& pe = phi_exact.array(mfi);
            auto const& bd = bcdata.array(mfi);
            auto const& a = acoef.array(mfi);
            ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                IntVect const iv(AMREX_D_DECL(i,j,k));
                RealVect x;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    // clamp the ghost cells onto the domain faces
                    Real const xc = problo[d] + (Real(iv[d])+Real(0.5))*dx[d];
                    x[d] = amrex::Clamp(xc, problo[d], probhi[d]);
                }
                pe(i,j,k) = exact_phi(x);
                bd(i,j,k) = pe(i,j,k);
                if (a.contains(i,j,k)) { a(i,j,k) = Real(1.0) + x[0]; }
            });
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                auto const& b = bcoef[idim].array(mfi);
                auto const& c = ccoef[idim].array(mfi);
                ParallelFor(amrex::surroundingNodes(mfi.validbox(), idim),
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    IntVect const iv(AMREX_D_DECL(i,j,k));
                    RealVect x;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        x[d] = problo[d] + (Real(iv[d]) + (d == idim ? Real(0.) : Real(0.5)))*dx[d];
                    }
                    b(i,j,k) = Real(1.0) + Real(0.5)*x[idim]*x[idim];
                    c(i,j,k) = velocity(idim, x);
                });
            }
        }

        // The ghost cells of phi as the boundary conditions make them
        MultiFab phi(ba, dm, 1, 1);
        MultiFab::Copy(phi, phi_exact, 0, 0, 1, 0);
        phi.FillBoundary(geom.periodicity());
        for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
            Box const& vbx = mfi.validbox();
            auto const& p = phi.array(mfi);
            auto const& bd = bcdata.const_array(mfi);
            for (OrientationIter oit; oit; ++oit) {
                Orientation const ori = oit();
                int const idim = ori.coordDir();
                if (vbx.smallEnd(idim) != domain.smallEnd(idim) && ori.isLow()) { continue; }
                if (vbx.bigEnd(idim) != domain.bigEnd(idim) && ori.isHigh()) { continue; }
                bool const dirichlet = (ori.isLow() ? lobc[idim] : hibc[idim]) == LinOpBCType::Dirichlet;
                IntVect const in = ori.isLow() ? IntVect::TheDimensionVector(idim)
                                               : -IntVect::TheDimensionVector(idim);
                ParallelFor(amrex::adjCell(vbx, ori), [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    IntVect const g(AMREX_D_DECL(i,j,k));
                    p(g) = dirichlet ? dirichlet_ghost(bd(g), p(g+in), p(g+2*in)) : p(g+in);
                });
            }
        }

        // rhs = a_scalar a phi - b_scalar div(b grad phi) - c_scalar div(c phi_upwind)
        MultiFab rhs(ba, dm, 1, 0);
        for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
            auto const& r = rhs.array(mfi);
            auto const& p = phi.const_array(mfi);
            auto const& a = acoef.const_array(mfi);
            GpuArray<Array4<Real const>,AMREX_SPACEDIM> b, c;
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                b[idim] = bcoef[idim].const_array(mfi);
                c[idim] = ccoef[idim].const_array(mfi);
            }
            ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                IntVect const iv(AMREX_D_DECL(i,j,k));
                Real v = ascalar * a(iv) * p(iv);
                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                    IntVect const e = IntVect::TheDimensionVector(idim);
                    Real const fm = b[idim](iv)   * (p(iv)   - p(iv-e)) * dxinv[idim];
                    Real const fp = b[idim](iv+e) * (p(iv+e) - p(iv))   * dxinv[idim];
                    Real const um = c[idim](iv);
                    Real const up = c[idim](iv+e);
                    Real const dm_ = um * (um > Real(0.) ? p(iv-e) : p(iv));
                    Real const dp_ = up * (up > Real(0.) ? p(iv) : p(iv+e));
                    v -= bscalar * (fp - fm) * dxinv[idim] + cscalar * (dp_ - dm_) * dxinv[idim];
                }
                r(iv) = v;
            });
        }

        HypreMLABecLap hypre({geom}, {ba}, {dm}, HypreSolverID::BoomerAMG);
        hypre.setVerbose(verbose);
        if (cscalar != Real(0.0)) {
            hypre.setDriftCoeffs(cscalar, {GetArrOfConstPtrs(ccoef)});
        }
        hypre.setup(ascalar, bscalar, {&acoef}, {GetArrOfConstPtrs(bcoef)},
                    lobc, hibc, {&bcdata});

        MultiFab sol(ba, dm, 1, 1);
        sol.setVal(Real(0.0));
        hypre.solve({&sol}, {&rhs}, tol, Real(0.0));

        MultiFab::Subtract(sol, phi_exact, 0, 0, 1, 0);
        Real const err = sol.norminf(0, 0) / phi_exact.norminf(0, 0);
        amrex::Print() << "HypreMLABecLap drift: relative max-norm difference "
                       << err << " after " << hypre.getNumIters() << " iterations\n";
        if (!(err <= max_error)) {
            amrex::Abort("HypreMLABecLapDrift: difference exceeds max_error");
        }
    }
    amrex::Finalize();
}
