#!/usr/bin/env python3
"""UDCT v7.06 Hessian-term sign audit — reproducibility script.

Companion to:
  "A sign audit of a Hessian term absent from the standard MOND equation:
   action-consistent virial work and conditional Jeans tests in three
   ultra-faint dwarfs" (29 September 2026).

This is a stand-alone extraction of the v7.05–v7.06 local-Taylor calculation.
It is not a new preregistered test, not an O8d/D8b rerun, and not an
AQUAL/QUMOND evaluation.  Needs numpy and scipy only.  Run --help for modes.
All accelerations/Hessians are SI; stellar radii are SI internally.

Sky coordinates and luminosities: Pace Local Volume Database (2025) and the
archived UDCT v5.61 object table.  The catalog Galactocentric distance is
descriptive only.  The Galaxy is evaluated at the position computed from
heliocentric distance and Galactic (l, b) with R0 = 8.2 kpc (McMillan 2017).
No dark halo and no LMC potential are included.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.optimize import brentq

G = 6.67430e-11
MSUN = 1.98847e30
PC = 3.085677581e16
KPC = 1000 * PC
A0 = 1.082401e-10
C = 299792458.0
XS0 = 4.0e9
ML = 2.0
POINT_BULGE_MSUN = 8.62e9

# Four visible McMillan17.ini disks: central surface density [Msun/kpc^2],
# radial scale, central hole, signed vertical scale [kpc]. A negative vertical
# scale denotes the sech^2 distribution used for the gas components.
DISKS = ((8.95679e8, 2.49955, 0.0, 0.300),
         (1.83444e8, 3.02134, 0.0, 0.900),
         (5.31319e7, 7.00000, 4.0, -0.085),
         (2.17995e9, 1.50000, 12.0, -0.045))


@dataclass(frozen=True)
class Dwarf:
    luminosity: float                 # L_sun; Pace LVDB / archived v5.61 table
    re_pc: float                      # projected half-light radius (pc)
    heliocentric_kpc: float           # used with (l, b) to place the object
    galactic_l_deg: float             # Galactic longitude (deg)
    galactic_b_deg: float             # Galactic latitude (deg)
    dgc_catalog_kpc: float            # catalog D_GC; descriptive, not integrated
    observed_kms: float               # unmatched-aperture historical reference


DWARFS = {
    "Leo IV": Dwarf(8472, 101.11, 151.36, 265.4576584, 56.5059706, 151.93, 3.3),
    "Pegasus III": Dwarf(3733, 82.12, 214.78, 69.8598973, -41.8262343, 212.84, 5.4),
    "Hydrus I": Dwarf(6546, 52.42, 27.54, 297.4162738, -36.7462917, 25.68, 2.7),
}
PROFILES = ("Plummer", "Tapered cusp")


@dataclass(frozen=True)
class Grid:
    radial: int = 360
    polar: int = 24
    azimuth: int = 36
    action: int = 160
    disk_radial: int = 120
    disk_azimuth: int = 64
    disk_height: int = 12


def nu_cut(x):
    return 1.0 + np.expm1(0.5 * np.log1p(1.0 / x)) / (1.0 + (x / 490.0) ** 2)


def galactocentric_position(obj: Dwarf):
    """Cartesian Galactocentric position (kpc), R0 = 8.2 kpc.

    Uses heliocentric distance and Galactic (l, b).  The catalog D_GC stored
    on the Dwarf object is not used here.
    """
    ll, bb = np.deg2rad((obj.galactic_l_deg, obj.galactic_b_deg))
    d = obj.heliocentric_kpc
    return np.array((8.2 - d * np.cos(bb) * np.cos(ll),
                     d * np.cos(bb) * np.sin(ll), d * np.sin(bb)))


def disk_mass_nodes(grid: Grid):
    qr, qw = leggauss(grid.disk_radial)
    radii, radial_w = 55.0 * (qr + 1.0), 55.0 * qw
    az = 2 * np.pi * np.arange(grid.disk_azimuth) / grid.disk_azimuth
    xx = np.outer(radii, np.cos(az)).ravel()
    yy = np.outer(radii, np.sin(az)).ravel()
    for sigma0, rd, rm, height in DISKS:
        sigma = sigma0 * np.exp(-radii / rd - rm / radii)
        base = np.repeat(sigma * radii * radial_w * 2 * np.pi / grid.disk_azimuth,
                         grid.disk_azimuth)
        if height > 0:
            zz, wz = laggauss(grid.disk_height)
            zz *= height               # two-sided exponential, Laguerre weights
        else:
            zq, wq = leggauss(grid.disk_height)
            t = (zq + 1) / 2
            zz = 2 * abs(height) * np.arctanh(t)
            wz = wq / 2               # two-sided sech^2, dt substitution
        for sign in (-1, 1):
            for z, w in zip(sign * zz, wz):
                yield np.column_stack((xx, yy, np.full_like(xx, z))), base * w / 2


def galactic_field(obj: Dwarf, grid: Grid):
    """Return grad(Phi_N) and Hessian from the SAME visible mass model."""
    loc = galactocentric_position(obj)
    e = np.zeros(3)
    h = np.zeros((3, 3))
    integrated_mass = POINT_BULGE_MSUN
    for coords, masses in disk_mass_nodes(grid):
        integrated_mass += np.sum(masses)
        for k in range(0, len(masses), 18000):
            delta = loc[None, :] - coords[k:k + 18000]
            mm = masses[k:k + 18000]
            d2 = np.einsum("ij,ij->i", delta, delta)
            inv3 = d2 ** -1.5
            e += np.einsum("i,ij->j", mm * inv3, delta)
            h += np.eye(3) * np.sum(mm * inv3) - 3 * np.einsum(
                "i,ij,ik->jk", mm * inv3 / d2, delta, delta)
    d2 = loc @ loc
    e += POINT_BULGE_MSUN * loc / d2 ** 1.5
    h += POINT_BULGE_MSUN * (np.eye(3) / d2 ** 1.5 -
                              3 * np.outer(loc, loc) / d2 ** 2.5)
    return e * G * MSUN / KPC ** 2, h * G * MSUN / KPC ** 3, integrated_mass


@lru_cache(maxsize=1)
def cusp_calibration():
    q = np.geomspace(1e-6, 200.0, 10000)  # r/Re

    def compute(a):
        density = (a / q) / (1 + q / a) ** 3 * np.exp(-(q / 10) ** 4)
        shell = q * q * density
        enclosed = np.r_[0., cumulative_trapezoid(shell, q)]
        cylinder = np.where(q <= 1, 1,
                            1 - np.sqrt(np.maximum(1 - 1 / q ** 2, 0)))
        projected_fraction = simpson(shell * cylinder, x=q) / enclosed[-1]
        return projected_fraction, enclosed

    a = brentq(lambda aa: compute(aa)[0] - 0.5, 0.05, 3, xtol=1e-10)
    _, enclosed = compute(a)
    rh_re = np.interp(0.5 * enclosed[-1], enclosed, q)
    return q, a, rh_re, enclosed / enclosed[-1], enclosed[-1]


def radial_mesh(profile: str, grid: Grid):
    outer = 80.0 if profile == "Plummer" else 40.0
    return np.sort(np.unique(np.r_[np.geomspace(0.002, outer, grid.radial),
                                   1.0, 6.0]))


def model_state(obj: Dwarf, profile: str, ratios, e, h, grid: Grid,
                p: float = 4.0, xs: float = XS0):
    """Angle-averaged leading force, P_nn and tr(P); radial coordinate in SI."""
    mass = ML * obj.luminosity * MSUN
    re = obj.re_pc * PC
    if profile == "Plummer":
        b = re
        rh = b / np.sqrt(2 ** (2 / 3) - 1)
    elif profile == "Tapered cusp":
        _, a_re, rh_re, _, _ = cusp_calibration()
        b, rh = a_re * re, rh_re * re
    else:
        raise ValueError(profile)
    r = np.asarray(ratios) * rh
    polar, wpolar = leggauss(grid.polar)
    az = 2 * np.pi * np.arange(grid.azimuth) / grid.azimuth
    n = np.stack((np.repeat(np.sqrt(1 - polar ** 2), grid.azimuth) *
                  np.tile(np.cos(az), grid.polar),
                  np.repeat(np.sqrt(1 - polar ** 2), grid.azimuth) *
                  np.tile(np.sin(az), grid.polar),
                  np.repeat(polar, grid.azimuth)), axis=1)
    weights = np.repeat(wpolar, grid.azimuth) / (2 * grid.azimuth)
    ne, nhn = n @ e, np.einsum("ki,ij,kj->k", n, h, n)
    hn = n @ h.T
    ehn, hn2 = hn @ e, np.einsum("ki,ki->k", hn, hn)

    if profile == "Plummer":
        gn = G * mass * r / (r * r + b * b) ** 1.5
        lr = G * mass * (b * b - 2 * r * r) / (r * r + b * b) ** 2.5
        lt = G * mass / (r * r + b * b) ** 1.5
        rho = (1 + (r / b) ** 2) ** -2.5
        dlogrho = -5 * r / (r * r + b * b)
    else:
        q, a_re, _, cum, total = cusp_calibration()
        rq = r / re
        rho_q = (a_re / rq) / (1 + rq / a_re) ** 3 * np.exp(-(rq / 10) ** 4)
        rho_mass = mass * rho_q / (4 * np.pi * re ** 3 * total)
        contained = np.interp(rq, q, cum, left=0, right=1)
        gn = G * mass * contained / r ** 2
        lt = gn / r
        lr = 4 * np.pi * G * rho_mass - 2 * lt
        rho = (b / r) / (1 + r / b) ** 3 * np.exp(-(r / (10 * re)) ** 4)
        dlogrho = -1 / r - 3 / (r + b) - 4 * r ** 3 / (10 * re) ** 4

    t2 = (lr[:, None] ** 2 + 2 * lt[:, None] ** 2 +
          2 * lt[:, None] * np.trace(h) +
          2 * (lr - lt)[:, None] * nhn + np.trace(h @ h))
    tau = np.sqrt(2 / 3 * np.maximum(t2, 0))
    rr = r[:, None]
    x2 = (gn[:, None] ** 2 + e @ e + 2 * gn[:, None] * ne +
          2 * rr * gn[:, None] * nhn + 2 * rr * ehn + rr * rr * hn2)
    x = np.sqrt(np.maximum(x2, 1e-200)) / A0
    x_tidal = C * C * tau / (A0 * A0 * x * x)
    switch = 1 / (1 + (x_tidal / xs) ** p)
    nu = 1 + switch * (nu_cut(x) - 1)
    lead = (nu * (gn[:, None] + ne + rr * nhn)) @ weights

    # Derivative of the SAME action Q that defines nu. Transform the integral
    # t in [0,1] using u=t/(t+epsilon), epsilon=X/Xs. This resolves the sharp
    # transition at small t. Never independently rescale P relative to nu.
    uq, uw = leggauss(grid.action)
    uq, uw = (uq + 1) / 2, uw / 2
    w = np.zeros(x.shape)
    for start in range(0, x.size, 1500):
        xx = x.ravel()[start:start + 1500]
        X = x_tidal.ravel()[start:start + 1500]
        eps = X / xs
        upper = 1 / (1 + eps)
        u = upper[:, None] * uq[None, :]
        t = eps[:, None] * u / (1 - u)
        ss = 1 / (1 + ((1 - u) / u) ** p)
        resp = nu_cut(xx[:, None] * np.sqrt(t)) - 1
        integral = np.sum(uw[None, :] * upper[:, None] * eps[:, None] /
                          (1 - u) ** 2 * ss * (1 - ss) * resp, axis=1)
        w.ravel()[start:start + len(xx)] = -p * C * C / X * integral
    pnn = (w * 2 / (3 * tau) * (lr[:, None] + nhn)) @ weights
    trp = (w * 2 / (3 * tau) * (lr + 2 * lt + np.trace(h))[:, None]) @ weights
    return dict(r=r, rh=rh, b=b, gn=gn, rho=rho, dlogrho=dlogrho,
                lead=lead, pnn=pnn, trp=trp)


def work_from_state(st):
    """Normalized work via radial integration by parts; keep boundary term."""
    r, rho, gn = st["r"], st["rho"], st["gn"]
    pnn, trp = st["pnn"], st["trp"]
    base = simpson(rho * r ** 3 * gn, x=r)
    leading = simpson(rho * r ** 3 * st["lead"], x=r) / base
    derivative = 0.5 * simpson(st["dlogrho"] * rho * r ** 3 * pnn +
                               rho * r ** 2 * trp, x=r) / base
    boundary = -0.5 * (rho[-1] * r[-1] ** 3 * pnn[-1] -
                       rho[0] * r[0] ** 3 * pnn[0]) / base
    direct_force = st["lead"] - 0.5 * (np.gradient(pnn, r, edge_order=2) +
                                        (3 * pnn - trp) / r)
    direct = simpson(rho * r ** 3 * direct_force, x=r) / base
    return dict(virial=float(leading + derivative + boundary),
                leading=float(leading), derivative=float(derivative),
                boundary=float(boundary), direct=float(direct)), direct_force


def aperture_jeans(st, inward_force, re_pc):
    """Conditional spherical, isotropic Jeans solve with outer pressure zero."""
    r, rho = st["r"], st["rho"]
    pressure = -np.r_[0., cumulative_trapezoid((rho * inward_force)[::-1],
                                               r[::-1])][::-1]
    # The illustrative taper underflows to exactly zero far outside the body.
    # Its pressure is zero there too; that 0/0 tail is not a Jeans datum.
    sigma2 = np.divide(pressure, rho, out=np.zeros_like(pressure),
                       where=rho > 1e-280)  # (m/s)^2
    re = re_pc * PC
    # Fraction of a spherical shell in the line-of-sight cylinder R < Re.
    cylinder = np.where(r <= re, 1.0,
                        1 - np.sqrt(np.maximum(1 - (re / r) ** 2, 0)))
    ap2 = (simpson(pressure * cylinder * r * r, x=r) /
           simpson(rho * cylinder * r * r, x=r))
    return dict(min_sigma2_kms2=float(np.min(sigma2) / 1e6),
                sigma2_at_6rh_kms2=float(np.interp(6.0, r / st["rh"], sigma2) / 1e6),
                formal_aperture_rms_kms=float(np.sqrt(max(ap2, 0)) / 1000),
                has_negative_pressure=bool(np.any(pressure < 0))), sigma2


def one(obj: Dwarf, profile: str, e, h, grid: Grid, p=4.0, xs=XS0,
        alpha=1.0, jeans=False):
    st = model_state(obj, profile, radial_mesh(profile, grid),
                     alpha * e, alpha * h, grid, p=p, xs=xs)
    work, force = work_from_state(st)
    result = dict(work, profile=profile, p=p, xs=xs, alpha=alpha,
                  local_at_rh=float(np.interp(1.0, st["r"] / st["rh"], force) /
                                    np.interp(1.0, st["r"] / st["rh"], st["gn"])))
    if jeans:
        jr, sigma2 = aperture_jeans(st, force, obj.re_pc)
        result["jeans"] = jr
        result["jeans"]["min_4_to_9rh_kms2"] = float(
            np.min(sigma2[(st["r"] / st["rh"] >= 4) &
                          (st["r"] / st["rh"] <= 9)]) / 1e6)
        result["jeans"]["force_at_6rh_over_newton"] = float(
            np.interp(6.0, st["r"] / st["rh"], force / st["gn"]))
    return result


def run(args):
    grid = Grid(radial=args.radial, polar=args.polar, azimuth=args.azimuth,
                action=args.action, disk_radial=args.disk_radial,
                disk_azimuth=args.disk_azimuth, disk_height=args.disk_height)
    if min(grid.radial, grid.polar, grid.azimuth, grid.action,
           grid.disk_radial, grid.disk_azimuth, grid.disk_height) < 4:
        raise ValueError("Each quadrature dimension must be at least four")
    if args.object == "all":
        names = tuple(DWARFS)
    else:
        names = (args.object,)
    outputs = []
    for name in names:
        obj = DWARFS[name]
        e, h, mass = galactic_field(obj, grid)
        for profile in PROFILES:
            record = dict(object=name, profile=profile,
                          galaxy_integrated_msun=float(mass),
                          e_m_per_s2=e.tolist(), h_per_s2=h.tolist())
            if args.mode in ("baseline", "all"):
                record["baseline"] = one(obj, profile, e, h, grid)
            if args.mode in ("switch", "all"):
                record["switch"] = {str(p): one(obj, profile, e, h, grid, p=p)
                                     for p in (1, 2, 3, 4, 6, 8)}
            if args.mode in ("high-scale", "all"):
                record["high_scale"] = {str(xs): one(obj, profile, e, h, grid,
                                                      xs=xs, jeans=True)
                                        for xs in (4e13, 4e17)}
            if args.mode in ("alpha", "all"):
                # First root in the small-field interval, as printed in v7.06.
                f = lambda alpha: one(obj, profile, e, h, grid,
                                      alpha=alpha)["virial"]
                record["alpha_crossing"] = brentq(f, 0, 0.2, xtol=1e-7)
            if args.mode == "xs-crossing":
                f = lambda logxs: one(obj, profile, e, h, grid,
                                      xs=10 ** logxs)["virial"]
                record["high_xs_crossing"] = 10 ** brentq(
                    f, np.log10(XS0), 13.1, xtol=1e-6)
            outputs.append(record)
    return dict(status="post-outcome exploratory audit; historical O8d remains 0/22",
                grid=grid.__dict__, outputs=outputs)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "switch", "high-scale",
                                            "alpha", "xs-crossing", "all"),
                        default="baseline")
    parser.add_argument("--object", choices=("all", *DWARFS), default="all")
    parser.add_argument("--radial", type=int, default=None)
    parser.add_argument("--polar", type=int, default=None)
    parser.add_argument("--azimuth", type=int, default=None)
    parser.add_argument("--action", type=int, default=None)
    parser.add_argument("--disk-radial", type=int, default=None)
    parser.add_argument("--disk-azimuth", type=int, default=None)
    parser.add_argument("--disk-height", type=int, default=None)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Use a faster grid (180 x 16 x 24 x 96 action; disk 40 x 24 x 8). "
             "Baseline V/V_N still matches the printed table for these objects.")
    args = parser.parse_args()
    smoke = dict(radial=180, polar=16, azimuth=24, action=96,
                 disk_radial=40, disk_azimuth=24, disk_height=8)
    paper = dict(radial=Grid.radial, polar=Grid.polar, azimuth=Grid.azimuth,
                 action=Grid.action, disk_radial=Grid.disk_radial,
                 disk_azimuth=Grid.disk_azimuth, disk_height=Grid.disk_height)
    base = smoke if args.smoke else paper
    for key in paper:
        val = getattr(args, key)
        setattr(args, key, base[key] if val is None else val)
    print(json.dumps(run(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    cli()
