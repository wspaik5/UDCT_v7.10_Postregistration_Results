#!/usr/bin/env python3
"""UDCT v7.10 -- implementation of the pre-registered GQUMOND sensitivity test.

Companion to:
  "UDCT v7.10 -- Pre-registration: Sensitivity of the v7.09 GQUMOND example-A
   virial and Jeans results to the screening function f, the QUMOND function Q,
   and the inner stellar profile, with a focus on the Hydrus I central region"
   (4 October 2026).

What this script is
-------------------
The v7.09 physical implementation with the following three changes:

  (i)   the constitutive functions: f(u) = u^n/(1+u^n) with n in {1/2, 1, 2} and
        Q'(Y^2) = nu(Y) with nu in {nu0, nu1 (simple), nu2 (standard)};
  (ii)  its own copy of the tapered-cusp calibration and geometry, which is the
        v7.06 cusp when eps = 0 and the softened cusp when eps > 0;
  (iii) a parameter for the inner radial mesh limit (0.002 r_h in v7.06/v7.09).

Everything else -- objects, Plummer profile, Galactic Taylor field, quadrature,
integration-by-parts evaluation of V, zero-outer-pressure Jeans integral -- is taken
from the UNMODIFIED v7.06 script, which is loaded from the same folder and refused
unless its full SHA-256 matches the frozen dependency.

Pre-execution implementation amendment: 5 October 2026 (Auckland).
Reporting logic follows the reviewed PDF Sections 4-7: strict negative work,
zero boundary cases, invalid-data rejection, and gate-R claim suppression.
No physical law, input, mesh or registered case is changed by this amendment.

Status of this file: the registered calculation has NOT been run by the author.
--selftest evaluates analytic identities, geometry regressions and the decision
logic on SYNTHETIC records; it evaluates no registered quantity (no V/V_N, no Jeans
pressure, no effective mass, no dispersion).

Needs numpy and scipy.  Run --help for options.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
from datetime import datetime, timezone
import scipy
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import cumulative_trapezoid, quad, simpson
from scipy.optimize import brentq

# --------------------------------------------------------------------------
# Load the frozen v7.06 script (hash-checked)
# --------------------------------------------------------------------------
V706_NAME = "UDCT_v7_06_Hessian_Term_Sign_Audit_reproduce.py"
V706_EXPECTED_SHA256 = "a64440e1350ae8bf10eb3188ec34829b6d586abf275de5cdc841a50ad3478284"
IMPLEMENTATION_REVISION = "2026-10-05-reviewed-rules"


def _load_v706():
    default = Path(__file__).resolve().with_name(V706_NAME)
    path = Path(os.environ.get("UDCT_V706_SCRIPT", default))
    if not path.is_file():
        raise FileNotFoundError(
            f"Frozen v7.06 script not found: {path}\n"
            f"Place the unmodified '{V706_NAME}' next to this file "
            f"(or set UDCT_V706_SCRIPT).")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != V706_EXPECTED_SHA256:
        raise RuntimeError(
            f"SHA-256 of {path.name} is {digest}; the registration requires "
            f"{V706_EXPECTED_SHA256}. Refusing to run on a modified "
            f"v7.06 script.")
    spec = importlib.util.spec_from_file_location("udct_v706_frozen", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, digest


V706, V706_SHA256 = _load_v706()
G, MSUN, PC, A0, ML = V706.G, V706.MSUN, V706.PC, V706.A0, V706.ML
DWARFS, PROFILES, Grid = V706.DWARFS, V706.PROFILES, V706.Grid

# --------------------------------------------------------------------------
# Registered constants (Sections 2-5 of the registration)
# --------------------------------------------------------------------------
ELL0_GRID_PC = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
ELL0_HYDRUS_PC = (1.0, 3.0, 10.0)             # the v7.09 negative-pressure cases
EPS_GRID_RH = (0.01, 0.03, 0.1, 0.3)          # Part B softening, units of baseline r_h
RMIN_DEFAULT = 0.002                          # inner mesh limit in v7.06/v7.09 (r_h)
RMIN_PART_C = 0.0005                          # Part C
TRUNCATE_RH = 5.0                             # v7.09 Section 3(b) follow-up
OUTWARD_WARNING = 10.0
RANGE_RH = (0.1, 10.0)
HYDRUS = "Hydrus I"
CUSP = "Tapered cusp"

GRID_SPECS = {
    "smoke": dict(radial=180, polar=16, azimuth=24,
                  disk_radial=40, disk_azimuth=24, disk_height=8),
    "default": dict(radial=360, polar=24, azimuth=36,
                    disk_radial=120, disk_azimuth=64, disk_height=12),
    "refined": dict(radial=720, polar=48, azimuth=72,
                    disk_radial=240, disk_azimuth=128, disk_height=24),
}

# Gate R reference values: v7.09 post-registration report (default-grid V/V_N,
# Table 4 for GQUMOND; Table 1 for the QUMOND control; refined min sigma_r^2, Table 3).
REF_GQ = {
    ("Leo IV", "Plummer"): (14.014603, 14.014590, 14.014469, 14.013099, 14.001059, 13.864456),
    ("Leo IV", CUSP): (13.896442, 13.896381, 13.895853, 13.890129, 13.846229, 13.480055),
    ("Pegasus III", "Plummer"): (19.373649, 19.373620, 19.373365, 19.370464, 19.344974, 19.057039),
    ("Pegasus III", CUSP): (19.217905, 19.217789, 19.216783, 19.205843, 19.122094, 18.461326),
    ("Hydrus I", "Plummer"): (3.994096, 3.994088, 3.994013, 3.993156, 3.985628, 3.900006),
    ("Hydrus I", CUSP): (3.556879, 3.556836, 3.556455, 3.552177, 3.516186, 3.153367),
}
REF_CONTROL = {
    ("Leo IV", "Plummer"): 14.014605, ("Leo IV", CUSP): 13.896448,
    ("Pegasus III", "Plummer"): 19.373652, ("Pegasus III", CUSP): 19.217916,
    ("Hydrus I", "Plummer"): 3.994097, ("Hydrus I", CUSP): 3.556883,
}
REF_MIN_SIGMA2 = {1.0: -0.113062, 3.0: -0.435382, 10.0: -1.571442}
GATE_R_TOL_VIRIAL = 1e-5
GATE_R_TOL_SIGMA2 = 1e-3


# --------------------------------------------------------------------------
# Constitutive functions
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Law:
    name: str
    n: float          # f(u) = u^n / (1 + u^n)
    nu: str           # "nu0" | "nu1" | "nu2"


BASE_LAW = Law("BASE", 1.0, "nu0")
PART_A_LAWS = (Law("A-f1/2", 0.5, "nu0"), Law("A-f2", 2.0, "nu0"),
               Law("A-Qs", 1.0, "nu1"), Law("A-Qd", 1.0, "nu2"))


def nu_val(kind, y):
    """nu(Y) with Q'(Y^2) = nu(Y)."""
    if kind == "nu0":                       # campaign law (1 + 1/Y)^(1/2)
        return np.sqrt(1.0 + 1.0 / y)
    if kind == "nu1":                       # "simple": 1/2 + (1/4 + 1/Y)^(1/2)
        return 0.5 + np.sqrt(0.25 + 1.0 / y)
    if kind == "nu2":                       # "standard"
        return np.sqrt(0.5 * (1.0 + np.sqrt(1.0 + 4.0 / (y * y))))
    raise ValueError(kind)


def d_val(kind, yy):
    """D(Y) = Q - W Q' with W = Y^2, Q(W) = int_0^W nu(sqrt s) ds
             = int_0^Y 2 y nu(y) dy - Y^2 nu(Y)   (closed forms, checked in --selftest)."""
    if kind == "nu0":
        return 0.5 * (np.sqrt(yy * (yy + 1.0)) - np.arcsinh(np.sqrt(yy)))
    if kind == "nu1":
        rr = np.sqrt(yy * yy + 4.0 * yy)
        return rr - 4.0 * np.arcsinh(0.5 * np.sqrt(yy))
    if kind == "nu2":
        t = 0.5 * (yy * yy + yy * np.sqrt(yy * yy + 4.0))   # x^2, with Y = x^2/sqrt(1+x^2)
        x = np.sqrt(t)
        return np.arcsinh(x) - x / np.sqrt(1.0 + x * x)
    raise ValueError(kind)


def q_closed(kind, w):
    """Q(W) = int_0^W nu(sqrt s) ds in closed form (used by --selftest only; checked
    there against numerical quadrature)."""
    yy = np.sqrt(w)
    if kind == "nu0":
        return 0.5 * (2 * yy + 1) * np.sqrt(yy * (yy + 1)) - 0.5 * np.arcsinh(np.sqrt(yy))
    if kind == "nu1":
        rr = np.sqrt(yy * yy + 4.0 * yy)
        return 0.5 * yy * yy + 0.5 * (yy + 2.0) * rr - 4.0 * np.arcsinh(0.5 * np.sqrt(yy))
    if kind == "nu2":
        x = np.sqrt(0.5 * (yy * yy + yy * np.sqrt(yy * yy + 4.0)))
        return x * np.sqrt(1 + x * x) + np.arcsinh(x) - 2 * x / np.sqrt(1 + x * x)
    raise ValueError(kind)


def q_prime(kind, w):
    return nu_val(kind, np.sqrt(w))


def gq_kernel(g2, s, ell0_m, law: Law):
    """Constitutive law at points with |grad psi|^2 = g2 and psi,ij psi,ij = s.

    P = f(u) Q(Z/f), f = u^n/(1+u^n); returns (nu_eff, wh) with P_ij = wh * psi,ij:
      nu_eff = Q'(W) + D(W) u f'(u) / Z,           W = Z/f,  D = Q - W Q'
      wh     = -2 a0^2 D(W) u f'(u) / s,           u f'(u) = n u^n / (1+u^n)^2
    ell0_m = None is the QUMOND control (f == 1): nu_eff = Q'(Z), wh = 0.
    """
    z = g2 / A0 ** 2
    if ell0_m is None:
        return q_prime(law.nu, z), np.zeros_like(z)
    u = g2 / (ell0_m ** 2 * s)
    un = np.maximum(u ** law.n, 1e-300)
    w = z * (1.0 + 1.0 / un)                       # W = Z / f(u)
    uf = law.n * un / (1.0 + un) ** 2              # u f'(u)
    common = d_val(law.nu, np.sqrt(w)) * uf
    nu = q_prime(law.nu, w) + common / z
    wh = -2.0 * A0 ** 2 * common / s
    return nu, wh


# --------------------------------------------------------------------------
# Tapered cusp with optional central softening (registration Section 2, Part B)
#   rho(q) ~ (a / sqrt(q^2 + eps_q^2)) (1 + q/a)^-3 exp[-(q/10)^4],  q = r / R_e
# eps_q = eps_rh * r_h(baseline cusp)/R_e.   eps_q = 0 is the v7.06 cusp.
# The calibration copies the v7.06 numerics (grid, trapezoid/Simpson, brentq).
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def cusp_calibration_soft(eps_q: float):
    q = np.geomspace(1e-6, 200.0, 10000)

    def compute(a):
        density = (a / np.sqrt(q * q + eps_q * eps_q)) / (1 + q / a) ** 3 * np.exp(-(q / 10) ** 4)
        shell = q * q * density
        enclosed = np.r_[0., cumulative_trapezoid(shell, q)]
        cylinder = np.where(q <= 1, 1, 1 - np.sqrt(np.maximum(1 - 1 / q ** 2, 0)))
        projected_fraction = simpson(shell * cylinder, x=q) / enclosed[-1]
        return projected_fraction, enclosed

    a = brentq(lambda aa: compute(aa)[0] - 0.5, 0.05, 3, xtol=1e-10)
    _, enclosed = compute(a)
    rh_re = np.interp(0.5 * enclosed[-1], enclosed, q)
    return q, a, rh_re, enclosed / enclosed[-1], enclosed[-1]


def baseline_rh_re() -> float:
    return float(V706.cusp_calibration()[2])


def eps_q_from_rh(eps_rh: float) -> float:
    return eps_rh * baseline_rh_re()


def geometry(obj, profile, ratios, e, h, grid, eps_rh=0.0):
    """Profile / Galaxy geometry.  `ratios` are in units of the BASELINE (eps = 0)
    three-dimensional half-mass radius r_h of the profile (also for eps > 0)."""
    mass = ML * obj.luminosity * MSUN
    re = obj.re_pc * PC
    if profile == "Plummer":
        if eps_rh != 0.0:
            raise ValueError("softening is defined for the tapered cusp only")
        b = re
        rh = b / np.sqrt(2 ** (2 / 3) - 1)
    elif profile == CUSP:
        _, a0_re, rh_re, _, _ = V706.cusp_calibration()
        rh = rh_re * re
        b = a0_re * re
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
        if eps_rh == 0.0:
            q, a_re, _, cum, total = V706.cusp_calibration()
            eps_q = 0.0
        else:
            eps_q = eps_q_from_rh(eps_rh)
            q, a_re, _, cum, total = cusp_calibration_soft(eps_q)
        b = a_re * re
        rq = r / re
        soft = np.sqrt(rq * rq + eps_q * eps_q)
        rho_q = (a_re / soft) / (1 + rq / a_re) ** 3 * np.exp(-(rq / 10) ** 4)
        rho_mass = mass * rho_q / (4 * np.pi * re ** 3 * total)
        contained = np.interp(rq, q, cum, left=0, right=1)
        gn = G * mass * contained / r ** 2
        lt = gn / r
        lr = 4 * np.pi * G * rho_mass - 2 * lt
        rsoft = np.sqrt(r * r + (eps_q * re) ** 2)
        rho = (b / rsoft) / (1 + r / b) ** 3 * np.exp(-(r / (10 * re)) ** 4)
        dlogrho = -r / (r * r + (eps_q * re) ** 2) - 3 / (r + b) - 4 * r ** 3 / (10 * re) ** 4

    t2 = (lr[:, None] ** 2 + 2 * lt[:, None] ** 2 +
          2 * lt[:, None] * np.trace(h) +
          2 * (lr - lt)[:, None] * nhn + np.trace(h @ h))
    rr = r[:, None]
    x2 = (gn[:, None] ** 2 + e @ e + 2 * gn[:, None] * ne +
          2 * rr * gn[:, None] * nhn + 2 * rr * ehn + rr * rr * hn2)
    radial_grad = gn[:, None] + ne + rr * nhn
    return dict(r=r, rh=rh, b=b, gn=gn, lr=lr, lt=lt, rho=rho, dlogrho=dlogrho,
                n=n, weights=weights, t2=t2, g2=x2, radial_grad=radial_grad,
                trh=np.trace(h), nhn=nhn)


def model_state(obj, profile, ratios, e, h, grid, law, ell0_pc, eps_rh=0.0):
    geo = geometry(obj, profile, ratios, e, h, grid, eps_rh)
    g2 = np.maximum(geo["g2"], 1e-200)
    s = np.maximum(geo["t2"], 1e-200)
    ell0_m = None if ell0_pc is None else ell0_pc * PC
    nu, wh = gq_kernel(g2, s, ell0_m, law)
    wts = geo["weights"]
    lr, lt = geo["lr"], geo["lt"]
    lead = (nu * geo["radial_grad"]) @ wts
    pnn = (wh * (lr[:, None] + geo["nhn"])) @ wts
    trp = (wh * (lr + 2 * lt + geo["trh"])[:, None]) @ wts
    return dict(r=geo["r"], rh=geo["rh"], b=geo["b"], gn=geo["gn"],
                rho=geo["rho"], dlogrho=geo["dlogrho"],
                lead=lead, pnn=pnn, trp=trp)


def mesh(profile, grid, truncate_rh=None, rmin=RMIN_DEFAULT):
    """Radial mesh in units of baseline r_h.  rmin = 0.002 and truncate_rh = None
    reproduces V706.radial_mesh; truncate_rh reproduces the v7.09 follow-up mesh."""
    if truncate_rh is None:
        outer = 80.0 if profile == "Plummer" else 40.0
        return np.sort(np.unique(np.r_[np.geomspace(rmin, outer, grid.radial), 1.0, 6.0]))
    return np.sort(np.unique(np.r_[np.geomspace(rmin, truncate_rh, grid.radial), 1.0]))


def one(obj, e, h, grid, profile, law, ell0_pc, eps_rh=0.0, rmin=RMIN_DEFAULT,
        truncate_rh=None):
    ratios = mesh(profile, grid, truncate_rh, rmin)
    st = model_state(obj, profile, ratios, e, h, grid, law, ell0_pc, eps_rh)
    work, force = V706.work_from_state(st)
    x = st["r"] / st["rh"]
    ratio = force / st["gn"]
    sel = (x >= RANGE_RH[0]) & (x <= RANGE_RH[1])
    meff = st["r"] ** 2 * force / G / MSUN
    mstar_total = ML * obj.luminosity
    k = int(np.argmin(np.where(sel, meff, np.inf)))
    total_abs = abs(work["leading"]) + abs(work["derivative"]) + abs(work["boundary"])
    jr, sigma2 = V706.aperture_jeans(st, force, obj.re_pc)
    neg = sigma2 < 0
    kmin = int(np.argmin(sigma2))
    out_force = ratio < 0
    jr = dict(jr)
    jr["min_sigma2_at_r_over_rh"] = float(x[kmin])
    jr["min_sigma2_boundary_limited"] = bool(neg.any() and kmin == 0)
    jr["negative_band_r_over_rh"] = ([float(x[neg].min()), float(x[neg].max())]
                                     if neg.any() else None)
    jr["outward_force_band_r_over_rh"] = ([float(x[out_force].min()), float(x[out_force].max())]
                                          if out_force.any() else None)
    return dict(
        profile=profile, law=law.name, ell0_pc=ell0_pc, eps_rh=eps_rh, rmin_rh=rmin,
        truncate_rh=truncate_rh,
        virial=work["virial"], leading=work["leading"],
        derivative=work["derivative"], boundary=work["boundary"], direct=work["direct"],
        derivative_abs_fraction=float(abs(work["derivative"]) / total_abs),
        local_force_over_gstar_at_rh=float(np.interp(1.0, x, ratio)),
        force_negative_in_range=bool(np.any(ratio[sel] < 0)),
        min_force_over_gstar_in_range=float(np.min(ratio[sel])),
        meff_min_msun=float(meff[k]),
        meff_min_over_mstar_total=float(meff[k] / mstar_total),
        meff_min_at_r_over_rh=float(x[k]),
        meff_negative=bool(meff[k] < 0),
        warning_outward_gt10x_newtonian=bool(np.any(ratio[sel] < -OUTWARD_WARNING)),
        jeans=jr)


# --------------------------------------------------------------------------
# Case enumeration (registration Section 2)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Spec:
    part: str                # "BASE" | "A" | "B" | "C"
    key: str
    law: Law
    name: str
    profile: str
    eps_rh: float
    ell0_pc: object          # float, or None for the QUMOND control
    rmin: float


def make_key(law_name, name, profile, eps_rh, ell0_pc, rmin=RMIN_DEFAULT):
    tag = "ctrl" if ell0_pc is None else f"l0={ell0_pc:g}"
    k = f"{law_name}|{name}|{profile}|eps={eps_rh:g}|{tag}"
    if rmin != RMIN_DEFAULT:
        k += f"|rmin={rmin:g}"
    return k


def enumerate_specs(parts, names, ell0_list):
    specs = []

    def add(part, law, name, profile, eps, l0, rmin=RMIN_DEFAULT):
        specs.append(Spec(part, make_key(law.name, name, profile, eps, l0, rmin),
                          law, name, profile, eps, l0, rmin))

    for name in names:
        if "BASE" in parts:                              # v7.09 re-run: gate R + f-variant controls
            for profile in PROFILES:
                add("BASE", BASE_LAW, name, profile, 0.0, None)
                for l0 in ell0_list:
                    add("BASE", BASE_LAW, name, profile, 0.0, l0)
        if "A" in parts:
            for law in PART_A_LAWS:
                for profile in PROFILES:
                    if law.nu != "nu0":                  # Q variants carry their own QUMOND control
                        add("A", law, name, profile, 0.0, None)
                    for l0 in ell0_list:
                        add("A", law, name, profile, 0.0, l0)
        if "B" in parts:
            for eps in EPS_GRID_RH:
                add("B", BASE_LAW, name, CUSP, eps, None)
                for l0 in ell0_list:
                    add("B", BASE_LAW, name, CUSP, eps, l0)
        if "C" in parts and name == HYDRUS:
            for l0 in ELL0_HYDRUS_PC:
                add("C", BASE_LAW, name, CUSP, 0.0, l0, RMIN_PART_C)
    return specs


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------
def make_grid(label):
    return Grid(**GRID_SPECS[label])


def run_grid(label, specs, names):
    grid = make_grid(label)
    out = {}
    for name in names:
        obj = DWARFS[name]
        e, h, mass = V706.galactic_field(obj, grid)
        for sp in specs:
            if sp.name != name:
                continue
            if sp.part == "C" and label != "refined":
                continue                                  # Part C is defined on the refined grid
            case = one(obj, e, h, grid, sp.profile, sp.law, sp.ell0_pc, sp.eps_rh, sp.rmin)
            if case["warning_outward_gt10x_newtonian"]:
                case["truncated_5rh_followup"] = one(
                    obj, e, h, grid, sp.profile, sp.law, sp.ell0_pc, sp.eps_rh,
                    RMIN_DEFAULT, TRUNCATE_RH)
            out[sp.key] = case
    return out


# --------------------------------------------------------------------------
# Registered decision rules (registration Sections 4-5)
# --------------------------------------------------------------------------
def finite_values(res, key, labels, field):
    """Missing or nonfinite primary values are invalid, never an ordinary sign."""
    try:
        vals = []
        for lab in labels:
            value = res[lab][key]
            for name in field:
                value = value[name]
            if isinstance(value, (bool, str)) or not np.isscalar(value):
                return None
            value = float(value)
            if not np.isfinite(value):
                return None
            vals.append(value)
        return vals
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def virial_status(res, key, labels, control=False):
    vals = finite_values(res, key, labels, ("virial",))
    if vals is None:
        return "invalid"
    if all(v > 0 for v in vals):
        return "pos"
    if control and all(v <= 0 for v in vals):
        return "nonpos"
    if all(v < 0 for v in vals):
        return "neg"
    if any(v == 0 for v in vals):
        return "boundary"
    return "disagree"


def pressure_status(res, key):
    vals = finite_values(res, key, ("default", "refined"),
                         ("jeans", "min_sigma2_kms2"))
    if vals is None:
        return "invalid"
    d, r = (v < 0 for v in vals)
    if d and r:
        return "yes"
    if not d and not r:
        return "no"
    return "unresolved"


def classify_global(res, specs):
    labels = ("smoke", "default", "refined")
    gq, controls = {}, {}
    for sp in specs:
        if sp.part in ("A", "B") and sp.ell0_pc is not None:
            gq[sp.key] = (virial_status(res, sp.key, labels), sp)
        elif sp.part in ("BASE", "A", "B") and sp.ell0_pc is None:
            controls[sp.key] = virial_status(res, sp.key, labels, control=True)

    def control_of(sp):
        if sp.part == "B":
            key = make_key("BASE", sp.name, sp.profile, sp.eps_rh, None)
        elif sp.law.nu == "nu0":
            key = make_key("BASE", sp.name, sp.profile, 0.0, None)
        else:
            key = make_key(sp.law.name, sp.name, sp.profile, 0.0, None)
        return controls.get(key, "invalid")

    all_statuses = {**controls, **{k: st for k, (st, _) in gq.items()}}
    invalid = sorted(k for k, st in all_statuses.items() if st == "invalid")
    # Every A/B primary pressure record must also be finite before a claim.
    invalid_pressure = sorted(sp.key for sp in specs if sp.part in ("A", "B")
                             and pressure_status(res, sp.key) == "invalid")
    unresolved_controls = sorted(k for k, st in controls.items()
                                 if st in ("boundary", "disagree"))
    nonpositive_controls = sorted(k for k, st in controls.items() if st == "nonpos")
    negatives = sorted(k for k, (st, sp) in gq.items()
                       if st == "neg" and control_of(sp) == "pos")
    boundaries = sorted(k for k, (st, _) in gq.items() if st == "boundary")
    disagreements = sorted(k for k, st in all_statuses.items() if st == "disagree")
    if invalid or invalid_pressure or not gq or not controls:
        final = "I"
    elif nonpositive_controls:
        final = "G3"
    elif unresolved_controls:
        final = "I"  # These controls can change the priority G3 category.
    elif negatives:
        final = "G2"  # Other unresolved GQ cases cannot remove this detection.
    elif all(st == "pos" for st, _ in gq.values()):
        final = "G1"
    else:
        final = "I"
    return dict(outcome=final, negative_gqumond_cases=negatives,
                nonpositive_control_cases=nonpositive_controls,
                boundary_cases=boundaries, disagreeing_cases=disagreements,
                unresolved_control_cases=unresolved_controls,
                invalid_cases=invalid, invalid_pressure_cases=invalid_pressure,
                gqumond_case_count=len(gq), control_case_count=len(controls))


def classify_local_a(res, specs):
    """J1 / J2 on the Hydrus I baseline tapered cusp (registration Section 4)."""
    per_variant = {}
    for law in PART_A_LAWS:
        keys = [make_key(law.name, HYDRUS, CUSP, 0.0, l0) for l0 in ELL0_GRID_PC]
        per_variant[law.name] = {k: pressure_status(res, k) for k in keys}

    def outcome(unres_as):
        npv = []
        for v in per_variant.values():
            sts = [unres_as if s == "unresolved" else s for s in v.values()]
            npv.append(any(s == "yes" for s in sts))
        return "J1" if all(npv) else "J2"

    a, b = outcome("yes"), outcome("no")
    statuses = [s for v in per_variant.values() for s in v.values()]
    invalid = "invalid" in statuses
    # J2 requires all relevant cases resolved in the reviewed PDF.
    final = "I" if invalid or a != b or (a == "J2" and "unresolved" in statuses) else a
    return dict(outcome=final, optimistic=a, pessimistic=b, statuses=per_variant)


def classify_local_b(res, specs):
    """K1 / K2 / K3 on the Hydrus I softened cusp (registration Section 4)."""
    nsets = {}
    for l0 in ELL0_HYDRUS_PC:
        nsets[l0] = {eps: pressure_status(res, make_key("BASE", HYDRUS, CUSP, eps, l0))
                     for eps in EPS_GRID_RH}

    def outcome(unres_as):
        N = {l0: {eps for eps, s in d.items()
                  if (unres_as if s == "unresolved" else s) == "yes"}
             for l0, d in nsets.items()}
        if any(0.3 in v for v in N.values()):
            return "K2"
        if not any(0.1 in v for v in N.values()):
            return "K1"
        return "K3"

    a, b = outcome("yes"), outcome("no")
    invalid = any(s == "invalid" for v in nsets.values() for s in v.values())
    return dict(outcome=a if a == b and not invalid else "I", optimistic=a, pessimistic=b,
                interpretation="Discrete sampled softening pattern; no exact threshold or causal isolation",
                statuses={str(l0): {str(k): v for k, v in d.items()} for l0, d in nsets.items()})


def new_pressure_cases(res, specs):
    """Reported in full regardless of outcome: negative-pressure cases other than
    the three Hydrus I baseline-cusp cases already known from v7.09."""
    known = {make_key("BASE", HYDRUS, CUSP, 0.0, l0) for l0 in ELL0_HYDRUS_PC}
    out = []
    for sp in specs:
        if sp.part == "C" or sp.ell0_pc is None:
            continue
        if sp.key in known:
            continue
        try:
            st = pressure_status(res, sp.key)
        except KeyError:
            continue
        if st != "no":
            out.append(dict(key=sp.key, status=st))
    return out


def gate_r(res):
    """Reproduction of 42 baseline virials and three refined pressure minima."""
    worst_v, worst_s, missing, invalid = 0.0, 0.0, [], []
    refs = []
    for (name, profile), vals in REF_GQ.items():
        for l0, ref in zip(ELL0_GRID_PC, vals):
            refs.append(("default", make_key("BASE", name, profile, 0.0, l0),
                         ("virial",), ref, "v"))
    for (name, profile), ref in REF_CONTROL.items():
        refs.append(("default", make_key("BASE", name, profile, 0.0, None),
                     ("virial",), ref, "v"))
    for l0, ref in REF_MIN_SIGMA2.items():
        refs.append(("refined", make_key("BASE", HYDRUS, CUSP, 0.0, l0),
                     ("jeans", "min_sigma2_kms2"), ref, "s"))
    for lab, key, field, ref, kind in refs:
        if key not in res.get(lab, {}):
            missing.append(lab + "|" + key)
            continue
        values = finite_values(res, key, (lab,), field)
        if values is None:
            invalid.append(lab + "|" + key)
            continue
        error = abs(values[0] / ref - 1)
        if kind == "v":
            worst_v = max(worst_v, error)
        else:
            worst_s = max(worst_s, error)
    ok = (not missing and not invalid and worst_v <= GATE_R_TOL_VIRIAL
          and worst_s <= GATE_R_TOL_SIGMA2)
    return dict(passed=bool(ok), max_rel_diff_virial=worst_v,
                max_rel_diff_min_sigma2=worst_s, missing=missing, invalid=invalid,
                tol_virial=GATE_R_TOL_VIRIAL, tol_min_sigma2=GATE_R_TOL_SIGMA2)


# --------------------------------------------------------------------------
# --selftest: analytic identities, geometry regression, decision logic on
# SYNTHETIC records.  No registered quantity is evaluated.
# --------------------------------------------------------------------------
def selftest(verbose=True):
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    if verbose:
        print("UDCT v7.10 self-test (analytic identities, geometry, decision logic on "
              "synthetic records; no registered quantity)")
    check("v7.06 script hash matches the registered prefix", True, f"({V706_SHA256[:16]})")

    laws = (BASE_LAW,) + PART_A_LAWS

    # S1: closed forms of D = Q - W Q' against numerical quadrature of Q
    worst = {k: 0.0 for k in ("nu0", "nu1", "nu2")}
    for kind in worst:
        for yy in (1e-3, 1e-2, 0.1, 1.0, 10.0, 1e2, 1e3, 1e4):
            q = quad(lambda y: 2 * y * nu_val(kind, y), 0, yy, epsabs=0, epsrel=1e-13, limit=500)[0]
            ref = q - yy * yy * nu_val(kind, yy)
            worst[kind] = max(worst[kind], abs(d_val(kind, yy) / ref - 1))
    worst_q = 0.0
    for kind in worst:
        for yy in (1e-3, 1e-1, 1.0, 30.0, 1e3):
            q = quad(lambda y: 2 * y * nu_val(kind, y), 0, yy, epsabs=0, epsrel=1e-13, limit=500)[0]
            worst_q = max(worst_q, abs(q_closed(kind, yy * yy) / q - 1))
    check("Q closed forms vs numerical integral (nu0, nu1, nu2)", worst_q < 1e-9,
          f"(max rel. err {worst_q:.1e})")
    check("D = Q - W Q' closed forms vs numerical integral (nu0, nu1, nu2)",
          max(worst.values()) < 1e-6,
          "(max rel. err " + ", ".join(f"{k} {v:.1e}" for k, v in worst.items()) + ")")

    # S2: nu_eff and P_ij coefficient vs finite differences of P(g2, s), every variant
    def pfun(law, g2, s, ell0):
        u = g2 / (ell0 ** 2 * s)
        f = u ** law.n / (1 + u ** law.n)
        return f * q_closed(law.nu, g2 / A0 ** 2 / f)

    rng = np.random.default_rng(710)
    for law in laws:
        worst_nu = worst_wh = 0.0
        for _ in range(25):
            x = 10 ** rng.uniform(-2, 2.5)
            u = 10 ** rng.uniform(-2.5, 3)
            ell0 = 10 ** rng.uniform(15, 18)
            g2 = (x * A0) ** 2
            s = g2 / (ell0 ** 2 * u)
            nu, wh = gq_kernel(np.array(g2), np.array(s), ell0, law)
            dg, ds = 1e-4 * g2, 1e-4 * s
            pg = (pfun(law, g2 + dg, s, ell0) - pfun(law, g2 - dg, s, ell0)) / (2 * dg)
            ps = (pfun(law, g2, s + ds, ell0) - pfun(law, g2, s - ds, ell0)) / (2 * ds)
            worst_nu = max(worst_nu, abs(nu / (A0 ** 2 * pg) - 1))
            worst_wh = max(worst_wh, abs(wh / (2 * A0 ** 2 * ps) - 1))
        check(f"[{law.name}] nu_eff = a0^2 dP/d(grad psi)^2 (finite difference)",
              worst_nu < 1e-5, f"(max rel. err {worst_nu:.2e})")
        check(f"[{law.name}] P_ij coefficient = 2 a0^2 dP/ds (finite difference)",
              worst_wh < 1e-4, f"(max rel. err {worst_wh:.2e})")

    # S3: limits (gate L)
    g2 = np.array((0.7 * A0) ** 2)
    for law in laws:
        nu_c, wh_c = gq_kernel(g2, np.array(1.0), None, law)
        s_big = g2 / (1e-9 ** 2 * 1e12)
        nu_g, wh_g = gq_kernel(g2, s_big, 1e-9, law)
        check(f"[{law.name}] f -> 1: nu_eff -> QUMOND Q' and Hessian factor -> 0",
              abs(nu_g / nu_c - 1) < 1e-4 and abs(wh_g) * s_big / A0 ** 2 < 1e-4)
        check(f"[{law.name}] QUMOND control: Hessian tensor identically zero", wh_c == 0)
    for kind in ("nu0", "nu1", "nu2"):
        yd = np.array([1e-6, 1e-4])
        yu = np.array([1e6, 1e8])
        check(f"[{kind}] deep-MOND limit nu ~ Y^(-1/2), high-Y limit nu -> 1",
              np.allclose(nu_val(kind, yd) * np.sqrt(yd), 1.0, rtol=2e-2) and
              np.allclose(nu_val(kind, yu), 1.0, rtol=1e-5))

    # S4: baseline law reproduces the v7.09 kernel (closed forms, f = u/(1+u), nu0)
    def v709_kernel(g2, s, ell0_m):
        z = g2 / A0 ** 2
        u = g2 / (ell0_m ** 2 * s)
        w = z * (1.0 + 1.0 / u)
        yy = np.sqrt(w)
        d = 0.5 * (np.sqrt(yy * (yy + 1.0)) - np.arcsinh(np.sqrt(yy)))
        common = d * u / (1.0 + u) ** 2
        return np.sqrt(1.0 + 1.0 / np.sqrt(w)) + common / z, -2.0 * A0 ** 2 * common / s

    worst = 0.0
    for _ in range(40):
        x = 10 ** rng.uniform(-2, 3)
        u = 10 ** rng.uniform(-3, 4)
        ell0 = 10 ** rng.uniform(15, 18)
        g2 = np.array((x * A0) ** 2)
        s = np.array(g2 / (ell0 ** 2 * u))
        a, b = gq_kernel(g2, s, ell0, BASE_LAW), v709_kernel(g2, s, ell0)
        worst = max(worst, abs(a[0] / b[0] - 1), abs(a[1] / b[1] - 1))
    check("baseline law reproduces the v7.09 kernel", worst < 1e-12, f"(max rel. diff {worst:.1e})")

    # S5: geometry regression (gate G, eps = 0) and explicit construction (eps >= 0)
    rng2 = np.random.default_rng(7101)
    e = np.array([3e-11, -2e-11, 1e-11])
    hh = 3e-35 * rng2.normal(size=(3, 3))
    h = 0.5 * (hh + hh.T)
    obj = DWARFS["Leo IV"]
    grid = Grid(radial=12, polar=4, azimuth=4)
    worst = 0.0
    for prof, eps in (("Plummer", 0.0), (CUSP, 0.0), (CUSP, 0.1)):
        geo = geometry(obj, prof, np.geomspace(0.01, 8, 12), e, h, grid, eps)
        for i in (0, 5, 11):
            for k in (0, 7, 15):
                nvec = geo["n"][k]
                grad = geo["gn"][i] * nvec + e + geo["r"][i] * (h @ nvec)
                psi_ij = (geo["lr"][i] * np.outer(nvec, nvec) +
                          geo["lt"][i] * (np.eye(3) - np.outer(nvec, nvec)) + h)
                worst = max(worst,
                            abs(geo["g2"][i, k] / (grad @ grad) - 1),
                            abs(geo["t2"][i, k] / np.sum(psi_ij * psi_ij) - 1),
                            abs(geo["radial_grad"][i, k] / (nvec @ grad) - 1))
    check("geometry: |grad psi|^2, psi,ij psi,ij, n.grad psi vs explicit construction",
          worst < 1e-9, f"(max rel. err {worst:.2e})")

    worst = 0.0
    g06 = Grid(radial=15, polar=4, azimuth=4, action=8)
    ratios = np.geomspace(0.05, 6, 15)
    for prof in PROFILES:
        ref = V706.model_state(obj, prof, ratios, e, h, g06)
        geo = geometry(obj, prof, ratios, e, h, g06, 0.0)
        for key in ("r", "gn", "rho", "dlogrho"):
            worst = max(worst, float(np.max(np.abs(geo[key] / ref[key] - 1))))
        worst = max(worst, abs(geo["rh"] / ref["rh"] - 1))
    check("gate G: eps = 0 geometry identical to the frozen v7.06 model_state",
          worst < 1e-12, f"(max rel. diff {worst:.1e})")

    q0 = V706.cusp_calibration()
    q1 = cusp_calibration_soft(0.0)
    worst = max(abs(q1[1] / q0[1] - 1), abs(q1[2] / q0[2] - 1),
                float(np.max(np.abs(q1[3] - q0[3]))), abs(q1[4] / q0[4] - 1))
    check("gate G: softened-cusp calibration at eps = 0 equals V706.cusp_calibration "
          "(a/R_e, r_h/R_e, enclosed mass)", worst < 1e-10, f"(max diff {worst:.1e})")
    check("baseline cusp: a/R_e = 0.7201434 and r_h = 1.3419267 R_e (v7.05)",
          abs(q0[1] - 0.7201434) < 5e-7 and abs(q0[2] - 1.3419267) < 5e-6,
          f"(a/R_e = {q0[1]:.7f}, r_h/R_e = {q0[2]:.7f})")

    # S6: softened density -- enclosed mass vs direct integration, projected half-light radius
    worst_m = worst_h = 0.0
    for eps_rh in EPS_GRID_RH:
        eq = eps_q_from_rh(eps_rh)
        q, a, rh_re, cum, total = cusp_calibration_soft(eq)

        def dens(t):
            return (a / np.sqrt(t * t + eq * eq)) / (1 + t / a) ** 3 * np.exp(-(t / 10) ** 4)

        tot = quad(lambda t: t * t * dens(t), 0, 60, epsabs=0, epsrel=1e-12, limit=500,
                   points=[eq, 1.0, 10.0])[0]
        for rq in (0.01, 0.1, 0.5, 1.0, 3.0, 8.0):
            m = quad(lambda t: t * t * dens(t), 0, rq, epsabs=0, epsrel=1e-12, limit=500)[0] / tot
            worst_m = max(worst_m, abs(np.interp(rq, q, cum) - m))
        proj = quad(lambda t: t * t * dens(t) * (1.0 if t <= 1 else 1 - np.sqrt(1 - 1 / t ** 2)),
                    0, 60, epsabs=0, epsrel=1e-12, limit=500, points=[eq, 1.0, 10.0])[0] / tot
        worst_h = max(worst_h, abs(proj - 0.5))
    check("gate G: softened cusp enclosed mass vs direct integration of the density",
          worst_m < 1e-5, f"(max abs diff {worst_m:.1e})")
    check("softened cusp keeps half of the projected mass inside R_e",
          worst_h < 1e-5, f"(max |fraction - 0.5| {worst_h:.1e})")

    # S7: mesh reproduces the frozen v7.06 mesh and the v7.09 truncated mesh
    gd = Grid()
    worst = max(float(np.max(np.abs(mesh(p, gd) - V706.radial_mesh(p, gd)))) for p in PROFILES)
    check("mesh(rmin = 0.002) identical to V706.radial_mesh", worst == 0.0, f"({worst:.1e})")
    trunc = np.sort(np.unique(np.r_[np.geomspace(0.002, TRUNCATE_RH, gd.radial), 1.0]))
    check("truncated 5 r_h mesh identical to the v7.09 follow-up mesh",
          float(np.max(np.abs(mesh(CUSP, gd, TRUNCATE_RH) - trunc))) == 0.0)
    check("Part C mesh limit 0.0005 r_h lies below the v7.09 limit",
          mesh(CUSP, gd, None, RMIN_PART_C)[0] < RMIN_DEFAULT)

    # S8: registered case counts
    full = enumerate_specs(("BASE", "A", "B", "C"), tuple(DWARFS), ELL0_GRID_PC)
    cnt = lambda part, gq: sum(1 for s in full if s.part == part and (s.ell0_pc is not None) == gq)
    check("case counts: BASE 36 + 6, Part A 144 + 12, Part B 72 + 12, Part C 3",
          (cnt("BASE", True), cnt("BASE", False), cnt("A", True), cnt("A", False),
           cnt("B", True), cnt("B", False), cnt("C", True)) == (36, 6, 144, 12, 72, 12, 3))

    # S9: decision logic on SYNTHETIC records (no physics is evaluated here)
    check_logic(check)
    if verbose:
        print("self-test:", "ALL PASSED" if ok else "FAILED")
    return 0 if ok else 1


def _mock(specs, virial=lambda sp: 1.0, sigma2=lambda sp, lab: 0.1,
          labels=("smoke", "default", "refined")):
    res = {}
    for lab in labels:
        res[lab] = {}
        for sp in specs:
            if sp.part == "C" and lab != "refined":
                continue
            res[lab][sp.key] = dict(virial=virial(sp),
                                    jeans=dict(min_sigma2_kms2=sigma2(sp, lab)))
    return res


def check_logic(check):
    specs = enumerate_specs(("BASE", "A", "B"), tuple(DWARFS), ELL0_GRID_PC)
    hydrus_cusp = lambda sp: sp.name == HYDRUS and sp.profile == CUSP and sp.ell0_pc in ELL0_HYDRUS_PC

    res = _mock(specs)
    check("logic: all positive -> G1", classify_global(res, specs)["outcome"] == "G1")

    victim = next(sp for sp in specs if sp.part == "A" and sp.ell0_pc is not None and sp.law.name == "A-f2")
    res = _mock(specs, virial=lambda sp: -1.0 if sp.key == victim.key else 1.0)
    check("logic: one negative GQUMOND case with positive control -> G2",
          classify_global(res, specs)["outcome"] == "G2")

    ctrl = next(sp for sp in specs if sp.part == "A" and sp.ell0_pc is None and sp.law.name == "A-Qs")
    res = _mock(specs, virial=lambda sp: -1.0 if sp.key == ctrl.key else 1.0)
    check("logic: a negative QUMOND control -> G3", classify_global(res, specs)["outcome"] == "G3")

    res = _mock(specs)
    res["refined"][victim.key]["virial"] = -1.0           # grids disagree on a sign
    out = classify_global(res, specs)
    check("logic: grid disagreement that could change the outcome -> indeterminate (I)",
          out["outcome"] == "I" and victim.key in out["disagreeing_cases"])

    res = _mock(specs)
    for sp in specs:                                      # a negative case in Part B, control positive
        if sp.part == "B" and sp.ell0_pc == 10.0 and sp.name == HYDRUS and sp.eps_rh == 0.3:
            for lab in res:
                res[lab][sp.key]["virial"] = -2.0
    check("logic: negative Part B case with positive softened-profile control -> G2",
          classify_global(res, specs)["outcome"] == "G2")

    # Reviewed-PDF boundary and invalid-data regressions.
    check("logic: complete primary matrix has 216 GQUMOND and 30 controls",
          classify_global(_mock(specs), specs)["gqumond_case_count"] == 216
          and classify_global(_mock(specs), specs)["control_case_count"] == 30)
    res = _mock(specs, virial=lambda sp: 0.0 if sp.key == victim.key else 1.0)
    out = classify_global(res, specs)
    check("logic: exact-zero GQUMOND is I, never a negative detection",
          out["outcome"] == "I" and victim.key in out["boundary_cases"]
          and not out["negative_gqumond_cases"])
    res = _mock(specs, virial=lambda sp: 0.0 if sp.key == ctrl.key else 1.0)
    check("logic: exact-zero control gives G3",
          classify_global(res, specs)["outcome"] == "G3")
    res["smoke"][ctrl.key]["virial"] = -1.0
    check("logic: control negative/zero mixture remains nonpositive (G3)",
          classify_global(res, specs)["outcome"] == "G3")
    res["refined"][ctrl.key]["virial"] = 1.0
    check("logic: nonpositive/positive control disagreement gives I",
          classify_global(res, specs)["outcome"] == "I")
    for bad in (float("nan"), float("inf"), -float("inf"), None):
        res = _mock(specs)
        res["default"][victim.key]["virial"] = bad
        check("logic: invalid virial cannot produce a claim",
              classify_global(res, specs)["outcome"] == "I")
    res = _mock(specs)
    del res["smoke"][victim.key]
    check("logic: missing primary record gives I",
          classify_global(res, specs)["outcome"] == "I")
    res = _mock(specs)
    res["default"][victim.key]["jeans"]["min_sigma2_kms2"] = float("nan")
    check("logic: invalid pressure is not nonnegative",
          pressure_status(res, victim.key) == "invalid"
          and classify_global(res, specs)["outcome"] == "I")

    # J: all variants negative-pressure vs one clean variant
    neg = lambda sp, lab: -0.5 if hydrus_cusp(sp) and sp.part in ("A", "BASE") else 0.1
    res = _mock(specs, sigma2=neg)
    check("logic: negative pressure in all four Part A variants -> J1",
          classify_local_a(res, specs)["outcome"] == "J1")
    neg2 = lambda sp, lab: -0.5 if hydrus_cusp(sp) and sp.part in ("A", "BASE") and sp.law.name != "A-f2" else 0.1
    res = _mock(specs, sigma2=neg2)
    check("logic: one variant without negative pressure -> J2",
          classify_local_a(res, specs)["outcome"] == "J2")
    res = _mock(specs, sigma2=lambda sp, lab: (-0.5 if lab == "default" else 0.1)
                if (hydrus_cusp(sp) and sp.part == "A" and sp.law.name == "A-f2") else neg2(sp, lab))
    check("logic: unresolved default/refined disagreement that decides J -> I",
          classify_local_a(res, specs)["outcome"] == "I")

    # K: three-way partition
    def kmock(neg_eps):
        return lambda sp, lab: (-0.5 if (sp.part == "B" and sp.name == HYDRUS and sp.ell0_pc in ELL0_HYDRUS_PC
                                         and sp.eps_rh in neg_eps) else 0.1)
    check("logic: negative pressure at eps = 0.3 -> K2",
          classify_local_b(_mock(specs, sigma2=kmock({0.01, 0.03, 0.1, 0.3})), specs)["outcome"] == "K2")
    check("logic: negative pressure only at eps <= 0.03 -> K1",
          classify_local_b(_mock(specs, sigma2=kmock({0.01, 0.03})), specs)["outcome"] == "K1")
    check("logic: negative pressure at 0.1 but not at 0.3 -> K3",
          classify_local_b(_mock(specs, sigma2=kmock({0.01, 0.03, 0.1})), specs)["outcome"] == "K3")
    check("logic: no negative pressure at any eps -> K1",
          classify_local_b(_mock(specs, sigma2=kmock(set())), specs)["outcome"] == "K1")

    # gate R on exact reference values passes; on perturbed values it fails
    base_specs = [sp for sp in specs if sp.part == "BASE"]
    res = {"default": {}, "refined": {}}
    for sp in base_specs:
        if sp.ell0_pc is None:
            v = REF_CONTROL[(sp.name, sp.profile)]
        else:
            v = REF_GQ[(sp.name, sp.profile)][ELL0_GRID_PC.index(sp.ell0_pc)]
        res["default"][sp.key] = dict(virial=v, jeans=dict(min_sigma2_kms2=0.0))
    for l0 in ELL0_HYDRUS_PC:
        res["refined"][make_key("BASE", HYDRUS, CUSP, 0.0, l0)] = dict(
            virial=1.0, jeans=dict(min_sigma2_kms2=REF_MIN_SIGMA2[l0]))
    check("logic: gate R passes on the reference values", gate_r(res)["passed"])
    k0 = make_key("BASE", "Leo IV", "Plummer", 0.0, 1.0)
    res["default"][k0]["virial"] *= 1.001
    check("logic: gate R fails on a perturbed value", not gate_r(res)["passed"])
    # Nonfinite reproduction data must not silently pass max/error checks.
    res["default"][k0]["virial"] = float("nan")
    check("logic: gate R rejects nonfinite references", not gate_r(res)["passed"])
    check("logic: gate R rejects missing grids", not gate_r({})["passed"])
    res = _mock(specs)
    ka = make_key("A-f2", HYDRUS, CUSP, 0.0, 1.0)
    res["default"][ka]["jeans"]["min_sigma2_kms2"] = float("nan")
    check("logic: invalid Part A pressure gives J=I", classify_local_a(res, specs)["outcome"] == "I")
    kb = make_key("BASE", HYDRUS, CUSP, 0.3, 10.0)
    res["default"][kb]["jeans"]["min_sigma2_kms2"] = float("inf")
    check("logic: invalid Part B pressure gives K=I", classify_local_b(res, specs)["outcome"] == "I")
    check("logic: exact-zero pressure is nonnegative",
          pressure_status(_mock(specs, sigma2=lambda sp, lab: 0.0), ka) == "no")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run(args):
    if selftest(verbose=False) != 0:
        raise SystemExit("self-test failed (gates S, G, L): refusing to run the registered "
                         "calculation.  Run --selftest for details.")
    names = tuple(DWARFS) if args.object == "all" else (args.object,)
    parts = tuple(args.parts.split(","))
    ell0_list = ELL0_GRID_PC if args.ell0 is None else (args.ell0,)
    labels = tuple(args.grids.split(","))
    specs = enumerate_specs(parts, names, ell0_list)
    complete = (args.object == "all" and set(parts) == {"BASE", "A", "B", "C"} and
                args.ell0 is None and set(labels) == {"smoke", "default", "refined"})
    result = dict(
        status="UDCT v7.10 pre-registered test implementation; see the registration PDF "
               "for decision rules",
        implementation_revision=IMPLEMENTATION_REVISION,
        run_started_utc=datetime.now(timezone.utc).isoformat(),
        environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                         platform=platform.platform()),
        command_line=sys.argv,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        v706_sha256=V706_SHA256, parts=list(parts), grids=list(labels),
        complete_registered_set=complete, selftest_passed=True)
    res = {}
    for lab in labels:
        res[lab] = run_grid(lab, specs, names)
    result["cases"] = res
    if complete:
        result["gate_R"] = gate_r(res)
        result["G"] = classify_global(res, specs)
        result["J"] = classify_local_a(res, specs)
        result["K"] = classify_local_b(res, specs)
        result["new_pressure_or_work_cases_to_report"] = new_pressure_cases(res, specs)
        if not result["gate_R"]["passed"]:
            result["provisional_classification_before_gate_R"] = {
                label: result[label] for label in ("G", "J", "K")}
            for label in ("G", "J", "K"):
                result[label] = dict(outcome="I", reason="gate R failed; no variant claim")
        result["note"] = ("Outcomes are claimable only if gate_R.passed is true; indeterminate (I) "
                          "outcomes carry no claim.  Report the deviation log with the results.")
    else:
        result["note"] = "partial run: no gate evaluation and no outcome classification"
    return result


def json_safe(value):
    """Archive nonfinite floats as null; invalid statuses remain in the audit."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def cli():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true",
                   help="analytic identities, geometry and decision logic on synthetic "
                        "records; evaluates no registered quantity")
    p.add_argument("--parts", default="BASE,A,B,C",
                   help="comma list of BASE (v7.09 re-run, gate R), A, B, C (default: all)")
    p.add_argument("--grids", default="smoke,default,refined",
                   help="comma list of registered grids (default: all three)")
    p.add_argument("--object", choices=("all", *DWARFS), default="all")
    p.add_argument("--ell0", type=float, default=None,
                   help="single l0 in pc (partial run; no classification)")
    args = p.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    print(json.dumps(json_safe(run(args)), indent=2, allow_nan=False))


if __name__ == "__main__":
    cli()
