"""
================================================================================
 LFP-ETGNet  |  Module 1: Electrode microstructure reconstruction,
                         graph abstraction and coupled multi-mechanism aging
================================================================================
 Physics-grounded virtual electrode generator for automotive LiFePO4 (LFP)
 power-battery cathodes.  All microstructural statistics are calibrated to
 published teardown / electron-microscopy studies of commercial Chinese
 prismatic LFP cells (BYD Blade / CATL-class), and all aging-rate constants are
 calibrated to published LFP cycling data.

 Key physical facts used as calibration anchors
 ---------------------------------------------
 * Olivine LiFePO4, space group Pnma, a=6.008, b=10.334, c=4.693 Angstrom.
 * LFP <-> FePO4 two-phase reaction, relative cell-volume change 6.8 %.
 * Intrinsic electronic conductivity ~1e-9 S/cm, Li solid diffusivity ~1e-14 cm^2/s.
 * Commercial LFP cathode: nano primary particles 100-300 nm aggregated into
   secondary particles of a few micron; carbon coating 2-5 nm; CNT/carbon-black
   conductive network; electrode (coating) thickness ~55 um.
 * Fresh SEI ~ 25 nm -> aged SEI ~ 50 nm on LFP particles (TEM).
 * Capacity fade of automotive LFP: ~0.25 %/100 cycles at 25 C, 1C, 100 % DOD;
   ~0.55 %/100 cycles at 45 C; Arrhenius acceleration, intrinsic Ea ~ 40 kJ/mol
   and effective Ea = 30 kJ/mol (see Ea_eff below).
 * Degradation channels: the interphase channel (loss of lithium inventory,
   LLI), which is the only one independent of the electrode network, and the
   network channel (contact fatigue), which manifests locally as conductivity
   loss (CL) and topologically as loss of active material (LAM).  LAM and CL
   are therefore not independent mechanisms: the graph separates them.
================================================================================
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# ----------------------------------------------------------------------------
#  Global physical parameter set (all lengths in um unless stated)
# ----------------------------------------------------------------------------
P = dict(
    # ---- geometry ---------------------------------------------------------
    L_RVE        = 6.0,      # representative volume element edge (um)
    #   Calendered automotive LFP cathodes are 30-40 % porous; 55 % active
    #   material plus 7 % carbon/binder leaves 38 % porosity.  The value also
    #   has to be packable: a jittered-lattice plus spring relaxation cannot
    #   reach a packing fraction close to random close packing, and phi = 0.60
    #   left residual particle overlap (a negative surface-to-surface gap).
    #   At 0.55 the relaxed packing is overlap-free.
    phi_solid    = 0.550,    # active-material volume fraction (calendered)
    porosity     = 0.380,    # electrode porosity
    phi_cbd      = 0.070,    # carbon-black / binder volume fraction
    # ---- active particle size distribution (bimodal) ----------------------
    #   Both modes are kept well below the RVE edge: the edge is derived from
    #   the sampled solid volume, and a representative volume element is only
    #   meaningful if its largest heterogeneity spans a small fraction of it
    #   (the constraint dia_max_frac below is enforced self-consistently).
    d_prim_med   = 0.20,     # primary LFP particle median diameter (um)
    d_prim_sig   = 0.25,     # lognormal sigma
    d_sec_med    = 0.45,     # secondary-agglomerate median diameter (um)
    d_sec_sig    = 0.20,
    w_sec        = 0.20,     # number fraction of secondary particles
    dia_min      = 0.080,    # smallest resolvable particle (um)
    dia_max      = 1.200,    # largest admissible particle (um)
    dia_max_frac = 0.30,     # max particle diameter as a fraction of the edge
    # ---- conductive additive ---------------------------------------------
    #   Carbon black forms 40-100 nm primary aggregates that cluster into
    #   coarser agglomerates.  The pore throats of a calendered electrode are
    #   only tens of nanometres wide, so a coarse-grained domain of a few
    #   hundred nanometres cannot be placed in the pore space at the true
    #   additive volume fraction.  The additive is therefore represented by
    #   (i) a moderate population of coarse-grained domains, each standing for
    #   one carbon-black cluster occupying an interstice, and (ii) the thin
    #   carbon coating on the particle surfaces, which is carried by the
    #   contact-quality edge feature (t_carbon below).
    d_CBD        = 0.045,    # carbon-black primary aggregate diameter (um)
    d_cbd_domain = 0.200,    # coarse-grained additive domain diameter (um)
    n_cbd_train  = 300,      # coarse-grained additive domains per RVE
    t_carbon     = 0.003,    # carbon coating thickness on LFP (um = 3 nm)
    # ---- interphase -------------------------------------------------------
    t_SEI0       = 0.005,    # as-built SEI thickness (um = 5 nm)
    t_SEI_ref    = 0.030,    # film thickness on a mean-size particle (30 nm)
    t_SEI_max    = 0.060,    # upper physical bound on the film (um = 60 nm)
    alpha_sei    = 0.35,     # surface-to-volume exponent of local growth
    gamma_sei    = 0.35,     # exponent of the local reaction current on growth
    # ---- electrochemistry -------------------------------------------------
    dV_frac      = 0.068,    # LFP/FePO4 relative volume change
    D_s          = 1.0e-14,  # solid-phase Li diffusivity (cm^2/s)
    c_max        = 0.0228,   # max Li concentration in LFP (mol/cm^3)
    V_molar      = 43.8,     # molar volume LFP (cm^3/mol)
    Ea_SEI       = 40.0e3,   # intrinsic SEI-growth activation energy (J/mol)
    #   Reported values for interphase growth span 30-40 kJ/mol.  The intrinsic
    #   value above is not used to scale the protocol: the effective value
    #   below is the one that reproduces the measured 25 -> 45 C fade-rate ratio
    #   of ~2.1 for automotive LFP, and it is lower because it lumps the
    #   temperature dependence of the diffusion-limited and linear terms
    #   together with their C-rate and DOD couplings.  Keeping both here, with
    #   the effective one applied and the intrinsic one recorded, prevents the
    #   two from being duplicated in different modules and drifting apart, which
    #   is what happened before this parameter set was consolidated.
    Ea_eff       = 30.0e3,   # effective Arrhenius energy applied to the protocol
    R_gas        = 8.314,
    T_ref        = 298.15,
    # ---- reference local electrochemistry ---------------------------------
    #   Used for the node attributes only, so that the graph branch describes
    #   the electrode under a nominal condition and stays independent of the
    #   operating protocol, which enters the prediction through the
    #   trajectory branch.
    i_1C         = 2.0e-3,   # area-specific current density at 1C (A/cm^2)
    beta_ref     = 0.45,     # through-thickness reaction tilt at the reference
    soc_ref      = 0.50,     # cell-level state of charge at the prediction
    # ---- contact / percolation -------------------------------------------
    gap_direct   = 0.006,    # direct solid-solid contact tolerance (um = 6 nm)
    gap_wetted   = 0.060,    # electrolyte-bridged (ionically wetted) gap (um)
    r_neck_ref   = 0.50,     # reference inter-particle neck size (um)
    m_fatigue    = 0.80,     # contact fatigue exponent (Basquin-like)
    s_fatigue    = 0.45,     # log-scatter of the neck fatigue life
    # ---- electrical network ----------------------------------------------
    sigma_AM     = 1.0e-7,   # effective AM+C-coating conductivity (S/cm)
    sigma_CBD    = 20.0,     # carbon-black network conductivity (S/cm)
)


# ============================================================================
#  1.  Microstructure reconstruction
# ============================================================================
def _relax_overlaps(centres: np.ndarray, radii: np.ndarray, box: float,
                    iters: int = 900, step: float = 0.42,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Relax a jittered lattice packing to a non-overlapping configuration.

    Uses a pairwise-spring relaxation with periodic boundary conditions, which
    is fast and yields packings with realistic coordination numbers for
    polydisperse sphere assemblies.  Both partners of an overlapping pair are
    displaced by ``step`` times the overlap along the contact normal, so an
    isolated pair contracts its overlap by a factor (1 - 2*step) per sweep.
    """
    rng = rng or np.random.default_rng(0)
    c = centres.copy()
    r = radii.copy()
    cut = 2.0 * r.max()
    for it in range(iters):
        tree = cKDTree(c, boxsize=box)
        pairs = tree.query_pairs(cut, output_type="ndarray")
        if pairs.size == 0:
            break
        d = c[pairs[:, 1]] - c[pairs[:, 0]]
        d -= box * np.round(d / box)                    # minimum image
        dist = np.linalg.norm(d, axis=1) + 1e-12
        overlap = radii[pairs[:, 0]] + radii[pairs[:, 1]] - dist
        bad = overlap > 1e-6
        if not np.any(bad):
            break
        p = pairs[bad]
        ov = overlap[bad]
        u = d[bad] / dist[bad, None]
        disp = np.zeros_like(c)
        np.add.at(disp, p[:, 0], -u * (step * ov)[:, None])
        np.add.at(disp, p[:, 1], +u * (step * ov)[:, None])
        c = (c + disp) % box
    return c


def _pack_random(centres: np.ndarray, radii: np.ndarray, box: float,
                 rng: np.random.Generator, n_grow: int = 14,
                 iters_per: int = 70, final_iters: int = 900,
                 step: float = 0.42) -> np.ndarray:
    """Compress a random point set into a dense, overlap-free packing.

    Seeding on a lattice is unsuitable here: a lattice whose spacing exceeds
    the particle diameter leaves the small particles mutually separated, so the
    reconstructed electrode contains almost no solid-solid contacts and the
    graph degenerates to a purely ionic network.  Instead the centres are drawn
    from a uniform random set and the radii are grown in steps, relaxing the
    overlaps between steps.  Because the relaxation stops as soon as the
    overlap vanishes, neighbouring particles are left exactly in contact, which
    is what makes the direct-contact edge class populated.
    """
    c = (centres + rng.normal(0.0, 0.05 * box, centres.shape)) % box
    for g in range(n_grow + 1):
        s = 0.12 + 0.88 * (g / n_grow)
        c = _relax_overlaps(c, s * radii, box, iters=iters_per, step=step,
                            rng=rng)
    return _relax_overlaps(c, radii, box, iters=final_iters, step=step, rng=rng)


def reconstruct_electrode(rng: np.random.Generator, p: dict = P,
                          n_target: int = 900) -> dict:
    """Reconstruct a representative volume element of a calendered LFP cathode.

    Strategy — the RVE size is *derived* from the physical particle-size
    distribution rather than assumed: the target particle count and the
    measured PSD fix the solid volume, which together with the calendered
    active-material volume fraction fixes the RVE edge length.

    Returns particle centres / radii (active material) and carbon-black domains.
    """
    # ---- sample the bimodal PSD ------------------------------------------
    n_sec = int(round(p["w_sec"] * n_target))
    n_prim = n_target - n_sec
    d_prim = rng.lognormal(np.log(p["d_prim_med"]), p["d_prim_sig"], n_prim)
    d_sec = rng.lognormal(np.log(p["d_sec_med"]), p["d_sec_sig"], n_sec)
    dia = np.clip(np.concatenate([d_prim, d_sec]), p["dia_min"], p["dia_max"])

    # ---- self-consistent RVE edge length ----------------------------------
    #   The edge follows from the sampled solid volume and the calendered
    #   active-material volume fraction.  It is then checked against the
    #   representativeness criterion that no single particle may span more
    #   than ``dia_max_frac`` of the edge; the diameters are capped and the
    #   edge recomputed until the two are mutually consistent.  Without this
    #   step a single agglomerate can be larger than the periodic cell, which
    #   produces self-overlap across the boundary and unphysical hubs in the
    #   contact graph.
    for _ in range(12):
        V_solid = float(np.sum(np.pi / 6.0 * dia ** 3))
        box = (V_solid / p["phi_solid"]) ** (1.0 / 3.0)
        capped = np.minimum(dia, p["dia_max_frac"] * box)
        if np.array_equal(capped, dia):
            break
        dia = capped
    V_solid = float(np.sum(np.pi / 6.0 * dia ** 3))
    box = (V_solid / p["phi_solid"]) ** (1.0 / 3.0)
    n_part = len(dia)
    rad = dia / 2.0

    # ---- seed and compress to a dense random packing ----------------------
    centres = rng.random((n_part, 3)) * box
    centres = _pack_random(centres, rad, box, rng)

    # ---- carbon-black / binder domains (coarse-grained network nodes) -----
    #   Each domain must lie entirely inside the pore space.  The constraint is
    #   expressed on the distance to the nearest surface, not to the nearest
    #   centre, because a large particle slightly further away than the nearest
    #   neighbour can still swallow the point - which is what produced
    #   overlapping active-material / additive pairs earlier.  The radius of a
    #   domain is drawn from the admissible range and then clipped by the local
    #   pore half-width, so no domain penetrates an active particle.
    d_cbd = p["d_cbd_domain"]
    V_box = box ** 3
    r_lo, r_hi = 0.55 * d_cbd / 2.0, 0.90 * d_cbd / 2.0
    n_cbd = int(p["n_cbd_train"])
    tree_am = cKDTree(centres, boxsize=box)
    kk = int(min(64, len(centres)))
    #   the neighbour search must reach far enough to include any particle that
    #   could contain the candidate, i.e. r_domain + r_max.  Truncating at the
    #   admissible clearance instead silently misses the large agglomerates and
    #   lets domains penetrate them.
    qbound = r_hi + rad.max()
    accepted = np.zeros((n_cbd, 3))
    acc_r = np.zeros(n_cbd)
    n_acc = 0
    for _ in range(160):
        if n_acc >= n_cbd:
            break
        pts = rng.random((max(n_cbd, 64) * 20, 3)) * box
        d, ix = tree_am.query(pts, k=kk, distance_upper_bound=qbound)
        d = np.atleast_2d(d)
        ix = np.atleast_2d(ix)
        good = np.isfinite(d)
        s = np.min(np.where(good, d - rad[np.clip(ix, 0, len(rad) - 1)], np.inf),
                   axis=1)
        sel = (s >= r_lo) & (s <= 3.0 * r_hi)
        pts, s = pts[sel], s[sel]
        if pts.shape[0] == 0:
            continue
        rdraw = np.minimum(rng.uniform(r_lo, r_hi, pts.shape[0]), s)
        for cc, rr in zip(pts, rdraw):
            if n_acc >= n_cbd:
                break
            if n_acc:
                dd, _ = cKDTree(accepted[:n_acc], boxsize=box).query(cc, k=1)
                if dd < rr + r_hi:          # conservative: no future overlap
                    continue
            accepted[n_acc] = cc
            acc_r[n_acc] = rr
            n_acc += 1
    cbd_c = accepted[:n_acc]
    cbd_r = acc_r[:n_acc]
    #   achieved clearance between every domain and the active-particle surface;
    #   must be non-negative, otherwise the domain penetrates a particle and the
    #   graph acquires spurious direct contacts
    if n_acc:
        dd, jj = cKDTree(centres, boxsize=box).query(cbd_c, k=int(min(64, len(centres))))
        min_clear = float(np.min(np.min(dd - rad[jj], axis=1) - cbd_r))
    else:
        min_clear = float("nan")

    # ---- packing validation ----------------------------------------------
    #   A negative surface-to-surface gap between two active particles is
    #   unphysical and would create spurious contacts in the graph, so the
    #   residual overlap is measured and returned for every realisation.
    tree_v = cKDTree(centres, boxsize=box)
    pr = tree_v.query_pairs(2.0 * rad.max(), output_type="ndarray")
    if pr.size:
        dd = centres[pr[:, 1]] - centres[pr[:, 0]]
        dd -= box * np.round(dd / box)
        dist = np.linalg.norm(dd, axis=1)
        max_ov = float(np.max(rad[pr[:, 0]] + rad[pr[:, 1]] - dist))
    else:
        max_ov = 0.0

    return dict(box=box, centres=centres, radii=rad, dia=dia,
                cbd_centres=cbd_c, cbd_radii=cbd_r, max_overlap=max_ov,
                min_clearance=min_clear)


# ============================================================================
#  2.  Graph abstraction  G = (V, E)
# ============================================================================
def build_graph(micro: dict, p: dict = P,
                rng: np.random.Generator | None = None) -> dict:
    """Abstract the electrode microstructure into a topological graph.

    Nodes  : active-material particles (AM) and carbon-black domains (CBD).
    Edges  : (i) direct solid-solid electronic contacts,
             (ii) electrolyte-bridged (ionically wetted) pathways.
    """
    rng = rng or np.random.default_rng(1)
    box = micro["box"]
    c_am, r_am = micro["centres"], micro["radii"]
    c_cb, r_cb = micro["cbd_centres"], micro["cbd_radii"]
    n_am, n_cb = len(c_am), len(c_cb)

    centres = np.vstack([c_am, c_cb]) if n_cb else c_am
    radii = np.concatenate([r_am, r_cb]) if n_cb else r_am
    kind = np.array([0] * n_am + [1] * n_cb)                 # 0=AM, 1=CBD
    n_node = len(centres)

    tree = cKDTree(centres, boxsize=box)
    max_r = radii.max()
    pairs = tree.query_pairs(2.0 * max_r + p["gap_wetted"], output_type="ndarray")
    d = centres[pairs[:, 1]] - centres[pairs[:, 0]]
    d -= box * np.round(d / box)
    gap = np.linalg.norm(d, axis=1) - radii[pairs[:, 0]] - radii[pairs[:, 1]]

    # ---- edge classification ---------------------------------------------
    #   direct solid contact  : gap <= gap_direct  (electronic percolation)
    #   wetted pathway        : gap <= gap_wetted  (electrolyte / ionic path)
    is_direct = gap <= p["gap_direct"]
    is_wetted = gap <= p["gap_wetted"]
    keep = is_direct | is_wetted

    # ---- heterogeneous contact resistance ---------------------------------
    #   The conductance of a solid-solid neck is limited by the constriction
    #   at the contact.  The Hertzian contact radius grows as the square root
    #   of the penetration depth pen = gap_direct - gap, so the normalised
    #   contact quality is taken as
    #        q_ij = 0.05 + 0.95 * sqrt( clip(pen / gap_direct, 0, 1) ),
    #   which is monotone in the penetration and spans the full (0, 1] range
    #   instead of a narrow band.  Wetted-only edges carry no electronic
    #   conductance and are handled separately through the ionic weight.
    r_eff = 2.0 * radii[pairs[:, 0]] * radii[pairs[:, 1]] / (
        radii[pairs[:, 0]] + radii[pairs[:, 1]] + 1e-12)
    pen = np.clip(p["gap_direct"] - gap, 0.0, None)
    sigma_c = np.where(
        is_direct,
        np.clip(0.05 + 0.95 * np.sqrt(np.clip(pen / p["gap_direct"], 0.0, 1.0)),
                0.0, 1.0),
        0.0)
    # wetted-only edges carry a small ionic conductance weighting
    ionic = np.where(~is_direct & is_wetted, 0.12, 0.0)

    edges = pairs[keep]
    gap_e = gap[keep]
    direct_e = is_direct[keep]
    sigma_c = sigma_c[keep]
    ionic = ionic[keep]
    r_eff = r_eff[keep]

    # ---- electrode through-thickness depth (0 = current collector) --------
    depth = centres[:, 2] / box

    return dict(box=box, centres=centres, radii=radii, dia=2 * radii, kind=kind,
                n_am=n_am, n_cb=n_cb, n_node=n_node,
                edges=edges, gap=gap_e, direct=direct_e,
                sigma_c=sigma_c, ionic=ionic, r_eff=r_eff, depth=depth,
                c_am=c_am, r_am=r_am, c_cb=c_cb, r_cb=r_cb)


# ============================================================================
#  3.  Through-thickness current distribution (electrolyte transport limit)
# ============================================================================
def local_current_profile(depth: np.ndarray, c_rate: float | None = None,
                          p: dict = P, beta: float | None = None) -> np.ndarray:
    """Normalised local reaction current density  j(z)/j_avg .

    Commercial LFP electrodes exhibit a strongly non-uniform reaction
    distribution because the OCP plateau is flat: the electrolyte ohmic drop
    tilts the reaction towards the separator.  A first-order tilt is used,
        j(z)/j_avg = 1 + beta (2 z - 1),
    with beta growing with C-rate (beta = 0.25 + 0.10 * C_rate) when a rate is
    supplied, and equal to the reference value ``p['beta_ref']`` otherwise.
    The mean of j over a uniform depth distribution is unity by construction,
    so the profile rescales rather than biases the local rates.
    """
    if beta is None:
        beta = p["beta_ref"] if c_rate is None else 0.25 + 0.10 * float(c_rate)
    beta = min(float(beta), 0.80)
    j = 1.0 + beta * (2.0 * depth - 1.0)
    return np.clip(j, 0.05, None)


def li_gradient_feature(r_am: np.ndarray, j_local: np.ndarray,
                        soc_local: np.ndarray, p: dict = P,
                        c_rate: float = 1.0) -> np.ndarray:
    """Solid-phase Li-concentration gradient scale  |grad c|  per particle.

    From the single-particle-model relation, the characteristic gradient over
    a particle of radius R carrying an area-specific current density i is

        |grad c| ~ (R / 5) * i / (D_s F c_max) ,

    in mol cm^-4, with i = i_1C * C_rate * (j/j_avg).  The mild dependence on
    the local state of charge accounts for the growth of the diffusion
    limitation as the particle approaches full (de)lithiation.
    """
    F = 96485.3
    r_cm = r_am * 1e-4                                   # um -> cm
    i_local = p["i_1C"] * float(c_rate) * j_local         # A/cm^2
    grad = (r_cm / 5.0) * i_local / (p["D_s"] * F * p["c_max"])
    grad = grad * (0.6 + 0.8 * np.clip(soc_local, 0, 1))
    return grad


def volume_strain(r_am: np.ndarray, soc_local: np.ndarray, p: dict = P,
                  dod: float = 1.0) -> np.ndarray:
    """Linear radial strain amplitude of each particle over a (de)lithiation
    swing of width ``dod``.

    The LFP -> FePO4 two-phase reaction changes the cell volume by 6.8 %, so
    the linear strain over a full swing is eps_lin = (dV/V)/3 ~ 2.27 %; a
    partial swing scales the amplitude in proportion to its width.  The local
    state of charge sets the position within the swing but not its amplitude,
    so it does not enter here.
    """
    amp = (p["dV_frac"] / 3.0) * float(np.clip(dod, 0.0, 1.0))
    return amp * np.ones_like(r_am)


# ============================================================================
#  4.  Coupled degradation model (LLI / LAM / CL) with graph-topology evolution
# ============================================================================
def _percolating_nodes(n_node: int, edges: np.ndarray, alive: np.ndarray,
                       kind: np.ndarray, depth: np.ndarray,
                       subnet: np.ndarray | None = None) -> np.ndarray:
    """Return a boolean mask of the nodes belonging to the *giant connected
    component* of the surviving conducting network.

    Active-material particles outside the giant component have lost their
    conducting path to the current collector; they are electrochemically
    isolated and constitute the loss of active material (LAM).

    ``subnet`` restricts the transport to a subset of the edge set.  Electronic
    isolation must be evaluated on the solid-solid contacts only: a particle
    that retains electrolyte-wetted paths but no electronic contact cannot be
    charged or discharged, so an ionic path cannot keep it electrochemically
    active.  Omitting this restriction makes the network far harder to
    disconnect than it is in reality.
    """
    if edges.size == 0:
        return np.zeros(n_node, bool)
    m = alive if subnet is None else (alive & subnet)
    e = edges[m]
    if e.size == 0:
        return np.zeros(n_node, bool)
    adj = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n_node, n_node))
    n_comp, lab = connected_components(adj, directed=False)
    counts = np.bincount(lab, minlength=n_comp)
    giant = int(np.argmax(counts))
    return lab == giant


def simulate_aging(graph: dict, protocol: dict, rng: np.random.Generator,
                   p: dict = P, n_check: int = 20,
                   n_pred: int | None = None) -> dict:
    """Simulate coupled LFP ageing and evolve the conductive-network topology.

    Degradation modes
    -----------------
    LLI : solid-electrolyte-interphase growth, diffusion limited -> sqrt(t)
          plus a linear side-reaction term, with Arrhenius temperature
          acceleration at the *effective* energy Ea_eff = 30 kJ/mol (the
          intrinsic interphase value of 40 kJ/mol is recorded in ``P`` but is
          not the one applied; see the note on ``Ea_eff``).
    LAM : particles that lose their electronic percolation path are
          electrochemically isolated  -> computed by graph connectivity.
    CL  : contact fatigue; the cyclic particle volume change (6.8 %) opens
          micro-gaps at particle-particle necks and progressively removes the
          corresponding graph edges.

    LAM and CL are not independent mechanisms: a ruptured neck raises the
    series resistance (CL) and may isolate a particle (LAM).  They are the
    topological and the local manifestation of the same contact-fatigue
    process, and the graph is what separates them.  Only the interphase
    channel (LLI) is independent of the network channel.

    Because the damage rates are monotonic in cycle number, the state at
    cycle n is obtained analytically and the graph percolation (the only
    expensive step) is evaluated on a coarse checkpoint grid.  If ``n_pred``
    is given, the surviving network and the percolation state are also
    evaluated at that cycle, which is the state the node attributes must use.
    """
    box = graph["box"]
    n_node, edges = graph["n_node"], graph["edges"]
    kind = graph["kind"]
    depth = graph["depth"]
    direct = graph["direct"]          # electronic (solid-solid) subnetwork
    n_am = graph["n_am"]
    r_am = graph["r_am"]

    T = protocol["T_celsius"] + 273.15
    c_rate = protocol["c_rate"]
    dod = protocol["dod"]
    #   The state-of-charge window enters the model through its width only,
    #   via the depth-of-discharge factor that scales the three rate constants
    #   and the strain amplitude.  Its position is recorded in the protocol for
    #   traceability but is deliberately not applied: on the flat LFP/FePO4
    #   plateau the fade of a window of a given width is insensitive to where
    #   the window sits, and a position dependence would require data that the
    #   calibration does not have.  (These two keys were previously unpacked
    #   here and then never read -- the same dormant-quantity defect that the
    #   parameter set was consolidated to remove -- so they are now declared
    #   explicitly as unused instead of silently suggesting they enter the
    #   ledger.)
    assert "soc_lo" in protocol and "soc_hi" in protocol, \
        "protocol must record the state-of-charge window (metadata only)"
    n_eol = protocol["n_eol"]

    # ---- local electrochemistry at the reference condition ----------------
    #   The node attributes must not encode the operating protocol: the
    #   protocol enters the prediction through the trajectory branch.  The
    #   through-thickness reaction profile is therefore evaluated at the
    #   nominal reference rate, and the local state of charge follows from the
    #   reaction tilt together with the particle size.  The size term enters
    #   with a positive coefficient on (r_ref/r - 1), i.e. a smaller particle
    #   sits at a higher local state of charge for the same local current.  That
    #   is the direction the transport physics gives: the concentration
    #   polarisation within a particle scales with j*R/(5 D_s F c_max), so at a
    #   fixed current the surface of a large particle is driven further from the
    #   equilibrium composition than that of a small one, while the small
    #   particle stays closer to it.  (An earlier comment here stated the
    #   opposite, which contradicted both the expression below and the
    #   single-particle relation used two lines further down.)
    j_local = local_current_profile(depth[:n_am], p=p, beta=p["beta_ref"])
    r_ref_am = float(r_am.mean()) if r_am.size else 1.0
    soc_local = np.clip(p["soc_ref"] + 0.5 * (j_local - 1.0)
                        + 0.10 * (r_ref_am / np.maximum(r_am, 1e-6) - 1.0),
                        0.05, 0.95)
    grad_li = li_gradient_feature(r_am, j_local, soc_local, p)
    #   Reported for inspection, not applied to the fatigue ledger.  The
    #   per-edge fatigue strain is the geometric ratio (r_i + r_j)/r_ref below,
    #   and the depth-of-discharge amplitude enters the network channel through
    #   the lumped power law on k_cl (exponent 1.45) rather than as a second,
    #   explicit factor on eps.  Multiplying both would double-count the
    #   amplitude.  The value is kept because it is the physically meaningful
    #   quantity to quote when describing the mechanism, but a reader should
    #   not expect it to appear in the rate expression.
    eps_amp = volume_strain(r_am, soc_local, p, dod=dod)

    # ---- edge-level cyclic strain amplitude --------------------------------
    #   The cyclic displacement at an inter-particle neck scales with the sum
    #   of the participating radii.  The amplitude of the (de)lithiation
    #   swing is a cell-level quantity and enters through k_CL, which carries
    #   the depth-of-discharge acceleration; the per-edge quantity is
    #   therefore the purely geometric factor
    #        eps_ij = (r_i + r_j) / r_ref ,   r_ref = 0.5 um .
    R_REF = 0.5                                    # um
    if edges.size:
        e = edges
        i0 = np.minimum(e[:, 0], n_am - 1)
        i1 = np.minimum(e[:, 1], n_am - 1)
        eps_edge = (r_am[i0] + r_am[i1]) / R_REF
        eps_edge = np.clip(eps_edge, 1e-3, None)
    else:
        eps_edge = np.zeros(0)

    # ---- rate coefficients -------------------------------------------------
    #   The values supplied in the protocol are already scaled by the
    #   Arrhenius factor and the C-rate / DOD acceleration factors, so that the
    #   same numbers are used both by the graph-topology simulator and by the
    #   synthetic early-life measurement series.
    k_lli = protocol["k_lli_ref"]
    k_lin = protocol.get("k_lin_ref", 0.0)
    k_cl = protocol["k_cl_ref"]
    # legacy key, no longer used in the capacity ledger: the structural loss of
    # active material is computed from percolation rather than prescribed
    k_lam = protocol.get("k_lam_ref", 0.0)

    Q0 = protocol["Q0"]

    # ---- analytic contact-failure cycle per edge --------------------------
    #   Basquin-type damage accumulation  D_ij(n) = n * k_cl * eps_ij^m ,
    #   the conductive neck ruptures at D = 1, i.e. n_break = 1 / rate_ij.
    # ---- stochastic neck-fatigue life -------------------------------------
    #   Fatigue life scatters strongly between nominally identical necks, and
    #   the scatter also removes an artefact of the deterministic model: with
    #   a single rupture cycle per edge a modest change of the rate
    #   coefficient moves the whole edge population across the percolation
    #   threshold at once, which makes the population response unphysically
    #   abrupt.
    if edges.size:
        fat = np.exp(rng.normal(0.0, p["s_fatigue"], edges.shape[0]))
        rate_edge = k_cl * eps_edge ** p["m_fatigue"] * fat
        n_break_edge = 1.0 / np.maximum(rate_edge, 1e-12)
        order = np.argsort(n_break_edge)
        n_break_sorted = n_break_edge[order]
    else:
        fat = np.zeros(0)
        order = np.zeros(0, dtype=int)
        n_break_sorted = np.zeros(0)

    def alive_at(n: int) -> np.ndarray:
        if edges.size == 0:
            return np.zeros(0, bool)
        k = np.searchsorted(n_break_sorted, n, side="right")
        a = np.zeros(edges.shape[0], bool)
        a[order[k:]] = True
        return a

    # ---- checkpoint sweep -------------------------------------------------
    rec = {k: [] for k in ("n", "Q", "SOH", "R_rel", "LLI", "LAM", "CL",
                           "edge_ret", "edge_ret_all", "T_max", "n_alive")}
    perc_last = None
    alive_last = None
    lam_baseline = None
    n = 0
    while n <= n_eol:
        nn = max(n, 1)
        # two-term LLI: diffusion-limited SEI growth (sqrt) + linear side reactions
        q_lli = k_lli * np.sqrt(nn) + k_lin * nn
        alive = alive_at(nn)
        if n % n_check == 0 or n == 1 or n >= n_eol:
            perc = _percolating_nodes(n_node, edges, alive, kind, depth,
                                      subnet=direct)
            iso_now = 1.0 - perc[:n_am].mean()
            if lam_baseline is None:
                # the as-built RVE retains a small fraction of geometrically
                # isolated particles (finite-size effect); degradation is the
                # increase of isolation above this as-built reference
                lam_baseline = iso_now
            lam_am = max(0.0, iso_now - lam_baseline)
            perc_last, alive_last = perc, alive
        else:
            lam_am = rec["LAM"][-1]
        soh = float(np.clip(1.0 - q_lli - lam_am, 0.0, 1.0))
        #   the series resistance rises with the interphase (SEI on both
        #   electrodes), with isolated active material and with the loss of
        #   electronic contacts; the contact loss is counted on the electronic
        #   subnetwork, because losing an electrolyte-wetted path does not by
        #   itself raise the electronic resistance
        f_el = float(alive[direct].mean()) if direct.any() else 1.0
        cl = float(1.0 - f_el)
        r_rel = 1.0 + 2.6 * q_lli + 1.9 * lam_am + 0.55 * cl
        tmax = protocol["T_celsius"] + protocol["dt_rise"] * (1.0 + 0.6 * cl) * c_rate
        rec["n"].append(nn); rec["Q"].append(soh * Q0); rec["SOH"].append(soh)
        rec["R_rel"].append(r_rel); rec["LLI"].append(q_lli)
        rec["LAM"].append(lam_am); rec["CL"].append(cl)
        rec["edge_ret"].append(f_el)
        rec["edge_ret_all"].append(float(alive.mean()))
        rec["T_max"].append(tmax); rec["n_alive"].append(int(alive.sum()))
        if soh <= 0.80:
            break
        n += 1
        if n > n_eol:
            break

    for k in rec:
        rec[k] = np.asarray(rec[k])

    # ---- prediction-point topology ----------------------------------------
    #   The node attributes must describe the electrode *at the prediction
    #   cycle*.  Taking the end-of-life surviving network would inject the
    #   future rupture history into the input and leak the target.
    n_pt = max(int(n_pred), 1) if n_pred else int(rec["n"][-1])
    alive_pred = alive_at(n_pt)
    perc_pred = _percolating_nodes(n_node, edges, alive_pred, kind, depth,
                                   subnet=direct)
    q_lli_pred = float(k_lli * np.sqrt(n_pt) + k_lin * n_pt)

    return dict(record=rec, alive=alive_last, percolating=perc_last,
                alive_pred=alive_pred, percolating_pred=perc_pred,
                n_pred=int(n_pt), q_lli_pred=q_lli_pred,
                eps_amp=eps_amp, grad_li=grad_li, j_local=j_local,
                soc_local=soc_local, k_lli=k_lli, k_lam=k_lam, k_cl=k_cl,
                fat_scatter=fat)


# ============================================================================
#  5.  Node / edge feature assembly for the GNN
# ============================================================================
def assemble_node_features(graph: dict, sim: dict, soc_ref: float,
                           p: dict = P, use_pred: bool = True) -> np.ndarray:
    """8 morphological / physical node descriptors (analogous to the
    pore-descriptor set used for concrete, but for an LFP electrode):
       0 particle diameter              (um)
       1 inter-particle spacing         (um)
       2 interphase thickness           (nm)
       3 Li-concentration gradient      (norm.)
       4 coordination number            (count)
       5 contact quality (mean sigma_c) (-)
       6 through-thickness depth        (-)
       7 local state of charge          (-)

    With ``use_pred`` (the default) every connectivity attribute is evaluated
    on the network that survives at the prediction cycle, so the attributes
    contain no information about ruptures that have not yet occurred.
    """
    n = graph["n_node"]
    centres, radii, box = graph["centres"], graph["radii"], graph["box"]
    edges = graph["edges"]
    alive = sim["alive_pred"] if use_pred else sim["alive"]

    # inter-particle spacing: mean surface-to-surface gap of live neighbours
    spacing = np.full(n, np.nan)
    z = np.zeros(n)
    sig = np.zeros(n)
    cnt = np.zeros(n)
    e = edges[alive]
    if e.size:
        gap = graph["gap"][alive]
        sc = graph["sigma_c"][alive]
        acc = np.zeros(n); acc2 = np.zeros(n)
        np.add.at(acc, e[:, 0], gap); np.add.at(acc2, e[:, 1], gap)
        np.add.at(sig, e[:, 0], sc);  np.add.at(sig, e[:, 1], sc)
        np.add.at(cnt, e[:, 0], 1);   np.add.at(cnt, e[:, 1], 1)
        spacing = np.where(cnt > 0, (acc + acc2) / np.maximum(cnt, 1), graph["gap"].max())
    else:
        spacing = np.full(n, graph["gap"].max() if graph["gap"].size else p["gap_wetted"])
    z = cnt
    contact_q = np.where(cnt > 0, sig / np.maximum(cnt, 1), 0.0)

    # ---- interphase thickness at the prediction point ---------------------
    #   Two physical drivers enter.  Surface-limited growth makes the local
    #   film thickness scale with the surface-to-volume ratio of the active
    #   phase, (r_ref / r_i)^alpha.  The local side-reaction rate that grows the
    #   film, however, is set by the local reaction current, and for LFP that
    #   current is strongly non-uniform through the electrode thickness because
    #   the flat open-circuit-potential plateau prevents the reaction from
    #   homogenising; the film is therefore thickest on the small particles
    #   nearest the separator.  The second factor, (j_local / j_mean)^gamma, is
    #   what makes the attribute depend on the through-thickness position as
    #   well as on the particle size.
    #
    #   Without the current factor the attribute collapses onto the particle
    #   diameter: since alpha is applied to the same r that forms attribute 1,
    #   the rank correlation between attribute 3 and attribute 1 is exactly
    #   -1.000 over the whole population (no clipping occurs in the realised
    #   size range), so the network would receive eight inputs of which only
    #   seven are independent, and the manuscript would describe it as eight
    #   distinct measurable quantities.
    #
    #   The absolute scale is anchored to the measured thickness on a
    #   mean-size particle at the mean current (t_SEI_ref = 30 nm, between the
    #   as-built 5 nm and the fully aged 50-60 nm reported by TEM), and the
    #   result is bounded by those two measured limits.  The attribute is a
    #   property of the electrode at the prediction point and carries no
    #   information about the cell's future.
    r_am_all = radii[:graph["n_am"]]
    r_ref_am = float(r_am_all.mean()) if r_am_all.size else p["t_SEI_ref"]
    sv = (r_ref_am / np.maximum(r_am_all, 1e-6)) ** p["alpha_sei"]
    jl = np.asarray(sim["j_local"], float)
    if jl.size == r_am_all.size and float(np.mean(jl)) > 1e-12:
        jl_norm = jl / float(np.mean(jl))
    else:
        jl_norm = np.ones(r_am_all.size)
    sv = sv * jl_norm ** p["gamma_sei"]
    t_sei = np.full(n, p["t_SEI0"] * 1e3)
    t_sei[:graph["n_am"]] = np.clip(p["t_SEI_ref"] * 1e3 * sv,
                                    p["t_SEI0"] * 1e3,
                                    p["t_SEI_max"] * 1e3)

    grad = sim["grad_li"]
    grad_full = np.zeros(n); grad_full[:len(grad)] = grad
    soc_full = np.zeros(n)
    soc_full[:len(sim["soc_local"])] = np.clip(sim["soc_local"], 0, 1)
    # CBD nodes inherit the local average from their AM neighbours
    cbd = np.where(graph["kind"] == 1)[0]
    if cbd.size:
        tree = cKDTree(centres[:graph["n_am"]])
        _, idx = tree.query(centres[cbd], k=min(4, graph["n_am"]))
        idx = np.atleast_2d(idx)
        grad_full[cbd] = grad_full[idx].mean(axis=1)
        soc_full[cbd] = soc_full[idx].mean(axis=1)
        t_sei[cbd] = p["t_SEI0"] * 1e3

    F = np.stack([2 * radii, spacing, t_sei, grad_full,
                  z, contact_q, graph["depth"], soc_full], 1).astype(np.float32)
    return F


def graph_descriptors(graph: dict, sim: dict,
                      use_pred: bool = True) -> dict:
    """Global (readout-level) topological descriptors used for reporting.

    By default they are evaluated at the prediction cycle, so that the
    reported topology is the one available when the prognosis is made.
    """
    n_am = graph["n_am"]
    alive = sim["alive_pred"] if use_pred else sim["alive"]
    perc = sim["percolating_pred"] if use_pred else sim["percolating"]
    direct = graph["direct"]
    deg = np.zeros(graph["n_node"])
    e = graph["edges"][alive]
    if e.size:
        np.add.at(deg, e[:, 0], 1); np.add.at(deg, e[:, 1], 1)
    ret_el = float(alive[direct].mean()) if direct.any() else 1.0
    return dict(
        n_node=graph["n_node"], n_edge=int(alive.sum()),
        n_edge_el=int((alive & direct).sum()),
        #   retention of the electronic contacts: the quantity that governs
        #   both the series resistance and the isolation of active material
        edge_retention=ret_el,
        edge_retention_all=float(alive.mean()),
        mean_degree=float(deg[:n_am].mean()),
        isolated_frac=float(1.0 - perc[:n_am].mean()),
        mean_diameter=float(graph["dia"][:n_am].mean()),
    )
