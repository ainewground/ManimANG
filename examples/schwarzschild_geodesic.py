"""블랙홀 시공간 곡률과 빛의 측지선: 플람 포물면 위의 일반상대론적 영측지선과 광자구.

슈바르츠실트 블랙홀 주변의 적도면 공간 기하학은 플람 포물면(Flamm's Paraboloid)
z(r) = -2√(r_s(r - r_s))로 3차원 유클리드 공간에 등거리 매장(isometric embedding)됩니다.
빛(광자)은 휘어진 시공간의 영측지선 방정식 d²u/dφ² + u = (3/2) r_s u² (단, u = 1/r)를 따르며,
충격 계수 b가 임계값 b_crit = (3√3/2) r_s ≈ 2.598 r_s보다 크면 중력 렌즈 산란을 겪고,
임계값 부근에서는 광자구(r = 1.5 r_s)를 한 바퀴 이상 휘감아 돌며, 임계값보다 작으면 사건의 지평선
(r = r_s) 안으로 포획됩니다.
"""

from collections.abc import Callable

import numpy as np

import manimang as m

RS = 1.0  # Schwarzschild radius r_s = 2GM/c² = 1
R_MAX = 4.1
B_CRIT = 1.5 * np.sqrt(3.0) * RS  # 2.598076...
SCALE = 0.54  # screen units per r_s (keeps HUD margins completely clear)
OFFSET = np.array([0.0, -0.05, 0.38])
N_STEPS = 220

# Impact parameters b and regime labels (0: lensing escape, 1: photon sphere orbit, 2: capture)
RAYS: list[tuple[float, int]] = [
    (3.45, 0),
    (2.95, 0),
    (2.68, 0),
    (2.6015, 1),
    (2.5983, 1),
    (2.32, 2),
    (1.75, 2),
]
REGIME_COLORS = ["#38bdf8", "#fbbf24", "#fb7185"]
SURFACE_STOPS = ["#0f172a", "#1e3a8a", "#0284c7", "#f59e0b", "#f43f5e"]


def colormap(values: np.ndarray, stops: list[str], opacity: float = 0.92) -> np.ndarray:
    """RGBA rows for normalized values in [0, 1]."""
    rgb = np.array([m.ManimColor(s).to_rgb() for s in stops])
    x = np.clip(values, 0.0, 1.0) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[:, None]
    out = np.full((len(values), 4), opacity)
    out[:, :3] = rgb[i] * (1.0 - f) + rgb[i + 1] * f
    return out


def flamm_z(r: np.ndarray | float) -> np.ndarray:
    """Flamm's paraboloid height z(r) = 2√(r_s(r - r_s)) - offset so the throat sinks downward."""
    r_clamped = np.maximum(r, RS)
    z_rim = 2.0 * np.sqrt(RS * (R_MAX - RS))
    return np.asarray(2.0 * np.sqrt(RS * (r_clamped - RS)) - 0.62 * z_rim, dtype=float)


def to_screen(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Convert Schwarzschild coordinates (x, y, z) to centered 3D screen coordinates."""
    return OFFSET + SCALE * np.stack([x, y, z], axis=-1)


def build_flamm_mesh(nr: int = 52, nphi: int = 96) -> tuple[m.MeshMobject, np.ndarray]:
    """Smooth lit 3D MeshMobject for Flamm's paraboloid colored by curvature scalar."""
    # Concentrate radial samples near the throat r = r_s where slope is steep
    s = np.linspace(0.0, 1.0, nr + 1)
    rs = RS + (R_MAX - RS) * (s**1.75)
    phis = np.linspace(0.0, m.TAU, nphi + 1)
    rr, pp = np.meshgrid(rs, phis, indexing="ij")
    xx = rr * np.cos(pp)
    yy = rr * np.sin(pp)
    zz = flamm_z(rr)
    pts = to_screen(xx, yy, zz)

    idx = np.arange((nr + 1) * (nphi + 1)).reshape(nr + 1, nphi + 1)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    tris = np.stack(
        [np.stack([a, b, c], axis=-1), np.stack([a, c, d], axis=-1)], axis=2
    ).reshape(-1, 3)

    # Color by relativistic potential / Kretschmann curvature intensity
    strength = (RS / rr.ravel()) ** 1.6
    strength_norm = (strength - strength.min()) / np.ptp(strength)
    rgba = colormap(strength_norm, SURFACE_STOPS, opacity=0.88)

    mesh = m.MeshMobject(pts.reshape(-1, 3), tris, shade_in_3d=True)
    mesh.paint = mesh.paint.but(fill=rgba)
    return mesh, pts


def integrate_null_geodesic(b: float) -> np.ndarray:
    """Integrate the exact general-relativistic null geodesic d²u/dφ² + u = 1.5 r_s u²
    using RK4 from the outer rim r_0 = R_MAX, resampled uniformly to N_STEPS + 1 points
    on Flamm's paraboloid."""
    r0 = R_MAX - 0.02
    y0 = min(b * 0.92, r0 - 0.15)
    x0 = -float(np.sqrt(r0 * r0 - y0 * y0))
    phi0 = float(np.arctan2(y0, x0))
    u = 1.0 / r0
    # Exact GR null condition (du/dφ)² = 1/b² - u²(1 - r_s u), approaching throat (du/dφ < 0 for dφ < 0)
    du = -float(np.sqrt(max(1.0 / (b * b) - (u * u) * (1.0 - RS * u), 1e-8)))

    dphi = -0.004
    xs, ys, rs_list = [x0], [y0], [r0]
    phi = phi0
    for step in range(3200):

        def rhs(u_val: float, du_val: float) -> tuple[float, float]:
            return du_val, 1.5 * RS * (u_val**2) - u_val

        k1_u, k1_du = rhs(u, du)
        k2_u, k2_du = rhs(u + 0.5 * dphi * k1_u, du + 0.5 * dphi * k1_du)
        k3_u, k3_du = rhs(u + 0.5 * dphi * k2_u, du + 0.5 * dphi * k2_du)
        k4_u, k4_du = rhs(u + dphi * k3_u, du + dphi * k3_du)

        u += (dphi / 6.0) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)
        du += (dphi / 6.0) * (k1_du + 2 * k2_du + 2 * k3_du + k4_du)
        phi += dphi
        if u <= 0.0:
            break
        r = 1.0 / u
        if r <= 1.02 * RS or (step > 25 and r > R_MAX):
            r_clip = float(np.clip(r, 1.02 * RS, R_MAX))
            xs.append(r_clip * float(np.cos(phi)))
            ys.append(r_clip * float(np.sin(phi)))
            rs_list.append(r_clip)
            break
        xs.append(r * float(np.cos(phi)))
        ys.append(r * float(np.sin(phi)))
        rs_list.append(r)

    x_arr = np.array(xs)
    y_arr = np.array(ys)
    r_arr = np.array(rs_list)
    z_arr = flamm_z(r_arr) + 0.06  # lift slightly above surface so curves gleam clearly
    raw_pts = to_screen(x_arr, y_arr, z_arr)

    # Resample by arc length to N_STEPS + 1 points
    diffs = np.linalg.norm(np.diff(raw_pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(diffs)])
    targets = np.linspace(0.0, cum[-1], N_STEPS + 1)
    resampled = np.column_stack(
        [np.interp(targets, cum, raw_pts[:, dim]) for dim in range(3)]
    )
    return resampled


def photon_beads(
    centers: np.ndarray, radius: float = 0.065
) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and triangles of glowing photon spheres at `centers`."""
    lon, lat = np.meshgrid(
        np.linspace(0, m.TAU, 9)[:-1], np.linspace(0, np.pi, 6), indexing="ij"
    )
    unit = np.stack(
        [np.sin(lat) * np.cos(lon), np.sin(lat) * np.sin(lon), np.cos(lat)], axis=-1
    ).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(8), np.arange(5), indexing="ij")
    a, b = i * 6 + j, (i + 1) % 8 * 6 + j
    tris = np.concatenate(
        [np.stack([a, b, b + 1], axis=-1), np.stack([a, b + 1, a + 1], axis=-1)]
    ).reshape(-1, 3)
    verts = (centers[:, None, :] + radius * unit[None, :, :]).reshape(-1, 3)
    offsets = len(unit) * np.arange(len(centers))[:, None, None]
    return verts, (tris[None, :, :] + offsets).reshape(-1, 3)


def live_decimal(number: m.DecimalNumber, fn: Callable[[], float]) -> m.DecimalNumber:
    def _update(d: m.DecimalNumber) -> None:
        val = fn()
        if round(val, d.num_decimal_places) != round(
            d.get_value(), d.num_decimal_places
        ):
            d.set_value(val)

    number.add_updater(_update)
    return number


class SchwarzschildGeodesic(m.ThreeDScene):
    voice = m.Edge(voice="ko-KR-SunHiNeural")

    def construct(self) -> None:
        flamm_mesh, _ = build_flamm_mesh()
        paths = [integrate_null_geodesic(b) for b, _ in RAYS]
        regimes = [reg for _, reg in RAYS]

        # Event horizon ring (r = r_s = 1.0) and Photon sphere ring (r = 1.5 r_s)
        angles = np.linspace(0.0, m.TAU, 120)
        horizon_pts = to_screen(
            RS * np.cos(angles),
            RS * np.sin(angles),
            np.full_like(angles, float(flamm_z(RS)) + 0.03),
        )
        horizon_ring = m.VMobject(
            stroke_color="#f43f5e", stroke_width=3.2, shade_in_3d=True
        )
        horizon_ring.set_points_as_corners(horizon_pts)

        r_ph = 1.5 * RS
        photon_ring_pts = to_screen(
            r_ph * np.cos(angles),
            r_ph * np.sin(angles),
            np.full_like(angles, float(flamm_z(r_ph)) + 0.035),
        )
        photon_ring = m.VMobject(
            stroke_color="#fbbf24",
            stroke_width=2.6,
            stroke_opacity=0.9,
            shade_in_3d=True,
        )
        photon_ring.set_points_as_corners(photon_ring_pts)

        progress = m.ValueTracker(0.0)

        # Geodesic trajectory curves and leading photon spheres
        curves = m.VGroup(
            *[
                m.VMobject(
                    stroke_color=REGIME_COLORS[reg],
                    stroke_width=3.0 if reg == 1 else 2.3,
                    stroke_opacity=0.92,
                    shade_in_3d=True,
                )
                for reg in regimes
            ]
        )

        def update_curves(group: m.Mobject) -> None:
            upto = max(2, int(progress.get_value() * N_STEPS) + 1)
            for curve_mob, path in zip(group.submobjects, paths, strict=True):
                assert isinstance(curve_mob, m.VMobject)
                curve_mob.set_points_as_corners(path[:upto])

        update_curves(curves)
        curves.add_updater(update_curves)

        head_pts0 = np.array([p[0] for p in paths])
        v0, t0 = photon_beads(head_pts0)
        v_per = len(v0) // len(paths)
        head_rgba = np.ones((len(v0), 4))
        for idx_ray, reg in enumerate(regimes):
            head_rgba[idx_ray * v_per : (idx_ray + 1) * v_per, :3] = m.ManimColor(
                REGIME_COLORS[reg]
            ).to_rgb()
        photons = m.MeshMobject(v0, t0, shade_in_3d=True)
        photons.paint = photons.paint.but(fill=head_rgba)

        def update_photons(mob: m.Mobject) -> None:
            f = np.clip(progress.get_value(), 0.0, 1.0) * N_STEPS
            k = min(int(f), N_STEPS - 1)
            w = f - k
            heads = np.array([(1.0 - w) * p[k] + w * p[k + 1] for p in paths])
            mob.set_points(photon_beads(heads)[0])

        photons.add_updater(update_photons)

        # Fixed-in-frame HUD (placed cleanly in the 4 corners)
        title = m.Text("블랙홀 시공간 곡률과 빛의 측지선", font_size=30).to_corner(
            m.UL, buff=0.38
        )
        subtitle = (
            m.Text("플람 포물면 위의 일반상대론적 영측지선과 광자구", font_size=19)
            .set_color(m.GREY_B)
            .next_to(title, m.DOWN, aligned_edge=m.LEFT, buff=0.12)
        )

        geodesic_eq = m.MathTex(
            r"\frac{d^2 u}{d\phi^2} + u = \frac{3}{2}\,r_s u^2, \quad u = \frac{1}{r}",
            font_size=24,
        ).to_corner(m.UR, buff=0.38)

        bcrit_eq = (
            m.MathTex(
                r"b_{\mathrm{crit}} = \frac{3\sqrt{3}}{2}\,r_s \approx 2.60\,r_s",
                font_size=22,
            )
            .set_color("#fbbf24")
            .next_to(geodesic_eq, m.DOWN, aligned_edge=m.RIGHT, buff=0.16)
        )

        rmin_label = m.Text("임계 광자 최소 반경 r_min =", font_size=20).set_color(
            "#fbbf24"
        )
        # Track the running minimum radial distance min_{s<=t} r(s)/r_s of the critical ray
        crit_pts_unscaled = (paths[4] - OFFSET) / SCALE
        crit_radii = np.hypot(crit_pts_unscaled[:, 0], crit_pts_unscaled[:, 1])
        crit_running_min = np.minimum.accumulate(crit_radii)

        def current_crit_r() -> float:
            f = np.clip(progress.get_value(), 0.0, 1.0) * N_STEPS
            k = min(int(f), N_STEPS)
            return float(crit_running_min[k])

        rmin_num = live_decimal(
            m.DecimalNumber(current_crit_r(), num_decimal_places=2, font_size=23),
            current_crit_r,
        )
        rmin_row = (
            m.VGroup(rmin_label, rmin_num)
            .arrange(m.RIGHT, buff=0.12)
            .next_to(bcrit_eq, m.DOWN, aligned_edge=m.RIGHT, buff=0.16)
        )

        flamm_eq = m.MathTex(
            r"z(r) = 2\sqrt{r_s(r - r_s)}, \quad r_{\mathrm{ph}} = 1.5\,r_s",
            font_size=24,
        ).to_corner(m.DL, buff=0.38)

        legend = (
            m.VGroup(
                m.Text("● 중력 렌즈 산란 (b > 2.60 rₛ)", font_size=18).set_color(
                    "#38bdf8"
                ),
                m.Text("● 광자구 궤도 선회 (b ≈ 2.60 rₛ)", font_size=18).set_color(
                    "#fbbf24"
                ),
                m.Text("● 사건의 지평선 포획 (b < 2.60 rₛ)", font_size=18).set_color(
                    "#fb7185"
                ),
            )
            .arrange(m.DOWN, aligned_edge=m.RIGHT, buff=0.10)
            .to_corner(m.DR, buff=0.38)
        )

        self.add_fixed_in_frame_mobjects(
            title, subtitle, geodesic_eq, bcrit_eq, rmin_row, flamm_eq, legend
        )
        self.remove(geodesic_eq, bcrit_eq, rmin_row, legend)

        # Camera choreography
        self.set_camera_orientation(
            phi=60 * m.DEGREES,
            theta=-65 * m.DEGREES,
            zoom=0.82,
        )
        self.add(flamm_mesh, horizon_ring, photon_ring)
        self.begin_ambient_camera_rotation(rate=0.07)

        # Act 1: Introduce Flamm's paraboloid and the photon sphere
        self.say(
            "블랙홀 주변의 휘어진 적도면 시공간을 [플람 포물면]으로 펼치면, 노란색 광자구와 붉은 사건의 지평선이 나타납니다.",
            m.Indicate(photon_ring, color="#fbbf24"),
        )

        # Act 2: Show the exact relativistic null geodesic equation & impact regimes
        self.add(curves, photons)
        self.say(
            "빛은 일반상대성이론의 [영측지선 방정식]을 따르며, 충격 계수에 따라 세 가지 운명으로 갈라집니다.",
            m.AnimationGroup(
                m.FadeIn(geodesic_eq, shift=0.15 * m.DOWN),
                m.FadeIn(bcrit_eq),
                m.FadeIn(rmin_row),
                m.FadeIn(legend, shift=0.15 * m.UP),
            ),
        )

        # Act 3: Propagate the 7 light rays across the curved spacetime funnel
        self.say(
            "바깥쪽 빛은 중력 렌즈처럼 휘어지고, [임계 광자는 광자구를 한 바퀴 선회]하며, 안쪽 빛은 사건의 지평선으로 빨려 들어갑니다.",
            progress.animate.set_value(1.0),
        )

        # Act 4: Top-down orbital view of the Einstein deflection & photon sphere loop
        self.stop_ambient_camera_rotation()
        self.move_camera(
            phi=24 * m.DEGREES,
            theta=-90 * m.DEGREES,
            zoom=0.86,
            run_time=2.8,
        )
        self.wait(1.0)


if __name__ == "__main__":
    SchwarzschildGeodesic().render("schwarzschild_geodesic.mp4")
