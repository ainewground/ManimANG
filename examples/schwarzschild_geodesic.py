"""Gravitational lensing and the photon sphere on Flamm's Schwarzschild paraboloid.

Null geodesics in the equatorial plane of a Schwarzschild black hole satisfy
(du/dφ)² = 1/b² − u²(1 − r_s u) with u = 1/r and critical impact parameter
b_c = (3√3 / 2) r_s ≈ 2.598 r_s. Embedded on Flamm's paraboloid z(r) = −2√(r_s(r − r_s)),
seven 3D tubular light rays with impact parameters spanning all three relativistic regimes —
hyperbolic gravitational lensing (b > b_c), unstable circular trapping on the photon sphere
r_ph = 1.5 r_s (b ≈ b_c), and plunge across the event horizon r = r_s (b < b_c) — race around
the curvature well alongside the live effective potential barrier V_eff(r).
"""

import numpy as np

import manimgx as m

RS = 0.65
B_CRIT = 1.5 * np.sqrt(3.0) * RS  # ≈ 1.6887
R_MIN = RS * 1.02
R_MAX = 3.45
RES = (84, 48)
FUNNEL_STOPS = ["#090d16", "#1e1b4b", "#3b0764", "#9d174d", "#f59e0b", "#fef08a"]

FUNNEL_MAT = m.Material(metallic=0.32, roughness=0.22, reflectance=0.78)
GOLD_MAT = m.Material(metallic=0.65, roughness=0.14, reflectance=0.92)
RAY_MAT = m.Material(metallic=0.22, roughness=0.18, reflectance=0.80)
HORIZON_MAT = m.Material(metallic=0.0, roughness=0.95, reflectance=0.02)


def colormap(values: np.ndarray, stops: list[str]) -> np.ndarray:
    """RGBA rows for normalized values in [0, 1]."""
    rgb = np.array([m.ManimColor(s).to_rgb() for s in stops])
    x = np.clip(values, 0.0, 1.0) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[:, None]
    out = np.ones((len(values), 4))
    out[:, :3] = rgb[i] * (1.0 - f) + rgb[i + 1] * f
    return out


def flamm_z(r: np.ndarray | float) -> np.ndarray | float:
    """Flamm's paraboloid embedding depth z(r): deep throat at r = r_s, rising to rim at r_max."""
    r_safe = np.maximum(r, RS)
    return -1.35 + 1.28 * np.sqrt(RS * (r_safe - RS))


def funnel_surface(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """u is azimuth φ ∈ [0, 2π], v is radius r ∈ [R_MIN, R_MAX]."""
    x = v * np.cos(u)
    y = v * np.sin(u)
    z = flamm_z(v)
    return np.stack([x, y, z], axis=-1)


def grid_mesh(points: np.ndarray) -> m.MeshMobject:
    """Smooth lit 3D mesh over an (nu + 1, nv + 1, 3) grid."""
    nu, nv = points.shape[0] - 1, points.shape[1] - 1
    idx = np.arange((nu + 1) * (nv + 1)).reshape(nu + 1, nv + 1)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    cells = np.stack([np.stack([a, b, c], -1), np.stack([a, c, d], -1)], 2)
    return m.MeshMobject(points.reshape(-1, 3), cells.reshape(-1, 3), shade_in_3d=True)


def tube(
    points: np.ndarray, radius: float, sides: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and triangles of a 3D cylindrical tube along a polyline."""
    tangent = np.gradient(points, axis=0)
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent /= np.maximum(norms, 1e-8)
    ref = (
        np.array([0.3, 0.5, 0.8])
        if abs(tangent[0, 2]) < 0.85
        else np.array([0.8, 0.2, 0.1])
    )
    seed = np.cross(tangent[0], ref)
    normals = [seed / max(float(np.linalg.norm(seed)), 1e-8)]
    for t in tangent[1:]:
        n = normals[-1] - np.dot(normals[-1], t) * t
        nl = float(np.linalg.norm(n))
        normals.append(n / nl if nl > 1e-8 else normals[-1])
    n_ = np.array(normals)
    b_ = np.cross(tangent, n_)
    a = np.linspace(0, m.TAU, sides, endpoint=False)
    ring = (
        np.cos(a)[None, :, None] * n_[:, None] + np.sin(a)[None, :, None] * b_[:, None]
    )
    verts = (points[:, None] + radius * ring).reshape(-1, 3)
    i = np.arange(len(points) - 1)[:, None]
    j = np.arange(sides)[None, :]
    k = (j + 1) % sides
    tris = np.stack(
        [
            np.stack([i * sides + j, (i + 1) * sides + j, (i + 1) * sides + k], -1),
            np.stack([i * sides + j, (i + 1) * sides + k, i * sides + k], -1),
        ],
        2,
    ).reshape(-1, 3)
    return verts, tris


def integrate_null_geodesic(
    b: float, y_sign: float = 1.0, n_steps: int = 240
) -> np.ndarray:
    """Integrate the exact 2nd-order Schwarzschild null geodesic d²u/dφ² + u = (3/2) r_s u²."""
    r0 = R_MAX
    u = 1.0 / r0
    phi = np.pi - float(np.arcsin(np.clip(b / r0, -0.98, 0.98)))
    radicand = max(1.0 / (b * b) - u * u * (1.0 - RS * u), 0.0)
    du = float(np.sqrt(radicand))

    dphi = 0.024
    pts: list[list[float]] = []
    for _ in range(n_steps):
        r = 1.0 / max(u, 1e-4)
        if r > R_MAX * 1.03 and len(pts) > 20:
            break
        if r <= RS * 1.03:
            r_clamp = RS * 1.03
            pts.append(
                [
                    float(r_clamp * np.cos(phi)),
                    float(y_sign * r_clamp * np.sin(phi)),
                    float(flamm_z(r_clamp)) + 0.04,
                ]
            )
            break
        pts.append(
            [
                float(r * np.cos(phi)),
                float(y_sign * r * np.sin(phi)),
                float(flamm_z(r)) + 0.04,
            ]
        )

        def accel(u_val: float) -> float:
            return -u_val + 1.5 * RS * u_val * u_val

        k1_u = du
        k1_du = accel(u)
        k2_u = du + 0.5 * dphi * k1_du
        k2_du = accel(u + 0.5 * dphi * k1_u)
        k3_u = du + 0.5 * dphi * k2_du
        k3_du = accel(u + 0.5 * dphi * k2_u)
        k4_u = du + dphi * k3_du
        k4_du = accel(u + dphi * k3_u)

        u += (dphi / 6.0) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)
        du += (dphi / 6.0) * (k1_du + 2 * k2_du + 2 * k3_du + k4_du)
        phi -= dphi

    return np.array(pts)


class SchwarzschildGeodesic(m.ThreeDScene):
    voice = m.QwenTTS("Aiden", language="Korean", seed=20261017, speed=1.18)

    def construct(self) -> None:
        # ── Studio PBR Camera & Lighting Setup ──
        self.camera.tone_mapping = "agx"
        self.camera.ambient_occlusion = 1.0
        self.camera.ambient_occlusion_radius = 0.40
        self.camera.bloom = 0.10
        self.camera.exposure = 1.14

        self.set_camera_orientation(
            phi=56 * m.DEGREES,
            theta=-52 * m.DEGREES,
            zoom=1.08,
            frame_center=np.array([-0.45, 0.15, -0.22]),
        )

        self.add(
            m.EnvironmentLight(intensity=0.78),
            m.SunLight(
                4.2 * m.UP + 3.0 * m.LEFT + 5.5 * m.OUT,
                intensity=1.40,
                shadows=True,
            ),
            m.PointLight(
                np.array([0.0, 0.0, float(flamm_z(1.5 * RS)) + 0.35]),
                color="#fbbf24",
                intensity=4.5,
            ),
        )

        # ── 3D Flamm's Paraboloid PBR Mesh ──
        us = np.linspace(0.0, m.TAU, RES[0] + 1)
        vs = np.linspace(R_MIN, R_MAX, RES[1] + 1)
        uu, vv = np.meshgrid(us, vs, indexing="ij")
        pts_grid = funnel_surface(uu, vv)
        funnel = grid_mesh(pts_grid)
        potential = 1.0 - (vv.ravel() - R_MIN) / (R_MAX - R_MIN)
        funnel.paint = funnel.paint.but(
            fill=colormap(potential**1.35, FUNNEL_STOPS[::-1])
        )
        funnel.set_opacity(0.90).set_material(FUNNEL_MAT)

        # ── Event Horizon Black Hole Sphere & Golden 3D Photon Sphere Torus ──
        z_horizon = float(flamm_z(R_MIN))
        black_hole = (
            m.Sphere(radius=RS * 0.96, resolution=(32, 24))
            .set_color("#030712")
            .set_material(HORIZON_MAT)
            .move_to(np.array([0.0, 0.0, z_horizon - 0.08]))
        )

        r_ph = 1.5 * RS
        z_ph = float(flamm_z(r_ph)) + 0.035
        ph_pts = np.array(
            [
                [r_ph * np.cos(t), r_ph * np.sin(t), z_ph]
                for t in np.linspace(0, m.TAU, 96)
            ]
        )
        ph_verts, ph_tris = tube(ph_pts, 0.032, sides=12)
        photon_ring = m.MeshMobject(
            ph_verts, ph_tris, shade_in_3d=True, fill_color="#fbbf24"
        )
        photon_ring.set_material(GOLD_MAT)

        # ── 7 Relativistic Null Geodesics as 3D Cylindrical PBR Tubes + Animated Photons ──
        ray_specs = [
            (2.42, +1.0, "#38bdf8", 0.022),  # hyperbolic lensing (+y)
            (2.05, -1.0, "#38bdf8", 0.022),  # strong gravitational lensing (-y)
            (1.80, +1.0, "#60a5fa", 0.024),  # strong deflection (+y)
            (B_CRIT * 1.002, +1.0, "#fbbf24", 0.030),  # critical photon-sphere orbit
            (B_CRIT * 0.997, -1.0, "#f59e0b", 0.028),  # critical spiral capture
            (1.42, +1.0, "#fb7185", 0.024),  # direct capture (+y)
            (1.16, -1.0, "#f43f5e", 0.024),  # plunge into horizon (-y)
        ]

        lensing_tubes = m.Group()
        critical_tubes = m.Group()
        capture_tubes = m.Group()
        trajectories: list[tuple[np.ndarray, str]] = []

        for b_val, y_sgn, col, rad in ray_specs:
            traj = integrate_null_geodesic(b_val, y_sign=y_sgn)
            trajectories.append((traj, col))
            rv, rt = tube(traj, rad, sides=10)
            ray_mesh = m.MeshMobject(rv, rt, shade_in_3d=True, fill_color=col)
            ray_mesh.set_material(RAY_MAT)
            if b_val > B_CRIT * 1.03:
                lensing_tubes.add(ray_mesh)
            elif b_val < B_CRIT * 0.97:
                capture_tubes.add(ray_mesh)
            else:
                critical_tubes.add(ray_mesh)

        # Continuous 60Hz Photon Wavefront Orbs Racing Along Each Geodesic
        phase_clock = m.ValueTracker(0.0)
        clock_driver = m.Mobject()
        clock_driver.add_updater(lambda mob, dt: phase_clock.increment_value(dt * 0.52))

        photon_orbs = m.Group()
        for idx, (traj, col) in enumerate(trajectories):
            orb = (
                m.Sphere(radius=0.072, resolution=(16, 12))
                .set_color(col)
                .set_material(GOLD_MAT)
            )

            def update_photon(
                mob: m.Mobject,
                pts: np.ndarray = traj,
                offset: float = idx * 0.14,
            ) -> None:
                s = (float(phase_clock.get_value()) + offset) % 1.0
                pos_f = s * (len(pts) - 1)
                k = min(int(pos_f), len(pts) - 2)
                alpha = pos_f - k
                p = (1.0 - alpha) * pts[k] + alpha * pts[k + 1]
                mob.move_to(p)

            orb.add_updater(update_photon)
            photon_orbs.add(orb)

        # ── Clean HUD & Live 2D Effective Potential Plot V_eff(r) ──
        title = m.Text(
            "블랙홀 중력 렌즈와 광자 구", font_size=30, weight="BOLD"
        ).to_corner(m.UL, buff=0.38)
        subtitle = (
            m.Text(
                "플람 포물면 위의 영측지선(Null Geodesic)과 유효 퍼텐셜 장벽",
                font_size=18,
            )
            .set_color(m.GREY_B)
            .next_to(title, m.DOWN, aligned_edge=m.LEFT, buff=0.12)
        )

        geodesic_eq = (
            m.MathTex(
                r"\left(\frac{du}{d\varphi}\right)^{\!2} = \frac{1}{b^2} - u^2(1 - r_s u)",
                font_size=26,
            )
            .set_color("#fde047")
            .to_corner(m.UR, buff=0.40)
        )
        bcrit_eq = (
            m.MathTex(
                r"b_c = \frac{3\sqrt{3}}{2}\,r_s, \quad r_{\text{ph}} = 1.5\,r_s",
                font_size=24,
            )
            .set_color("#38bdf8")
            .next_to(geodesic_eq, m.DOWN, aligned_edge=m.RIGHT, buff=0.18)
        )

        # Compact regime legend under bcrit_eq
        leg_lens = m.VGroup(
            m.Dot(radius=0.055, color="#38bdf8"),
            m.Text("중력 렌즈 산란 (b > b_c)", font_size=15).set_color(m.GREY_A),
        ).arrange(m.RIGHT, buff=0.10)
        leg_crit = m.VGroup(
            m.Dot(radius=0.055, color="#fbbf24"),
            m.Text("광자 구 궤도 포획 (b ≈ b_c)", font_size=15).set_color(m.GREY_A),
        ).arrange(m.RIGHT, buff=0.10)
        leg_capt = m.VGroup(
            m.Dot(radius=0.055, color="#fb7185"),
            m.Text("사건의 지평선 추락 (b < b_c)", font_size=15).set_color(m.GREY_A),
        ).arrange(m.RIGHT, buff=0.10)
        legend = (
            m.VGroup(leg_lens, leg_crit, leg_capt)
            .arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.10)
            .next_to(bcrit_eq, m.DOWN, aligned_edge=m.RIGHT, buff=0.22)
        )

        # 2D Effective Potential Plot V_eff(r) in Bottom-Left
        pot_axes = (
            m.Axes(
                x_range=[0.6, 3.4, 0.7],
                y_range=[0.0, 0.42, 0.1],
                x_length=3.1,
                y_length=1.55,
                tips=False,
                axis_config={"stroke_width": 1.6, "color": m.GREY_B},
            )
            .to_corner(m.DL, buff=0.42)
            .shift(0.32 * m.UP)
        )
        pot_label = (
            m.Text("유효 퍼텐셜 장벽 V_eff(r)", font_size=16)
            .set_color("#fde047")
            .next_to(pot_axes, m.UP, aligned_edge=m.LEFT, buff=0.12)
        )
        pot_curve = pot_axes.plot(
            lambda r: (1.0 - RS / r) / (r * r) if r >= RS else 0.0,
            x_range=[RS, 3.35],
            color="#fbbf24",
            stroke_width=3.2,
        )
        peak_dot = m.Dot(
            pot_axes.c2p(1.5 * RS, (1.0 - 1.0 / 1.5) / ((1.5 * RS) ** 2)),
            radius=0.075,
            color="#fbbf24",
        )

        self.add_fixed_in_frame_mobjects(
            title,
            subtitle,
            geodesic_eq,
            bcrit_eq,
            legend,
            pot_axes,
            pot_label,
            pot_curve,
            peak_dot,
        )
        self.remove(
            geodesic_eq,
            bcrit_eq,
            legend,
            pot_axes,
            pot_label,
            pot_curve,
            peak_dot,
        )

        # ── Act 1 (Close-Up Hook): Flamm's Funnel & Golden Photon Sphere ──
        self.add(black_hole, clock_driver)
        self.say(
            "블랙홀 주변의 휘어진 시공간을 3차원 [플람 포물면]으로 펼치면, "
            "중력장 정점에 [황금빛 광자 구]가 나타납니다.",
            m.Create(funnel),
            m.FadeIn(photon_ring),
        )

        # ── Act 2: Effective Potential Barrier & Gravitational Lensing Tubes ──
        self.say(
            "[유효 퍼텐셜 장벽]보다 바깥으로 스쳐 지나가는 푸른 빛줄기는, "
            "곡률 우물을 따라 휘어지며 [중력 렌즈 현상]을 만듭니다.",
            m.FadeIn(
                m.VGroup(
                    geodesic_eq, bcrit_eq, pot_axes, pot_label, pot_curve, peak_dot
                )
            ),
            m.FadeIn(lensing_tubes),
        )

        # ── Act 3: Critical Photon Sphere Trapping & Event Horizon Plunge ──
        self.begin_ambient_camera_rotation(rate=0.14)
        self.say(
            "장벽 꼭대기에 도달한 [임계 광선]은 광자 구에 포획되어 공전하고, "
            "안쪽의 [붉은 광선]은 사건의 지평선 너머로 추락합니다.",
            m.FadeIn(m.VGroup(critical_tubes, legend)),
            m.FadeIn(m.Group(capture_tubes, photon_orbs)),
        )

        # ── Act 4: Top-Down Relativistic Orbit Climax ──
        closing = (
            m.Text(
                "임계 충격 변수 b_c 가 빛의 탈출과 영원한 포획의 경계를 가릅니다.",
                font_size=21,
            )
            .set_color("#fef08a")
            .to_edge(m.DOWN, buff=0.28)
            .shift(1.1 * m.RIGHT)
        )
        self.add_fixed_in_frame_mobjects(closing)
        self.remove(closing)

        self.say(
            "결국 [임계 충격 변수] 하나가 빛의 탈출과 영원한 포획을 가르는 "
            "[시공간의 기하학적 경계선]이 됩니다.",
            m.FadeIn(closing, shift=0.15 * m.UP),
            bcrit_eq.animate.set_color("#fbbf24"),
        )
        self.move_camera(
            phi=44 * m.DEGREES,
            zoom=1.02,
            frame_center=np.array([-0.35, 0.10, -0.18]),
            run_time=1.8,
        )
        self.wait(0.8)


if __name__ == "__main__":
    SchwarzschildGeodesic().render("schwarzschild_geodesic.mp4")
