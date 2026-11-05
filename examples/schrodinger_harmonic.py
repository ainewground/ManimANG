"""Quantum harmonic oscillator coherent wavepacket: 3D complex helix and Ehrenfest sloshing.

A Glauber coherent state |α⟩ = exp(−|α|²/2) Σ_n (α^n / √n!) |n⟩ is built by superposing the
first N = 12 Hermite-Gauss eigenstates ψ_n(x) of the 1D quantum harmonic oscillator
iℏ ∂Ψ/∂t = (−ℏ²/2m ∂²/∂x² + ½mω²x²) Ψ. Plotted in 3D space (x, Re Ψ, Im Ψ) above a lit studio
floor that catches real shadows, the complex wavefunction winds as a gleaming 3D cylindrical
corkscrew while its floor shadow |Ψ(x, t)|² oscillates sinusoidally without spreading,
satisfying Ehrenfest's theorem ⟨x⟩(t) = √2 α₀ cos(ω t).
"""

import math

import numpy as np

import manimgx as m

N_MAX = 12
X_MIN, X_MAX = -3.6, 3.6
N_PTS = 200
X_GRID = np.linspace(X_MIN, X_MAX, N_PTS)
N_RIBBONS = 68
RIBBON_X = np.linspace(-3.1, 3.1, N_RIBBONS)

HELIX_MAT = m.Material(metallic=0.28, roughness=0.16, reflectance=0.84)
PROB_MAT = m.Material(metallic=0.35, roughness=0.18, reflectance=0.86)
FLOOR_MAT = m.Material(metallic=0.10, roughness=0.38, reflectance=0.52)
FIN_MAT = m.Material(metallic=0.12, roughness=0.26, reflectance=0.68)


def hermite_gauss_basis(x: np.ndarray, n_max: int) -> np.ndarray:
    """Orthonormal Hermite-Gauss eigenstates ψ_n(x) for n = 0 .. n_max − 1."""
    psi = np.zeros((n_max, len(x)))
    psi[0] = (np.pi**-0.25) * np.exp(-0.5 * x * x)
    if n_max > 1:
        psi[1] = np.sqrt(2.0) * x * psi[0]
    for n in range(1, n_max - 1):
        psi[n + 1] = (
            np.sqrt(2.0 / (n + 1)) * x * psi[n] - np.sqrt(n / (n + 1)) * psi[n - 1]
        )
    return psi


BASIS_GRID = hermite_gauss_basis(X_GRID, N_MAX)
BASIS_RIBBON = hermite_gauss_basis(RIBBON_X, N_MAX)


def coherent_weights(alpha0: float, mix: float) -> np.ndarray:
    """Interpolate from pure eigenstate n = 2 (mix = 0) to coherent state |α₀⟩ (mix = 1)."""
    c_pure = np.zeros(N_MAX)
    c_pure[2] = 1.0

    ns = np.arange(N_MAX)
    facts = np.array([math.factorial(int(n)) for n in ns], dtype=float)
    c_coh = np.exp(-0.5 * alpha0 * alpha0) * (alpha0**ns) / np.sqrt(facts)
    c_coh /= np.linalg.norm(c_coh)

    w = (1.0 - mix) * c_pure + mix * c_coh
    return w / np.linalg.norm(w)


def eval_psi(basis: np.ndarray, alpha0: float, mix: float, t: float) -> np.ndarray:
    """Complex wavefunction Ψ(x, t) = Σ_n c_n ψ_n(x) exp(−i(n + ½)t)."""
    weights = coherent_weights(alpha0, mix)
    energies = np.arange(N_MAX) + 0.5
    phases = np.exp(-1j * energies * t)
    return (weights * phases) @ basis


def phase_color(angle: float) -> m.ManimColor:
    """Map complex argument arg(Ψ) ∈ [−π, π] to a vivid cyclic spectrum."""
    u = (angle + np.pi) / (2.0 * np.pi)
    r = 0.52 + 0.46 * np.cos(2.0 * np.pi * (u - 0.05))
    g = 0.52 + 0.46 * np.cos(2.0 * np.pi * (u - 0.38))
    b = 0.56 + 0.42 * np.cos(2.0 * np.pi * (u - 0.72))
    return m.ManimColor(np.clip([r, g, b], 0.0, 1.0))


def tube(
    points: np.ndarray, radius: float, sides: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and triangles of a 3D cylindrical tube along a polyline."""
    tangent = np.gradient(points, axis=0)
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent /= np.maximum(norms, 1e-8)
    ref = (
        np.array([0.0, 0.0, 1.0])
        if abs(tangent[0, 2]) < 0.85
        else np.array([0.0, 1.0, 0.0])
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


def floor_grid_mesh(
    x_range: tuple[float, float] = (-4.0, 4.0),
    y_range: tuple[float, float] = (-2.3, 2.3),
    z_level: float = -1.32,
) -> m.MeshMobject:
    """Lit 3D studio floor plane at z = z_level to catch shadow maps from the 3D helix."""
    us = np.linspace(*x_range, 24)
    vs = np.linspace(*y_range, 16)
    uu, vv = np.meshgrid(us, vs, indexing="ij")
    zz = np.full_like(uu, z_level)
    pts = np.stack([uu, vv, zz], axis=-1)
    nu, nv = pts.shape[0] - 1, pts.shape[1] - 1
    idx = np.arange((nu + 1) * (nv + 1)).reshape(nu + 1, nv + 1)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    cells = np.stack([np.stack([a, b, c], -1), np.stack([a, c, d], -1)], 2)
    mesh = m.MeshMobject(
        pts.reshape(-1, 3),
        cells.reshape(-1, 3),
        shade_in_3d=True,
        fill_color="#0f172a",
    )
    return mesh.set_material(FLOOR_MAT)


class SchrodingerHarmonic(m.ThreeDScene):
    voice = m.QwenTTS("Aiden", language="Korean", seed=20261017, speed=1.18)

    def construct(self) -> None:
        # ── Studio PBR Camera & Lighting Setup ──
        self.camera.tone_mapping = "agx"
        self.camera.ambient_occlusion = 1.0
        self.camera.ambient_occlusion_radius = 0.38
        self.camera.bloom = 0.08
        self.camera.exposure = 1.12

        self.set_camera_orientation(
            phi=64 * m.DEGREES,
            theta=-54 * m.DEGREES,
            zoom=1.16,
            frame_center=np.array([0.35, 0.0, -0.15]),
        )

        self.add(
            m.EnvironmentLight(intensity=0.82),
            m.SunLight(
                3.5 * m.UP + 2.5 * m.LEFT + 6.0 * m.OUT,
                intensity=1.40,
                shadows=True,
            ),
            m.PointLight(
                np.array([2.2, -1.8, 2.0]),
                color="#38bdf8",
                intensity=3.6,
            ),
        )

        # ── Continuous 60Hz Quantum Time & Coherent Mixing Trackers ──
        t_clock = m.ValueTracker(0.0)
        mix_tracker = m.ValueTracker(0.0)
        alpha0 = 1.38
        scale_amp = 1.28
        y_wall = 1.82
        z_floor = -1.28

        clock_driver = m.Mobject()
        clock_driver.add_updater(lambda mob, dt: t_clock.increment_value(dt * 2.15))

        # ── Studio Floor Plane, Axis Spine & Parabolic Potential Tube ──
        studio_floor = floor_grid_mesh(z_level=z_floor - 0.04)

        spine_pts = np.column_stack(
            [X_GRID, np.zeros_like(X_GRID), np.zeros_like(X_GRID)]
        )
        sp_v, sp_t = tube(spine_pts, 0.015, sides=8)
        x_spine = m.MeshMobject(
            sp_v, sp_t, shade_in_3d=True, fill_color="#94a3b8"
        ).set_material(FIN_MAT)

        pot_pts = np.column_stack(
            [
                X_GRID,
                np.full_like(X_GRID, y_wall),
                z_floor + 0.19 * (X_GRID**2),
            ]
        )
        pv, pt = tube(pot_pts, 0.020, sides=8)
        potential_well = m.MeshMobject(
            pv, pt, shade_in_3d=True, fill_color="#64748b"
        ).set_material(FIN_MAT)

        # ── 3D Cylindrical Complex Corkscrew Tube Ψ(x, t) = (x, Re Ψ, Im Ψ) ──
        def current_helix_pts() -> np.ndarray:
            psi = eval_psi(
                BASIS_GRID,
                alpha0,
                float(mix_tracker.get_value()),
                float(t_clock.get_value()),
            )
            return np.column_stack([X_GRID, scale_amp * psi.real, scale_amp * psi.imag])

        hv0, ht0 = tube(current_helix_pts(), 0.038, sides=12)
        helix_tube = m.MeshMobject(
            hv0, ht0, shade_in_3d=True, fill_color="#38bdf8"
        ).set_material(HELIX_MAT)
        helix_tube.add_updater(
            lambda mob: mob.set_points(tube(current_helix_pts(), 0.038, sides=12)[0])
        )

        # ── Phase-Colored Radial Struts Connecting x-Spine to Complex Helix ──
        struts = m.VGroup()
        for k in range(N_RIBBONS):
            strut = m.Line3D(
                start=np.array([RIBBON_X[k], 0.0, 0.0]),
                end=np.array([RIBBON_X[k], 0.0, 0.2]),
                thickness=0.014,
                color="#38bdf8",
            )
            struts.add(strut)

        def update_struts(grp: m.VGroup) -> None:
            psi_r = eval_psi(
                BASIS_RIBBON,
                alpha0,
                float(mix_tracker.get_value()),
                float(t_clock.get_value()),
            )
            re_r = scale_amp * psi_r.real
            im_r = scale_amp * psi_r.imag
            ang_r = np.angle(psi_r)
            for k, line_mob in enumerate(grp):
                p0 = np.array([RIBBON_X[k], 0.0, 0.0])
                p1 = np.array([RIBBON_X[k], re_r[k], im_r[k]])
                if float(np.linalg.norm(p1 - p0)) < 0.03:
                    p1 = p0 + np.array([0.0, 0.0, 0.03])
                line_mob.become(
                    m.Line3D(
                        start=p0,
                        end=p1,
                        thickness=0.014,
                        color=phase_color(float(ang_r[k])),
                    )
                )

        struts.add_updater(update_struts)

        # ── 3D Probability Density Tube |Ψ(x, t)|² & Center-of-Mass <x>(t) Sphere ──
        def current_prob_pts() -> np.ndarray:
            psi = eval_psi(
                BASIS_GRID,
                alpha0,
                float(mix_tracker.get_value()),
                float(t_clock.get_value()),
            )
            dens = np.abs(psi) ** 2
            return np.column_stack(
                [
                    X_GRID,
                    np.full_like(X_GRID, y_wall - 0.08),
                    z_floor + 1.95 * dens,
                ]
            )

        pb_v0, pb_t0 = tube(current_prob_pts(), 0.030, sides=10)
        prob_tube = m.MeshMobject(
            pb_v0, pb_t0, shade_in_3d=True, fill_color="#fbbf24"
        ).set_material(PROB_MAT)
        prob_tube.add_updater(
            lambda mob: mob.set_points(tube(current_prob_pts(), 0.030, sides=10)[0])
        )

        # Classical Expectation Marker Sphere <x>(t) on the Floor
        expectation_orb = (
            m.Sphere(radius=0.11, resolution=(20, 14))
            .set_color("#fbbf24")
            .set_material(PROB_MAT)
        )

        def update_expectation(mob: m.Mobject) -> None:
            psi = eval_psi(
                BASIS_GRID,
                alpha0,
                float(mix_tracker.get_value()),
                float(t_clock.get_value()),
            )
            dens = np.abs(psi) ** 2
            dx = X_GRID[1] - X_GRID[0]
            x_mean = float(np.sum(X_GRID * dens) * dx)
            mob.move_to(np.array([x_mean, y_wall - 0.08, z_floor + 0.08]))

        expectation_orb.add_updater(update_expectation)

        # ── Clean HUD & Live 2D Phase-Space Portrait (<x>(t), <p>(t)) ──
        title = m.Text(
            "양자 조화 진동자와 결맞음 파속", font_size=30, weight="BOLD"
        ).to_corner(m.UL, buff=0.38)
        subtitle = (
            m.Text(
                "3D 복소 나선 파동함수 Ψ(x,t) 와 에렌페스트 정리에 따른 고전적 왕복",
                font_size=18,
            )
            .set_color(m.GREY_B)
            .next_to(title, m.DOWN, aligned_edge=m.LEFT, buff=0.12)
        )

        schrod_eq = (
            m.MathTex(
                r"i\hbar\frac{\partial \Psi}{\partial t} = \left(-\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2} + \frac{1}{2}m\omega^2 x^2\right)\Psi",
                font_size=24,
            )
            .set_color("#7dd3fc")
            .to_corner(m.UR, buff=0.40)
        )
        coherent_eq = (
            m.MathTex(
                r"|\alpha(t)\rangle = e^{-|\alpha_0|^2/2}\sum_{n=0}^{N}\frac{\alpha_0^n}{\sqrt{n!}}\,e^{-i(n+\frac{1}{2})\omega t}|n\rangle",
                font_size=22,
            )
            .set_color("#fde047")
            .next_to(schrod_eq, m.DOWN, aligned_edge=m.RIGHT, buff=0.18)
        )

        x_lbl = m.MathTex(r"\langle x \rangle(t) =", font_size=25).set_color("#fbbf24")
        x_num = m.DecimalNumber(
            0.0, num_decimal_places=2, font_size=25, color="#fbbf24", include_sign=True
        )
        x_num.add_updater(
            lambda d: d.set_value(
                float(
                    np.sum(
                        X_GRID
                        * (
                            np.abs(
                                eval_psi(
                                    BASIS_GRID,
                                    alpha0,
                                    float(mix_tracker.get_value()),
                                    float(t_clock.get_value()),
                                )
                            )
                            ** 2
                        )
                    )
                    * (X_GRID[1] - X_GRID[0])
                )
            )
        )
        readout = (
            m.VGroup(x_lbl, x_num)
            .arrange(m.RIGHT, buff=0.12)
            .next_to(coherent_eq, m.DOWN, aligned_edge=m.RIGHT, buff=0.20)
        )

        # Live 2D Phase-Space Portrait (<x>, <p>) in Bottom-Left
        phase_axes = m.Axes(
            x_range=[-2.4, 2.4, 1.2],
            y_range=[-2.4, 2.4, 1.2],
            x_length=2.3,
            y_length=2.3,
            tips=False,
            axis_config={"stroke_width": 1.6, "color": m.GREY_B},
        ).to_corner(m.DL, buff=0.45)
        phase_title = (
            m.Text("위상 공간 (⟨x⟩, ⟨p⟩) 고전 궤도", font_size=15)
            .set_color(m.GREY_B)
            .next_to(phase_axes, m.UP, aligned_edge=m.LEFT, buff=0.12)
        )
        r_orbit = 1.65
        phase_circle = m.ParametricFunction(
            lambda s: phase_axes.c2p(r_orbit * np.cos(s), -r_orbit * np.sin(s)),
            t_range=(0.0, m.TAU),
            color="#38bdf8",
            stroke_width=2.6,
        )
        phase_rider = m.Dot(radius=0.075, color="#fbbf24")
        phase_rider.add_updater(
            lambda d: d.move_to(
                phase_axes.c2p(
                    float(mix_tracker.get_value())
                    * r_orbit
                    * np.cos(float(t_clock.get_value())),
                    -float(mix_tracker.get_value())
                    * r_orbit
                    * np.sin(float(t_clock.get_value())),
                )
            )
        )

        self.add_fixed_in_frame_mobjects(
            title,
            subtitle,
            schrod_eq,
            coherent_eq,
            readout,
            phase_axes,
            phase_title,
            phase_circle,
            phase_rider,
        )
        self.remove(
            schrod_eq,
            coherent_eq,
            readout,
            phase_axes,
            phase_title,
            phase_circle,
            phase_rider,
        )

        # ── Act 1 (Close-Up Hook): 3D Cylindrical Complex Helix & Cast Shadows ──
        self.add(studio_floor, x_spine, potential_well, struts, clock_driver)
        self.say(
            "양자 조화 진동자의 파동함수는 실수부와 허수부를 합친 "
            "3차원 공간에서 [복소 나선 기둥]으로 회전하며 바닥에 그림자를 드리웁니다.",
            m.FadeIn(helix_tube),
        )

        # ── Act 2: Stationary Eigenstate n = 2 vs Probability Density Shadow ──
        self.say(
            "단일 고유 상태에서는 나선이 제자리에서 돌기 때문에, "
            "바닥에 투영된 [확률 밀도 곡선]은 시간에 따라 멈춰 있습니다.",
            m.FadeIn(
                m.Group(
                    prob_tube,
                    expectation_orb,
                    schrod_eq,
                    coherent_eq,
                    readout,
                )
            ),
        )

        # ── Act 3: Coherent Superposition & Ehrenfest Classical Oscillation ──
        self.begin_ambient_camera_rotation(rate=0.12)
        self.say(
            "하지만 열두 개의 고유 상태를 포아송 분포로 [결맞음 중첩]하면, "
            "파속이 퍼지지 않고 [위상 공간의 원 궤도]를 따라 고전 입자처럼 왕복합니다.",
            mix_tracker.animate.set_value(1.0),
            m.FadeIn(m.VGroup(phase_axes, phase_title, phase_circle, phase_rider)),
        )

        # ── Act 4: Climax Angle & Ehrenfest Theorem Closing Synthesis ──
        closing = (
            m.Text(
                "에렌페스트 정리: 미시적 위상 간섭이 거시적 고전 역학의 왕복 운동을 창발합니다.",
                font_size=21,
            )
            .set_color("#fef08a")
            .to_edge(m.DOWN, buff=0.28)
            .shift(1.0 * m.RIGHT)
        )
        self.add_fixed_in_frame_mobjects(closing)
        self.remove(closing)

        self.say(
            "이것이 [에렌페스트 정리]가 보여주는 양자역학에서 "
            "[고전 역학으로의 기하학적 다리]입니다.",
            m.FadeIn(closing, shift=0.15 * m.UP),
            coherent_eq.animate.set_color("#fbbf24"),
        )
        self.move_camera(
            phi=54 * m.DEGREES,
            zoom=1.06,
            frame_center=np.array([0.25, 0.0, -0.10]),
            run_time=1.8,
        )
        self.wait(0.8)


if __name__ == "__main__":
    SchrodingerHarmonic().render("schrodinger_harmonic.mp4")
