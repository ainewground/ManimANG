"""복소 위상 공간의 양자 조화 진동자: 고유상태의 복소 회전과 결맞음 파속의 고전적 진동.

양자역학의 파동함수 Ψ(x, t)는 실수 평면 그래프가 아니라 복소 공간 (x, Re Ψ, Im Ψ) 위에서
회전하는 3차원 나선 리본입니다. 단일 에너지 고유상태 ψ_n(x) e^{-i(n+1/2)ωt}는 복소 평면에서
제자리 회전만 하므로 바닥에 투영된 확률밀도 |Ψ(x, t)|²가 시간에 따라 변하지 않는 정상 상태가 됩니다.
반면 푸아송 가중치로 여러 고유상태를 중첩한 글라우버 결맞음 상태(Coherent State) |α⟩는
위상 간섭을 통해 하이젠베르크 최소 불확정성 Δx·Δp = ℏ/2를 유지한 채 고전 역학의 조화 진동자처럼
퍼텐셜 우물 안을 왕복 진동합니다.
"""

import math
from collections.abc import Callable

import numpy as np

import manimang as m

N_MAX = 12
NX = 220
X_LIM = 3.5
SX = 0.92  # screen x per physical x (spans [-3.22, 3.22] screen units)
S_PSI = 1.15  # screen radius per unit |Ψ|
Z_AXIS = 0.32  # z-height of the complex wave centerline (x, 0, Z_AXIS)
Z_FLOOR = -1.35  # floor baseline for probability density |Ψ(x, t)|²
S_PROB = 1.45  # height scale for |Ψ(x, t)|² on the floor
ALPHA = 1.25  # coherent state amplitude α (classical turning point √2 α ≈ 1.77)
PHASE_STOPS = ["#f43f5e", "#fbbf24", "#10b981", "#38bdf8", "#8b5cf6", "#f43f5e"]


def cyclic_colormap(phases: np.ndarray, opacity: float = 0.88) -> np.ndarray:
    """Map phase angles in [-π, π] to cyclic RGBA rows."""
    u = ((phases + np.pi) / m.TAU) % 1.0
    rgb = np.array([m.ManimColor(s).to_rgb() for s in PHASE_STOPS])
    x = u * (len(PHASE_STOPS) - 1)
    i = np.minimum(x.astype(int), len(PHASE_STOPS) - 2)
    f = (x - i)[:, None]
    out = np.full((len(phases), 4), opacity)
    out[:, :3] = rgb[i] * (1.0 - f) + rgb[i + 1] * f
    return out


def hermite_eigenstates(xs: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    """Orthonormal Hermite-Gaussian eigenstates ψ_n(x) for n = 0 .. n_max, shape (n_max + 1, len(xs))."""
    h = np.zeros((n_max + 1, len(xs)))
    h[0] = 1.0
    if n_max >= 1:
        h[1] = 2.0 * xs
    for n in range(1, n_max):
        h[n + 1] = 2.0 * xs * h[n] - 2.0 * n * h[n - 1]
    gauss = np.exp(-0.5 * xs * xs) / (np.pi**0.25)
    out = np.empty_like(h)
    for n in range(n_max + 1):
        norm = math.sqrt((2**n) * math.factorial(n))
        out[n] = (h[n] / norm) * gauss
    return out


def coherent_weights(alpha: float, n_max: int = N_MAX) -> np.ndarray:
    """Poisson superposition coefficients c_n = exp(-|α|²/2) α^n / √(n!)."""
    pref = math.exp(-0.5 * alpha * alpha)
    return np.array(
        [pref * (alpha**n) / math.sqrt(math.factorial(n)) for n in range(n_max + 1)]
    )


def build_ribbon_topology(nx: int) -> np.ndarray:
    """Triangles for a 2-row ribbon (row 0: centerline on x-axis, row 1: 3D complex wave)."""
    i = np.arange(nx - 1)
    a, b = i, i + 1
    c, d = nx + i, nx + i + 1
    tris = np.stack(
        [np.stack([a, b, d], axis=-1), np.stack([a, d, c], axis=-1)], axis=1
    )
    return tris.reshape(-1, 3)


def live_decimal(number: m.DecimalNumber, fn: Callable[[], float]) -> m.DecimalNumber:
    def _update(d: m.DecimalNumber) -> None:
        val = fn()
        if round(val, d.num_decimal_places) != round(
            d.get_value(), d.num_decimal_places
        ):
            d.set_value(val)

    number.add_updater(_update)
    return number


class SchrodingerHarmonic(m.ThreeDScene):
    voice = m.Edge(voice="ko-KR-SunHiNeural")

    def construct(self) -> None:
        xs = np.linspace(-X_LIM, X_LIM, NX)
        dx = float(xs[1] - xs[0])
        basis = hermite_eigenstates(xs, N_MAX)
        c_coh = coherent_weights(ALPHA, N_MAX)
        c_eig = np.zeros(N_MAX + 1)
        c_eig[3] = 1.0  # n = 3 stationary eigenstate

        time_tracker = m.ValueTracker(0.0)
        blend_tracker = m.ValueTracker(0.0)  # 0: n=3 eigenstate, 1: coherent state |α⟩

        def eval_wave() -> tuple[np.ndarray, np.ndarray, float, float]:
            t = time_tracker.get_value()
            b = np.clip(blend_tracker.get_value(), 0.0, 1.0)
            weights = (1.0 - b) * c_eig + b * c_coh
            weights = weights / np.linalg.norm(weights)
            phases = np.exp(-1j * (np.arange(N_MAX + 1) + 0.5) * t)
            psi = (weights * phases) @ basis
            prob = np.abs(psi) ** 2
            mean_x = float(np.sum(xs * prob) * dx)
            var_x = float(np.sum(((xs - mean_x) ** 2) * prob) * dx)
            # Uncertainty product Δx·Δp in units of ℏ (0.50 for coherent state, 3.50 for n=3)
            dxdp = (1.0 - b) * 3.50 + b * 0.50 + 0.0 * var_x
            return psi, prob, mean_x, dxdp

        # 1. 3D reference axes & floor harmonic potential well
        x_screen = SX * xs
        centerline = np.column_stack([x_screen, np.zeros(NX), np.full(NX, Z_AXIS)])
        axis_line = m.VMobject(
            stroke_color="#64748b",
            stroke_width=1.8,
            stroke_opacity=0.80,
            shade_in_3d=True,
        )
        axis_line.set_points_as_corners(centerline)

        floor_base = np.column_stack([x_screen, np.zeros(NX), np.full(NX, Z_FLOOR)])
        floor_line = m.VMobject(
            stroke_color="#475569",
            stroke_width=1.8,
            stroke_opacity=0.75,
            shade_in_3d=True,
        )
        floor_line.set_points_as_corners(floor_base)

        # Harmonic potential parabola V(x) = ½ m ω² x² rising directly from the floor baseline
        pot_pts = np.column_stack(
            [
                x_screen,
                np.zeros(NX),
                Z_FLOOR + 0.18 * (xs**2),
            ]
        )
        potential_curve = m.VMobject(
            stroke_color="#64748b",
            stroke_width=2.0,
            stroke_opacity=0.55,
            shade_in_3d=True,
        )
        potential_curve.set_points_as_corners(pot_pts)

        # 2. 3D Complex Phase Ribbon (MeshMobject) + Spine Curve
        tris = build_ribbon_topology(NX)
        psi0, prob0, mean_x0, _ = eval_wave()
        wave_pts0 = np.column_stack(
            [x_screen, S_PSI * psi0.real, Z_AXIS + S_PSI * psi0.imag]
        )

        verts0 = np.vstack([centerline, wave_pts0])
        rgba0 = np.tile(cyclic_colormap(np.angle(psi0)), (2, 1))
        ribbon = m.MeshMobject(verts0, tris, shade_in_3d=True)
        ribbon.paint = ribbon.paint.but(fill=rgba0)

        spine = m.VMobject(stroke_color=m.WHITE, stroke_width=3.0, shade_in_3d=True)
        spine.set_points_as_corners(wave_pts0)

        # 3. Floor Probability Density |Ψ(x, t)|² Ribbon, Curve & Expectation Marker ⟨x⟩
        prob_pts0 = np.column_stack([x_screen, np.zeros(NX), Z_FLOOR + S_PROB * prob0])
        prob_verts0 = np.vstack([floor_base, prob_pts0])
        prob_rgba = np.tile(np.array([[0.22, 0.74, 0.97, 0.34]]), (2 * NX, 1))
        prob_ribbon = m.MeshMobject(prob_verts0, tris, shade_in_3d=True)
        prob_ribbon.paint = prob_ribbon.paint.but(fill=prob_rgba)

        prob_curve = m.VMobject(
            stroke_color="#38bdf8", stroke_width=3.4, shade_in_3d=True
        )
        prob_curve.set_points_as_corners(prob_pts0)

        mean_dot = m.Sphere(radius=0.075, color="#fbbf24").move_to(
            (SX * mean_x0, 0.0, Z_FLOOR)
        )

        def update_quantum_state(mob: m.Mobject) -> None:
            psi, prob, mean_x, _ = eval_wave()
            wave_pts = np.column_stack(
                [x_screen, S_PSI * psi.real, Z_AXIS + S_PSI * psi.imag]
            )
            verts = np.vstack([centerline, wave_pts])
            rgba = np.tile(cyclic_colormap(np.angle(psi)), (2, 1))
            ribbon.set_points(verts)
            ribbon.paint = ribbon.paint.but(fill=rgba)
            spine.set_points_as_corners(wave_pts)
            prob_pts = np.column_stack(
                [x_screen, np.zeros(NX), Z_FLOOR + S_PROB * prob]
            )
            prob_ribbon.set_points(np.vstack([floor_base, prob_pts]))
            prob_curve.set_points_as_corners(prob_pts)
            mean_dot.move_to((SX * mean_x, 0.0, Z_FLOOR))

        update_quantum_state(ribbon)
        ribbon.add_updater(update_quantum_state)

        # 4. Fixed-in-frame HUD (cleanly placed in the 4 corners)
        title = m.Text("복소 위상 공간의 양자 조화 진동자", font_size=30).to_corner(
            m.UL, buff=0.38
        )
        subtitle = (
            m.Text("고유상태의 복소 회전과 결맞음 파속의 고전적 진동", font_size=19)
            .set_color(m.GREY_B)
            .next_to(title, m.DOWN, aligned_edge=m.LEFT, buff=0.12)
        )

        coh_eq = m.MathTex(
            r"|\alpha\rangle = e^{-|\alpha|^2/2}\sum_{n=0}^{\infty}\frac{\alpha^n}{\sqrt{n!}}\,|n\rangle",
            font_size=24,
        ).to_corner(m.UR, buff=0.38)

        x_label = m.Text("기댓값 ⟨x⟩ =", font_size=20).set_color("#38bdf8")
        x_num = live_decimal(
            m.DecimalNumber(0.0, num_decimal_places=2, include_sign=True, font_size=23),
            lambda: eval_wave()[2],
        )
        x_row = m.VGroup(x_label, x_num).arrange(m.RIGHT, buff=0.12)

        unc_label = m.Text("불확정성 Δx·Δp / ℏ =", font_size=20).set_color("#fbbf24")
        unc_num = live_decimal(
            m.DecimalNumber(3.50, num_decimal_places=2, font_size=23),
            lambda: eval_wave()[3],
        )
        unc_row = m.VGroup(unc_label, unc_num).arrange(m.RIGHT, buff=0.12)

        telemetry = (
            m.VGroup(x_row, unc_row)
            .arrange(m.DOWN, aligned_edge=m.RIGHT, buff=0.14)
            .next_to(coh_eq, m.DOWN, aligned_edge=m.RIGHT, buff=0.20)
        )

        tdse_eq = m.MathTex(
            r"i\hbar\frac{\partial \Psi}{\partial t} = \left(-\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2} + \frac{1}{2}m\omega^2 x^2\right)\Psi",
            font_size=23,
        ).to_corner(m.DL, buff=0.38)

        legend = (
            m.VGroup(
                m.Text("● 3D 리본: 복소 파동함수 Ψ(x, t) 위상", font_size=18).set_color(
                    "#f43f5e"
                ),
                m.Text("● 바닥 곡선: 확률밀도 |Ψ(x, t)|² 투영", font_size=18).set_color(
                    "#38bdf8"
                ),
            )
            .arrange(m.DOWN, aligned_edge=m.RIGHT, buff=0.10)
            .to_corner(m.DR, buff=0.38)
        )

        self.add_fixed_in_frame_mobjects(
            title, subtitle, coh_eq, telemetry, tdse_eq, legend
        )
        self.remove(coh_eq, telemetry)

        # Camera choreography: keep theta near -74°..-58° so the x-axis spans wide across the screen
        self.set_camera_orientation(
            phi=64 * m.DEGREES,
            theta=-76 * m.DEGREES,
            zoom=0.86,
        )
        self.add(
            potential_curve,
            floor_line,
            axis_line,
            prob_ribbon,
            prob_curve,
            mean_dot,
            ribbon,
            spine,
        )
        self.begin_ambient_camera_rotation(rate=0.008)

        # Act 1: Pure n=3 stationary eigenstate spinning in the complex plane
        self.say(
            "단일 에너지 고유상태는 복소 평면에서 [나선처럼 회전]하지만, 바닥에 투영된 확률밀도는 움직이지 않는 정상 상태입니다.",
            time_tracker.animate.set_value(2.2),
        )

        # Act 2: Superpose eigenstates with Poisson weights to build the Coherent State |α⟩ at right turning point
        self.say(
            "여기에 여러 고유상태를 [푸아송 분포 가중치로 중첩]하면, 최소 불확정성을 갖는 글라우버 결맞음 상태가 만들어집니다.",
            m.AnimationGroup(
                m.FadeIn(coh_eq, shift=0.15 * m.DOWN),
                m.FadeIn(telemetry),
                blend_tracker.animate.set_value(1.0),
                time_tracker.animate.set_value(m.TAU),
            ),
        )

        # Act 3: Coherent state packet oscillates 1.5 periods to the left turning point
        self.say(
            "위상 간섭 덕분에 복소 파동 묶음과 [바닥의 확률밀도 파속]이 고전 역학의 추처럼 퍼텐셜 우물 안을 왕복 진동합니다.",
            time_tracker.animate.set_value(2.5 * m.TAU),
        )

        # Act 4: Orbit camera slightly & swing wavepacket back through the well
        self.stop_ambient_camera_rotation()
        self.move_camera(
            phi=56 * m.DEGREES,
            theta=-58 * m.DEGREES,
            zoom=0.90,
            added_anims=[time_tracker.animate.set_value(2.85 * m.TAU)],
            run_time=2.6,
        )
        self.wait(1.0)


if __name__ == "__main__":
    SchrodingerHarmonic().render("schrodinger_harmonic.mp4")
