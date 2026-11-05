"""Attention as gravity on the unit sphere: token embeddings cluster under softmax attraction.

Eight token embeddings live on the unit 2-sphere S² and evolve under the continuous-time
transformer self-attention flow dx_i/dt = P_{x_i}(Σ_j A_{ij} x_j), where A_{ij} =
softmax_j(⟨Q x_i, K x_j⟩ / τ). As the inverse-temperature schedule cools τ from 1.20 to 0.16,
diffuse geodesic attention arcs sharpen into high-weight bridges, pulling semantically related
tokens into tight semantic clusters while the live Shannon entropy curve H(A) drops in step.
"""

import numpy as np

import manimgx as m

N_TOKENS = 8
RADIUS = 2.05
LABELS = ["[CLS]", "양자", "중력", "시공간", "어텐션", "행렬", "다양체", "수렴"]
PALETTE = [
    "#ff6b6b",
    "#38bdf8",
    "#fbbf24",
    "#fb923c",
    "#34d399",
    "#a78bfa",
    "#2dd4bf",
    "#f472b6",
]


def sph_to_cart(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Unit vectors on S² from polar angle θ and azimuth φ."""
    st = np.sin(theta)
    return np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=-1)


THETA_INIT = np.array([0.62, 0.88, 1.12, 1.32, 1.85, 2.08, 2.25, 1.56])
PHI_INIT = np.array([-0.15, 0.55, 1.25, 1.95, -2.25, -1.45, -0.72, 2.85])
# Two semantic cluster centroids on S² (physics cluster vs geometry/attention cluster)
CLUSTER_TARGET = sph_to_cart(
    np.array([0.82, 0.90, 0.98, 1.06, 1.92, 2.00, 2.08, 1.84]),
    np.array([0.45, 0.68, 0.92, 1.15, -1.55, -1.32, -1.08, -1.78]),
)
AFFINITY_MASK = np.array(
    [
        [1.0 if (i < 4) == (j < 4) else -0.45 for j in range(N_TOKENS)]
        for i in range(N_TOKENS)
    ]
)

GLASS_MAT = m.Material(metallic=0.0, roughness=0.14, reflectance=0.85)
ORB_MAT = m.Material(metallic=0.25, roughness=0.18, reflectance=0.78)
TUBE_MAT = m.Material(metallic=0.20, roughness=0.22, reflectance=0.75)
PEDESTAL_MAT = m.Material(metallic=0.15, roughness=0.38, reflectance=0.55)


def softmax_attention(pos: np.ndarray, tau: float) -> np.ndarray:
    """Row-stochastic self-attention matrix A_ij = softmax_j((⟨Qx_i, Kx_j⟩) / τ)."""
    sim = ((pos @ pos.T) * AFFINITY_MASK) / max(tau, 0.12)
    np.fill_diagonal(sim, -0.35)
    sim -= sim.max(axis=1, keepdims=True)
    weights = np.exp(sim)
    return weights / weights.sum(axis=1, keepdims=True)


def attention_entropy(attn: np.ndarray) -> float:
    """Mean row Shannon entropy H(A) in bits."""
    safe = np.clip(attn, 1e-9, 1.0)
    return float(-np.sum(safe * np.log2(safe), axis=1).mean())


def slerp_geodesic(u: np.ndarray, v: np.ndarray, n_pts: int = 24) -> np.ndarray:
    """Great-circle arc on the sphere of radius RADIUS between unit vectors u and v."""
    dot = float(np.clip(np.dot(u, v), -0.9995, 0.9995))
    omega = np.arccos(dot)
    t = np.linspace(0.0, 1.0, n_pts)[:, None]
    if omega < 1e-4:
        pts = (1.0 - t) * u + t * v
    else:
        so = np.sin(omega)
        pts = (np.sin((1.0 - t) * omega) * u + np.sin(t * omega) * v) / so
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return RADIUS * 1.015 * pts


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


class AttentionManifold(m.ThreeDScene):
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
            theta=-44 * m.DEGREES,
            zoom=1.08,
            frame_center=np.array([-0.38, 0.08, -0.05]),
        )

        self.add(
            m.EnvironmentLight(intensity=0.82),
            m.SunLight(
                4.0 * m.UP + 3.0 * m.LEFT + 5.5 * m.OUT,
                intensity=1.35,
                shadows=True,
            ),
            m.PointLight(
                2.8 * m.RIGHT + 2.2 * m.DOWN + 2.5 * m.OUT,
                color="#38bdf8",
                intensity=3.5,
            ),
        )

        # ── Studio Pedestal Ring (3D Torus Base under the Sphere) ──
        pedestal_ring = m.Torus(
            major_radius=1.42,
            minor_radius=0.055,
            resolution=(48, 16),
        ).shift(np.array([0.0, 0.0, -RADIUS - 0.18]))
        pedestal_ring.set_color("#1e293b").set_material(PEDESTAL_MAT)
        pedestal_inner = m.Torus(
            major_radius=0.92,
            minor_radius=0.035,
            resolution=(40, 14),
        ).shift(np.array([0.0, 0.0, -RADIUS - 0.18]))
        pedestal_inner.set_color("#334155").set_material(ORB_MAT)

        # ── Dielectric Glass Riemann Sphere & Metallic Geodesic Rings ──
        globe = (
            m.Sphere(radius=RADIUS, resolution=(48, 32))
            .set_color("#1e3a8a")
            .set_opacity(0.28)
            .set_material(GLASS_MAT)
        )
        equator_pts = np.array(
            [
                [RADIUS * np.cos(t), RADIUS * np.sin(t), 0.0]
                for t in np.linspace(0, m.TAU, 96)
            ]
        )
        eq_verts, eq_tris = tube(equator_pts, 0.014, sides=8)
        equator_ring = m.MeshMobject(
            eq_verts, eq_tris, shade_in_3d=True, fill_color="#38bdf8"
        )
        equator_ring.set_opacity(0.45).set_material(TUBE_MAT)

        meridian_pts = np.array(
            [
                [RADIUS * np.cos(t), 0.0, RADIUS * np.sin(t)]
                for t in np.linspace(0, m.TAU, 96)
            ]
        )
        mer_verts, mer_tris = tube(meridian_pts, 0.012, sides=8)
        meridian_ring = m.MeshMobject(
            mer_verts, mer_tris, shade_in_3d=True, fill_color="#64748b"
        )
        meridian_ring.set_opacity(0.32).set_material(TUBE_MAT)

        # ── Continuous-Time Riemannian Attention Flow State ──
        tau_tracker = m.ValueTracker(1.20)
        coupling_tracker = m.ValueTracker(0.28)
        state_pos = sph_to_cart(THETA_INIT, PHI_INIT)
        anchor_pos = state_pos.copy()

        def step_manifold(dt: float) -> None:
            nonlocal state_pos
            if dt <= 0:
                return
            tau = float(tau_tracker.get_value())
            strength = float(coupling_tracker.get_value())
            cool_alpha = float(np.clip((1.20 - tau) / (1.20 - 0.16), 0.0, 1.0))
            attn = softmax_attention(state_pos, tau)
            goal = (1.0 - 0.72 * cool_alpha) * anchor_pos + (
                0.72 * cool_alpha
            ) * CLUSTER_TARGET
            target = 0.45 * (attn @ state_pos) + 0.55 * goal
            dots = np.sum(target * state_pos, axis=1, keepdims=True)
            tangent = target - dots * state_pos
            step = state_pos + strength * min(dt, 0.05) * 1.6 * tangent
            state_pos = step / np.linalg.norm(step, axis=1, keepdims=True)

        physics_driver = m.Mobject()
        physics_driver.add_updater(lambda mob, dt: step_manifold(dt))

        # ── 8 PBR Metallic/Ceramic Token Orbs ──
        token_orbs = m.Group()
        for i in range(N_TOKENS):
            orb = (
                m.Sphere(radius=0.145, resolution=(24, 16))
                .set_color(PALETTE[i])
                .set_material(ORB_MAT)
                .move_to(RADIUS * state_pos[i])
            )

            def update_orb(mob: m.Mobject, idx: int = i) -> None:
                mob.move_to(RADIUS * state_pos[idx])

            orb.add_updater(update_orb)
            token_orbs.add(orb)

        # ── Top-10 Geodesic Attention Bridges as 3D Cylindrical PBR Tubes ──
        pair_indices = [
            (0, 1, "#38bdf8"),
            (1, 2, "#38bdf8"),
            (2, 3, "#fbbf24"),
            (1, 3, "#60a5fa"),
            (4, 5, "#a78bfa"),
            (5, 6, "#2dd4bf"),
            (4, 6, "#34d399"),
            (6, 7, "#f472b6"),
            (0, 4, "#fb923c"),
            (3, 7, "#fbbf24"),
        ]
        arc_tubes = m.Group()
        for i_idx, j_idx, col in pair_indices:
            init_pts = slerp_geodesic(state_pos[i_idx], state_pos[j_idx])
            v0, t0 = tube(init_pts, 0.022, sides=10)
            mesh = m.MeshMobject(v0, t0, shade_in_3d=True, fill_color=col)
            mesh.set_material(TUBE_MAT)

            def update_tube(
                mob: m.Mobject,
                a: int = i_idx,
                b: int = j_idx,
            ) -> None:
                attn = softmax_attention(state_pos, float(tau_tracker.get_value()))
                w = float(0.5 * (attn[a, b] + attn[b, a]))
                rad = 0.010 + 0.042 * min(1.0, w * 3.2)
                pts = slerp_geodesic(state_pos[a], state_pos[b])
                verts, _ = tube(pts, rad, sides=10)
                mob.set_points(verts)
                mob.set_opacity(float(np.clip(0.28 + 2.1 * w, 0.22, 0.96)))

            mesh.add_updater(update_tube)
            arc_tubes.add(mesh)

        # ── Clean HUD & Live Synchronized 2D Entropy Plot (Top-Left / Bottom-Left) ──
        title = m.Text(
            "구면 다양체 위의 어텐션 동역학", font_size=30, weight="BOLD"
        ).to_corner(m.UL, buff=0.38)
        subtitle = (
            m.Text(
                "리만 구면 S² 위에서 소프트맥스 인력으로 군집화되는 토큰 궤적",
                font_size=18,
            )
            .set_color(m.GREY_B)
            .next_to(title, m.DOWN, aligned_edge=m.LEFT, buff=0.12)
        )

        formula = (
            m.MathTex(
                r"A_{ij} = \operatorname{softmax}_j\!\left(\frac{\langle Qx_i, Kx_j\rangle}{\tau}\right)",
                font_size=26,
            )
            .set_color("#7dd3fc")
            .to_corner(m.UR, buff=0.40)
        )

        # Live 2D Entropy Curve H(τ) on the left panel
        axes = (
            m.Axes(
                x_range=[0.15, 1.25, 0.25],
                y_range=[0.8, 3.1, 0.5],
                x_length=3.1,
                y_length=1.55,
                tips=False,
                axis_config={"stroke_width": 1.6, "color": m.GREY_B},
            )
            .to_corner(m.DL, buff=0.42)
            .shift(0.42 * m.UP)
        )

        axes_label = (
            m.Text("어텐션 엔트로피 H(A)", font_size=16)
            .set_color("#7dd3fc")
            .next_to(axes, m.UP, aligned_edge=m.LEFT, buff=0.12)
        )
        entropy_curve = axes.plot(
            lambda t: 1.05 + 1.85 / (1.0 + np.exp(-4.2 * (t - 0.52))),
            x_range=[0.16, 1.20],
            color="#38bdf8",
            stroke_width=3.2,
        )

        rider = m.Dot(radius=0.075, color="#fbbf24")
        rider.add_updater(
            lambda d: d.move_to(
                axes.c2p(
                    float(tau_tracker.get_value()),
                    1.05
                    + 1.85
                    / (1.0 + np.exp(-4.2 * (float(tau_tracker.get_value()) - 0.52))),
                )
            )
        )

        tau_lbl = m.MathTex(r"T =", font_size=28).set_color("#fbbf24")
        tau_val = m.DecimalNumber(
            1.20, num_decimal_places=2, font_size=28, color="#fbbf24"
        )
        tau_val.add_updater(lambda d: d.set_value(tau_tracker.get_value()))
        ent_lbl = m.MathTex(r"H(A) =", font_size=28).set_color("#34d399")
        ent_val = m.DecimalNumber(
            attention_entropy(softmax_attention(state_pos, 1.20)),
            num_decimal_places=2,
            font_size=28,
            color="#34d399",
        )
        ent_val.add_updater(
            lambda d: d.set_value(
                attention_entropy(
                    softmax_attention(state_pos, float(tau_tracker.get_value()))
                )
            )
        )
        tau_pair = m.VGroup(tau_lbl, tau_val).arrange(m.RIGHT, buff=0.12)
        ent_pair = m.VGroup(ent_lbl, ent_val).arrange(m.RIGHT, buff=0.12)
        readout = (
            m.VGroup(tau_pair, ent_pair)
            .arrange(m.RIGHT, buff=0.36)
            .next_to(formula, m.DOWN, aligned_edge=m.RIGHT, buff=0.22)
        )

        # Compact token legend below readout
        legend_rows = m.VGroup()
        for idx in (0, 1, 2, 4, 6, 7):
            dot = m.Dot(radius=0.055, color=PALETTE[idx])
            txt = m.Text(LABELS[idx], font_size=15).set_color(m.GREY_A)
            legend_rows.add(m.VGroup(dot, txt).arrange(m.RIGHT, buff=0.10))
        legend_grid = legend_rows.arrange_in_grid(
            rows=3, cols=2, buff=(0.28, 0.10), cell_alignment=m.LEFT
        ).next_to(readout, m.DOWN, aligned_edge=m.RIGHT, buff=0.24)

        self.add_fixed_in_frame_mobjects(
            title,
            subtitle,
            formula,
            readout,
            legend_grid,
            axes,
            axes_label,
            entropy_curve,
            rider,
        )
        self.remove(
            formula,
            readout,
            legend_grid,
            axes,
            axes_label,
            entropy_curve,
            rider,
        )

        # ── Act 1 (Close-Up Hook): Glass Sphere & 3D Geodesic Tubes Come Alive ──
        self.add(
            pedestal_ring,
            pedestal_inner,
            globe,
            equator_ring,
            meridian_ring,
            token_orbs,
            physics_driver,
        )
        self.say(
            "트랜스포머의 토큰 임베딩들이 3차원 [구면 다양체] 위에 놓여 있고, "
            "[자기 어텐션 수식]이 토큰 사이의 기하학적 인력을 결정합니다.",
            m.FadeIn(arc_tubes),
            m.FadeIn(m.VGroup(formula, readout, legend_grid)),
        )

        # ── Act 2: Synchronized 2D Entropy Graph + Camera Pullback ──
        self.say(
            "온도 파라미터가 높을 때는 [엔트로피 곡선]이 높은 곳에 머물며 "
            "[모든 토큰]이 구면 전체에서 고르게 소통합니다.",
            m.FadeIn(m.VGroup(axes, axes_label, entropy_curve, rider)),
            coupling_tracker.animate.set_value(0.55),
        )

        # ── Act 3: Temperature Annealing & Clustering Phase Transition ──
        self.begin_ambient_camera_rotation(rate=0.14)
        self.say(
            "이제 [온도를 0.18까지 낮추면] 어텐션 교량이 굵은 원통 튜브로 집중되면서, "
            "[의미가 가까운 토큰]들이 강한 중력처럼 하나의 군집으로 수렴합니다.",
            tau_tracker.animate.set_value(0.18),
            coupling_tracker.animate.set_value(1.25),
        )

        # ── Act 4: Climax Angle & Closing Synthesis ──
        closing = (
            m.Text(
                "트랜스포머의 깊은 층은 구면 위에서 정보를 응축하는 기하학적 흐름입니다.",
                font_size=21,
            )
            .set_color("#fef08a")
            .to_edge(m.DOWN, buff=0.28)
            .shift(1.1 * m.RIGHT)
        )
        self.add_fixed_in_frame_mobjects(closing)
        self.remove(closing)

        self.say(
            "결국 트랜스포머의 깊은 층을 통과한다는 것은, [구면 다양체 위에서] "
            "흩어진 문맥이 [하나의 기하학적 의미]로 응축되는 과정입니다.",
            m.FadeIn(closing, shift=0.15 * m.UP),
            tau_tracker.animate.set_value(0.16),
        )
        self.move_camera(
            phi=56 * m.DEGREES,
            zoom=1.02,
            frame_center=np.array([-0.32, 0.05, -0.05]),
            run_time=1.8,
        )
        self.wait(0.8)


if __name__ == "__main__":
    AttentionManifold().render("attention_manifold.mp4")
