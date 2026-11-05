"""트랜스포머 어텐션의 기하학: 구면 다양체 위의 자기-어텐션 흐름과 의미 군집화.

트랜스포머의 자기-어텐션(Self-Attention)을 연속 시간 동역학계로 보면, 단위 구면 S² 위의
토큰 임베딩 x_i들이 어텐션 가중치 A_ij(τ) = softmax(<q_i, k_j> / (√d_k · τ))를 따라 서로를
끌어당기는 리만 기하학적 흐름 dx_i/dt = P_{x_i}(∑_j A_ij(x) v_j)를 이룹니다 (Geshkovski et al., 2023).
온도 τ가 낮아질수록 균일하게 퍼져 있던 어텐션 엔트로피 H(A)가 감소하고, 구면 위에서 무질서하게
섞여 있던 토큰들이 측지선 아크를 따라 세 개의 선명한 의미 군집으로 응집됩니다.
"""

from collections.abc import Callable

import numpy as np

import manimang as m

N_TOKENS = 36  # 12 tokens per semantic cluster
STEPS = 160
RADIUS = 1.72  # sphere radius in screen units (keeps HUD margins completely clear)
CENTER = np.array([0.0, -0.08, 0.0])
CLUSTER_COLORS = ["#38bdf8", "#fbbf24", "#f43f5e"]  # 기하·공간, 대수·수식, 동역학·시간
TAU_START, TAU_END = 1.80, 0.18


def normalize(v: np.ndarray) -> np.ndarray:
    """Unit vectors along the last axis."""
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def attention_matrix(
    x: np.ndarray, anchors: np.ndarray, labels: np.ndarray, tau: float
) -> np.ndarray:
    """Exact scaled dot-product attention matrix A_ij(τ) on the sphere."""
    # Query and Key projections combine current position x_i with semantic anchor affinity
    q = normalize(0.65 * x + 0.55 * anchors[labels])
    k = q
    logits = (q @ k.T) / max(tau, 0.05)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def entropy(a: np.ndarray) -> float:
    """Mean row-wise Shannon entropy H(A) in nats."""
    clipped = np.clip(a, 1e-9, 1.0)
    return float(-np.mean(np.sum(clipped * np.log(clipped), axis=1)))


def alignment(x: np.ndarray, anchors: np.ndarray, labels: np.ndarray) -> float:
    """Mean cosine similarity between each token and its semantic cluster centroid."""
    return float(np.mean(np.sum(x * anchors[labels], axis=1)))


def simulate_attention_flow() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """RK4 integration of the Riemannian Transformer Attention ODE on S²."""
    rng = np.random.default_rng(42)
    # Three semantic anchors well-separated on the sphere
    anchors = normalize(
        np.array(
            [
                [0.82, 0.42, 0.38],
                [-0.78, 0.48, 0.40],
                [0.0, -0.86, -0.51],
            ]
        )
    )
    labels = np.repeat(np.arange(3), N_TOKENS // 3)
    # Initial token embeddings: widely dispersed across S² with mild bias toward anchors
    raw = 0.28 * anchors[labels] + 0.92 * rng.standard_normal((N_TOKENS, 3))
    x = normalize(raw)

    taus = np.geomspace(TAU_START, TAU_END, STEPS + 1)
    traj = np.empty((STEPS + 1, N_TOKENS, 3))
    attns = np.empty((STEPS + 1, N_TOKENS, N_TOKENS))
    entropies = np.empty(STEPS + 1)
    alignments = np.empty(STEPS + 1)

    dt = 0.042
    for step, tau in enumerate(taus):
        a = attention_matrix(x, anchors, labels, float(tau))
        traj[step] = x
        attns[step] = a
        entropies[step] = entropy(a)
        alignments[step] = alignment(x, anchors, labels)
        if step == STEPS:
            break

        def field(pos: np.ndarray, t_val: float) -> np.ndarray:
            att = attention_matrix(pos, anchors, labels, t_val)
            v = normalize(0.55 * pos + 0.65 * anchors[labels])
            pull = att @ v
            # Tangent projection P_x(u) = u - <u, x> x on S²
            tangent = pull - np.sum(pull * pos, axis=1, keepdims=True) * pos
            return 1.35 * tangent

        t0 = float(tau)
        t1 = float(taus[step + 1])
        tm = 0.5 * (t0 + t1)
        k1 = field(x, t0)
        k2 = field(normalize(x + 0.5 * dt * k1), tm)
        k3 = field(normalize(x + 0.5 * dt * k2), tm)
        k4 = field(normalize(x + dt * k3), t1)
        x = normalize(x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))

    return traj, attns, taus, entropies, alignments, labels


def sphere_beads(
    centers: np.ndarray, radius: float = 0.075
) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and triangles of lit 3D spheres at `centers` merged into one MeshMobject."""
    lon, lat = np.meshgrid(
        np.linspace(0, m.TAU, 11)[:-1], np.linspace(0, np.pi, 7), indexing="ij"
    )
    unit = np.stack(
        [np.sin(lat) * np.cos(lon), np.sin(lat) * np.sin(lon), np.cos(lat)], axis=-1
    ).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(10), np.arange(6), indexing="ij")
    a, b = i * 7 + j, (i + 1) % 10 * 7 + j
    tris = np.concatenate(
        [np.stack([a, b, b + 1], axis=-1), np.stack([a, b + 1, a + 1], axis=-1)]
    ).reshape(-1, 3)
    verts = (centers[:, None, :] + radius * unit[None, :, :]).reshape(-1, 3)
    offsets = len(unit) * np.arange(len(centers))[:, None, None]
    return verts, (tris[None, :, :] + offsets).reshape(-1, 3)


def geodesic_arc(p: np.ndarray, q: np.ndarray, n: int = 12) -> np.ndarray:
    """Points along the spherical geodesic arc from unit vector p to unit vector q."""
    s = np.linspace(0.0, 1.0, n)[:, None]
    chord = (1.0 - s) * p[None, :] + s * q[None, :]
    return CENTER + RADIUS * normalize(chord)


def live_decimal(number: m.DecimalNumber, fn: Callable[[], float]) -> m.DecimalNumber:
    """Update a DecimalNumber only when its displayed digits change."""

    def _update(d: m.DecimalNumber) -> None:
        val = fn()
        if round(val, d.num_decimal_places) != round(
            d.get_value(), d.num_decimal_places
        ):
            d.set_value(val)

    number.add_updater(_update)
    return number


class AttentionManifold(m.ThreeDScene):
    voice = m.Edge(voice="ko-KR-SunHiNeural")

    def construct(self) -> None:
        traj, attns, taus, entropies, alignments, labels = simulate_attention_flow()
        progress = m.ValueTracker(0.0)

        def state_at(p: float) -> tuple[np.ndarray, float, float, float, int]:
            f = np.clip(p, 0.0, 1.0) * STEPS
            k = min(int(f), STEPS - 1)
            w = f - k
            pos = normalize((1.0 - w) * traj[k] + w * traj[k + 1])
            tau_val = float((1.0 - w) * taus[k] + w * taus[k + 1])
            ent_val = float((1.0 - w) * entropies[k] + w * entropies[k + 1])
            aln_val = float((1.0 - w) * alignments[k] + w * alignments[k + 1])
            return CENTER + RADIUS * pos, tau_val, ent_val, aln_val, k

        # 1. Wireframe S² manifold grid
        wireframe = m.VGroup()
        angles = np.linspace(0, m.TAU, 96)
        for lat_deg in (-60, -30, 0, 30, 60):
            lat = np.radians(lat_deg)
            r_lat, z_lat = RADIUS * np.cos(lat), RADIUS * np.sin(lat)
            pts = CENTER + np.column_stack(
                [
                    r_lat * np.cos(angles),
                    r_lat * np.sin(angles),
                    np.full_like(angles, z_lat),
                ]
            )
            ring = m.VMobject(
                stroke_color="#334155",
                stroke_width=1.4 if lat_deg != 0 else 2.1,
                stroke_opacity=0.55,
                shade_in_3d=True,
            )
            ring.set_points_as_corners(pts)
            wireframe.add(ring)
        for lon_deg in range(0, 180, 30):
            lon = np.radians(lon_deg)
            pts = CENTER + RADIUS * np.column_stack(
                [
                    np.cos(angles) * np.cos(lon),
                    np.cos(angles) * np.sin(lon),
                    np.sin(angles),
                ]
            )
            meridian = m.VMobject(
                stroke_color="#334155",
                stroke_width=1.3,
                stroke_opacity=0.45,
                shade_in_3d=True,
            )
            meridian.set_points_as_corners(pts)
            wireframe.add(meridian)

        # 2. Geodesic attention arcs between top-attending token pairs
        pair_indices = [
            (i, (i + 1) % (N_TOKENS // 3) + c * (N_TOKENS // 3))
            for c in range(3)
            for i in range(c * (N_TOKENS // 3), (c + 1) * (N_TOKENS // 3))
        ]
        arcs = m.VGroup(
            *[
                m.VMobject(
                    stroke_color=CLUSTER_COLORS[labels[i]],
                    stroke_width=2.2,
                    stroke_opacity=0.55,
                    shade_in_3d=True,
                )
                for i, _ in pair_indices
            ]
        )

        def update_arcs(group: m.Mobject) -> None:
            pts, _, _, _, step_idx = state_at(progress.get_value())
            unit_pts = (pts - CENTER) / RADIUS
            a_mat = attns[step_idx]
            for arc_mob, (i, j) in zip(group.submobjects, pair_indices, strict=True):
                assert isinstance(arc_mob, m.VMobject)
                arc_mob.set_points_as_corners(geodesic_arc(unit_pts[i], unit_pts[j]))
                weight = float(a_mat[i, j])
                arc_mob.set_stroke(
                    width=1.2 + 5.5 * min(weight * 4.0, 1.0),
                    opacity=0.22 + 0.65 * min(weight * 4.0, 1.0),
                )

        update_arcs(arcs)
        arcs.add_updater(update_arcs)

        # 3. Geodesic trails on S²
        trails = m.VGroup(
            *[
                m.VMobject(
                    stroke_color=CLUSTER_COLORS[labels[i]],
                    stroke_width=1.8,
                    stroke_opacity=0.65,
                    shade_in_3d=True,
                )
                for i in range(N_TOKENS)
            ]
        )

        def update_trails(group: m.Mobject) -> None:
            upto = max(2, int(progress.get_value() * STEPS) + 1)
            for i, trail_mob in enumerate(group.submobjects):
                assert isinstance(trail_mob, m.VMobject)
                trail_mob.set_points_as_corners(CENTER + RADIUS * traj[:upto, i])

        update_trails(trails)
        trails.add_updater(update_trails)

        # 4. Lit 3D token spheres (MeshMobject)
        init_pts, _, _, _, _ = state_at(0.0)
        verts0, tris0 = sphere_beads(init_pts)
        verts_per_bead = len(verts0) // N_TOKENS
        bead_rgba = np.ones((len(verts0), 4))
        for i in range(N_TOKENS):
            rgb = m.ManimColor(CLUSTER_COLORS[labels[i]]).to_rgb()
            bead_rgba[i * verts_per_bead : (i + 1) * verts_per_bead, :3] = rgb
        tokens_mesh = m.MeshMobject(verts0, tris0, shade_in_3d=True)
        tokens_mesh.paint = tokens_mesh.paint.but(fill=bead_rgba)

        def update_tokens(mob: m.Mobject) -> None:
            pts, _, _, _, _ = state_at(progress.get_value())
            mob.set_points(sphere_beads(pts)[0])

        tokens_mesh.add_updater(update_tokens)

        # 5. Fixed-in-frame HUD (carefully spaced in the four corners)
        title = m.Text("트랜스포머 어텐션의 기하학", font_size=30).to_corner(
            m.UL, buff=0.38
        )
        subtitle = (
            m.Text("구면 다양체 위의 자기-어텐션 흐름과 의미 군집화", font_size=19)
            .set_color(m.GREY_B)
            .next_to(title, m.DOWN, aligned_edge=m.LEFT, buff=0.12)
        )

        attn_eq = m.MathTex(
            r"\operatorname{Attn}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}\,\tau}\right)V",
            font_size=24,
        ).to_corner(m.UR, buff=0.38)

        tau_label = m.Text("온도 τ =", font_size=21).set_color("#38bdf8")
        tau_num = live_decimal(
            m.DecimalNumber(TAU_START, num_decimal_places=2, font_size=24),
            lambda: state_at(progress.get_value())[1],
        )
        tau_row = m.VGroup(tau_label, tau_num).arrange(m.RIGHT, buff=0.14)

        ent_label = m.Text("엔트로피 H =", font_size=21).set_color("#fbbf24")
        ent_num = live_decimal(
            m.DecimalNumber(float(entropies[0]), num_decimal_places=2, font_size=24),
            lambda: state_at(progress.get_value())[2],
        )
        ent_row = m.VGroup(ent_label, ent_num).arrange(m.RIGHT, buff=0.14)

        metrics_box = (
            m.VGroup(tau_row, ent_row)
            .arrange(m.DOWN, aligned_edge=m.RIGHT, buff=0.14)
            .next_to(attn_eq, m.DOWN, aligned_edge=m.RIGHT, buff=0.22)
        )

        flow_eq = m.MathTex(
            r"\frac{dx_i}{dt} = P_{x_i}\!\left(\sum_{j=1}^{N} A_{ij}(x)\,v_j\right)",
            font_size=25,
        ).to_corner(m.DL, buff=0.38)

        align_label = m.Text("군집 응집도 =", font_size=21).set_color("#f43f5e")
        align_num = live_decimal(
            m.DecimalNumber(float(alignments[0]), num_decimal_places=2, font_size=24),
            lambda: state_at(progress.get_value())[3],
        )
        align_row = (
            m.VGroup(align_label, align_num)
            .arrange(m.RIGHT, buff=0.14)
            .to_corner(m.DR, buff=0.38)
        )

        self.add_fixed_in_frame_mobjects(
            title, subtitle, attn_eq, metrics_box, flow_eq, align_row
        )
        self.remove(attn_eq, metrics_box, flow_eq, align_row)

        # Camera setup & choreography
        self.set_camera_orientation(
            phi=64 * m.DEGREES,
            theta=-55 * m.DEGREES,
            zoom=0.84,
        )
        self.add(wireframe, tokens_mesh)
        self.begin_ambient_camera_rotation(rate=0.09)

        # Act 1: Introduce the token manifold on S² and the attention equation
        self.say(
            "트랜스포머의 토큰 임베딩들이 3차원 구면 다양체 위에 놓여 있고, [자기 어텐션 수식]이 토큰 사이의 연결 강도를 결정합니다.",
            m.FadeIn(attn_eq, shift=0.15 * m.DOWN),
        )

        # Act 2: Show the geodesic attention arcs and Riemannian flow equation
        self.add(arcs, trails)
        self.say(
            "온도 파라미터와 엔트로피가 표시되면, [리만 어텐션 미분방정식]을 따라 각 토큰이 측지선 궤적을 그리며 이동하기 시작합니다.",
            m.AnimationGroup(
                m.FadeIn(metrics_box),
                m.FadeIn(flow_eq, shift=0.15 * m.UP),
                m.FadeIn(align_row, shift=0.15 * m.UP),
            ),
        )

        # Act 3: Cool down temperature τ from 1.80 to 0.18 — tokens cluster on S²
        self.say(
            "온도가 낮아질수록 어텐션 엔트로피가 급격히 줄어들고, [무질서하던 토큰들이 세 개의 선명한 의미 군집]으로 응집됩니다.",
            progress.animate.set_value(1.0),
        )

        # Act 4: Orbit camera to inspect the separated semantic polytopes
        self.stop_ambient_camera_rotation()
        self.move_camera(
            phi=48 * m.DEGREES,
            theta=-20 * m.DEGREES,
            zoom=0.88,
            run_time=2.8,
        )
        self.wait(1.0)


if __name__ == "__main__":
    AttentionManifold().render("attention_manifold.mp4")
