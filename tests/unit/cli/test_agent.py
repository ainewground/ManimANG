"""Autonomous self-correction pipeline (`manimang agent` / `manimgx.cli.agent`)."""

from pathlib import Path

from typer.testing import CliRunner

from manimgx.cli import app
from manimgx.cli.agent import (
    AgentAttempt,
    build_feedback_prompt,
    heuristic_repair,
    run_agent_pipeline,
)

BROKEN_3D_SCENE = """import manimang as m
m.config.pixel_width = 320
m.config.pixel_height = 180
m.config.frame_rate = 10


class BrokenSphere(m.ThreeDScene):
    def construct(self) -> None:
        self.set_camera_orientation(phi=65 * m.DEGREES, theta=-45 * m.DEGREES)
        sphere = m.Sphere(radius=2.0).shift([1.5, -1.5, 1.5])
        formula = m.MathTex(r"E = mc^2", font_size=10)
        cut_box = m.Square(side_length=1.0).shift(9 * m.RIGHT)
        self.add(sphere, formula, cut_box)
        self.wait(0.2)
"""


def test_heuristic_repair_fixes_3d_covered_small_and_cut() -> None:
    issues = [
        ("covered", "MathTex('E = mc^2') is 100% covered by Sphere()"),
        ("small", "MathTex('E = mc^2') is 7 px tall"),
        ("cut", "Square() is 100% outside the frame"),
    ]
    repaired = heuristic_repair(BROKEN_3D_SCENE, issues, None)
    assert "add_fixed_in_frame_mobjects(formula)" in repaired
    assert "font_size=36" in repaired
    assert "2.5 * m.RIGHT" in repaired


def test_build_feedback_prompt_includes_diagnostic_issues() -> None:
    attempt = AgentAttempt(
        iteration=1,
        code=BROKEN_3D_SCENE,
        report="1 issue",
        issues=(("covered", "MathTex is 100% covered by Sphere"),),
        error=None,
        storyboard_paths=(),
    )
    msg = build_feedback_prompt("Explain sphere volume", [attempt])
    assert "covered: MathTex is 100% covered by Sphere" in msg
    assert "Attempt #1" in msg or "attempt #1" in msg


def test_agent_pipeline_self_corrects_broken_3d_scene(tmp_path: Path) -> None:
    scene_file = tmp_path / "self_heal_3d.py"
    result = run_agent_pipeline(
        "Fix 3D layout issues",
        scene_path=scene_file,
        initial_code=BROKEN_3D_SCENE,
        max_iterations=3,
        render_video=True,
    )
    assert len(result.history) == 2
    assert not result.history[0].passed
    first_kinds = {kind for kind, _ in result.history[0].issues}
    assert "covered" in first_kinds
    assert "cut" in first_kinds
    assert result.history[1].passed
    assert result.passed
    assert result.video_path is not None
    assert result.video_path.exists()


def test_cli_agent_command_from_prompt(tmp_path: Path) -> None:
    runner = CliRunner()
    out_py = tmp_path / "korean_sphere.py"
    res = runner.invoke(
        app,
        ["agent", "구의 부피 공식 설명", "-o", str(out_py), "--no-render"],
    )
    assert res.exit_code == 0, res.output
    assert "PASSED (0 issues)" in res.output
    assert out_py.exists()
