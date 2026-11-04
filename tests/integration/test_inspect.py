"""Inspection uses the video's scene state, with the same checks at default and chosen times."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from manimgx.cli import app
from manimgx.cli.scenes import scene
from manimgx.cli.storyboard import CAPTION, GAP, tile
from manimgx.rendering.film import Frame


def source(
    tmp_path: Path, body: str, base: str = "Scene", size: tuple[int, int] = (320, 180)
) -> Path:
    path = tmp_path / "scene.py"
    path.write_text(
        "import manimgx as m\n"
        f"m.config.pixel_width, m.config.pixel_height = {size[0]}, {size[1]}\n"
        "m.config.frame_rate = 10\n"
        f"class Example(m.{base}):\n"
        "    def construct(self) -> None:\n"
        + "\n".join(f"        {line}" for line in body.splitlines())
        + "\n",
        encoding="utf-8",
    )
    return path


def test_selected_times_check_the_picture_during_the_animation(tmp_path: Path) -> None:
    path = source(
        tmp_path,
        'moving = m.Text("moving").shift(4 * m.LEFT)\n'
        'fixed = m.Text("fixed")\n'
        "self.add(moving, fixed)\n"
        "self.play(moving.animate.shift(8 * m.RIGHT), run_time=2, rate_func=m.linear)",
    )
    runner = CliRunner()
    default = runner.invoke(app, ["inspect", str(path)])
    assert default.exit_code == 0, default.output
    assert "no problems in 1 sample" in default.output
    chosen = runner.invoke(app, ["inspect", str(path), "-t", "1.09"])
    assert chosen.exit_code == 1, chosen.output
    assert "t=1s" in chosen.output  # the frame on screen at 1.09 s is frame 10
    assert "moving (Text 'moving')" in chosen.output
    assert "texts overlap" in chosen.output
    assert "scene.py:9" in chosen.output
    with Image.open(tmp_path / "Example.storyboard.png") as image:
        pixels = np.array(image)
        assert ((pixels == [255, 64, 64]).all(axis=-1)).any()  # numbered red boxes


def test_selected_times_stop_after_their_play(tmp_path: Path) -> None:
    path = source(
        tmp_path,
        "self.add(m.Square())\nself.wait(2)\nraise RuntimeError('too far')",
    )
    chosen = CliRunner().invoke(app, ["inspect", str(path), "-t", "0.5"])
    assert chosen.exit_code == 0, chosen.output
    assert "2.00 s, 1 plays" in chosen.output
    ending = CliRunner().invoke(app, ["inspect", str(path), "-t", "end"])
    assert ending.exit_code == 1
    assert "too far" in ending.output


def test_inspection_stops_when_a_wait_ends_on_the_requested_frame(
    tmp_path: Path,
) -> None:
    path = source(
        tmp_path,
        "self.add(m.Square())\n"
        "self.wait(stop_condition=lambda: True)\n"
        "raise RuntimeError('the frame was already inspected')",
    )
    result = CliRunner().invoke(app, ["inspect", str(path), "-t", "0"])
    assert result.exit_code == 0, result.output


def test_default_inspection_deduplicates_pictures_but_checks_every_play(
    tmp_path: Path,
) -> None:
    path = source(
        tmp_path, 'self.add(m.Text("tiny", font_size=8))\nself.wait()\nself.wait()'
    )
    result = CliRunner().invoke(app, ["inspect", str(path)])
    assert result.exit_code == 1, result.output
    assert "t=1–2s" in result.output
    assert "#0" in result.output
    assert "#1" in result.output
    with Image.open(tmp_path / "Example.storyboard.png") as image:
        assert image.size == (960, 568)


def test_the_last_frame_keeps_local_names_and_problem_numbers(tmp_path: Path) -> None:
    path = source(
        tmp_path,
        'tiny = m.Text("tiny", font_size=8)\nself.add(tiny)\nself.wait()',
    )
    result = CliRunner().invoke(app, ["inspect", str(path), "-t", "end,0.5,1"])
    assert result.exit_code == 1, result.output
    assert "layout: 1 problem" in result.output
    assert "t=0.5–1s" in result.output
    assert "tiny (Text 'tiny')" in result.output


@pytest.mark.parametrize(
    "size", [(320, 180), (640, 360), (180, 320), (320, 320), (240, 180), (180, 240)]
)
def test_a_text_is_as_tall_in_every_video_shape(
    tmp_path: Path, size: tuple[int, int]
) -> None:
    """A video is 1080p when its short side is 1080 px, so a text measures the same at 1080p in
    a video of any shape and resolution."""
    body = 'self.add(m.Text("Hello", font_size=12))\nself.wait()'
    result = CliRunner().invoke(
        app, ["inspect", str(source(tmp_path, body, size=size))]
    )
    assert result.exit_code == 1, result.output
    assert "text is 16 px tall at 1080p" in result.output


def test_a_camera_frame_stretched_narrow_leaves_a_texts_height(tmp_path: Path) -> None:
    """The camera maps the frame's width and its height onto the video's apart: stretching the
    frame's width changes how wide a text is, not how tall."""
    body = (
        "self.camera.frame.stretch_to_fit_width(4)\n"
        'self.add(m.Text("Hello", font_size=12))\nself.wait()'
    )
    path = source(tmp_path, body, base="MovingCameraScene", size=(180, 320))
    result = CliRunner().invoke(app, ["inspect", str(path)])
    assert result.exit_code == 1, result.output
    assert "text is 16 px tall at 1080p" in result.output


def test_notes_do_not_fail_inspection(tmp_path: Path) -> None:
    path = source(tmp_path, "self.add(m.Line(10 * m.LEFT, 10 * m.RIGHT))\nself.wait()")
    result = CliRunner().invoke(app, ["inspect", str(path)])
    assert result.exit_code == 0, result.output
    assert "layout: 0 problems, 1 note" in result.output


def test_end_inspects_changes_after_the_last_play(tmp_path: Path) -> None:
    path = source(
        tmp_path,
        'tiny = m.Text("tiny", font_size=8)\nself.add(tiny)\nself.wait()\ntiny.scale(5)',
    )
    result = CliRunner().invoke(app, ["inspect", str(path), "-t", "end"])
    assert result.exit_code == 0, result.output
    assert "no problems in 1 sample" in result.output


def test_explicit_times_keep_held_pictures_and_split_sheets(tmp_path: Path) -> None:
    path = source(tmp_path, "self.add(m.Square())\nself.wait(10)")
    output = tmp_path / "detail.png"
    result = CliRunner().invoke(
        app, ["inspect", str(path), "-t", "1,2,3", "-t", "4,5,6,end", "-o", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert "no problems in 7 samples" in result.output
    with Image.open(output) as image:
        assert image.size == (1928, 1720)
    with Image.open(tmp_path / "detail-2.png") as image:
        assert image.size == (960, 568)


@pytest.mark.parametrize("times", [[], ["-t", "end"], ["-t", "0,end"]])
def test_a_scene_without_plays_still_gets_a_picture(
    tmp_path: Path, times: list[str]
) -> None:
    path = source(tmp_path, "self.add(m.Cube())", base="ThreeDScene")
    result = CliRunner().invoke(app, ["inspect", str(path), *times])
    assert result.exit_code == 0, result.output
    assert "layout: no problems in " in result.output
    assert (tmp_path / "Example.storyboard.png").is_file()


@pytest.mark.parametrize("time", ["-1", "nan", "inf", "later", ""])
def test_invalid_times_are_command_errors(tmp_path: Path, time: str) -> None:
    path = source(tmp_path, "self.wait()")
    result = CliRunner().invoke(app, ["inspect", str(path), f"--time={time}"])
    assert result.exit_code == 2, result.output
    assert not list(tmp_path.glob("*.png"))


def test_a_missing_frame_is_a_scene_error(tmp_path: Path) -> None:
    path = source(tmp_path, "self.wait()")
    result = CliRunner().invoke(app, ["inspect", str(path), "-t", "4"])
    assert result.exit_code == 1, result.output
    assert "there is no frame at 4" in result.output
    assert not list(tmp_path.glob("*.png"))


def test_the_output_must_be_a_png(tmp_path: Path) -> None:
    path = source(tmp_path, "self.wait()")
    result = CliRunner().invoke(
        app, ["inspect", str(path), "-o", str(tmp_path / "x.jpg")]
    )
    assert result.exit_code == 2, result.output
    assert "*.png" in result.output
    assert not (tmp_path / "x.jpg").exists()


@pytest.mark.parametrize(
    ("base", "updating", "custom_emit"),
    [
        ("Scene", False, False),
        ("Scene", True, False),
        ("Scene", True, True),
        ("ThreeDScene", True, False),
    ],
)
def test_sampling_preserves_the_video_with_tweens_holds_and_updaters(
    tmp_path: Path,
    base: str,
    updating: bool,
    custom_emit: bool,
) -> None:
    camera = (
        "self.set_camera_orientation(phi=60 * m.DEGREES, theta=-40 * m.DEGREES)\n"
        "self.begin_ambient_camera_rotation(rate=0.3)\n"
        if base == "ThreeDScene"
        else ""
    )
    shape = "Cube" if base == "ThreeDScene" else "Square"
    path = source(
        tmp_path,
        camera + f"moving = m.{shape}(fill_opacity=1).shift(2 * m.LEFT)\n"
        "self.add(moving)\n"
        + (
            "moving.add_updater(lambda mob: mob.shift(0.01 * m.UP))\n"
            if updating
            else ""
        )
        + "self.play(moving.animate.shift(4 * m.RIGHT), rate_func=m.linear)\n"
        "self.wait(0.25)",
        base=base,
    )
    if custom_emit:
        # With an updater, both takes emit each frame rather than batching the tween.
        with path.open("a", encoding="utf-8") as out:
            out.write(
                "    def _emit(self, repeat: int = 1) -> None:\n"
                "        for mob in self.mobjects:\n"
                "            mob.set_color(m.RED)\n"
                "        super()._emit(repeat)\n"
            )

    frames: dict[int, bytes] = {}

    def keep(frame: Frame) -> None:
        pixels = frame.pixels()
        for index in range(frame.index, frame.index + frame.repeat):
            frames[index] = pixels

    normal = scene(path, None)().render(frames=keep)
    result = CliRunner().invoke(app, ["inspect", str(path), "-t", "1.3,0,1,0.5,1.2"])
    assert result.exit_code == 0, result.output
    assert normal.frame_count == 14
    with Image.open(tmp_path / "Example.storyboard.png") as sheet:
        for position, index in enumerate((0, 5, 10, 12, 13)):
            expected = tile(frames[index], 320, 180)
            row, column = divmod(position, 2)
            x = column * (expected.width + GAP)
            y = row * (expected.height + CAPTION + GAP) + CAPTION
            actual = sheet.crop((x, y, x + expected.width, y + expected.height))
            assert actual.tobytes() == expected.tobytes(), f"frame {index}"


@pytest.mark.parametrize(
    ("mesh_expr", "mesh_kind"),
    [
        ("m.Sphere(radius=1.8).shift(2 * m.OUT)", "Sphere"),
        ("m.Cube(side_length=3.0).shift(2 * m.OUT)", "Cube"),
        (
            "m.Surface(lambda u, v: [u, v, 2.0], u_range=(-2, 2), v_range=(-2, 2))",
            "Surface",
        ),
        ("m.Torus(major_radius=0.8, minor_radius=0.9).shift(2 * m.OUT)", "Torus"),
    ],
)
def test_3d_meshes_covering_text_are_detected_and_marked(
    tmp_path: Path, mesh_expr: str, mesh_kind: str
) -> None:
    path = source(
        tmp_path,
        "self.set_camera_orientation(phi=0, theta=-90 * m.DEGREES)\n"
        'formula = m.MathTex("E = mc^2")\n'
        f"blocker = {mesh_expr}\n"
        "self.add(formula, blocker)\n"
        "self.wait(0.1)",
        base="ThreeDScene",
    )
    result = CliRunner().invoke(app, ["inspect", str(path)])
    assert result.exit_code == 1, result.output
    assert f"blocker ({mesh_kind}): {mesh_kind} is drawn over" in result.output
    with Image.open(tmp_path / "Example.storyboard.png") as image:
        pixels = np.array(image)
        assert ((pixels == [255, 64, 64]).all(axis=-1)).any()


def test_3d_mesh_behind_text_or_behind_hud_text_is_not_flagged_as_covered(
    tmp_path: Path,
) -> None:
    path = source(
        tmp_path,
        "self.set_camera_orientation(phi=0, theta=-90 * m.DEGREES)\n"
        'front_text = m.Text("In Front").shift(2 * m.OUT)\n'
        'hud_title = m.Text("HUD Title").to_corner(m.UL)\n'
        "self.add_fixed_in_frame_mobjects(hud_title)\n"
        "back_sphere = m.Sphere(radius=1.5).shift(3 * m.IN)\n"
        "self.add(back_sphere, front_text)\n"
        "self.wait(0.1)",
        base="ThreeDScene",
    )
    result = CliRunner().invoke(app, ["inspect", str(path)])
    assert result.exit_code == 0, result.output
    assert "no problems in 1 sample" in result.output


@pytest.mark.parametrize(
    ("line_expr", "expected_kind"),
    [
        ("m.Line3D(3 * m.LEFT, 3 * m.RIGHT)", "Line3D"),
        ("m.Arrow3D(3 * m.LEFT, 3 * m.RIGHT)", "Arrow3D"),
        ("m.ThreeDAxes()", "NumberLine"),
    ],
)
def test_3d_lines_and_axes_crossing_text_are_detected(
    tmp_path: Path, line_expr: str, expected_kind: str
) -> None:
    path = source(
        tmp_path,
        "self.set_camera_orientation(phi=60 * m.DEGREES, theta=-45 * m.DEGREES)\n"
        'label = m.Text("Origin")\n'
        f"wire = {line_expr}\n"
        "self.add(wire, label)\n"
        "self.wait(0.1)",
        base="ThreeDScene",
    )
    result = CliRunner().invoke(app, ["inspect", str(path)])
    assert result.exit_code == 1, result.output
    assert f"{expected_kind} runs through the text" in result.output
    with Image.open(tmp_path / "Example.storyboard.png") as image:
        pixels = np.array(image)
        assert ((pixels == [255, 64, 64]).all(axis=-1)).any()


def test_3d_cut_small_and_overlap_with_hud_and_fixed_orientation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scene_3d_all.py"
    path.write_text(
        "import manimang as m\n"
        "m.config.pixel_width, m.config.pixel_height = 320, 180\n"
        "m.config.frame_rate = 10\n"
        "class Example(m.ThreeDScene):\n"
        "    def construct(self) -> None:\n"
        "        self.set_camera_orientation(phi=75 * m.DEGREES, theta=-90 * m.DEGREES)\n"
        "        clipped_sphere = m.Sphere(radius=2.0).shift(6.5 * m.RIGHT)\n"
        '        tiny_3d = m.Text("tiny", font_size=24).shift(2 * m.LEFT)\n'
        '        hud = m.Text("Centered")\n'
        "        self.add_fixed_in_frame_mobjects(hud)\n"
        '        billboard = m.Text("Billboard")\n'
        "        self.add_fixed_orientation_mobjects(billboard)\n"
        "        self.add(clipped_sphere, tiny_3d)\n"
        "        self.wait(0.1)\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["inspect", str(path)])
    assert result.exit_code == 1, result.output
    assert "clipped_sphere (Sphere):" in result.output
    assert "past the" in result.output
    assert "tiny_3d (Text 'tiny'): text is" in result.output
    assert "px tall at 1080p" in result.output
    assert "texts overlap" in result.output
    with Image.open(tmp_path / "Example.storyboard.png") as image:
        pixels = np.array(image)
        assert ((pixels == [255, 64, 64]).all(axis=-1)).any()
