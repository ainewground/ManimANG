"""Autonomous self-correction agent pipeline (`manimang agent`):
Prompt / Scene → Code Generation → 2D & 3D `inspect` Diagnostics → Self-Correction Loop → MP4 Render.

Works with cloud or local LLMs (Gemini, OpenAI-compatible / Ollama / vLLM, Anthropic) as well
as a built-in deterministic geometric repair engine (`heuristic_repair`) for offline self-healing
of 2D/3D layout defects (`covered`, `crossing`, `cut`, `overlap`, `small`).
"""

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

__all__ = [
    "AgentAttempt",
    "AgentResult",
    "agent",
    "heuristic_repair",
    "run_agent_pipeline",
    "verify_scene",
]

_PROBLEM_LINE_RE = re.compile(r"^\s+\[\d+\]\s+(.+)$", re.MULTILINE)
_NOTE_LINE_RE = re.compile(r"^\s+note:\s+(.+)$", re.MULTILINE)
_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def _classify_problem(line: str) -> str:
    if "px tall at 1080p" in line:
        return "small"
    if "is drawn over" in line:
        return "covered"
    if "through the text" in line:
        return "crossing"
    if "texts overlap" in line:
        return "overlap"
    return "cut"


SYSTEM_PROMPT = """You are the ManimANG Autonomous Animation Agent (ainewground.com).
Write a complete, self-contained Python scene file using `import manimang as m`.

Rules:
1. Define a `m.Scene` or `m.ThreeDScene` subclass with `def construct(self) -> None:`.
2. In `m.ThreeDScene`:
   - Put 2D titles, captions, and formulas in `self.add_fixed_in_frame_mobjects(mob)` and
     position them with `.to_edge(m.UP)` or `.to_edge(m.DOWN)` so 3D meshes (`Sphere`, `Cube`,
     `Surface`, `Torus`) never cover (`covered`) or cross (`crossing`) them.
   - Or orient 3D labels toward the camera with `self.add_fixed_orientation_mobjects(label)` so
     perspective tilt never makes them unreadably foreshortened (`small`).
3. Keep all objects inside the frame (`[-7.1, 7.1] x [-4.0, 4.0]`) with comfortable margins so
   nothing is `cut`, and separate labels with `.next_to(..., buff=0.35)` or `.arrange(..., buff=0.4)`
   so nothing has `overlap`.
4. Return ONLY valid Python code inside a ```python ... ``` block.
"""


@dataclass(frozen=True, slots=True)
class AgentAttempt:
    """One iteration of the `inspect` → self-correction loop."""

    iteration: int
    code: str
    report: str
    issues: tuple[tuple[str, str], ...]
    error: str | None
    storyboard_paths: tuple[Path, ...]

    @property
    def passed(self) -> bool:
        """Whether the scene ran without errors and `inspect` reported zero layout defects."""
        return self.error is None and len(self.issues) == 0


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Final outcome of `run_agent_pipeline`."""

    passed: bool
    iterations: int
    code: str
    scene_path: Path
    video_path: Path | None
    storyboard_paths: tuple[Path, ...]
    history: tuple[AgentAttempt, ...]


type Generator = Callable[[str, Sequence[AgentAttempt]], str]


def verify_scene(
    scene_path: Path,
    scene_name: str | None = None,
    storyboard_output: Path | None = None,
) -> tuple[str, list[tuple[str, str]], str | None, list[Path]]:
    """Run `manimgx inspect` on `scene_path` and return `(report, issues, error, storyboards)`."""
    cmd = [sys.executable, "-m", "manimang", "inspect", str(scene_path)]
    if scene_name:
        cmd.append(scene_name)
    if storyboard_output is not None:
        cmd.extend(["-o", str(storyboard_output)])
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    storyboards: list[Path] = []
    report_lines: list[str] = []
    for line in stdout.splitlines():
        if line.startswith("storyboard:"):
            raw_paths = line.removeprefix("storyboard:").strip().split(",")
            for p in raw_paths:
                cand = Path(p.strip())
                if cand.exists():
                    storyboards.append(cand)
        report_lines.append(line)
    report = "\n".join(report_lines).strip()
    issues = [
        (_classify_problem(m.group(1)), m.group(1).strip())
        for m in _PROBLEM_LINE_RE.finditer(report)
    ]
    for m in _NOTE_LINE_RE.finditer(report):
        if "outside the frame" in m.group(1):
            issues.append(("cut", m.group(1).strip()))
    error: str | None = None
    if proc.returncode != 0 and not issues:
        error = (
            stderr.strip() or report or f"inspect exited with code {proc.returncode}"
        )
    return report, issues, error, storyboards


def heuristic_repair(
    code: str,
    issues: Sequence[tuple[str, str]],
    error: str | None,
) -> str:
    """Deterministic geometric self-correction for common 2D & 3D layout defects.

    Repairs:
    - `covered` / `crossing` in `ThreeDScene`: pins covered text/math as a HUD overlay with
      `self.add_fixed_in_frame_mobjects(...)` at the top edge so 3D meshes/lines no longer
      occlude or cross it.
    - `small`: increases `font_size` (or replaces tiny `font_size=<n>` with `font_size=36`)
      and adds `self.add_fixed_orientation_mobjects(...)` in 3D scenes.
    - `cut`: scales oversized mobjects or moves out-of-frame objects back inside the frame.
    - `overlap`: increases buffer spacing or separates overlapping texts vertically.
    """
    fixed = code
    kinds = {kind for kind, _ in issues}
    is_three_d = "ThreeDScene" in fixed

    if "small" in kinds:
        fixed = re.sub(
            r"font_size\s*=\s*(\d+(?:\.\d+)?)",
            lambda m: f"font_size={max(36, int(float(m.group(1)) * 3))}",
            fixed,
        )

    if is_three_d and ("covered" in kinds or "crossing" in kinds or "small" in kinds):
        # Find Text / MathTex / Tex variables added to the scene without fixed_in_frame
        text_vars = re.findall(
            r"^\s*([a-zA-Z_]\w*)\s*=\s*(?:m\.)?(?:Text|MathTex|Tex|MarkupText)\b",
            fixed,
            flags=re.MULTILINE,
        )
        for idx, var in enumerate(dict.fromkeys(text_vars)):
            edge = "m.UP" if idx % 2 == 0 else "m.DOWN"
            if f"add_fixed_in_frame_mobjects({var})" not in fixed:
                pattern = re.compile(
                    rf"^([ \t]*{re.escape(var)}\s*=\s*[^\n]+)$",
                    flags=re.MULTILINE,
                )
                fixed = pattern.sub(
                    rf"\1\n        {var}.to_edge({edge})\n"
                    rf"        self.add_fixed_in_frame_mobjects({var})",
                    fixed,
                    count=1,
                )

    if "cut" in kinds:
        # Bring out-of-bounds shifts such as 9 * m.RIGHT or [10, 0, 0] inside the camera frame
        fixed = re.sub(
            r"\b([6-9]|\d{2,})(?:\.\d+)?\s*\*\s*(m\.(?:RIGHT|LEFT|UP|DOWN))",
            r"2.5 * \2",
            fixed,
        )
        fixed = re.sub(
            r"\.shift\(\s*\[\s*[-+]?(?:[6-9]|\d{2,})(?:\.\d+)?\s*,\s*[-+]?\d+(?:\.\d+)?\s*,\s*[-+]?\d+(?:\.\d+)?\s*\]\s*\)",
            ".to_edge(m.DOWN)",
            fixed,
        )

    if "overlap" in kinds:
        # If multiple Text/MathTex objects sit at ORIGIN without positioning, arrange them
        text_vars = list(
            dict.fromkeys(
                re.findall(
                    r"^\s*([a-zA-Z_]\w*)\s*=\s*(?:m\.)?(?:Text|MathTex|Tex)\b",
                    fixed,
                    flags=re.MULTILINE,
                )
            )
        )
        if (
            len(text_vars) >= 2
            and ".arrange(" not in fixed
            and ".next_to(" not in fixed
            and ".to_edge(" not in fixed
        ):
            first, second = text_vars[0], text_vars[1]
            fixed = re.sub(
                rf"^([ \t]*self\.add\([^\n]*\b{re.escape(second)}\b[^\n]*\))",
                f"        {second}.next_to({first}, m.DOWN, buff=0.5)\n\\1",
                fixed,
                count=1,
                flags=re.MULTILINE,
            )

    return fixed


def _extract_code(response: str) -> str:
    match = _CODE_BLOCK_RE.search(response)
    return (match.group(1).strip() if match else response.strip()) + "\n"


def _default_template_from_prompt(prompt: str) -> str:
    """Generate a Korean/English 3D or 2D ManimANG scene from `prompt` when no LLM key is set."""
    title = (
        prompt.strip().splitlines()[0][:36] if prompt.strip() else "ManimANG 3D Demo"
    )
    return (
        "import manimang as m\n\n\n"
        "class GeneratedScene(m.ThreeDScene):\n"
        "    def construct(self) -> None:\n"
        "        self.set_camera_orientation(phi=65 * m.DEGREES, theta=-45 * m.DEGREES)\n"
        "        axes = m.ThreeDAxes(x_length=5, y_length=5, z_length=4)\n"
        "        sphere = m.Sphere(radius=1.1).set_color(m.BLUE)\n"
        f"        title = m.Text({title!r}, font_size=36).to_edge(m.UP)\n"
        "        formula = m.MathTex(r'V = \\frac{4}{3}\\pi r^3', font_size=40).to_edge(m.DOWN)\n"
        "        self.add_fixed_in_frame_mobjects(title, formula)\n"
        "        self.add(axes)\n"
        "        self.play(m.Create(sphere), m.FadeIn(title), m.Write(formula))\n"
        "        self.wait(0.5)\n"
    )


def build_feedback_prompt(prompt: str, history: Sequence[AgentAttempt]) -> str:
    """Format the original user prompt along with any `inspect` diagnostics from prior attempts."""
    if not history:
        return f"Task: {prompt}\n\nWrite the complete ManimANG scene."
    last = history[-1]
    diag = (
        f"Runtime / compilation error:\n{last.error}"
        if last.error
        else "Layout issues detected by `manimang inspect`:\n"
        + "\n".join(f"- {kind}: {detail}" for kind, detail in last.issues)
    )
    return (
        f"Task: {prompt}\n\n"
        f"Previous attempt #{last.iteration} code:\n```python\n{last.code}```\n\n"
        f"{diag}\n\n"
        "Fix every reported issue so `manimang inspect` reports 0 issues. Return the full corrected Python file."
    )


def llm_generator(
    prompt: str,
    history: Sequence[AgentAttempt],
    *,
    model: str | None = None,
) -> str:
    """Generate or repair scene code via Gemini, OpenAI-compatible API, or heuristic fallback."""
    user_msg = build_feedback_prompt(prompt, history)
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if gemini_key:
        chosen = model or "gemini-2.5-flash"
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{chosen}"
            f":generateContent?key={gemini_key}"
        )
        body = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            reply = json.loads(resp.read().decode("utf-8"))
        text = reply["candidates"][0]["content"]["parts"][0]["text"]
        return _extract_code(str(text))

    if openai_key or os.environ.get("OPENAI_BASE_URL"):
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip(
            "/"
        )
        chosen = model or "gpt-4o"
        headers = {"Content-Type": "application/json"}
        if openai_key:
            headers["Authorization"] = f"Bearer {openai_key}"
        body = {
            "model": chosen,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        }
        req = urllib.request.Request(
            f"{base}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            reply = json.loads(resp.read().decode("utf-8"))
        text = reply["choices"][0]["message"]["content"]
        return _extract_code(str(text))

    if history:
        last = history[-1]
        return heuristic_repair(last.code, last.issues, last.error)
    return _default_template_from_prompt(prompt)


def run_agent_pipeline(
    prompt: str = "",
    *,
    scene_path: Path,
    initial_code: str | None = None,
    max_iterations: int = 3,
    render_video: bool = True,
    output_video: Path | None = None,
    generator: Generator | None = None,
) -> AgentResult:
    """Run the ManimANG generate → `inspect` → self-correct → `render` loop.

    Args:
        prompt: Natural-language description of the animation to create or repair.
        scene_path: File path where the scene Python code is written and inspected.
        initial_code: Optional starting Python code (for repairing an existing scene file).
        max_iterations: Maximum number of `inspect` + self-correction iterations.
        render_video: Whether to render the final MP4 video once `inspect` passes.
        output_video: Optional output MP4 path.
        generator: Optional custom code generator `(prompt, history) -> code`.

    Returns:
        [`AgentResult`][manimgx.cli.agent.AgentResult] with the verification history and
        output paths.
    """
    gen = generator or llm_generator
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[AgentAttempt] = []
    code = initial_code if initial_code is not None else gen(prompt, ())

    for iteration in range(1, max_iterations + 1):
        if iteration > 1:
            last = history[-1]
            repaired = heuristic_repair(last.code, last.issues, last.error)
            code = (
                repaired
                if repaired != last.code and generator is None
                else gen(prompt, history)
            )

        scene_path.write_text(code, encoding="utf-8")
        storyboard_target = scene_path.with_name(f"{scene_path.stem}.storyboard.png")
        report, issues, error, storyboards = verify_scene(
            scene_path, storyboard_output=storyboard_target
        )
        attempt = AgentAttempt(
            iteration=iteration,
            code=code,
            report=report,
            issues=tuple(issues),
            error=error,
            storyboard_paths=tuple(storyboards),
        )
        history.append(attempt)
        if attempt.passed:
            break

    final = history[-1]
    video_path: Path | None = None
    if final.passed and render_video:
        target = output_video or scene_path.with_suffix(".mp4")
        cmd = [
            sys.executable,
            "-m",
            "manimang",
            "render",
            str(scene_path),
            "-o",
            str(target),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0 and target.exists():
            video_path = target

    return AgentResult(
        passed=final.passed,
        iterations=len(history),
        code=final.code,
        scene_path=scene_path,
        video_path=video_path,
        storyboard_paths=final.storyboard_paths,
        history=tuple(history),
    )


def agent(
    target: Annotated[
        str,
        typer.Argument(
            help="A natural-language prompt, or path to an existing .py scene file to self-correct."
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output scene .py or .mp4 path (default: generated_scene.py).",
        ),
    ] = None,
    max_iterations: Annotated[
        int,
        typer.Option(
            "--max-iterations",
            "-n",
            min=1,
            max=10,
            help="Maximum inspect & self-correction iterations.",
        ),
    ] = 3,
    no_render: Annotated[
        bool,
        typer.Option(
            "--no-render",
            help="Only run the inspect & self-correction loop without encoding MP4.",
        ),
    ] = False,
) -> None:
    """Generate or repair a scene with autonomous 2D/3D `inspect` self-correction."""
    candidate = Path(target)
    if candidate.suffix == ".py" and candidate.exists():
        scene_path = candidate if output is None or output.suffix == ".mp4" else output
        initial_code = candidate.read_text(encoding="utf-8")
        prompt = f"Repair layout issues in {candidate.name}"
    else:
        scene_path = (
            output.with_suffix(".py")
            if output is not None
            else Path("generated_scene.py")
        )
        initial_code = None
        prompt = target

    video_out = (
        output
        if output is not None and output.suffix == ".mp4"
        else scene_path.with_suffix(".mp4")
    )
    result = run_agent_pipeline(
        prompt,
        scene_path=scene_path,
        initial_code=initial_code,
        max_iterations=max_iterations,
        render_video=not no_render,
        output_video=video_out,
    )
    for attempt in result.history:
        status = (
            "PASSED (0 issues)"
            if attempt.passed
            else f"FAILED ({len(attempt.issues)} issues)"
            if attempt.error is None
            else f"ERROR ({attempt.error.splitlines()[0]})"
        )
        typer.echo(f"[Iteration {attempt.iteration}/{max_iterations}] {status}")
        for kind, detail in attempt.issues:
            typer.echo(f"  - {kind}: {detail}")
    typer.echo(str(result.scene_path.resolve()))
    for sb in result.storyboard_paths:
        typer.echo(str(sb.resolve()))
    if result.video_path is not None:
        typer.echo(str(result.video_path.resolve()))
    if not result.passed:
        raise typer.Exit(1)
