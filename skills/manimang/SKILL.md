---
name: manimang
description: Create, inspect, and self-correct 2D/3D Studio PBR mathematical and scientific animations in Python with ManimANG (including 3D depth occlusion checks, Korean/multilingual local TTS sync, and autonomous agent repair).
license: MIT
---

# ManimANG

Use ManimANG (`import manimang as m` or `import manimgx as m`) to author, verify, and render
mathematical and 3D Studio PBR animations with zero layout defects and synchronized speech.

## Author a 3D Studio PBR scene

Enable Filament Studio PBR lighting (`tone_mapping = "agx"`, `ambient_occlusion = 1.0`,
`bloom = 0.08`, `EnvironmentLight`, `SunLight(shadows=True)`) and attach `m.Material(...)`
to 3D surfaces, spheres, and 3D cylindrical tubes:

```python
import manimgx as m


class StudioSphere(m.ThreeDScene):
    def construct(self) -> None:
        self.camera.tone_mapping = "agx"
        self.camera.ambient_occlusion = 1.0
        self.set_camera_orientation(phi=64 * m.DEGREES, theta=-45 * m.DEGREES)
        self.add(
            m.EnvironmentLight(intensity=0.8),
            m.SunLight(
                3.5 * m.UP + 2.5 * m.LEFT + 5.5 * m.OUT, intensity=1.3, shadows=True
            ),
        )
        sphere = (
            m.Sphere(radius=1.2, resolution=(48, 32), stroke_width=0)
            .set_color("#38bdf8")
            .set_material(m.Material(metallic=0.25, roughness=0.15, reflectance=0.85))
        )
        formula = m.MathTex(r"V = \frac{4}{3}\pi r^3").to_corner(m.UR)
        self.add_fixed_in_frame_mobjects(formula)
        self.play(m.Create(sphere), m.Write(formula))
        self.wait(1)
```

## 2D & 3D Inspect, Self-Correction, and Delivery

```sh
manimang inspect scene.py StudioSphere
manimang render scene.py StudioSphere -o final.mp4
```

`manimang inspect` projects 3D meshes through the camera's 4×4 view-projection matrix with
reversed-Z depth ordering to catch `covered`, `crossing`, `small`, `cut`, and `overlap`
defects across both `Scene` and `ThreeDScene`. Use `manimang agent` to run the closed-loop
`Prompt → Code → 3D Inspect → Self-Correction → MP4` pipeline automatically.
