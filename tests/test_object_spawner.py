"""Tests for multi-object spawning system."""

import numpy as np
import pytest

from envs.object_spawner import (
    COLOR_NAMES,
    COLORS,
    SHAPES,
    ObjectSpec,
    generate_object_specs,
    get_object_qpos_slice,
    get_object_qvel_slice,
    inject_objects_into_xml,
    _make_geom_size,
    _surface_z,
)

_SCENE_XML = "envs/assets/tabletop_scene.xml"


class TestObjectSpec:
    """ObjectSpec dataclass works correctly."""

    def test_qpos_len(self):
        spec = ObjectSpec(
            name="test", shape="box", color_name="red",
            rgba=(1, 0, 0, 1), size_attr="0.02 0.02 0.02",
        )
        assert spec.qpos_len == 7

    def test_qvel_len(self):
        spec = ObjectSpec(
            name="test", shape="box", color_name="red",
            rgba=(1, 0, 0, 1), size_attr="0.02 0.02 0.02",
        )
        assert spec.qvel_len == 6


class TestGeomHelpers:
    """Geom size and surface height helpers."""

    def test_box_size(self):
        assert "0.02 0.02 0.02" == _make_geom_size("box")

    def test_cylinder_size(self):
        assert "0.02 0.02" == _make_geom_size("cylinder")

    def test_sphere_size(self):
        assert "0.02" == _make_geom_size("sphere")

    def test_unknown_shape_raises(self):
        with pytest.raises(ValueError):
            _make_geom_size("pyramid")

    def test_surface_z_box(self):
        z = _surface_z("box")
        assert z == pytest.approx(0.435, abs=0.001)

    def test_surface_z_cylinder(self):
        z = _surface_z("cylinder")
        assert z == pytest.approx(0.435, abs=0.001)

    def test_surface_z_sphere(self):
        z = _surface_z("sphere")
        assert z == pytest.approx(0.435, abs=0.001)


class TestGenerateSpecs:
    """generate_object_specs produces correct output."""

    def test_correct_count(self):
        rng = np.random.default_rng(42)
        specs = generate_object_specs(3, rng)
        assert len(specs) == 3

    def test_unique_colors_enforced(self):
        rng = np.random.default_rng(42)
        specs = generate_object_specs(5, rng, unique_colors=True)
        colors = [s.color_name for s in specs]
        assert len(set(colors)) == 5

    def test_too_many_unique_colors_raises(self):
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError):
            generate_object_specs(7, rng, unique_colors=True)

    def test_non_unique_colors_allowed(self):
        rng = np.random.default_rng(42)
        # Should not raise even with 6 objects
        specs = generate_object_specs(6, rng, unique_colors=False)
        assert len(specs) == 6

    def test_shapes_are_valid(self):
        rng = np.random.default_rng(42)
        specs = generate_object_specs(4, rng)
        for s in specs:
            assert s.shape in SHAPES

    def test_colours_are_valid(self):
        rng = np.random.default_rng(42)
        specs = generate_object_specs(4, rng)
        for s in specs:
            assert s.color_name in COLOR_NAMES

    def test_positions_are_spaced(self):
        rng = np.random.default_rng(42)
        specs = generate_object_specs(4, rng)
        for i in range(len(specs)):
            for j in range(i + 1, len(specs)):
                dist = np.linalg.norm(
                    specs[i].init_pos[:2] - specs[j].init_pos[:2]
                )
                assert dist >= 0.05, f"Objects {i},{j} too close: {dist:.3f}"

    def test_deterministic_with_same_seed(self):
        specs1 = generate_object_specs(3, np.random.default_rng(99))
        specs2 = generate_object_specs(3, np.random.default_rng(99))
        for s1, s2 in zip(specs1, specs2):
            assert s1.name == s2.name
            np.testing.assert_array_equal(s1.init_pos, s2.init_pos)

    def test_allowed_shapes_filter(self):
        rng = np.random.default_rng(42)
        specs = generate_object_specs(
            4, rng, allowed_shapes=["sphere"],
        )
        for s in specs:
            assert s.shape == "sphere"


class TestInjectObjects:
    """XML injection produces valid MuJoCo XML."""

    def test_inject_returns_string(self):
        rng = np.random.default_rng(42)
        specs = generate_object_specs(2, rng)
        xml = inject_objects_into_xml(_SCENE_XML, specs)
        assert isinstance(xml, str)
        assert "<mujoco" in xml

    def test_old_block_removed(self):
        rng = np.random.default_rng(42)
        specs = generate_object_specs(2, rng)
        xml = inject_objects_into_xml(
            _SCENE_XML, specs, remove_existing_block=True,
        )
        assert 'name="block_red"' not in xml

    def test_old_block_kept(self):
        rng = np.random.default_rng(42)
        specs = generate_object_specs(1, rng)
        xml = inject_objects_into_xml(
            _SCENE_XML, specs, remove_existing_block=False,
        )
        assert 'name="block_red"' in xml

    def test_new_objects_present(self):
        rng = np.random.default_rng(42)
        specs = generate_object_specs(3, rng)
        xml = inject_objects_into_xml(_SCENE_XML, specs)
        for spec in specs:
            assert f'name="{spec.name}"' in xml

    def test_loads_in_mujoco(self):
        """Injected XML loads without error in MuJoCo."""
        import mujoco
        rng = np.random.default_rng(42)
        specs = generate_object_specs(3, rng)
        xml = inject_objects_into_xml(_SCENE_XML, specs)
        model = mujoco.MjModel.from_xml_string(xml)
        assert model is not None


class TestQposSlices:
    """qpos/qvel slice helpers."""

    def test_first_object_qpos(self):
        sl = get_object_qpos_slice(0)
        assert sl == slice(9, 16)

    def test_second_object_qpos(self):
        sl = get_object_qpos_slice(1)
        assert sl == slice(16, 23)

    def test_first_object_qvel(self):
        sl = get_object_qvel_slice(0)
        assert sl == slice(9, 15)
