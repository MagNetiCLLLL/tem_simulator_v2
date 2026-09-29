"""Geometric projection checks; these do not claim physical STEM contrast."""
import math

import numpy as np
import pytest
from PySide6.QtGui import QVector3D

from temsim.gui.sample_projection import orthographic_pixel_size, orthographic_projection


def test_column_sites_at_different_depths_project_to_identical_xy():
    viewport = (0, 0, 1000, 600)
    matrix = orthographic_projection(viewport, viewport, 10., 60.)
    projected = np.array([
        matrix.map(QVector3D(1.2, -0.8, z)).toTuple() for z in (-25., -5., 0., 8.)
    ])
    np.testing.assert_allclose(projected[:, :2], np.tile(projected[0, :2], (4, 1)))
    assert np.all(np.diff(projected[:, 2]) < 0)
    assert np.all(np.abs(projected[:, 2]) < 1)


@pytest.mark.parametrize("width,height", [(1000, 600), (600, 1000)])
def test_equal_world_lengths_have_equal_pixel_lengths(width, height):
    viewport = (0, 0, width, height)
    matrix = orthographic_projection(viewport, viewport, 10., 60.)
    point = matrix.map(QVector3D(1., 1., -5.))
    horizontal, vertical = point.x() * width / 2, point.y() * height / 2
    assert horizontal == pytest.approx(vertical, rel=1e-6)
    assert horizontal == pytest.approx(1 / orthographic_pixel_size(10., 60., width), rel=1e-6)


def test_cropped_viewport_keeps_projection_location():
    viewport = (20, 30, 1000, 600)
    crop = (220, 130, 400, 300)
    point = QVector3D(1., -.5, -5.)
    full = orthographic_projection(viewport, viewport, 10., 60.).map(point)
    partial = orthographic_projection(crop, viewport, 10., 60.).map(point)
    full_pixels = np.array([viewport[0] + (full.x()+1)*viewport[2]/2,
                            viewport[1] + (full.y()+1)*viewport[3]/2])
    crop_pixels = np.array([crop[0] + (partial.x()+1)*crop[2]/2,
                            crop[1] + (partial.y()+1)*crop[3]/2])
    np.testing.assert_allclose(crop_pixels, full_pixels, atol=1e-4)


def test_orthographic_zoom_halves_diameters_when_distance_doubles():
    first = orthographic_pixel_size(10., 60., 1000)
    assert first == pytest.approx(20 * math.tan(math.pi/6) / 1000)
    assert orthographic_pixel_size(20., 60., 1000) == pytest.approx(2 * first)
