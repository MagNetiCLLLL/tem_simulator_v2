"""Parallel sample projection with the existing camera orbit/zoom controls."""
import math

from PySide6.QtGui import QMatrix4x4


def orthographic_pixel_size(distance, field_of_view_deg, width):
    """World units per screen pixel, independent of atom position/depth."""
    return (2.0 * max(float(distance), 1e-9)
            * math.tan(math.radians(float(field_of_view_deg)) * 0.5)
            / max(float(width), 1.0))


def orthographic_projection(region, viewport, distance, field_of_view_deg):
    """Keep transverse scale independent of depth, including viewport crops."""
    x0, y0, width, height = viewport
    width, height = max(float(width), 1.), max(float(height), 1.)
    distance = max(float(distance), 1e-9)
    half_width = orthographic_pixel_size(distance, field_of_view_deg, width) * width * 0.5
    half_height = half_width * height / width
    left = half_width * ((region[0]-x0)*2./width-1.)
    right = half_width * ((region[0]+region[2]-x0)*2./width-1.)
    bottom = half_height * ((region[1]-y0)*2./height-1.)
    top = half_height * ((region[1]+region[3]-y0)*2./height-1.)
    # A parallel camera can zoom through its focus without discarding the
    # near half of the lattice; depth ordering remains handled by OpenGL.
    depth = max(distance*1000., 1.)
    matrix = QMatrix4x4()
    matrix.ortho(left, right, bottom, top, -depth, depth)
    return matrix
