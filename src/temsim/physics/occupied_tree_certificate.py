"""Bounded full-port certificates for an executed adaptive interval.

For leaf i, M_i maps ROOT incoming amplitudes to its coarse/fine two-port
difference, and f_i is its fraction of the root action length (sum f_i=1).
Cauchy--Schwarz gives sum_i ||M_i x|| <= ||R x||, where QR compression of
stack_i(M_i/sqrt(f_i)) supplies R. R checks every root input, not only the
previous occupied vector. It conservatively bounds the same local numerical
indicators; it is NOT a rigorous bound on error against the physical continuum.

The executed complete scattering operator is untouched. Only intermediate
comparison operators are released. A failed certificate regenerates the
recorded partition and refines it before the next global tip boundary solve.
"""
import math

import numpy as np
from scipy.linalg import qr

from temsim.physics.scattering_load import compose


def compress_tree(tree, left, right, work):
    from temsim.physics.occupied_axial_refinement import internal_incoming, apply_two_port
    if not tree.expanded or tree.certificate is not None:
        return None
    count = len(left)
    eye, zero = np.eye(count, dtype=complex), np.zeros((count, count), complex)
    factor, partition = None, []
    released = 0
    def visit(node, a, b, root=False):
        nonlocal factor, released
        work.check()
        if not root:
            released += sum(block.nbytes for block in node.operator)
        if node.children is not None:
            first, second = node.children
            backward, forward = internal_incoming(first.operator, second.operator, a, b)
            visit(first, a, backward)
            visit(second, forward, b)
        else:
            if node.error_operator is None:
                raise ValueError("An unverified adaptive leaf cannot be compressed")
            released += sum(block.nbytes for block in node.error_operator)
            fraction = (node.end-node.start)/(tree.end-tree.start)
            block = apply_two_port(node.error_operator, a, b)/math.sqrt(fraction)
            factor = block if factor is None else qr(np.vstack((factor, block)),
                mode="r", check_finite=False, overwrite_a=True)[0][:2*count].copy()
            partition.append((node.start, node.end, node.depth))
    visit(tree, np.c_[eye, zero], np.c_[zero, eye], root=True)
    if factor is None or not np.all(np.isfinite(factor)):
        raise ValueError("Invalid full-port adaptive error certificate")
    work.retain(factor.nbytes)
    tree.certificate, tree.partition = factor, tuple(partition)
    tree.children = None
    tree.error_operator = None
    work.release(released)
    return certificate_indicator(tree, left, right)


def certificate_indicator(tree, left, right):
    # QR works on full complex matrices. A small roundoff cushion is an
    # increase of the error bound, never a reduction of the error estimate.
    incoming = np.r_[left, right]
    factor = tree.certificate
    error = np.linalg.norm(factor@incoming)
    error += 64*np.finfo(float).eps*np.linalg.norm(factor)*np.linalg.norm(incoming)
    return float(math.sqrt(tree.kappa)*error)


def restore_partition(tree, work):
    """Re-execute all saved leaves; never recreate a source or fit a wave."""
    from temsim.physics.occupied_axial_refinement import RefinableSlab, evaluate_interval
    entries = iter(tree.partition)
    entry = next(entries)
    def rebuild(start, end, depth, root=False):
        nonlocal entry
        work.check()
        if entry == (start, end, depth):
            operator = evaluate_interval(tree.sample, tree.width, tree.kappa, start, end, work)
            node = RefinableSlab(tree.sample, tree.width, tree.kappa, operator,
                                 start=start, end=end, depth=depth)
            entry = next(entries, None)
        else:
            if entry is None or depth >= work.settings.maximum_depth+1:
                raise ValueError("Invalid saved adaptive partition")
            midpoint = (start+end)/2
            children = (rebuild(start, midpoint, depth+1), rebuild(midpoint, end, depth+1))
            node = RefinableSlab(tree.sample, tree.width, tree.kappa,
                                 compose(children[0].operator, children[1].operator),
                                 start=start, end=end, depth=depth)
            node.children, node.expanded = children, True
        if not root:
            work.retain(sum(block.nbytes for block in node.operator))
        return node
    work.release(tree.certificate.nbytes)
    tree.certificate = None
    rebuilt = rebuild(tree.start, tree.end, tree.depth, root=True)
    if entry is not None:
        raise ValueError("Incomplete saved adaptive partition")
    tree.children, tree.operator, tree.expanded = rebuilt.children, rebuilt.operator, rebuilt.expanded
    tree.partition = None
