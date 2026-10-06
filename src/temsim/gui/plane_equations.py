"""Symbolic, display-only explanations of first-order plane diagnostics.

Keep the selected-plane conditions aligned with scan_geometry's classifier
and the search conditions aligned with conjugate_planes. This module neither
evaluates those conditions nor imports the optical solvers.
"""

from html import escape


_SELECTED_CLASSIFICATIONS = {
    "image": (
        "Image plane",
        "||B||<sub>2</sub> &le; &epsilon;<sub>B</sub> and "
        "||A||<sub>2</sub> &gt; &epsilon;<sub>A</sub>",
        "r(z) &asymp; A r<sub>0</sub> + d: input momentum has negligible "
        "position response within tolerance.",
    ),
    "diffraction": (
        "Diffraction plane",
        "||A||<sub>2</sub> &le; &epsilon;<sub>A</sub> and "
        "||B||<sub>2</sub> &gt; &epsilon;<sub>B</sub>",
        "r(z) &asymp; B &eta;<sub>0</sub> + d: input position has negligible "
        "position response within tolerance.",
    ),
    "mixed": (
        "Mixed plane",
        "||A||<sub>2</sub> &gt; &epsilon;<sub>A</sub> and "
        "||B||<sub>2</sub> &gt; &epsilon;<sub>B</sub>",
        "Both input position and momentum contribute to output position.",
    ),
    "degenerate": (
        "Degenerate plane",
        "||A||<sub>2</sub> &le; &epsilon;<sub>A</sub> and "
        "||B||<sub>2</sub> &le; &epsilon;<sub>B</sub>",
        "Both responses fall below their tolerances; neither image nor "
        "diffraction classification is established.",
    ),
}

_SEARCH_CLASSIFICATIONS = {
    "image": (
        "Image conjugate",
        "&sigma;<sub>max</sub>(B) + e<sub>B</sub> &le; "
        "&epsilon;<sub>B</sub> and rank(A) = 2",
        "r(z) &asymp; A r<sub>0</sub> + d: a two-dimensional first-order "
        "image is resolved within tolerance.",
    ),
    "line_focus": (
        "Line focus",
        "&sigma;<sub>min</sub>(B) + e<sub>B</sub> &le; "
        "&epsilon;<sub>B</sub> &lt; &sigma;<sub>max</sub>(B)",
        "Only one transverse combination is focused; not a point image "
        "or a two-dimensional image.",
    ),
    "approximate": (
        "Approximate conjugate candidate",
        "z<sub>*</sub> &isin; local arg min<sub>z</sub> "
        "&sigma;<sub>max</sub>(B(z))",
        "The full image condition, including uncertainty and rank(A), is "
        "not established. This label is not caused by C<sub>s</sub>.",
    ),
}

_UNCLASSIFIED = {
    "specimen": "Object/reference plane; no downstream classification is asserted.",
    "reference": "Reference plane; the search excludes its own identity map.",
    "upstream": "Upstream of the specimen; this diagnostic has no downstream classification.",
    "unavailable": "The required optical response is unavailable.",
    "not_calculated": "This plane has not been calculated in the captured result.",
    "pending": "The classification is pending.",
}


def plane_equation_tooltip(
    kind: str,
    *,
    reference: str = "specimen",
    conjugate_search: bool = False,
    stale: bool = False,
) -> str:
    """Return a compact Qt rich-text fragment without substituting state values.

    ``kind`` is an existing diagnostic label, never a request to classify a
    plane. Unknown and unfinished labels show the model without claiming that
    an optical condition has been solved. ``reference`` is escaped as text.
    """
    normalized = str(kind).strip().lower().replace("-", "_").replace(" ", "_")
    search = conjugate_search or normalized in {"line_focus", "approximate"}
    classifications = _SEARCH_CLASSIFICATIONS if search else _SELECTED_CLASSIFICATIONS
    classification = classifications.get(normalized)
    heading = "First-order plane equations"
    if stale:
        heading += " &mdash; previous result (stale)"
    lines = [
        f"<b>{heading}</b>",
        f"Reference: {escape(str(reference))}; X/Y are column transverse axes.",
        "<b>r(z) = A(z) r<sub>0</sub> + B(z) &eta;<sub>0</sub> + d(z)</b>",
        "Centred form: &Delta;r = A &Delta;r<sub>0</sub> + "
        "B &Delta;&eta;<sub>0</sub>.",
        "r, r<sub>0</sub>, d: positions [m]; &eta;<sub>0</sub> = "
        "(p<sub>x,can</sub>, p<sub>y,can</sub>)/p<sub>0</sub>: canonical transverse momentum "
        "normalised by reference momentum (angle-equivalent).",
        "A = J<sub>img</sub> is dimensionless; B = J<sub>diff</sub> [m/rad].",
    ]
    if classification is None:
        explanation = _UNCLASSIFIED.get(normalized, "No solved classification is available.")
        lines.append(f"<b>Model only.</b> {explanation}")
    else:
        title, condition, explanation = classification
        lines.extend((f"<b>{title}:</b> {condition}.", explanation))
        lines.append(
            "||M||<sub>2</sub> = &sigma;<sub>max</sub>(M): "
            "spectral norm of the full 2 &times; 2 matrix."
        )
        if search:
            lines.append(
                "&epsilon;<sub>B</sub>: image tolerance; e<sub>B</sub>: "
                "cached-resolution + roundoff estimate, not a physical error bound."
            )
        else:
            lines.append(
                "&epsilon;<sub>A</sub>, &epsilon;<sub>B</sub>: the classifier's "
                "position- and momentum-response tolerances."
            )
    lines.extend((
        "Even with C<sub>s</sub>, this is the first-order reference. Higher "
        "orders would add H<sub>2</sub> + H<sub>3</sub> + &hellip;; "
        "they are not evaluated here.",
        "First-order map of captured instrument settings; not calculated crystal "
        "diffraction intensity. A/B alone do not determine wave intensity.",
    ))
    return '<div style="max-width:550px; white-space:normal">' + "<br>".join(lines) + "</div>"
