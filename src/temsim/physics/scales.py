def length_text(m):
    a=abs(m)
    if a<1e-6:return f'{m*1e9:.4g} nm'
    if a<1e-3:return f'{m*1e6:.4g} um'
    return f'{m*1e3:.4g} mm'
def description(m,width):
    if m['mode']=='image':return f"IMAGE | M={m['magnification']:.4g}x | specimen FOV={length_text(m['object_full_m'])} | camera={width:.4g} mm"
    return f"DIFFRACTION | camera length={m['effective_camera_length_m']:.4g} m | half scale +/-{m['mrad_half']:.4g} mrad = +/-{m['g_half_inv_nm']:.4g} 1/nm | camera={width:.4g} mm"
