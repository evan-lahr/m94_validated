import math
import numpy as np

def fwc94(cmu, cukw):
    """
    Calculate wave-current friction factor using FWC94 model.
    
    Args:
    cmu (float): Constant factor.
    cukw (float): Current velocity times wave number.
    
    Returns:
    float: Wave-current friction factor.
    
    Reference:
    Madsen (1994), Equations 32 and 33.
    """
    fwc = 0.00999  # meaningless (small) return value
    if cukw <= 0.:
        print("ERROR: cukw too small in fwc94: ", cukw)
        return fwc
    if cukw < 0.2:
        fwc = math.exp(7.02 * (0.2**-0.078) - 8.82)
        print("WARNING: cukw very small in fwc94: ", cukw)
    if 0.2 <= cukw <= 100.:
        fwc = cmu * math.exp(7.02 * (cukw**-0.078) - 8.82)
    elif 100. < cukw <= 10000.:
        fwc = cmu * math.exp(5.61 * (cukw**-0.109) - 7.30)
    elif cukw > 10000.:
        fwc = cmu * math.exp(5.61 * (10000.**-0.109) - 7.30)
    return fwc

def waven(T, h):
    """
    Calculate the wavenumber k for gravity waves using Newton's method.

    Parameters:
    T : float or np.ndarray
        Wave period [s]. Can be a scalar or an array.
    h : float
        Water depth [m].

    Returns:
    k : float or np.ndarray
        Wavenumber [1/m]. Matches the shape of T.
    """

    T = np.atleast_1d(T)  # Ensure T is an array
    jmax = 20
    xacc = 0.0001
    g = 9.80665
    w = 2 * np.pi / T
    w2h = w**2 * h / g

    # Initial guess for kh
    kh = w2h.copy()

    for _ in range(jmax):
        tx = np.tanh(kh)
        f = kh * tx - w2h
        df = kh * (1 - tx**2) + tx
        dx = f / df
        kh -= dx

        if np.all(np.abs(dx) < xacc):
            break
    else:
        raise ValueError("waven exceeded maximum iterations")

    k = kh / h
    return k

def uRMS_ubr_Soulsby(Hs, Tp, h):
    """
    Calculate orbital velocity under JONSWAP spectra using Soulsby approximation.

    Args:
    Hs (float): Significant wave height (m).
    Tp (float): Wave period (s).
    h (float): Water depth (m).

    Returns:
    float: RMS orbital velocity (m/s).

    Reference:
    Soulsby (2006), "Simplified calculation of wave orbital velocities",
    Report TR 155 Release 1.0 Feb 2006. HR Wallingford.
    Section 3.2, equation 28.

    Note:
    1.28*Tz ~= Tp
    sqrt(2)*uRMS = ubr (monochromatic wave with same variance)
    """
    import math
    Tz = Tp / 1.28
    term = - ((3.65 / Tz) * (math.sqrt(h / 9.81)))**2.1
    uRMS = (Hs / 4.) * math.sqrt(9.81 / h) * math.exp(term)
    ubr = math.sqrt(2) * uRMS
    return ubr

def ubr_linearwavetheory(T, H, h):
    """
    Calculate near-bottom wave-orbital velocity amplitude.

    Parameters:
    T : float or np.ndarray
        Significant wave period. Can be a scalar or an array.
    H : float or np.ndarray
        Significant wave height (2 * amplitude of surface wave). Matches the shape of T.
    h : float
        Water depth.

    Returns:
    ub : float or np.ndarray
        Near-bottom wave-orbital velocity amplitude. Matches the shape of T and H.
    """

    T = np.atleast_1d(T)
    H = np.atleast_1d(H)
    
    if T.shape != H.shape:
        raise ValueError("T and H must have the same shape")

    # Get the wave number using the waven function
    k = waven(T, h)

    # Calculate near-bottom wave-orbital velocity amplitude
    twopi = 2 * np.pi
    w = twopi / T
    kh = k * h
    amp = H / (2.0 * np.sinh(kh))
    ub = w * amp
    return ub

def m94(ubr, wr, ucr, zr, phiwc, kN, iverbose=1):
    """
    Grant-Madsen model from Madsen (1994) for wave-current interaction.
    
    Args:
    ubr (float): Rep. wave-orbital velocity amplitude outside wbl [m/s].
    wr (float): Rep. angular wave frequency = 2pi/T [rad/s].
    ucr (float): Current velocity at height zr [m/s].
    zr (float): Reference height for current velocity [m].
    phiwc (float): Angle between currents and waves at zr (radians).
    kN (float): Bottom roughness height (e.g., Nikuradse k) [m].
    iverbose (int, optional): Switch; when 1, extra output. Default is 0.
    
    Returns:
    list: Array with:
          ustrc (float): Current friction velocity u*c [m/s].
          ustrr (float): W-C combined friction velocity u*r [m/s].
          ustrwm (float): Wave max. friction velocity u*wm [m/s].
          dwc (float): Wave boundary layer thickness [m].
          fwc (float): Wave friction factor.
          zoa (float): Apparent bottom roughness [m].
    
    Reference:
    Madsen (1994)
    """
    MAXIT = 20
    vk = 0.41
    fwc = 0.4
    dwc = kN
    zo = kN / 30.
    zoa = zo
    rmu = [1.]
    Cmu = [1.]
    fwci = [.4]
    dwci = [kN]
    ustrwm2 = [0.01]
    ustrr2 = [0.01]
    ustrci = [0.01]

    if wr <= 0.:
        print("WARNING: Bad value for frequency in M94: ", wr)
        return [float('nan'), float('nan'), float('nan'), dwc, fwc, zoa]
    if ubr < 0.:
        print("WARNING: Bad value for orbital vel. in M94: ", ubr)
        return [float('nan'), float('nan'), float('nan'), dwc, fwc, zoa]
    if kN < 0.:
        print("WARNING: Weird value for roughness in M94: ", kN)
    if (zr < zoa or zr < 0.05) and iverbose == 1:
        print("WARNING: Low value for ref. level in M94: ", zr)

    if ubr <= 0.01:
        if ucr <= 0.01:
            ustrc = 0.
            ustrwm = 0.
            ustrr = 0.
            return [ustrc, ustrwm, ustrr, dwc, fwc, zoa]
        ustrc = ucr * vk / math.log(zr / zoa)
        ustrwm = 0.
        ustrr = ustrc
        return [ustrc, ustrwm, ustrr, dwc, fwc, zoa]

    cosphiwc = abs(math.cos(phiwc))
    rmu[0] = 0.
    Cmu[0] = 1.
    fwci[0] = fwc94(Cmu[0], (Cmu[0] * ubr / (kN * wr)))
    ustrwm2[0] = 0.5 * fwci[0] * ubr**2
    ustrr2[0] = Cmu[0] * ustrwm2[0]
    ustrr = math.sqrt(ustrr2[0])
    dwci[0] = kN

    if (Cmu[0] * ubr / (kN * wr)) >= 8.:
        dwci[0] = 2. * vk * ustrr / wr

    lnzr = math.log(zr / dwci[0])
    lndw = math.log(dwci[0] / zo)
    lnln = lnzr / lndw
    bigsqr = (-1. + math.sqrt(1. + ((4. * vk * lndw) / (lnzr**2)) * ucr / ustrr))
    ustrci[0] = 0.5 * ustrr * lnln * bigsqr

    for i in range(1, MAXIT):
        rmu.append(ustrci[i - 1]**2 / ustrwm2[i - 1])
        Cmu.append(math.sqrt(1. + 2. * rmu[i] * cosphiwc + rmu[i]**2))
        fwci.append(fwc94(Cmu[i], (Cmu[i] * ubr / (kN * wr))))
        ustrwm2.append(0.5 * fwci[i] * ubr**2)
        ustrr2.append(Cmu[i] * ustrwm2[i])
        ustrr = math.sqrt(ustrr2[i])
        dwci.append(kN)

        if (Cmu[i] * ubr / (kN * wr)) >= 8.:
            dwci[i] = 2. * vk * ustrr / wr

        lnzr = math.log(zr / dwci[i])
        lndw = math.log(dwci[i] / zo)
        lnln = lnzr / lndw
        bigsqr = (-1. + math.sqrt(1. + ((4. * vk * lndw) / (lnzr**2)) * ucr / ustrr))
        ustrci.append(0.5 * ustrr * lnln * bigsqr)

        diffw = abs((fwci[i] - fwci[i - 1]) / fwci[i])
        if diffw < 0.0005:
            break

    nit = i
    ustrwm = math.sqrt(ustrwm2[nit])
    ustrc = ustrci[nit]
    ustrr = math.sqrt(ustrr2[nit])
    zoa = math.exp(math.log(dwci[nit]) - (ustrc / ustrr) * math.log(dwci[nit] / zo))
    fwc = fwci[nit]
    dwc = dwci[nit]

    if iverbose == 1:
        for i in range(nit):
            print(f"i {i} fwc {fwci[i]} dwc {dwci[i]} u*c {ustrci[i]} u*wm {math.sqrt(ustrwm2[i])} u*rr {math.sqrt(ustrr2[i])}")

    ''' 
    print('ustrc: ',np.round(ustrc,4))
    print('ustrwm:',np.round(ustrwm,4))
    print('ustrr: ',np.round(ustrr,4))
    print('----------')
    '''
    return [ustrc, ustrwm, ustrr, dwc, fwc, zoa]



def calc_m94(h, wvht, dpd, wvdr, cspd, cdir, grsz, zr):
    """
    Calculate various parameters based on user input.
    
    h     - meters,  water depth
    wvht  - meters,  significant waveheight
    dpd   - seconds, dominant wave period
    wvdr  - degrees, wave direction
    cspd  - met/sec, current speed 1mab
    cdir  - degrees, current direction
    grsz  - microns, seafloor d50 grain size
    zr    - meters, current measurement height above bed
    
    """

    # Define constants
    rhow = 1027.0      # Density of seawater (kg/m^3)
    phiwc = 0          # angle between currents & waves fixed at zero
    Tz = dpd #/ 1.28   # zero-crossing period, seconds
    
    D = 1e-6 * grsz    # d50, microns to meters
    ks = D #2.5 * D    # grain roughness, meters
    zoa = ks / 30.0    # apparent bottom roughness, meters
    
    # wave estimate
    ubr = ubr_linearwavetheory(dpd, wvht, h)
    #ubr = ubr / 1.4
    #ubr = uRMS_ubr_Soulsby(wvht, dpd, h)

    # angular frequency
    w = 2.0 * math.pi / dpd    
    
    # Calculate shear parameters using Madsen's equations
    m94o = m94(ubr, w, cspd, zr, phiwc, ks, 0)
    
    # Calculate stresses and critical values
    tauc = rhow * m94o[0] * m94o[0]
    tauw = rhow * m94o[1] * m94o[1]
    taucw = rhow * m94o[2] * m94o[2]
    fwc = m94o[4]
    zoa = m94o[5]
    
    return tauw, tauc, taucw, fwc, zoa, ubr, ks, ubr, m94o[0], m94o[1], m94o[2]