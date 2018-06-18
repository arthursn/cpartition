from periodictable import elements


def w2x(wC, w={}, x={}, y=dict(Fe=1.), fullcomp=False):
    """
    Calculates the mole fraction of carbon from its weight fraction.

    Parameters
    ----------
    wC : float
        weight fraction of C
    w : dict, optional
        known full composition of the alloy in weight fraction
    x : dict, optional
        known full composition of the alloy in mole fraction
    y : dict, optional
        site fraction of the substitutional elements
    fullcomp : bool, optional
        if True, returns a dict with full composition.
        if False, returns only C composition
    """
    if w:
        w = {k: v*(1-wC)/(1-w['C']) for k, v in w.items()}
        w['C'] = wC
    elif x:
        w = x2w(0, x=x, fullcomp=True)
        w = {k: v*(1-wC) for k, v in w.items()}
        w['C'] = wC
    else:
        w = x2w(0, y=y, fullcomp=True)
        w = {k: v*(1-wC) for k, v in w.items()}
        w['C'] = wC

    x = {k: v/elements.symbol(k).mass for k, v in w.items()}
    sumx = sum(x.values())
    x = {k: v/sumx for k, v in x.items()}

    if fullcomp:
        return x
    else:
        return x['C']


def x2w(xC, x={}, w={}, y=dict(Fe=1.), fullcomp=False):
    """
    Calculates the weight fraction of carbon from its mole fraction.

    Parameters
    ----------
    xC : float
        mole fraction of C
    x : dict, optional
        known full composition of the alloy in mole fraction
    w : dict, optional
        known full composition of the alloy in weight fraction
    y : dict, optional
        site fraction of the substitutional elements
    fullcomp : bool, optional
        if True, returns a dict with full composition.
        if False, returns only C composition
    """
    if x:
        x = {k: v*(1-xC)/(1-x['C']) for k, v in x.items()}
        x['C'] = xC
    elif w:
        x = w2x(0, w=w, fullcomp=True)
        x = {k: v*(1-xC) for k, v in x.items()}
        x['C'] = xC
    else:
        x = {k: v*(1-xC) for k, v in y.items()}
        x['C'] = xC

    w = {k: v*elements.symbol(k).mass for k, v in x.items()}
    sumw = sum(w.values())
    w = {k: v/sumw for k, v in w.items()}

    if fullcomp:
        return w
    else:
        return w['C']


def x2wp(xC, x={}, w={}, y=dict(Fe=1.)):
    """
    Calculates the weight percentage of carbon from its mole fraction.

    Parameters
    ----------
    xC : float
        mole fraction of C
    x : dict, optional
        known full composition of the alloy in mole fraction
    w : dict, optional
        known full composition of the alloy in weight fraction
    y : dict, optional
        site fraction of the substitutional elements
    """
    return 100.*x2w(xC, x=x, w=w, y=y)


if __name__ == '__main__':
    x = dict(C=3.34414e-2, Fe=9.13324e-1, Si=4.79858e-2,
             Cu=3.26288e-3, Mn=1.98638e-3)

    w = dict(C=7.56827e-3, Fe=9.61074e-1, Si=2.53942e-2,
             Cu=3.90681e-3, Mn=2.05621e-3)

    y = dict(Cu=3.37577e-3, Fe=9.44924e-1, Mn=2.05511e-3, Si=4.96460e-2)

    print(x2w(1e-2, x=x))
    print(x2w(1e-2, w=w))
    print(x2w(1e-2, y=y))

    print(w2x(1e-2, w=w))
    print(w2x(1e-2, x=x))
    print(w2x(1e-2, y=y))
