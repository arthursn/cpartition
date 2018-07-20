# -*- coding: utf-8 -*-

from builtins import open
from collections import OrderedDict

import numpy as np
import pandas as pd
from fnmatch import fnmatch


def _isfloat(x):
    try:
        float(x)
    except:
        return False
    return True


def load_table_blocks(fname):
    """
    Load table generated by Thermo-Calc

    Parameters
    ----------
    fname : string
        File name

    Returns
    -------
    dflist : list of pandas DataFrame objects with the table
    data.
    regionlist : list with corresponding phase regions of 
    the table
    """
    regionlist = []  # list containing list of phases in each phase region
    dflist = []  # list of dataframes for each phase region

    with open(fname, errors='ignore') as f:
        getphases = False
        newregion = False

        for line in f:
            # removes leading and trailing chars ' ', '\t', '\n', and ','
            line = line.strip(' \t\n,')

            if 'Phase Region for' in line:
                # new phase region
                newregion = True
                # start getting list of phases in phase region
                getphases = True

                phases = []  # list of phases in phase region
                cnames = []  # column names
                data = []  # numeric data
                continue

            if 'col' in line:
                # stop getting list of phases in phase region
                getphases = False
                # get list of cnames
                cnames += [col.split('=')[1] for col in line.split(', ')]
                continue

            if getphases:
                # get phase in phase region and append to 'phases'
                phases.append(line)
                continue

            if line:
                try:
                    # split line into list and filter float values
                    arr = map(float, filter(_isfloat, line.split()))
                    data += list(arr)
                except Exception as ex:
                    print(ex)
                    pass
                continue

            if newregion:
                regionlist.append(phases)
                # reshape data as nrow x len(cnames) numpy array
                data = np.reshape(data, (-1, len(cnames)))
                dflist.append(pd.DataFrame(data=data, columns=cnames))
                newregion = False

        if newregion:
            # Do the same thing as the snippet above. This is necessary because
            # after finishing reading the file the last phase region might not
            # be appended to data lists
            regionlist.append(phases)
            # reshape data as nrow x len(cnames) numpy array
            data = np.reshape(data, (-1, len(cnames)))
            dflist.append(pd.DataFrame(data=data, columns=cnames))

    return dflist, regionlist


def load_table(fname, sort=None, fill=None, unique=True, ignorephaseregions=''):
    """
    Load table generated by Thermo-Calc

    Parameters
    ----------
    fname : string
        File name
    sort : str or list of str, optional
        Argument passed to pandas df.sort_values function.
        Name or list of names which refer to the axis items.
        Default: None
    fill : float, optional
        Fill NA/NaN values with the 'fill' value
        Default: None
    unique : boolean, optional
        If True, the returned DataFrame will contain unique
        values of the first variable passed in the argument 
        'sort'.
    ignorephaseregions : string, optional
        Ignore phase regions containing a phase whose name
        matches the provided string

    Returns
    -------
    Pandas DataFrame object
    """

    dflist, regionlist = load_table_blocks(fname)

    # Snippet that fix the weird behavior of pd.concat sorting the columns
    # of the dataframes
    columns = []
    newdflist = []
    for df, phases in zip(dflist, regionlist):
        ignore = False
        for ph in phases:
            if fnmatch(ph, ignorephaseregions):
                ignore = True
                break

        if not ignore:
            newdflist.append(df)
            for c in df.columns:
                if c not in columns:
                    columns.append(c)

    df = pd.concat(newdflist)[columns]

    if isinstance(fill, (int, float)):
        df = df.fillna(fill)

    if isinstance(sort, (str, list)):
        try:
            df = df.sort_values(by=sort)
        except KeyError:
            print('Invalid sorting key(s). Data not sorted.')
        except:
            raise

        if unique:
            if isinstance(sort, list):
                sort = sort[0]
            _, idx = np.unique(df[sort], True)
            df = df.iloc[idx]

    return df.reset_index(drop=True)


def plot_table(df, xaxis, ax=None, legend=True, colpattern='*', **kwargs):
    """
    Load table generated by Thermo-Calc and plots data

    Parameters
    ----------
    df : pandas DataFrame object
        DataFrame with table data
    xaxis : str
        Column name used as xaxis in the plot
    ax : AxesSubplot object, optional
        Axes where to plot the table. If not provided, the
        function will automatically create one
    legend : bool, optional
        If True, shows legend
        Default: True
    colpattern : string, optional
        Plot only columns that match colpattern in Unix
        shell-style wildcard
    **kwargs :
        Optional arguments passed to AxesSubplot.plot method

    Returns
    -------
    AxesSubplot object
    """

    if not ax:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

    for c, y in df.items():
        if c != xaxis and fnmatch(c, colpattern):
            ax.plot(df[xaxis], y, label=c, **kwargs)

    if legend:
        ax.legend()

    ax.set_xlabel(xaxis)

    return ax


def interp_table(x, xaxis, df):
    from scipy.interpolate import interp1d

    y = pd.DataFrame()
    if isinstance(x, (int, float)):
        x = [x]

    for c in df.columns:
        if c != xaxis:
            y[c] = interp1d(df[xaxis], df[c])(x)

    return y


def table_blocks_to_tctable(fname, dflist, regionlist):
    with open(fname, 'w') as f:
        for df, region in zip(dflist, regionlist):
            f.write('\n Phase Region for:\n')

            for phase in region:
                f.write('     {:}\n'.format(phase))

            for i, c in enumerate(df.columns):
                f.write(' col-{:d}={:},'.format(i+1, c))
            f.write('\n')

            for i, r in df.iterrows():
                f.write('  ')
                f.write('  '.join(np.char.mod('%.5E', r)))
                f.write('\n')


def table_to_excel(filein, fileout, sort=None, fill=None, **kwargs):
    """
    Convert Thermo-Calc table to Excel table

    Parameters
    ----------
    filein : string
        File name of the Thermo-Calc table used as input
    fileout : string
        File name of the Excel table used as output
    sort : str or list of str, optional
        Argument passed to pandas df.sort_values function.
        Name or list of names which refer to the axis items.
        Default: None
    fill : float, optional
        Fill NA/NaN values with the 'fill' value
        Default: None
    **kwargs :
        Optional arguments passed to pandas.DataFrame
        to_excel method
    """
    df = load_table(filein, sort=sort, fill=fill)
    df.to_excel(fileout, **kwargs)