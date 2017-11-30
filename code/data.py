"""
Module for conveniently loading relevant data
"""
import os
import pandas as pd
import numpy as np
import astropy.units as u
import astropy.coordinates as coords

__all__ = [
    "A17"
]

_DATADIR = os.path.dirname(os.path.dirname(__file__)[:-1])+'/data'

class A17(object):
    """
    Class representing raw and transformed data from Andrews et al. 2017

    Attributes
    ----------
    df : pandas.DataFrame
        Raw data as is.
    star : pandas.DataFrame
        Star table of unique stars
    pair : pandas.DataFrame
        Pair table of pair indices into star table and pair parameters.
    star_coords : astropy.coordinates.SkyCoord
        Star coordinates, row-by-row match to star table.
    """
    def __init__(self):

        df = pd.read_csv(_DATADIR+'/andrews17_tgas_catalog.csv')
        # make star table of unique stars
        star1 = df[['source_id_1', 'TYC_ID_1', 'ra_1', 'dec_1', 'mu_ra_1', 'mu_dec_1',
                    'mu_ra_err_1', 'mu_dec_err_1', 'plx_1', 'plx_err_1']]
        star1.columns = star1.columns.str.replace('_1', '')
        star2 = df[['source_id_2', 'TYC_ID_2', 'ra_2', 'dec_2', 'mu_ra_2', 'mu_dec_2',
                    'mu_ra_err_2', 'mu_dec_err_2', 'plx_2', 'plx_err_2']]
        star2.columns = star2.columns.str.replace('_2', '')
        star = pd.concat([star1, star2])
        star.drop_duplicates('TYC_ID', inplace=True)
        star['idx'] = np.arange(len(star))

        star_coords = coords.ICRS(
            star.ra.values*u.deg,
            star.dec.values*u.deg,
            1000./star.plx.values*u.pc)
        star['vra'], star['vdec'] = (star.mu_ra/star.plx*4.74,
                                     star.mu_dec/star.plx*4.74)
        star['gx'], star['gy'], star['gz'] = star_coords\
            .transform_to(coords.Galactic).cartesian.xyz.value

        # The catalog is actually a pair table; only keep pair quantities
        tmp = pd.merge(df, star[['idx', 'TYC_ID']],
                       left_on='TYC_ID_1', right_on='TYC_ID')\
                .drop('TYC_ID', axis=1)
        tmp = pd.merge(tmp, star[['idx', 'TYC_ID']],
                       left_on='TYC_ID_2', right_on='TYC_ID',
                       suffixes=('1', '2'))\
                .drop('TYC_ID', axis=1)
        pair = tmp[['idx1','idx2','P_log_flat','P_power_law', 'theta']].copy()
        pair_angsep = star_coords[pair.idx1].separation(star_coords[pair.idx2]).to(u.rad).value
        pair_projected_sep = pair_angsep*star_coords[pair.idx1].distance.to(u.pc).value
        pair_sep3d = star_coords[pair.idx1].separation_3d(star_coords[pair.idx2]).to(u.pc).value

        pair_dvra, pair_dvdec = (star.vra.values[pair.idx1]-star.vra.values[pair.idx2],
                                star.vdec.values[pair.idx1]-star.vdec.values[pair.idx2])
        pair_dvtan = np.hypot(pair_dvra, pair_dvdec)
        pair_dpm = np.hypot(star['mu_ra'].values[pair.idx1]-star['mu_ra'].values[pair.idx2],
                            star.mu_dec.values[pair.idx1]-star.mu_dec.values[pair.idx2])
        pair['angsep_rad'] = pair_angsep
        pair['projected_sep'] = pair_projected_sep
        pair['sep3d'] = pair_sep3d
        pair['dvra'], pair['dvdec'] = pair_dvra, pair_dvdec
        pair['dvtan'] = pair_dvtan
        pair['dpm'] = pair_dpm

        self.df = df
        self.star = star
        self.star_coords = star_coords
        self.pair = pair
