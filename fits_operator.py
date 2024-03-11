"""
SDSS 测光图像操作 Made By EggTargaryen

Before Using:
    pip install astropy opencv-python numpy reproject
"""

import os
import warnings
from typing import Union, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as trans_func
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from reproject import reproject_interp
from specutils import Spectrum1D, SpectralRegion
from specutils.analysis import equivalent_width
from specutils.fitting import fit_generic_continuum
from specutils.manipulation import extract_region
from tqdm import tqdm


def create_dir(path):
    """
    Create a directory if it does not exist
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")


def fits_reproject(
        target_fits_path: str,
        other_fits_path: list[str, ...],
        bands_order: list[str, ...],
        hdu_index: int,
        post_process: callable = None) -> np.ndarray:
    """
    Reproject a FITS image to a target header
    :param target_fits_path: str, target FITS image path
    :param other_fits_path: list[str, ...], other FITS image path
    :param bands_order: list[str, ...], order of bands
    :param hdu_index: int, HDU index
    :param post_process: callable, post process work flow, like SqrtStretch()(MinMaxInterval()(target_data, clip=False))
    :return: C x H x W numpy array
    """
    # Target
    warnings.simplefilter('ignore', AstropyWarning)
    target_hdu = fits.open(target_fits_path)[hdu_index]
    target_header = target_hdu.header
    target_data = target_hdu.data
    if post_process is not None:
        target_data = post_process(target_data)
    stack_img = {
        os.path.basename(target_fits_path).split('-')[1].lower(): np.expand_dims(target_data, axis=-1)
    }
    # Other
    for fits_path in other_fits_path:
        warnings.simplefilter('ignore', AstropyWarning)
        hdu = fits.open(fits_path)[hdu_index]
        reprojected_data, reprojected_footprint = reproject_interp(hdu, target_header)
        if post_process is not None:
            reprojected_data = post_process(reprojected_data)
        stack_img[os.path.basename(fits_path).split('-')[1].lower()] = np.expand_dims(reprojected_data, axis=-1)
    # stack img following bands_order
    stack_img_list = []
    for band in bands_order:
        stack_img_list.append(stack_img[band])
    stack_img = np.concatenate(stack_img_list, axis=-1)
    return stack_img


def from_world_2_pixel(fits_path: str, hdu_index: int, coord: SkyCoord) -> list[float, float]:
    """
    Convert world coordinate to pixel coordinate
    :param fits_path: str, path of FITS image, only for getting the header to convert coord
    :param hdu_index: int, index of HDU
    :param coord: SkyCoord, coordinate of object, this, ra and dec
    """
    # read fits file
    fits_file = fits.open(fits_path)
    # from wcs coord to pixel coord
    wcs = WCS(fits_file[hdu_index].header)
    obj_y, obj_x = wcs.world_to_pixel(coord)
    # close fits file
    fits_file.close()
    return [int(obj_x), int(obj_y)]


def crop_from_fits(
        fits_path: str,
        hdu_index: int,
        img: np.ndarray,
        bbox_size: int,
        obj_coord: SkyCoord) -> Union[None, np.ndarray]:
    """
    Crop a FITS image from a target coordinate
    :param fits_path: str, path of FITS image, only for getting the header to convert coord
    :param hdu_index: int, index of HDU
    :param img: C x H x W numpy array
    :param bbox_size: int, size of bbox
    :param obj_coord: SkyCoord, coordinate of object, this, ra and dec
    :return: C x H x W numpy array
    """
    obj_x, obj_y = from_world_2_pixel(fits_path, hdu_index, obj_coord)
    # crop by bbox_size
    try:
        crop_data = img[
                    :,
                    int(obj_x - bbox_size / 2):int(obj_x + bbox_size / 2),
                    int(obj_y - bbox_size / 2):int(obj_y + bbox_size / 2)]
        return crop_data
    except Exception as e:
        print('[Warning]: crop failed: {}'.format(e))
        return None


def print_cross_label_to_img(
        fits_path: str,
        hdu_index: int,
        img: np.ndarray,
        obj_coord: SkyCoord,
        save_path: str
):
    """
    Print cross label to image
    :param fits_path: str, path of FITS image, only for getting the header to convert coord
    :param hdu_index: int, index of HDU
    :param img: C x H x W numpy array
    :param obj_coord: SkyCoord, coordinate of object, this, ra and dec
    :param save_path: str, path to save image
    """
    coord_x, coord_y = from_world_2_pixel(fits_path, hdu_index=hdu_index, coord=obj_coord)
    # min-max normalization
    fits_ndarray = (img - np.min(img)) / (np.max(img) - np.min(img))
    # put obj cross on fits_ndarray
    fits_ndarray[:, int(coord_x), :] = 1
    fits_ndarray[:, :, int(coord_y)] = 1
    # save
    cv2.imwrite(save_path, np.transpose(fits_ndarray * 255, (1, 2, 0)))


def generate_img(fits_dir: str,
                 hdu_index: int,
                 target_band: str,
                 other_band: list[str, ...],
                 bbox_size: int,
                 obj_coord: SkyCoord,
                 band_name_match_rule: callable,
                 post_process: callable) -> Tuple[Union[None, np.ndarray], Union[None, np.ndarray]]:
    """
    Generate a image from FITS files with reproject and stack
    :param fits_dir: str, directory of FITS files
    :param hdu_index: int, HDU index
    :param target_band: str, target band
    :param other_band: list[str, ...], other bands
    :param bbox_size: int, size of bbox
    :param obj_coord: SkyCoord, coordinate of object, this, ra and dec
    :param band_name_match_rule: callable, a function to match band name, fits_path = band_name_match_rule(band_name, fits_dir)
    :param post_process: callable, post process work flow, like SqrtStretch()(MinMaxInterval()(target_data, clip=False))
    :return: crop_img, stack_img, where both are C x H x W numpy array, following the order of target_band + other_band
    """
    # get target fits path
    target_fits_path = band_name_match_rule(target_band, fits_dir)
    # get other fits path
    other_fits_path = []
    for band in other_band:
        other_fits_path.append(band_name_match_rule(band, fits_dir))
    # get img
    stack_img = fits_reproject(target_fits_path, other_fits_path, hdu_index, post_process)
    if stack_img is None:
        return None, None
    # crop
    crop_img = crop_from_fits(target_fits_path, hdu_index, stack_img, bbox_size, obj_coord)
    return crop_img, stack_img


def SDSS_photo_download_process(
        unique_id: str,
        base_url: str,
        run: str,
        rerun: str,
        camcol: str,
        field: str,
        save_dir: str,
        band: list[str, ...] = None
) -> None:
    if band is None:
        band = ['u', 'g', 'r', 'i', 'z']
    assert len(band) > 0, '[Error] band must be a list with at least one element'
    urls = ['{}/{}/{}/{}/frame-{}-{}-{}-{}.fits.bz2'.format(base_url, rerun, run, camcol, f, run.zfill(6), camcol,
                                                            field.zfill(4)) for f in band]
    create_dir(os.path.join(save_dir, '{}_{}_{}_{}_{}'.format(unique_id, rerun, run, camcol, field)))
    for url in urls:
        filename = os.path.basename(url)
        # check if file exists
        if os.path.exists(os.path.join(save_dir, '{}_{}_{}_{}_{}/{}'.format(unique_id, rerun, run, camcol, field,
                                                                            filename))):
            print('[Warning] {} already exists'.format(filename))
            continue
        else:
            try:
                os.system('wget {} -T 300 -c -O {}'.format(url, os.path.join(save_dir, '{}_{}_{}_{}_{}/{}'.format(
                    unique_id, rerun, run, camcol, field, filename)))
                          )
                print('[Info] {} downloaded'.format(filename))
            except Exception as e:
                print('[Error] {} download failed: {}'.format(filename, e))
                continue


def up_sample(img: np.ndarray, new_size: int) -> np.ndarray:
    """
    Upsample image to new size by torchvision.transforms.functional.resize
    :param img: H x W x C numpy array
    :param new_size:
    :return:
    """
    # to tensor
    img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)))
    # up sample
    img_tensor = trans_func.resize(img_tensor, [new_size, new_size], antialias=True)
    # to numpy
    return np.transpose(img_tensor.numpy(), (1, 2, 0))


def reproject_process(
        fits_dir: str,
        unique_id: str,
        target_band: str,
        other_bands: list[str, ...],
        bands_order: list[str, ...],
        crop_size: int,
        up_sample_size: Union[int, None],
        target_coord: SkyCoord,
        save_dir: str,
        hdu_index: int = 0,
        post_process: callable = None,
        fits_file_suffix: str = '.fits.bz2',
        padding_value: float = 0.0,
):
    try:
        if os.path.exists(os.path.join(save_dir, f'{unique_id}.npy')):
            print('[Warning] {} already exists'.format(unique_id))
        else:
            # check if there have target band + other bands fits file in fits_dir
            target_path = []
            other_paths = []
            for file_name in os.listdir(fits_dir):
                if file_name.endswith(fits_file_suffix):
                    file_band = file_name.split('-')[1].lower()
                    if file_band == target_band:
                        target_path.append(os.path.join(fits_dir, file_name))
                    if file_band in other_bands:
                        other_paths.append(os.path.join(fits_dir, file_name))
            if len(target_path) == 0:
                raise FileNotFoundError('No target band fits file in {}'.format(fits_dir))
            if len(other_paths) == 0 or len(other_paths) != len(other_bands):
                raise FileNotFoundError('No other bands fits file in {}'.format(fits_dir))
            # reproject
            stack_img = fits_reproject(
                target_fits_path=target_path[0],
                other_fits_path=other_paths,
                bands_order=bands_order,
                hdu_index=hdu_index,
                post_process=post_process
            )
            if stack_img is None:
                raise ValueError('Reproject failed')
            # get obj_coord
            obj_coord = from_world_2_pixel(
                fits_path=target_path[0],
                hdu_index=hdu_index,
                coord=target_coord
            )  # x, y
            if (obj_coord[0] < 0 or
                    obj_coord[1] < 0 or
                    obj_coord[0] > stack_img.shape[0] or
                    obj_coord[1] > stack_img.shape[1]):
                raise ValueError('Target coord is out of image')
            # padding
            bg_img = np.ones((stack_img.shape[0] + 2 * crop_size, stack_img.shape[1] + 2 * crop_size,
                              stack_img.shape[2])) * padding_value
            bg_img[crop_size: crop_size + stack_img.shape[0], crop_size: crop_size + stack_img.shape[1], :] = stack_img
            # crop
            crop_img = bg_img[
                       int(obj_coord[0]) + crop_size - crop_size // 2: int(obj_coord[0]) + crop_size + crop_size // 2,
                       int(obj_coord[1]) + crop_size - crop_size // 2: int(obj_coord[1]) + crop_size + crop_size // 2,
                       :]
            # check nan
            if np.isnan(crop_img).any():
                raise ValueError('{} crop_img contains nan'.format(unique_id))
            # check is all 0
            if np.all(crop_img == 0):
                raise ValueError('{} crop_img is all 0'.format(unique_id))
            # crop img shape is like H x W x C
            # up sample
            if up_sample_size is not None and up_sample_size != crop_size:
                crop_img = up_sample(crop_img, up_sample_size)
            # save
            # H x W x C -> C x H x W
            crop_img = np.transpose(crop_img, (2, 0, 1))
            # save
            create_dir(save_dir)
            np.save(os.path.join(save_dir, f'{unique_id}.npy'), crop_img)
    except Exception as e:
        print('[Error]: Skip! {}'.format(e))


def spectra_equivalent_width(spectra: Spectrum1D, spectra_region: SpectralRegion) -> Spectrum1D:
    cont_norm_spec = spectra / fit_generic_continuum(spectra)(spectra.spectral_axis)
    return equivalent_width(cont_norm_spec, regions=spectra_region)


def spectra_redshift_correction(spectra: Spectrum1D, redshift: float) -> None:
    spectra.set_redshift_to(redshift)
    spectra.shift_spectrum_to(redshift=0)


def read_spec_fits_file(fits_file_path: str, spec_format: str = 'SDSS-III/IV spec') -> Spectrum1D:
    # set ignore warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Spectrum1D.read(fits_file_path, format=spec_format)


def get_csv_header_col_name(csv_file_path: str) -> list:
    with open(csv_file_path, 'r') as f:
        header = f.readline()
    return header.split(',')


def read_lamost_lrs_spectrum(fits_path: str):
    """
    Read LAMOST LRS spectrum from fits file.
    :param fits_path:
    :return:
    """
    hdulist = fits.open(fits_path)
    header = hdulist[1].header
    flux = hdulist[1].data['FLUX']
    wavelength = hdulist[1].data['WAVELENGTH']
    return header, wavelength, flux


def fits_to_npy_process(
        sub_list: list[str, ...],
        npy_save_dir: str,
        spectra_region: SpectralRegion,
) -> None:
    with tqdm(total=len(sub_list), ncols=150) as pbar:
        for fits_file_path in sub_list:
            basename = os.path.basename(fits_file_path).split('.')[0]
            spectra = read_spec_fits_file(fits_file_path)
            spectra = extract_region(spectra, spectra_region)
            # from Spectrum1D to np.ndarray
            spectra_wavelength = spectra.spectral_axis.value
            spectra_flux = spectra.flux.value
            spectra = np.array([spectra_wavelength, spectra_flux], dtype=np.float32)
            np.save(os.path.join(
                npy_save_dir,
                f'{basename}.npy'
            ), spectra)
            pbar.update(1)


def LAMOST_spec_fits_to_npy(
        lamost_data_df: pd.DataFrame,
        sub_list: list[str, ...],
        npy_save_dir: str,
        spectra_region: list[float, float],
        filter: callable = None,
) -> None:
    for fits_file_path in tqdm(sub_list, ncols=150):
        basename = os.path.basename(fits_file_path).split('.')[0]
        # check if npy file exists
        if os.path.exists(os.path.join(npy_save_dir, f'{basename}.npy')):
            continue
        try:
            mjd = basename.split('-')[1]
            plan_id, sp_id = basename.split('-')[2].split('_sp')
            fiber_id = basename.split('-')[-1]
            # get target obsid from lamost_data_df
            # print('\n', mjd, plan_id, sp_id, fiber_id)
            target_obsid = lamost_data_df[
                (lamost_data_df['combined_lmjd'] == mjd) &
                (lamost_data_df['combined_planid'] == plan_id) &
                (lamost_data_df['combined_spid'] == sp_id) &
                (lamost_data_df['combined_fiberid'] == fiber_id)
                ]['combined_obsid'].values[0]
            header, wavelength, flux = read_lamost_lrs_spectrum(fits_file_path)
            # cut spectrum to keep only the region of interest
            wavelength_index = np.where(
                (wavelength >= spectra_region[0]) &
                (wavelength <= spectra_region[1])
            )
            wavelength = wavelength[wavelength_index]
            flux = flux[wavelength_index]
            # if flux is all 0 or has nan, skip
            if np.all(flux == 0) or np.isnan(flux).any():
                raise ValueError('{} flux is all 0 or has nan'.format(basename))
            # if wavelength len <= 0 or flux len <= 0, skip
            if len(wavelength) <= 0 or len(flux) <= 0:
                raise ValueError('{} wavelength len <= 0 or flux len <= 0'.format(basename))
            if filter is not None and filter(wavelength, flux):
                spectra = np.array([wavelength, flux], dtype=np.float32)
                np.save(os.path.join(
                    npy_save_dir,
                    f'{target_obsid}.npy'
                ), spectra)
            if filter is None:
                spectra = np.array([wavelength, flux], dtype=np.float32)
                np.save(os.path.join(
                    npy_save_dir,
                    f'{target_obsid}.npy'
                ), spectra)
        except Exception as e:
            print(f'[Error] {basename} failed: {e}')
            continue
