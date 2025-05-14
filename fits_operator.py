"""
Fits文件操作 Made By EggTargaryen
"""

import os
import warnings
from typing import Union, Tuple

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torchvision.transforms.functional as trans_func
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from reproject import reproject_interp
from rich import print
from scipy import interpolate
from specutils import Spectrum1D, SpectralRegion
from specutils.analysis import equivalent_width
from specutils.fitting import fit_generic_continuum
from specutils.manipulation import extract_region
from tqdm.rich import tqdm


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
    other_fits_path: list[str],
    bands_order: list[str],
    hdu_index: int,
    post_process: callable = None,
) -> np.ndarray:
    """
    Reproject a FITS image to a target header
    :param target_fits_path: str, target FITS image path
    :param other_fits_path: list[str], other FITS image path
    :param bands_order: list[str], order of bands
    :param hdu_index: int, HDU index
    :param post_process: callable, post process work flow, like SqrtStretch()(MinMaxInterval()(target_data, clip=False))
    :return: C x H x W numpy array
    """
    # Target
    warnings.simplefilter("ignore", AstropyWarning)
    target_hdu = fits.open(target_fits_path)[hdu_index]
    target_header = target_hdu.header
    target_data = target_hdu.data
    if post_process is not None:
        target_data = post_process(target_data)
    stack_img = {
        os.path.basename(target_fits_path)
        .split("-")[1]
        .lower(): np.expand_dims(target_data, axis=-1)
    }
    # Other
    for fits_path in other_fits_path:
        warnings.simplefilter("ignore", AstropyWarning)
        hdu = fits.open(fits_path)[hdu_index]
        reprojected_data, reprojected_footprint = reproject_interp(hdu, target_header)
        if post_process is not None:
            reprojected_data = post_process(reprojected_data)
        stack_img[os.path.basename(fits_path).split("-")[1].lower()] = np.expand_dims(
            reprojected_data, axis=-1
        )
    # stack img following bands_order
    stack_img_list = []
    for band in bands_order:
        stack_img_list.append(stack_img[band])
    stack_img = np.concatenate(stack_img_list, axis=-1)
    return stack_img


def from_world_2_pixel(fits_path: str, hdu_index: int, coord: SkyCoord) -> list[int]:
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
    fits_path: str, hdu_index: int, img: np.ndarray, bbox_size: int, obj_coord: SkyCoord
) -> Union[None, np.ndarray]:
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
            int(obj_x - bbox_size / 2) : int(obj_x + bbox_size / 2),
            int(obj_y - bbox_size / 2) : int(obj_y + bbox_size / 2),
        ]
        return crop_data
    except Exception as e:
        print("[Warning]: crop failed: {}".format(e))
        return None


def print_cross_label_to_img(
    fits_path: str, hdu_index: int, img: np.ndarray, obj_coord: SkyCoord, save_path: str
) -> None:
    """
    Print cross label to image
    :param fits_path: str, path of FITS image, only for getting the header to convert coord
    :param hdu_index: int, index of HDU
    :param img: C x H x W numpy array
    :param obj_coord: SkyCoord, coordinate of object, this, ra and dec
    :param save_path: str, path to save image
    """
    coord_x, coord_y = from_world_2_pixel(
        fits_path, hdu_index=hdu_index, coord=obj_coord
    )
    # min-max normalization
    fits_ndarray = (img - np.min(img)) / (np.max(img) - np.min(img))
    # put obj cross on fits_ndarray
    fits_ndarray[:, int(coord_x), :] = 1
    fits_ndarray[:, :, int(coord_y)] = 1
    # save
    cv2.imwrite(save_path, np.transpose(fits_ndarray * 255, (1, 2, 0)))


def generate_img(
    fits_dir: str,
    hdu_index: int,
    target_band: str,
    other_band: list[str],
    bbox_size: int,
    obj_coord: SkyCoord,
    band_name_match_rule: callable,
    post_process: callable,
) -> Tuple[Union[None, np.ndarray], Union[None, np.ndarray]]:
    """
    Generate an image from FITS files with reproject and stack
    :param fits_dir: str, directory of FITS files
    :param hdu_index: int, HDU index
    :param target_band: str, target band
    :param other_band: list[str], other bands
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
    stack_img = fits_reproject(
        target_fits_path, other_fits_path, hdu_index, post_process
    )
    if stack_img is None:
        return None, None
    # crop
    crop_img = crop_from_fits(
        target_fits_path, hdu_index, stack_img, bbox_size, obj_coord
    )
    return crop_img, stack_img


def SDSS_photo_download_process(
    unique_id: str,
    base_url: str,
    run: str,
    rerun: str,
    camcol: str,
    field: str,
    save_dir: str,
    band: list[str] = None,
) -> None:
    if band is None:
        band = ["u", "g", "r", "i", "z"]
    assert len(band) > 0, "[Error] band must be a list with at least one element"
    urls = [
        "{}/{}/{}/{}/frame-{}-{}-{}-{}.fits.bz2".format(
            base_url, rerun, run, camcol, f, run.zfill(6), camcol, field.zfill(4)
        )
        for f in band
    ]
    create_dir(
        os.path.join(
            save_dir, "{}_{}_{}_{}_{}".format(unique_id, rerun, run, camcol, field)
        )
    )
    for url in urls:
        filename = os.path.basename(url)
        # check if file exists
        if os.path.exists(
            os.path.join(
                save_dir,
                "{}_{}_{}_{}_{}/{}".format(
                    unique_id, rerun, run, camcol, field, filename
                ),
            )
        ):
            print("[Warning] {} already exists".format(filename))
            continue
        else:
            try:
                os.system(
                    "wget {} -T 300 -c -O {}".format(
                        url,
                        os.path.join(
                            save_dir,
                            "{}_{}_{}_{}_{}/{}".format(
                                unique_id, rerun, run, camcol, field, filename
                            ),
                        ),
                    )
                )
                print("[Info] {} downloaded".format(filename))
            except Exception as e:
                print("[Error] {} download failed: {}".format(filename, e))
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
    other_bands: list[str],
    bands_order: list[str],
    crop_size: int,
    up_sample_size: Union[int, None],
    target_coord: SkyCoord,
    save_dir: str,
    hdu_index: int = 0,
    post_process: callable = None,
    fits_file_suffix: str = ".fits.bz2",
    padding_value: float = 0.0,
    png_save_dir: str = None,
) -> None:
    try:
        if os.path.exists(os.path.join(save_dir, f"{unique_id}.npy")):
            print("[Warning] {} already exists".format(unique_id))
        else:
            # check if there have target band + other bands fits file in fits_dir
            target_path = []
            other_paths = []
            for file_name in os.listdir(fits_dir):
                if file_name.endswith(fits_file_suffix):
                    file_band = file_name.split("-")[1].lower()
                    if file_band == target_band:
                        target_path.append(os.path.join(fits_dir, file_name))
                    if file_band in other_bands:
                        other_paths.append(os.path.join(fits_dir, file_name))
            if len(target_path) == 0:
                raise FileNotFoundError(
                    "No target band fits file in {}".format(fits_dir)
                )
            if len(other_paths) == 0 or len(other_paths) != len(other_bands):
                raise FileNotFoundError(
                    "No other bands fits file in {}".format(fits_dir)
                )
            # reproject
            stack_img = fits_reproject(
                target_fits_path=target_path[0],
                other_fits_path=other_paths,
                bands_order=bands_order,
                hdu_index=hdu_index,
                post_process=post_process,
            )
            if stack_img is None:
                raise ValueError("Reproject failed")
            # get obj_coord
            obj_coord = from_world_2_pixel(
                fits_path=target_path[0], hdu_index=hdu_index, coord=target_coord
            )  # x, y
            if (
                obj_coord[0] < 0
                or obj_coord[1] < 0
                or obj_coord[0] > stack_img.shape[0]
                or obj_coord[1] > stack_img.shape[1]
            ):
                raise ValueError("Target coord is out of image")
            # padding
            bg_img = (
                np.ones(
                    (
                        stack_img.shape[0] + 2 * crop_size,
                        stack_img.shape[1] + 2 * crop_size,
                        stack_img.shape[2],
                    )
                )
                * padding_value
            )
            bg_img[
                crop_size : crop_size + stack_img.shape[0],
                crop_size : crop_size + stack_img.shape[1],
                :,
            ] = stack_img
            # crop
            crop_img = bg_img[
                int(obj_coord[0])
                + crop_size
                - crop_size // 2 : int(obj_coord[0])
                + crop_size
                + crop_size // 2,
                int(obj_coord[1])
                + crop_size
                - crop_size // 2 : int(obj_coord[1])
                + crop_size
                + crop_size // 2,
                :,
            ]
            # check nan
            if np.isnan(crop_img).any():
                raise ValueError("{} crop_img contains nan".format(unique_id))
            # check is all 0
            if np.all(crop_img == 0):
                raise ValueError("{} crop_img is all 0".format(unique_id))
            # crop img shape is like H x W x C
            # up sample
            if up_sample_size is not None and up_sample_size != crop_size:
                crop_img = up_sample(crop_img, up_sample_size)
            # save
            # H x W x C -> C x H x W
            crop_img = np.transpose(crop_img, (2, 0, 1))
            # save
            create_dir(save_dir)
            np.save(os.path.join(save_dir, f"{unique_id}.npy"), crop_img)
            # save png
            if png_save_dir:
                create_dir(png_save_dir)
                crop_img = (crop_img - crop_img.min()) / (
                    crop_img.max() - crop_img.min()
                )
                cv2.imwrite(
                    os.path.join(png_save_dir, f"{unique_id}.png"),
                    np.transpose(crop_img * 255, (1, 2, 0)),
                )
    except Exception as e:
        print("[Error]: Skip! {}".format(e))


def spectra_equivalent_width(
    spectra: Spectrum1D, spectra_region: SpectralRegion
) -> Spectrum1D:
    cont_norm_spec = spectra / fit_generic_continuum(spectra)(spectra.spectral_axis)
    return equivalent_width(cont_norm_spec, regions=spectra_region)


def spectra_redshift_correction(spectra: Spectrum1D, redshift: float) -> None:
    spectra.set_redshift_to(redshift)
    spectra.shift_spectrum_to(redshift=0)


def read_spec_fits_file(
    fits_file_path: str, spec_format: str = "SDSS-III/IV spec"
) -> Spectrum1D:
    # set ignore warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Spectrum1D.read(fits_file_path, format=spec_format)


def get_csv_header_col_name(csv_file_path: str) -> list:
    with open(csv_file_path, "r") as f:
        header = f.readline()
    return header.split(",")


def read_lamost_lrs_spectrum(
    fits_path: str,
    enable_calibration: bool = False,
    sdss_calibration_template_dir_obj: dict = None,
    mag_type: str = None,
    mag_list: list[float, float, float, float, float] = None,
):
    if not enable_calibration:
        hdulist = fits.open(fits_path)
        header = hdulist[1].header
        flux = hdulist[1].data["FLUX"]
        wavelength = hdulist[1].data["WAVELENGTH"]
    else:
        spec = LAMOSTSpec(
            fits_path,
            sdss_calibration_template_dir_obj=sdss_calibration_template_dir_obj,
            mag_type=mag_type,
            mag_list=mag_list,
        )
        header = spec.hdr
        wavelength, flux = spec.flux_calibration()
    return header, wavelength, flux


def fits_to_npy_process(
    sub_list: list[str],
    npy_save_dir: str,
    spectra_region: SpectralRegion,
) -> None:
    with tqdm(total=len(sub_list), ncols=150) as pbar:
        for fits_file_path in sub_list:
            basename = os.path.basename(fits_file_path).split(".")[0]
            spectra = read_spec_fits_file(fits_file_path)
            spectra = extract_region(spectra, spectra_region)
            # from Spectrum1D to np.ndarray
            spectra_wavelength = spectra.spectral_axis.value
            spectra_flux = spectra.flux.value
            spectra = np.array([spectra_wavelength, spectra_flux], dtype=np.float32)
            np.save(os.path.join(npy_save_dir, f"{basename}.npy"), spectra)
            pbar.update(1)


def LAMOST_spec_fits_to_npy(
    sub_list: list[str],
    obsid_index: int,
    file_path_index: int,
    npy_save_dir: str,
    spectra_region: list[float, float],
    filter: callable = None,
    enable_calibration: bool = False,
    sdss_calibration_template_dir_obj: dict = None,
    mag_type_index: int = None,
    mag_list_index: list[int, int, int, int, int] = None,
) -> None:
    for row_content in tqdm(sub_list, ncols=150):
        basename = str(row_content[obsid_index])
        # check if npy file exists
        if os.path.exists(os.path.join(npy_save_dir, f"{basename}.npy")):
            continue
        try:
            header, wavelength, flux = read_lamost_lrs_spectrum(
                row_content[file_path_index],
                enable_calibration=enable_calibration,
                sdss_calibration_template_dir_obj=sdss_calibration_template_dir_obj,
                mag_type=(
                    row_content[mag_type_index] if mag_type_index is not None else None
                ),
                mag_list=(
                    [
                        float(row_content[mag_list_index[0]]),
                        float(row_content[mag_list_index[1]]),
                        float(row_content[mag_list_index[2]]),
                        float(row_content[mag_list_index[3]]),
                        float(row_content[mag_list_index[4]]),
                    ]
                    if mag_list_index is not None
                    else None
                ),
            )
            # cut spectrum to keep only the region of interest
            wavelength_index = np.where(
                (wavelength >= spectra_region[0]) & (wavelength <= spectra_region[1])
            )
            wavelength = wavelength[wavelength_index]
            flux = flux[wavelength_index]
            # if flux is all 0 or has nan, skip
            if np.all(flux == 0) or np.isnan(flux).any():
                raise ValueError("flux is all 0 or has nan")
            # if wavelength len <= 0 or flux len <= 0, skip
            if len(wavelength) <= 0 or len(flux) <= 0:
                raise ValueError("wavelength len <= 0 or flux len <= 0")
            if filter is not None and filter(wavelength, flux):
                spectra = np.array([wavelength, flux], dtype=np.float32)
                np.save(os.path.join(npy_save_dir, f"{basename}.npy"), spectra)
            if filter is None:
                spectra = np.array([wavelength, flux], dtype=np.float32)
                np.save(os.path.join(npy_save_dir, f"{basename}.npy"), spectra)
        except Exception as e:
            print(f"[Error] {basename} failed: {e}")
            continue


def DECaLS_photo_download_process(
    ra: str,
    dec: str,
    pixscale: float,
    fits_save_dir: str,
    jpg_save_dir: str,
    bands: list[str],
    obsid: str = None,
    layer: str = "ls-dr10",
    download_jpg: bool = False,
) -> None:
    """
    Download DECaLS photo
    :param ra: the coordinate of the object
    :param dec: the coordinate of the object
    :param pixscale: pixel scale, e.g. when = 0.262, the image is 0.262 arcsec/pixel
    :param fits_save_dir: str, save directory
    :param jpg_save_dir: str, save directory, only used when download_jpg is True
    :param bands: list[str], bands to download, range from ['g', 'r', 'i', 'z']
    :param obsid: str, LAMOST obsid, default is None
    :param layer: survey code, default is 'ls-dr10'
    :param download_jpg: whether download jpg photo, default is False
    :return: None
    """
    bands = "".join(bands)
    fits_base_url = f"https://www.legacysurvey.org/viewer/fits-cutout?ra={ra}&dec={dec}&layer={layer}&pixscale={pixscale}&bands={bands}"
    jpg_base_url = f"https://www.legacysurvey.org/viewer/jpeg-cutout?ra={ra}&dec={dec}&layer={layer}&pixscale={pixscale}"
    create_dir(fits_save_dir)
    create_dir(jpg_save_dir)
    fits_filename = "{}_{}_{}.fits".format(ra, dec, layer)
    jpg_filename = "{}_{}_{}.jpg".format(ra, dec, layer)
    if obsid is not None:
        fits_filename = f"{obsid}_{fits_filename}"
        jpg_filename = f"{obsid}_{jpg_filename}"
    # fits download
    fits_exists_flag = False
    # check if file exists
    if os.path.exists(os.path.join(fits_save_dir, fits_filename)):
        print(f"[Warning] {fits_filename} already exists")
        fits_exists_flag = True
        return
    else:
        try:
            # use requests to download
            response = requests.get(fits_base_url, timeout=300)
            if response.status_code == 200:
                with open(os.path.join(fits_save_dir, fits_filename), "wb") as f:
                    f.write(response.content)
                    f.close()
                fits_exists_flag = True
            else:
                raise ConnectionError(response.status_code)
        except Exception as e:
            print(f"[Error] {fits_filename} download failed: {e}")
            return
    if fits_exists_flag and download_jpg:
        if os.path.exists(os.path.join(jpg_save_dir, jpg_filename)):
            print(f"[Warning] {jpg_filename} already exists")
            return
        else:
            try:
                # use requests to download
                response = requests.get(jpg_base_url, timeout=300)
                if response.status_code == 200:
                    with open(os.path.join(jpg_save_dir, jpg_filename), "wb") as f:
                        f.write(response.content)
                        f.close()
                else:
                    raise ConnectionError(response.status_code)
            except Exception as e:
                print(f"[Error] {jpg_filename} download failed: {e}")
                return


def DESI_fits_reader(
    fits_path: str,
    stack_bands: list[str],
    hdu_index: int = 0,
    post_process: callable = None,
    crop_size: int = None,
) -> Union[np.ndarray, None]:
    """
    Read DESI photometric fits file
    :param fits_path: str, path of fits file
    :param stack_bands: list[str], bands to stack, range from ['g', 'r', 'i', 'z']
    :param hdu_index: int, index of HDU, default is 0
    :param post_process: callable, post process work flow, like SqrtStretch()(MinMaxInterval()(target_data, clip=False))
    :param crop_size: int, size of bbox, default is None
    :return: C x H x W numpy array, or None
    """
    try:
        bands_projection = {
            "g": -1,
            "r": -1,
            "i": -1,
            "z": -1,
        }
        desi_fits = fits.open(os.path.join(fits_path))
        head = desi_fits[hdu_index].header
        # bands check
        if head["BANDS"]:
            exist_bands = str(head["BANDS"])
            band_exist_flag = 0
            for input_band in stack_bands:
                if input_band in exist_bands:
                    band_exist_flag += 1
            if band_exist_flag != len(stack_bands):
                raise ValueError(f"Not all bands in {stack_bands} exist in {fits_path}")
            for band_key in ["BAND{}".format(i) for i in range(len(exist_bands))]:
                bands_projection[head[band_key]] = int(band_key[-1])
            # desi_fits[hdu_index].data: C x H x W
            stack_img = [
                desi_fits[hdu_index].data[bands_projection[band]]
                for band in stack_bands
            ]
            stack_img = np.stack(stack_img, axis=0)
            if post_process is not None:
                stack_img = post_process(stack_img)
            if crop_size is not None:
                stack_img = stack_img[
                    :,
                    int(stack_img.shape[1] / 2 - crop_size / 2) : int(
                        stack_img.shape[1] / 2 + crop_size / 2
                    ),
                    int(stack_img.shape[2] / 2 - crop_size / 2) : int(
                        stack_img.shape[2] / 2 + crop_size / 2
                    ),
                ]
            # check nan
            if np.isnan(stack_img).any():
                raise ValueError("contains nan")
            # check is any channel is all 0
            for i, band in zip(range(stack_img.shape[0]), stack_bands):
                if np.all(stack_img[i] == 0):
                    raise ValueError(f"channel {band} is all 0")
            return stack_img
        else:
            raise ValueError(f"no BANDS key")
    except Exception as e:
        print(f"[Error] {fits_path} failed: {e}")
        return None


def read_LAMOST_spec_SNR(
    lamost_spec_file_path: str, snr_band: str = "SNRG"
) -> (str, str):
    """
    Read LAMOST spectrum SNR
    :param lamost_spec_file_path: LABOST spectrum file path
    :param snr_band: SNR band, default is 'SNRG'
    :return: obsid, snr
    """
    lamost_spec = fits.open(os.path.join(lamost_spec_file_path))
    obsid = lamost_spec[0].header["OBSID"]
    snr = lamost_spec[0].header[snr_band]
    lamost_spec.close()
    return str(obsid), str(snr)


class LAMOSTSpec:
    """
    A simple class to load LAMOST spectrum.
    """

    def __init__(
        self,
        fits_file_path: str,
        sdss_calibration_template_dir_obj: dict,
        redshift: float = None,
        version: str = "New",
        mag_type: str = None,
        mag_list: list[float, float, float, float, float] = None,
    ):
        hdu = fits.open(fits_file_path)
        basename = os.path.basename(fits_file_path)
        self.basename = basename
        hdr = hdu[0].header
        self.hdr = hdr
        self.wave_list = np.array([3557, 4825, 6261, 7672, 9097])  # ugriz
        self.mag_obj = {
            "selected_wave": [],
            "selected_mag": [],
            "selected_band": [],
        }
        self.filter_curve_list, self.filter_curve_fit_list = self.get_curve(
            filter_dir_obj=sdss_calibration_template_dir_obj
        )
        if redshift is None:
            try:
                redshift = hdr["Z"]
            except:
                raise ValueError(
                    "Redshift not provided. "
                    + "Please check the input parameters carefully."
                )
        self.redshift = redshift
        if mag_type is None or mag_list is None:
            if hdr["MAGTYPE"] == "ugriz":
                if hdr["MAG1"] > -900:
                    self.mag_obj["selected_wave"].append(3557)
                    self.mag_obj["selected_mag"].append(hdr["MAG1"])
                    self.mag_obj["selected_band"].append("u")
                if hdr["MAG2"] > -900:
                    self.mag_obj["selected_wave"].append(4825)
                    self.mag_obj["selected_mag"].append(hdr["MAG2"])
                    self.mag_obj["selected_band"].append("g")
                if hdr["MAG3"] > -900:
                    self.mag_obj["selected_wave"].append(6261)
                    self.mag_obj["selected_mag"].append(hdr["MAG3"])
                    self.mag_obj["selected_band"].append("r")
                if hdr["MAG4"] > -900:
                    self.mag_obj["selected_wave"].append(7672)
                    self.mag_obj["selected_mag"].append(hdr["MAG4"])
                    self.mag_obj["selected_band"].append("i")
                if hdr["MAG5"] > -900:
                    self.mag_obj["selected_wave"].append(9097)
                    self.mag_obj["selected_mag"].append(hdr["MAG5"])
                    self.mag_obj["selected_band"].append("z")
            else:
                raise ValueError(
                    "MAGTYPE not recognized. Unable to calibrate the spectrum."
                )
        elif mag_type is not None and mag_list is not None:
            if mag_list[0] is not None and -900 < mag_list[0] < 99:
                self.mag_obj["selected_wave"].append(3557)
                self.mag_obj["selected_mag"].append(mag_list[0])
                self.mag_obj["selected_band"].append("u")
            if mag_list[1] is not None and -900 < mag_list[1] < 99:
                self.mag_obj["selected_wave"].append(4825)
                self.mag_obj["selected_mag"].append(mag_list[1])
                self.mag_obj["selected_band"].append("g")
            if mag_list[2] is not None and -900 < mag_list[2] < 99:
                self.mag_obj["selected_wave"].append(6261)
                self.mag_obj["selected_mag"].append(mag_list[2])
                self.mag_obj["selected_band"].append("r")
            if mag_list[3] is not None and -900 < mag_list[3] < 99:
                self.mag_obj["selected_wave"].append(7672)
                self.mag_obj["selected_mag"].append(mag_list[3])
                self.mag_obj["selected_band"].append("i")
            if mag_list[4] is not None and -900 < mag_list[4] < 99:
                self.mag_obj["selected_wave"].append(9097)
                self.mag_obj["selected_mag"].append(mag_list[4])
                self.mag_obj["selected_band"].append("z")
        else:
            raise ValueError(
                "Header MAGTYPE not recognized, and mag_type and mag_list are not provided."
            )
        if (
            len(self.mag_obj["selected_wave"]) <= 1
            or len(self.mag_obj["selected_mag"]) <= 1
            or len(self.mag_obj["selected_band"]) <= 1
        ):
            raise ValueError("No enough magnitude information in the header.")

        if version == "New":
            data = hdu[1].data
            hdu.close()
            wave = data["WAVELENGTH"][0]
            flux = data["FlUX"][0]
            ivar = pd.Series(data["IVAR"][0])
        else:
            data = hdu[0].data
            hdu.close()
            wave = data[2]
            flux = data[0]
            ivar = pd.Series(data[1])
        ivar.replace(0, np.nan, inplace=True)
        ivar_safe = ivar.interpolate()
        err = 1.0 / np.sqrt(ivar_safe.values)
        flux *= 1e-17
        err *= 1e-17
        self.wave = wave
        self.flux = flux
        self.err = err
        self.flux_calibrated = False
        self.flux_rescaled = None
        self.err_rescaled = None

    @staticmethod
    def get_curve(filter_dir_obj: dict) -> tuple[dict, dict]:
        """
        filter_dir_obj: like {
            'u': 'u.dat',
            'g': 'g.dat',
            'r': 'r.dat',
            'i': 'i.dat',
            'z': 'z.dat',
        }
        """
        filter_fn_list = {
            "u": None,
            "g": None,
            "r": None,
            "i": None,
            "z": None,
        }
        filter_curve_list = {
            "u": None,
            "g": None,
            "r": None,
            "i": None,
            "z": None,
        }
        filter_curve_fit_list = {
            "u": None,
            "g": None,
            "r": None,
            "i": None,
            "z": None,
        }
        for file_key in filter_dir_obj:
            fn = os.path.join(filter_dir_obj[file_key])
            filter_fn_list[file_key] = fn
            filter_curve = np.loadtxt(str(fn))
            filter_curve_list[file_key] = filter_curve
            filter_f = interpolate.interp1d(filter_curve[:, 0], filter_curve[:, 1])
            filter_curve_fit_list[file_key] = filter_f
        return filter_curve_list, filter_curve_fit_list

    @staticmethod
    def synthetic_photo(
        model_wave: np.ndarray,
        model_flux: np.ndarray,
        filter_curve_list: dict,
        filter_curve_fit_list: dict,
        filter_array_index: list[str],
    ) -> np.ndarray:
        """
        work in the observed frame
        calculated the synthetic gri magnitudes from LAMOST spectra
        input flux is the original relative flux from LAMOST
        """
        c = 3e18  # in units of A/s
        photometry_list = np.zeros(len(filter_array_index))
        photometry_list_index = 0
        for filter_key in filter_array_index:
            filter_curve = filter_curve_list[filter_key]
            filter_curve_fit = filter_curve_fit_list[filter_key]
            filter_mask = (model_wave < filter_curve[-1, 0]) & (
                model_wave > filter_curve[0, 0]
            )
            wave = model_wave[filter_mask]
            flux = model_flux[filter_mask]
            transmission = filter_curve_fit(wave)
            n = len(flux)
            if n != 0 and n != 1:
                sum_flambda = np.trapz(flux * transmission * wave, wave)
                sum_transmission = np.trapz(transmission * c / wave, wave)
                photometry_list[photometry_list_index] = (
                    -2.5 * np.log10(sum_flambda / sum_transmission) - 48.6
                )
            else:
                photometry_list[photometry_list_index] = 0
            photometry_list_index += 1
        return photometry_list

    def flux_calibration(self, order=0):
        lamost_mag = self.synthetic_photo(
            self.wave,
            self.flux,
            self.filter_curve_list,
            self.filter_curve_fit_list,
            self.mag_obj["selected_band"],
        )
        sdss_mag = np.array(self.mag_obj["selected_mag"])
        mag_diff = sdss_mag - lamost_mag
        coeffs = np.polyfit(np.asarray(self.mag_obj["selected_wave"]), mag_diff, order)
        lamost_flux_rescaled = self.flux * 10 ** (
            np.polyval(coeffs, self.wave) * (-0.4)
        )
        lamost_flux_err = self.err * 10 ** (np.polyval(coeffs, self.wave) * (-0.4))
        self.flux_rescaled = lamost_flux_rescaled
        self.err_rescaled = lamost_flux_err
        self.flux_calibrated = True
        return self.wave, self.flux_rescaled
