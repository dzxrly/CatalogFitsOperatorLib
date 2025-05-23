import datetime
import os.path
import re

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from astroquery.esa.euclid import EuclidClass
from rich import print


def modify_adql_query(query: str, object_ids: list[str | int]) -> str:
    """
    Modify an ADQL query by adding object_id equality constraints to the main table's WHERE clause.

    Parameters
    ----------
    query : str
        Input ADQL query string to be modified.
    object_ids : list of str or int
        List of object IDs to include in the equality constraints.

    Returns
    -------
    str
        Modified ADQL query with the new object_id equality constraints added to the WHERE clause.
    """
    if not object_ids:
        raise ValueError("object_ids list cannot be empty")

    # Normalize whitespace to simplify parsing
    normalized_query = " ".join(query.split())

    # Find the main table alias (FROM clause)
    from_match = re.search(
        r"FROM\s+([^\s]+)\s+AS\s+([^\s]+)", normalized_query, re.IGNORECASE
    )
    if not from_match:
        raise ValueError("Could not identify main table and alias in FROM clause")

    main_table = from_match.group(1)
    main_alias = from_match.group(2)

    # Prepare new condition with object_id equality constraints
    # Convert IDs to strings and ensure proper quoting for strings
    conditions = [
        (
            f"{main_alias}.object_id = {id!r}"
            if isinstance(id, str)
            else f"{main_alias}.object_id = {id}"
        )
        for id in object_ids
    ]
    new_condition = " OR ".join(conditions)

    # Find and modify WHERE clause
    where_match = re.search(
        r"WHERE\s+(.+?)(?=(?:\s+INNER\s+JOIN|$))",
        normalized_query,
        re.IGNORECASE | re.DOTALL,
    )
    if where_match:
        # WHERE clause exists, append new condition
        original_where = where_match.group(1).strip()
        modified_where = f"WHERE {original_where} AND ({new_condition})"
        # Replace original WHERE clause with modified one
        modified_query = (
            normalized_query[: where_match.start()]
            + modified_where
            + normalized_query[where_match.end() :]
        )
    else:
        # No WHERE clause, add one after FROM clause
        from_end = from_match.end()
        modified_query = (
            normalized_query[:from_end]
            + f" WHERE {new_condition}"
            + normalized_query[from_end:]
        )
    return modified_query


class EuclidOps:
    def __init__(
        self, credentials_file: str, environment: str = "PDR", verbose: bool = False
    ):
        super().__init__()
        self.credentials_file = credentials_file
        self.verbose = verbose
        self.euclid = EuclidClass(environment=environment, verbose=self.verbose)
        # login
        self.euclid.login(
            credentials_file=credentials_file,
            verbose=self.verbose,
        )

    def launch_adql_query(
        self,
        adql_query: str,
        job_name: str | None = None,
        verbose: bool | None = None,
    ) -> pd.DataFrame:
        """
        Launch an ADQL query to the Euclid database and return the results as a pandas DataFrame.

        Parameters
        ----------
        adql_query : str
            The ADQL query to be executed
        job_name : str
            The name of the job. If None, a timestamp will be used.
        verbose : bool
            If True, print verbose output.
        """
        if job_name is None:
            job_name = f"QUERY_{datetime.datetime.now().timestamp()}"
        # launch the job
        job = self.euclid.launch_job_async(
            adql_query,
            name=job_name,
            dump_to_file=False,
            verbose=self.verbose if verbose is None else verbose,
        )
        # get the results
        res = job.get_results().to_pandas()
        return res

    def fetch_product_list(
        self,
        obs_id: str | int | None = None,
        tile_index: str | int | None = None,
        product_type: str = "DpdMerBksMosaic",
        verbose: bool = None,
        to_list: bool = False,
    ) -> pd.DataFrame | list[dict]:
        """
        Fetch the product list from the Euclid database.

        Parameters
        ----------
        obs_id : str | int | None
            The observation ID. If None, the tile index will be used.
        tile_index : str | int | None
            The tile index. If None, the observation ID will be used.
        product_type : str
            The type of product to be fetched. Default is "DpdMerBksMosaic".
        verbose : bool
            If True, print verbose output.
        to_list : bool
            If True, return the results as a list[dict]. Default is False.

        Returns
        -------
        pd.DataFrame | list[dict]
            The product list as a pandas DataFrame or a list[dict].
        """
        if obs_id is not None:
            product_list_results = self.euclid.get_product_list(
                observation_id=obs_id,
                product_type=product_type,
                verbose=self.verbose if verbose is None else verbose,
            ).to_pandas()
        elif tile_index is not None:
            product_list_results = self.euclid.get_product_list(
                tile_index=tile_index,
                product_type=product_type,
                verbose=self.verbose if verbose is None else verbose,
            ).to_pandas()
        else:
            raise ValueError("Either obs_id or tile_index must be provided.")
        # convert to list[dict] if required
        if to_list:
            product_list_results = product_list_results.to_dict(orient="records")
        return product_list_results

    def download_full_tile_product(
        self, file_name: str, save_dir: str, schema: str = "sedm", verbose: bool = None
    ) -> str | None:
        """
        Download the full tile product from the Euclid database.

        Parameters
        ----------
        file_name : str
            The name of the file to be downloaded.
        save_dir : str
            The directory where the file will be saved.
        schema : str
            The data release name (schema) in which the product should be searched
        verbose : bool
            If True, print verbose output.
        """
        try:
            path = self.euclid.get_product(
                file_name=file_name,
                output_file=save_dir,
                schema=schema,
                verbose=self.verbose if verbose is None else verbose,
            )
            return path
        except Exception as e:
            print(f"[ERROR] downloading product {file_name}: {e}")
            return None

    def download_cutout_by_full_info(
        self,
        ra: float | str,
        dec: float | str,
        radius: float | str,
        tile_id: str | int,
        data_release: str,
        data_type: str,
        instrument: str,
        data_server_url: str,
        file_name: str,
        save_dir: str,
        verbose: bool = None,
    ) -> str | None:
        """
        Download the cutout product from the Euclid database.

        Parameters
        ----------
        ra : float | str
            The right ascension of the cutout center.
        dec : float | str
            The declination of the cutout center.
        radius : float | str
            The radius of the cutout in arcseconds.
        tile_id : str | int
            The tile ID.
        data_release : str
            The data release name (schema) in which the product should be searched. Such as "Q1_R1".
        data_type : str
            The type of data to be downloaded. Such as "MER"
        instrument : str
            The instrument used to capture the data. Such as "VIS".
        data_server_url : str
            The URL of the data server. Such as "/euclid/repository_idr/iqr1"
        file_name : str
            The name of the file to be downloaded.
        save_dir : str
            The directory where the file will be saved.
        verbose : bool
            If True, print verbose output.
        """
        try:
            coord = SkyCoord(
                ra=ra,
                dec=dec,
                unit="deg",
                frame="icrs",
            )
            radius = float(radius) * u.arcsec
            saved_cutout_filepath = self.euclid.get_cutout(
                file_path=os.path.join(
                    str(data_server_url),
                    str(data_release),
                    str(data_type),
                    str(tile_id),
                    str(instrument),
                    str(file_name),
                ),
                instrument=instrument,
                id=tile_id,
                coordinate=coord,
                radius=radius,
                output_file=os.path.join(save_dir, f"{file_name}_CUTOUT.fits"),
                verbose=self.verbose if verbose is None else verbose,
            )
            if saved_cutout_filepath is not None:
                return saved_cutout_filepath[0]
            else:
                print(f"[ERROR] cutout product {file_name} not found.")
                return None
        except Exception as e:
            print(f"[ERROR] downloading cutout product: {e}")
            return None

    def download_cutout_by_product(
        self,
        ra: float | str,
        dec: float | str,
        radius: float | str,
        data_server_url: str,
        data_type: str,
        product_info: dict,
        file_name: str,
        save_dir: str,
        verbose: bool = None,
    ) -> str | None:
        """
        Download the cutout product from the Euclid database.

        Parameters
        ----------
        ra : float | str
            The right ascension of the cutout center.
        dec : float | str
            The declination of the cutout center.
        radius : float | str
            The radius of the cutout in arcseconds.
        data_server_url : str
            The URL of the data server. Such as "/euclid/repository_idr/iqr1"
        data_type : str
            The type of data to be downloaded. Such as "MER"
        product_info : dict
            The product information dictionary containing keys:
                "tile_index",
                "instrument_name",
                "release_name",
        file_name : str
            The name of the file to be downloaded.
        save_dir : str
            The directory where the file will be saved.
        verbose : bool
            If True, print verbose output.
        """
        try:
            coord = SkyCoord(
                ra=ra,
                dec=dec,
                unit="deg",
                frame="icrs",
            )
            radius = float(radius) * u.arcsec
            saved_cutout_filepath = self.euclid.get_cutout(
                file_path=os.path.join(
                    str(data_server_url),
                    str(product_info["release_name"]),
                    str(data_type),
                    str(product_info["tile_index"]),
                    str(product_info["instrument_name"]),
                    str(file_name),
                ),
                instrument=product_info["instrument_name"],
                id=product_info["tile_index"],
                coordinate=coord,
                radius=radius,
                output_file=os.path.join(save_dir, f"{file_name}_CUTOUT.fits"),
                verbose=self.verbose if verbose is None else verbose,
            )
            if saved_cutout_filepath is not None:
                return saved_cutout_filepath[0]
            else:
                print(f"[ERROR] cutout product {file_name} not found.")
                return None
        except Exception as e:
            print(f"[ERROR] downloading cutout product: {e}")
            return None

    def download_cutout_batch(
        self,
        ra: float | str,
        dec: float | str,
        radius: float | str,
        data_server_url: str,
        data_type: str,
        save_dir: str,
        include_bands: list[str],
        skip_when_band_not_found: bool = True,
        obs_id: str | int | None = None,
        tile_index: str | int | None = None,
        product_type: str = "DpdMerBksMosaic",
        verbose: bool = None,
    ) -> None:
        """
        Download a list of cutout products from the Euclid database.

        Parameters
        ----------
        ra : float | str
            The right ascension of the cutout center.
        dec : float | str
            The declination of the cutout center.
        radius : float | str
            The radius of the cutout in arcseconds.
        data_server_url : str
            The URL of the data server. Such as "/euclid/repository_idr/iqr1"
        data_type : str
            The type of data to be downloaded. Such as "MER"
        save_dir : str
            The directory where the cutout products will be saved.
        include_bands : list[str]
            List of bands to include in the cutout products. Choose from ["VIS", "NIR-Y/J/H", "DES-G/R/I/Z"].
        skip_when_band_not_found : bool
            If True, skip the cutout product if the band is not found. Default is True.
        obs_id : str | int | None
            The observation ID. If None, the tile index will be used.
        tile_index : str | int | None
            The tile index. If None, the observation ID will be used.
        product_type : str
            The type of product to be fetched. Default is "DpdMerBksMosaic".
        verbose : bool
            If True, print verbose output.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # fetch the product list
        product_list_results = self.fetch_product_list(
            obs_id=obs_id,
            tile_index=tile_index,
            product_type=product_type,
            verbose=verbose,
            to_list=True,
        )
        if verbose:
            print(product_list_results)
        _temp_res = []
        for _product in product_list_results:
            _leak_band_flag = True
            for band in include_bands:
                if band in _product["file_name"]:
                    _temp_res.append(_product)
                    _leak_band_flag = False
                    break
            if _leak_band_flag and not skip_when_band_not_found:
                raise ValueError(
                    f"[ERROR] Leak bands for obs_id: {obs_id} or tile_index: {tile_index}"
                )
        if len(_temp_res) == 0:
            raise ValueError(
                f"[ERROR] No product found for obs_id: {obs_id} or tile_index: {tile_index}"
            )
        # download the cutout products
        for _product in _temp_res:
            file_name = _product["file_name"]
            saved_cutout_filepath = self.download_cutout_by_product(
                ra=ra,
                dec=dec,
                radius=radius,
                data_server_url=data_server_url,
                data_type=data_type,
                product_info=_product,
                file_name=file_name,
                save_dir=save_dir,
                verbose=verbose,
            )
            if saved_cutout_filepath is None:
                raise ValueError(
                    f"[ERROR] cutout product {file_name} not found for obs_id: {obs_id} or tile_index: {tile_index}"
                )

    def download_spectrum(
        self,
        source_id: str | int,
        save_dir: str,
        retrieval_type: str = "ALL",
        schema: str = "sedm",
        verbose: bool = None,
    ) -> None:
        """
        Download the spectrum product from the Euclid database.

        Parameters
        ----------
        source_id : str | int
            The source ID.
        save_dir : str
            The directory where the spectrum product will be saved.
        retrieval_type : str
            The type of retrieval, choose in ["ALL", "SPECTRA_BGS", "SPECTRA_RGS"]. Default is "ALL".
        schema : str
            The data release name (schema) in which the product should be searched. Default is "sedm".
        verbose : bool
            If True, print verbose output.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # download the spectrum product
        try:
            self.euclid.get_spectrum(
                source_id=source_id,
                output_file=os.path.join(
                    save_dir,
                    f"{schema}_{source_id}_{retrieval_type}",
                    f"{schema}_{source_id}_{retrieval_type}.zip",
                ),
                retrieval_type=retrieval_type,
                schema=schema,
                verbose=self.verbose if verbose is None else verbose,
            )
        except Exception as e:
            print(f"[ERROR] downloading spectrum product: {e}")
