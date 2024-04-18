"""
星表操作 Made By EggTargaryen
"""

from astropy.io import fits
from rich import print
from tqdm.rich import tqdm


def get_header_from_fits(
        fits_file_header: fits.header.Header,
        col_name_keyword: str = 'TTYPE'
) -> list[str]:
    """
    :param fits_file_header: fits file header
    :param col_name_keyword: keyword for column names
    :return: list of column names
    """
    col_names = []
    for keyword in fits_file_header.keys():
        if col_name_keyword in keyword:
            col_names.append(fits_file_header[keyword])
    return col_names


def save_fits_catalog(
        fits_catalog_path: str,
        csv_catalog_save_path: str,
        removed_col_names: list[str] = []
) -> None:
    """
    convert fits catalog to csv catalog
    :param fits_catalog_path: the path of fits catalog
    :param csv_catalog_save_path: csv catalog save path
    :param removed_col_names: removed column names
    :return: None
    """
    catalog = fits.open(fits_catalog_path)
    header = get_header_from_fits(catalog[1].header)
    new_header = []
    for col_name in header:
        if col_name not in removed_col_names:
            new_header.append(col_name)
    keep_col_idx = [header.index(col_name) for col_name in new_header]
    print('[INFO] Header: ', new_header)
    data = catalog[1].data
    print('[INFO] Data Length: ', len(data))
    with open(csv_catalog_save_path, 'w') as f:
        f.write(','.join(new_header) + '\n')
        for row in tqdm(data, ncols=100, desc='[INFO] Saving CSV'):
            row = [
                str(row[idx]) if str(row[idx]) != '' else '-'
                for idx in keep_col_idx
            ]
            _line = ','.join(row).replace(' ', '_')
            if len(_line.split(',')) != len(new_header):
                print('\n[ERROR] Line: ', _line)
                continue
            f.write(_line + '\n')
        f.close()


def extract_ra_dec_col_from_csv_catalog(
        csv_catalog_path: str,
        ra_col_name: str,
        dec_col_name: str,
        split_char: str = ','
) -> list:
    """
    extract ra and dec column from csv catalog
    :param csv_catalog_path: csv catalog path
    :param ra_col_name: ra column name
    :param dec_col_name: dec column name
    :param split_char: split character
    :return: ra, dec list
    """
    ra_dec_list = []
    # read csv catalog
    with open(csv_catalog_path, 'r') as f:
        lines = f.readlines()
        f.close()
        header = lines[0].strip().split(split_char)
        ra_idx = header.index(ra_col_name)
        dec_idx = header.index(dec_col_name)
        if ra_idx == -1 or dec_idx == -1:
            raise ValueError(
                f'ra_col_name: {ra_col_name} or dec_col_name: {dec_col_name} '
                f'not in header: {header}'
            )
        for line in tqdm(lines[1:], ncols=100, desc='[INFO] Extracting RA/DEC'):
            line = line.strip().split(split_char)
            ra_dec_list.append([float(line[ra_idx]), float(line[dec_idx])])
    return ra_dec_list


def build_LAMOST_position_search_file(
        csv_catalog_path: str,
        ra_col_name: str,
        dec_col_name: str,
        save_path: str,
        radius_arcsec: float = 2.0,
        split_char: str = ','
) -> None:
    """
    build LAMOST search file
    :param csv_catalog_path: csv catalog path
    :param ra_col_name: ra column name
    :param dec_col_name: dec column name
    :param save_path: save path
    :param radius_arcsec: radius in arcsec
    :param split_char: split character
    :return: None
    """
    ra_dec_list = extract_ra_dec_col_from_csv_catalog(
        csv_catalog_path, ra_col_name, dec_col_name, split_char
    )
    with open(save_path, 'w') as f:
        # write header
        f.write('#ra,dec,radius\n')
        for ra, dec in tqdm(ra_dec_list, ncols=100, desc='[INFO] Building LAMOST Search File'):
            f.write(f'{ra},{dec},{radius_arcsec}\n')
        f.close()


def build_LAMOST_id_search_file(
        csv_catalog_path: str,
        id_col_name: str,
        save_path: str,
        split_char: str = ',',
        keep_header: bool = True
) -> None:
    """
    build LAMOST search file
    :param csv_catalog_path: csv catalog path
    :param id_col_name: id column name
    :param save_path: save path
    :param split_char: split character
    :param keep_header: keep header
    :return:
    """
    id_list = []
    # read csv catalog
    with open(csv_catalog_path, 'r') as f:
        lines = f.readlines()
        f.close()
        header = lines[0].strip().split(split_char)
        id_idx = header.index(id_col_name)
        if id_idx == -1:
            raise ValueError(
                f'id_col_name: {id_col_name} not in header: {header}'
            )
        for line in tqdm(lines[1:], ncols=100, desc='[INFO] Extracting ID'):
            line = line.strip().split(split_char)
            id_list.append(line[id_idx])
    with open(save_path, 'w') as f:
        if keep_header:
            # write header
            f.write('#id\n')
        for _id in tqdm(id_list, ncols=100, desc='[INFO] Building LAMOST Search File'):
            f.write(f'{_id}\n')
        f.close()


def byte_by_byte_table_description(
        table_header_description: dict,
        table_content: list,
        start_byte: int = 0,
        enable_strip: bool = True
) -> (list, list):
    """
    byte by byte table description
    :param table_header_description: table header description, \
    e.g. {'col_name': [start_byte, end_byte]} or {'col_name': [start_byte]}
    :param table_content: table content
    :param start_byte: start byte index of table content, 0 or other
    :param enable_strip: enable strip
    :return: header, content
    """
    # check table header description
    for col_name, byte_range in table_header_description.items():
        if len(byte_range) > 2 or len(byte_range) < 1:
            raise ValueError(
                f'byte_range: {byte_range} '
                f'length should be 1 or 2, but got {len(byte_range)}'
            )
    header = []
    content = []
    for row in tqdm(table_content, ncols=100, desc='[INFO] Byte by Byte Table Description'):
        _row = []
        for col_name, byte_range in table_header_description.items():
            _start_index = byte_range[0] - start_byte
            if len(byte_range) == 2:
                _end_index = byte_range[1] - start_byte + 1
                if enable_strip:
                    _row.append(row[_start_index:_end_index].strip())
                else:
                    _row.append(row[_start_index:_end_index])
            else:
                _row.append(row[_start_index])
        if len(_row) != len(table_header_description):
            raise ValueError(
                f'row: {_row} length should be '
                f'{len(table_header_description)}, but got {len(_row)}'
            )
        content.append(_row)
    for col_name in table_header_description.keys():
        header.append(col_name)
    return header, content


def build_SDSS_position_search_file(
        csv_catalog_path: str,
        id_col_name: str,
        ra_col_name: str,
        dec_col_name: str,
        save_path: str,
        split_char: str = ','
) -> None:
    """
    build SDSS search file
    :param csv_catalog_path: csv catalog path
    :param id_col_name: id column name
    :param ra_col_name: ra column name
    :param dec_col_name: dec column name
    :param save_path: save path
    :param split_char: split character
    :return:
    """
    content = []
    # read csv catalog
    with open(csv_catalog_path, 'r') as f:
        lines = f.readlines()
        f.close()
        header = lines[0].strip().split(split_char)
        id_idx = header.index(id_col_name)
        ra_idx = header.index(ra_col_name)
        dec_idx = header.index(dec_col_name)
        for line in tqdm(lines[1:], ncols=100, desc='[INFO] Extracting ID/RA/DEC'):
            line = line.strip().split(split_char)
            content.append([line[id_idx], line[ra_idx], line[dec_idx]])
    with open(save_path, 'w') as f:
        # write header
        f.write('name ra dec\n')
        for _id, ra, dec in tqdm(content, ncols=100, desc='[INFO] Building SDSS Search File'):
            f.write(f'{_id} {ra} {dec}\n')
        f.close()
