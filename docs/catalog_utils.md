# catalog_utils.py

这个库里主要包含针对星表的操作：

1. [get_header_from_fits](#get_header_from_fits)
2. [save_fits_catalog](#save_fits_catalog)
3. [extract_ra_dec_col_from_csv_catalog](#extract_ra_dec_col_from_csv_catalog)
4. [build_LAMOST_position_search_file](#build_LAMOST_position_search_file)
5. [build_LAMOST_id_search_file](#build_LAMOST_id_search_file)
6. [byte_by_byte_table_description](#byte_by_byte_table_description)
7. [build_SDSS_position_search_file](#build_SDSS_position_search_file)

---

## get_header_from_fits

从`.fits`格式的星表从读取以`col_name_keyword`为标记的列名称

### 参数

- `fits_file_header`：`fits.header.Header`；`.fits`格式星表的表头
- `col_name_keyword`：`str`，默认为`TTYPE`；列名的关键词

### 返回

`list[str, ...]`

## save_fits_catalog

将`.fits`格式星表转换为`.csv`格式

### 参数

- `fits_catalog_path`：`str`；`.fits`格式星表的路径
- `csv_catalog_save_path`：`str`；`.csv`格式星表的存储路径
- `removed_col_names`：`list[str, ...]`；忽略的列名称

### 返回

`None`

## extract_ra_dec_col_from_csv_catalog

从`.csv`格式的星表中导出坐标

### 参数

- `csv_catalog_path`：`str`；`.csv`格式星表的路径
- `ra_col_name`：`str`；赤经坐标列对应的列名称
- `dec_col_name`：`str`；赤纬坐标列对应的列名称
- `split_char`：`str`，默认为`,`；`.csv`星表的分隔符

### 返回

`list[list[float, float]]`

## build_LAMOST_position_search_file

构建用于LAMOST官网的，以坐标进行范围搜索的文件

### 参数

- `csv_catalog_path`：`str`；`.csv`格式星表的路径
- `ra_col_name`：`str`；赤经坐标列对应的列名称
- `dec_col_name`：`str`；赤纬坐标列对应的列名称
- `save_path`：`str`；导出文件的存储路径
- `radius_arcsec`：`float`，默认为`2.0`角秒；每个坐标的搜索半径
- `split_char`：`str`，默认为`,`；`.csv`星表的分隔符

### 返回

`None`

## build_LAMOST_id_search_file

构建用于LAMOST官网的，以LAMOST天体id进行搜索的文件

### 参数

- `csv_catalog_path`：`str`；`.csv`格式星表的路径
- `id_col_name`：`str`；id列对应的列名称
- `save_path`：`str`；导出文件的存储路径
- `split_char`：`str`，默认为`,`；`.csv`星表的分隔符
- `keep_header`：`bool`，默认为`True`；是否在文件中保存`#id`列名称

### 返回

`None`

## byte_by_byte_table_description

将天文数据发布中类似[apjs519525t4_mrt.txt](http://cdsarc.u-strasbg.fr/viz-bin/qcat?J/A+A/588/A87)的星表转换为`.csv`
格式，其中输入的原始星表只包含数据行！

### 参数

- `table_header_description`：`dict`
  ；原始星表的描述字典，其结构类似`{'col_name': [start_byte, end_byte]} or {'col_name': [start_byte]}`
  ，例如[apjs519525t4_mrt.txt](http://cdsarc.u-strasbg.fr/viz-bin/qcat?J/A+A/588/A87)
  中的`1- 38 A38 --- File     LAMOST 1D FITS file name`可写为`'File': [1, 38]`
- `table_content`：`list`；原始星表的数据行数组
- `start_byte`：`int`，默认为`0`；数据行起始位置，如无必要请勿修改
- `enable_strip`：`bool`，默认为`True`；是否自动删除每行前后的空格

### 返回

`(list, list)`

## build_SDSS_position_search_file

构建用于Casjob，以坐标进行范围搜索的文件

## 参数

- `csv_catalog_path`：`str`；`.csv`格式星表的路径
- `id_col_name`：`str`；id列对应的列名称
- `ra_col_name`：`str`；赤经坐标列对应的列名称
- `dec_col_name`：`str`；赤纬坐标列对应的列名称
- `save_path`：`str`；导出文件的存储路径
- `split_char`：`str`，默认为`,`；`.csv`星表的分隔符

### 返回

`None`

---

Made By