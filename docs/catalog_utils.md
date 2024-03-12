# catalog_utils.py

这个库里主要包含针对星表的操作

[TOC]

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

