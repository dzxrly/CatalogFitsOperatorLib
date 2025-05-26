# euclid_ops.py

包含针对Euclid数据的操作：

1. [EuclidClass](#EuclidClass)
2. [launch_adql_query](#launch_adql_query)
3. [fetch_product_list](#fetch_product_list)
4. [download_full_tile_product](#download_full_tile_product)
5. [download_cutout_by_full_info](#download_cutout_by_full_info)
6. [download_cutout_by_product](#download_cutout_by_product)
7. [download_cutout_batch](#download_cutout_batch)
8. [download_spectrum](#download_spectrum)

---

## EuclidClass

### 参数

- `credentials_file`：`str`；登录ESA Datalabs的凭证文件路径，该文件为纯文本文件，第一行为用户名，第二行为密码
- `environment`：`str`，默认为`PDR`；Euclid数据发布的名称
- `verbose`：`bool`，默认为`False`；是否打印详细信息

### 返回

`None`

## launch_adql_query

### 参数

- `adql_query`：`str`；ADQL查询语句
- `job_name`: `str | None`；查询任务名称, 默认为`QUERY_当前时间戳`
- `verbose`：`bool | None`；是否打印详细信息，默认为`None`，即跟随`EuclidClass`的设置

### 返回

- `pd.DataFrame`：查询结果

## fetch_product_list

### 参数

- `obs_id`: `str | int | None`；观测目标ID，即`observation id`，默认为`None`
- `tile_index`: `str | int | None`；观测分片ID，即`tile id`，默认为`None`。`obs_id`和`tile_id`不能同时为`None`且只能选择一项设置
- `product_type`: `str`；Euclid数据发布类型，默认为`DpdMerBksMosaic`，具体请参考[EuclidClass.get_product_list](https://astroquery.readthedocs.io/en/latest/_modules/astroquery/esa/euclid/core.html#EuclidClass.get_product_list)
- `verbose`：`bool | None`；是否打印详细信息，默认为`None`，即跟随`EuclidClass`的设置
- `to_list`：`bool`；是否将结果转换为字典类列表，默认为`False`

### 返回

- `pd.DataFrame | list[dict]`：该天区的Fits文件列表

## download_full_tile_product

### 参数

- `file_name`: `str`；Euclid Fits文件名称
- `save_dir`: `str`；保存路径
- `schema`: `str`；Euclid Schema，默认为`sedm`，具体请参考[EuclidClass.get_product](https://astroquery.readthedocs.io/en/latest/_modules/astroquery/esa/euclid/core.html#EuclidClass.get_product)
- `verbose`：`bool | None`；是否打印详细信息，默认为`None`，即跟随`EuclidClass`的设置

### 返回

- `str`：下载的Fits文件路径

## download_cutout_by_full_info

### 参数

- `ra`：`float | str`；CUTOUT的赤经中心点
- `dec`：`float | str`；CUTOUT的赤纬中心点
- `radius`：`float | str`；CUTOUT的半径（角秒）
- `tile_id`：`str | int`；观测分片ID
- `data_release`：`str`；数据发布名称，例如"Q1_R1"
- `data_type`：`str`；数据类型，例如"MER"
- `instrument`：`str`；使用的仪器，例如"VIS"
- `data_server_url`：`str`；数据服务器URL，例如"/euclid/repository_idr/iqr1"
- `file_name`：`str`；要下载的文件名
- `save_dir`：`str`；保存路径
- `verbose`：`bool | None`；是否打印详细信息，默认为`None`，即跟随`EuclidClass`的设置

### 返回

- `str | None`：下载的Fits文件路径，如果下载失败则返回`None`

## download_cutout_by_product

### 参数

- `ra`：`float | str`；CUTOUT的赤经中心点
- `dec`：`float | str`；CUTOUT的赤纬中心点
- `radius`：`float | str`；CUTOUT的半径（角秒）
- `data_server_url`：`str`；数据服务器URL，例如"/euclid/repository_idr/iqr1"
- `data_type`：`str`；数据类型，例如"MER"
- `product_info`：`dict`；产品信息字典，包含以下键：
    - `tile_index`：观测分片ID
    - `instrument_name`：仪器名称
    - `release_name`：发布名称
- `file_name`：`str`；要下载的文件名
- `save_dir`：`str`；保存路径
- `skip_when_exists`：`bool`；当文件已存在时是否跳过下载，默认为`True`
- `verbose`：`bool | None`；是否打印详细信息，默认为`None`，即跟随`EuclidClass`的设置

### 返回

- `str | None`：下载的Fits文件路径，如果下载失败则返回`None`

## download_cutout_batch

### 参数

- `ra`：`float | str`；CUTOUT的赤经中心点
- `dec`：`float | str`；CUTOUT的赤纬中心点
- `radius`：`float | str`；CUTOUT的半径（角秒）
- `data_server_url`：`str`；数据服务器URL，例如"/euclid/repository_idr/iqr1"
- `data_type`：`str`；数据类型，例如"MER"
- `save_dir`：`str`；保存路径
- `include_bands`：`list[str]`；要包含的波段列表，可选值为["VIS", "NIR-Y/J/H", "DES-G/R/I/Z"]
- `skip_when_band_not_found`：`bool`；当找不到指定波段时是否跳过，默认为`True`
- `skip_when_exists`：`bool`；当文件已存在时是否跳过下载，默认为`True`
- `obs_id`：`str | int | None`；观测目标ID，默认为`None`
- `tile_index`：`str | int | None`；观测分片ID，默认为`None`
- `product_type`：`str`；Euclid数据发布类型，默认为`DpdMerBksMosaic`
- `verbose`：`bool | None`；是否打印详细信息，默认为`None`，即跟随`EuclidClass`的设置

### 返回

- `None`

## download_spectrum

### 参数

- `source_id`：`str | int`；观测目标ID
- `save_dir`：`str`；保存路径
- `retrieval_type`：`str`；检索类型，可选值为["ALL", "SPECTRA_BGS", "SPECTRA_RGS"]，默认为"ALL"
- `schema`：`str`；数据发布名称，默认为"sedm"
- `verbose`：`bool | None`；是否打印详细信息，默认为`None`，即跟随`EuclidClass`的设置

### 返回

- `None`

---

Made By EggTargaryen
