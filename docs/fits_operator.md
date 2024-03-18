# fits_operator.py

该库中主要含有对`.fits`以及`.fits.gz`格式的光谱和测光数据的操作：

1. [create_dir](#create_dir)
2. [fits_reproject](#fits_reproject)
3. [from_world_2_pixel](#from_world_2_pixel)
4. [crop_from_fits](#crop_from_fits)
5. [print_cross_label_to_img](#print_cross_label_to_img)
6. [generate_img](#generate_img)
7. [SDSS_photo_download_process](#SDSS_photo_download_process)
8. [up_sample](#up_sample)
9. [reproject_process](#reproject_process)
10. [spectra_equivalent_width](#spectra_equivalent_width)
11. [spectra_redshift_correction](#spectra_redshift_correction)
12. [read_spec_fits_file](#read_spec_fits_file)
13. [get_csv_header_col_name](#get_csv_header_col_name)
14. [read_lamost_lrs_spectrum](#read_lamost_lrs_spectrum)
15. [fits_to_npy_process](#fits_to_npy_process)
16. [LAMOST_spec_fits_to_npy](#LAMOST_spec_fits_to_npy)
17. [DECaLS_photo_download_process](#DECaLS_photo_download_process)
18. [DESI_fits_reader](#DESI_fits_reader)

## create_dir

创建指定目录，如果目录已存在则忽略

### 参数

- `path`：`str`；目录路径

### 返回

`None`

## fits_reproject

`.fits`或`.fits.gz`格式的测光图像不同波段的对齐操作，返回多波段对齐后的通道堆叠结果

### 参数

- `target_fits_path`：`str`；对齐目标波段的测光图像路径
- `other_fits_path`：`list[str, ...]`；其他波段的测光图像的路径
- `bands_order`：`list[str, ...]`；波段堆叠顺序
- `hdu_index`：`int`；指定测光图像在`.fits`或`.fits.gz`文件中存储的`hdu`
- `post_process`：`function`；对齐后对该图像的后处理操作回调函数

### 返回

`np.ndarray`，形状为`(C, H, W)`的堆叠图像

## from_world_2_pixel

将天文坐标转换为基于图像像素的坐标

### 参数

- `fits_path`：`str`；`.fits`或`.fits.gz`格式测光图像的路径
- `hdu_index`：`int`；指定测光图像在`.fits`或`.fits.gz`文件中存储的`hdu`
- `coord`：`SkyCoord`；`Astropy`库中的`SkyCoord`类型的天文坐标

### 返回

`list[int]`

## crop_from_fits

在测光图像中裁剪出指定大小的区域

### 参数

- `fits_path`：`str`；`.fits`或`.fits.gz`格式测光图像的路径
- `hdu_index`：`int`；指定测光图像在`.fits`或`.fits.gz`文件中存储的`hdu`
- `img`：`np.ndarray`；以`numpy`数组保存的测光图像
- `bbox_size`：`int`；裁剪的尺寸
- `obj_coord`：`SkyCoord`；`Astropy`库中的`SkyCoord`类型的天文坐标

### 返回

`None`或`np.ndarray`，如果裁剪成功则返回`numpy`数组，否则返回`None`

## print_cross_label_to_img

在测光图像中画出对应坐标的十字瞄准线用以观察目标的位置

### 参数

- `fits_path`：`str`；`.fits`或`.fits.gz`格式测光图像的路径
- `hdu_index`：`int`；指定测光图像在`.fits`或`.fits.gz`文件中存储的`hdu`
- `obj_coord`：`SkyCoord`；`Astropy`库中的`SkyCoord`类型的天文坐标
- `save_path`：`str`；保存路径

### 返回

`None`

## generate_img

从多波段测光图像中完成对齐、后处理操作并裁剪对应大小的图像，相当于合并了上述操作

### 参数

- `fits_dir`：`str`；保存多个波段`.fits`或`.fits.gz`格式测光图像的根目录
- `hdu_index`：`int`；指定测光图像在`.fits`或`.fits.gz`文件中存储的`hdu`
- `target_band`：`str`；目标波段的名称
- `other_band`：`list[str, ...]`；其他波段的名称
- `bbox_size`：`int`；裁剪的尺寸
- `obj_coord`：`SkyCoord`；`Astropy`库中的`SkyCoord`类型的天文坐标
- `band_name_match_rule`：`(band_name: str, fits_dir: str) => {}`；用来从输入的根目录中匹配指定波段测光图像文件的回调函数
- `post_process`：`function`；对齐后对该图像的后处理操作回调函数

### 返回

`Tuple[Union[None, np.ndarray], Union[None, np.ndarray]]`，返回**裁剪后的测光图像**与**只经过堆叠的测光图像**
，如果不存在则对应返回`None`

## SDSS_photo_download_process

通过`run`、`rerun`、`camcol`和`field`参数下载SDSS测光图像

### 参数

- `unique_id`：`str`；唯一id，用来在保存时避免同名覆盖
- `base_url`：`str`；SDSS文件服务器的基础`url`
  ，例如[https://data.sdss.org/sas/dr17/eboss/photoObj/frames](https://data.sdss.org/sas/dr17/eboss/photoObj/frames)
- `run`：`str`；SDSS测光星表中的字段
- `rerun`：`str`；SDSS测光星表中的字段
- `camcol`：`str`；SDSS测光星表中的字段
- `field`：`str`；SDSS测光星表中的字段
- `save_dir`：`str`；下载后的保存根目录
- `band`：`list[str, ...]`，默认为`[u, g, r, i, z]`；需要下载的波段名称

### 返回

`None`

## up_sample

基于`torchvision`的上采样

### 参数

- `img`：`np.ndarray`；以`numpy`数组保存的测光图像
- `new_size`：`int`；上采样后的尺寸

### 返回

`np.ndarray`，返回上采样后的图像

## reproject_process

完成对齐、后处理、裁剪与上采样的综合测光图像操作函数

### 参数

- `fits_dir`：`str`；保存多个波段`.fits`或`.fits.gz`格式测光图像的根目录
- `unique_id`：`str`；唯一id，用来在保存时避免同名覆盖
- `target_band`：`str`；目标波段的名称
- `other_bands`：`list[str, ...]`；其他波段的名称
- `bands_order`：`list[str, ...]`；波段堆叠顺序
- `crop_size`：`int`；裁剪后的图像大小
- `up_sample_size`：`int`或`None`；上采样后的大小，如果设置为`None`则不进行上采样
- `target_coord`：`SkyCoord`；`Astropy`库中的`SkyCoord`类型的天文坐标
- `save_dir`：`str`；保存到的根目录
- `hdu_index`：`int`；指定测光图像在`.fits`或`.fits.gz`文件中存储的`hdu`
- `post_process`：`function`；对齐后对该图像的后处理操作回调函数
- `fits_file_suffix`：`str`，默认为`.fits.bz2`；测光图像的文件扩展名
- `padding_value`：`int`，默认为`0`；裁剪后如果存在空白则在空白部分填充的值

### 返回

`None`

## spectra_equivalent_width

光谱等效宽度计算

### 参数

- `spectra_region`：`SpectralRegion`；指定波长范围

### 返回

`Spectrum1D`

## spectra_redshift_correction

从原光谱中进行红移校正

### 参数

- `spectra`：`Spectrum1D`；输入的光谱数据
- `redshift`：`float`；红移值

### 返回

`None`

## read_spec_fits_file

使用`specutils`库以指定巡天项目格式读取光谱数据

### 参数

- `fits_file_path`：`str`；光谱数据的存储路径
- `spec_format`：`str`，默认为`SDSS-III/IV spec`；`specutils`库中存在光谱格式名称

### 返回

`Spectrum1D`，光谱数据

## get_csv_header_col_name

在不使用`pandas`库的情况下读取`.csv`文件的表头（适用场景是包含几百万甚至几千万条数据的`.csv`文件，这类文件使用`pandas`
库读取太慢）

### 参数

- `csv_file_path`：`str`；`.csv`文件存储路径

### 返回

`list`，表头数组

## read_lamost_lrs_spectrum

读取LAMOST低分辨率光谱

### 参数

- `fits_path`：`str`；`.fits`格式的光谱文件的存储路径

### 返回

`(光谱文件头, 通量ndarray, 波长ndarray)`

## fits_to_npy_process

将给定的光谱文件列表中的光谱从`.fits`文件转换为`.npy`文件

### 参数

- `sub_list`：` list[str, ...]`；文件存储路径列表
- `npy_save_dir`：`str`；`.npy`文件存储路径
- `spectra_region`：`SpectralRegion`；转换时按照波长切分光谱

### 返回

`None`

## LAMOST_spec_fits_to_npy

将LAMOST光谱转换为`.npy`文件

### 参数

- `sub_list`：` list[str, ...]`；文件存储路径列表
- `npy_save_dir`：`str`；`.npy`文件存储路径
- `spectra_region`：`SpectralRegion`；转换时按照波长切分光谱
- `filter`：`(wavelength: np.ndarray, flux: np.ndarray) => {}`；用于光谱数据后处理的回调函数

### 返回

`None`

## DECaLS_photo_download_process

DECaLS测光图像下载工具

### 参数

- `ra`：` str`；赤经坐标
- `dec`：` str`；赤纬坐标
- `pixscale`：`float`；DECaLS中指定的`pixscale`参数
- `fits_save_dir`：`str`；下载的`.fits`测光图像保存目录
- `jpg_save_dir`：`str`；下载的`.jpeg`测光图像保存目录
- `bands`：`list[str, ...]`；下载`.fits`测光图像时需要包含的波段，仅对`.fits`测光图像生效
- `obsid`：`str`，默认为`None`；LAMOST中的`obsid`，用于指定唯一名称防止覆盖，可以不写或采用其他生成方式
- `layer`：`str`，默认为`ls-dr10`；指定使用的DECaLS DR版本
- `download_jpg`：`bool`，默认为`False`；是否下载`.jpeg`测光图像

### 返回

`None`

## DESI_fits_reader

DESI测光图像转换`numpy`数组

### 参数

- `fits_path`：` str`；`.fits`文件存储路径
- `stack_bands`：`list[str]`；波段堆叠顺序
- `hdu_index`：`int`；指定测光图像在`.fits`或`.fits.gz`文件中存储的`hdu`
- `post_process`：`function`；对齐后对该图像的后处理操作回调函数

### 返回

`np.ndarray`或`None`，如果转换成功则返回`C, H, W`结构的数组，否则返回

---

Made By EggTargaryen