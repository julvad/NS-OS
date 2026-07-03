import os
from typing import Optional
import numpy as np
from osgeo import gdal


def split_s1_geotiff_gdal(
    in_tif_path:str,
    out_folder_path:str,
    out_pixel_size:int,
    out_tile_size:int,
    overlap:int,
    landmask_path:Optional[str]=None,
    custom_tile_name: Optional[str] = None,
    in_nodata_value: Optional[int]=0,
    ):
    """
    - Projects a single-band SAR GRD geotiff to its UTM zone.
    - Optionally applies landmask (binary geotiff [1=sea; 0=land])
    - Resample to a given pixel size (m)
    - Splits the raster into tiles of a given tile size (px).
    - Optionally splits with overlap (px)
    - Optionally use a custom tile name (str), otherwise uses 'tile'. Tile name is followed by <tile_id>
    Input: path to sentinel-1 single-pol-band raster
    Output: a set of geotiff tiles
    """

    # WARP USING GCPs to ensure proj is well defined EPSG:4326 NOTE: This might not be needed but some 
    src = gdal.Open(in_tif_path)

    wgs84_ds = gdal.Warp(
        "",
        src,
        format="MEM",
        dstSRS="EPSG:4326",
        resampleAlg="bilinear"
    )

    gt = wgs84_ds.GetGeoTransform() 

    minx = gt[0]
    maxy = gt[3]
    maxx = gt[0] + gt[1] * wgs84_ds.RasterXSize
    miny = gt[3] + gt[5] * wgs84_ds.RasterYSize

    if landmask_path:
        # ALIGN LANDMASK
        landmask = gdal.Open(landmask_path)

        landmask_aligned_wgs84 = gdal.Warp(
            "",
            landmask,
            format="MEM",
            dstSRS="EPSG:4326",
            width=wgs84_ds.RasterXSize,
            height=wgs84_ds.RasterYSize,
            outputBounds=[minx, miny, maxx, maxy],
            resampleAlg="nearest" # we can use nearest resampling because binary landmask
        )
        
        # apply landmask

        img = wgs84_ds.ReadAsArray()
        mask = landmask_aligned_wgs84.ReadAsArray()

        masked = np.where(mask == 1, img, 0)

        # Write back to MEM dataset
        driver = gdal.GetDriverByName("MEM")
        masked_ds = driver.Create(
            "",#no name for memory
            wgs84_ds.RasterXSize,
            wgs84_ds.RasterYSize,
            1,
            wgs84_ds.GetRasterBand(1).DataType
        )

        masked_ds.SetGeoTransform(wgs84_ds.GetGeoTransform())
        masked_ds.SetProjection(wgs84_ds.GetProjection())
        masked_ds.GetRasterBand(1).WriteArray(masked)
    else:
        masked_ds = wgs84_ds


    # Reproject to UTM
    gt = masked_ds.GetGeoTransform()

    center_lon = gt[0] + gt[1] * (masked_ds.RasterXSize / 2)
    center_lat = gt[3] + gt[5] * (masked_ds.RasterYSize / 2)

    zone = int((center_lon + 180) / 6) + 1
    epsg = 32600 + zone if center_lat >= 0 else 32700 + zone

    print(f"Using UTM EPSG:{epsg}")

    utm_ds = gdal.Warp(
        "",#no name for memory
        masked_ds,
        format="MEM",
        dstSRS=f"EPSG:{epsg}",
        xRes=out_pixel_size,
        yRes=out_pixel_size,
        resampleAlg="bilinear" #bilinear more precise but may affect nodata borders... minimal effect though
    )

    # save tiles
    save_tiles(
        utm_ds,
        out_dir=out_folder_path,
        custom_tile_name=custom_tile_name,
        tile_size=out_tile_size,
        overlap=overlap,
        nodata_value=in_nodata_value
    )


def save_tiles(
    utm_ds,
    out_dir,
    custom_tile_name,
    tile_size=512,
    overlap=256,
    nodata_value=0
):
    os.makedirs(out_dir, exist_ok=True)

    arr = utm_ds.ReadAsArray()
    gt = utm_ds.GetGeoTransform()
    proj = utm_ds.GetProjection()

    height, width = arr.shape

    stride = tile_size - overlap
    tile_id = 0

    driver = gdal.GetDriverByName("GTiff")

    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):

            tile = arr[y:y + tile_size, x:x + tile_size]

            # Skip if all nodata
            if (tile == nodata_value).all():
                continue

            # Compute new geotransform for tile
            new_gt = (
                gt[0] + x * gt[1] + y * gt[2],
                gt[1],
                gt[2],
                gt[3] + x * gt[4] + y * gt[5],
                gt[4],
                gt[5]
            )
            
            if not custom_tile_name:
                custom_tile_name = 'tile'
            out_path = os.path.join(out_dir, f"{custom_tile_name}_{tile_id:05d}.tif") # name the tile with five 0-padding

            ds_tile = driver.Create(
                out_path,
                tile_size,
                tile_size,
                1,
                utm_ds.GetRasterBand(1).DataType
            )

            ds_tile.SetGeoTransform(new_gt)
            ds_tile.SetProjection(proj)

            band = ds_tile.GetRasterBand(1)
            band.WriteArray(tile)
            band.SetNoDataValue(nodata_value)

            ds_tile.FlushCache()
            ds_tile = None

            tile_id += 1