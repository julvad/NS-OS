# Use smp3 env

# Model inference (predict)
import os
from glob import glob
import gc
import torch
from typing import Literal, Union, List, Optional
from my_code.inference import predict_pytorch, mosaic_with_rasterio, raster_to_polygon
from my_code.utils.tile_raster import split_s1_geotiff_gdal
from my_code.utils.transforms import sar_transform
from datetime import datetime


# ARGS FOR SCRIPT B
PADDING = 'auto' #'auto' or int
MIN_AREA_M2 = 2e4 #NOTE: experiment with 1e4 and more
MAX_AREA_M2 = 1e8

#--------------------------------------------------------------------------------------------
def inference_mosaic(
        tif_img_path:str, #ex. img_path.tif or [img1.tif, img2.tif]
        model: torch.nn.Module,
        transform,
        landmask_path:Optional[str]=None,
        predict_prob_or_class:Literal['probs','class']='probs',
        predict:bool=True,
        batch_size:int=8,
        mosaic_tiles:bool=True,
        padding: Union[int, str] = 'auto',
        mosaic_classify_softmax:bool=True,
        vectorize_output:bool=True,
        min_area_m2:int=1e4, 
        max_area_m2:int=1e8,
        vector_format:Literal['shp','geojson']='shp',
    ):
    """
    Outputs will be saved in the tif_img dir:
    - Tiles will be saved in the 'tiles' folder
    - Preds will be saved in 'preds' folder
    - Mosaic will be saved as mosaic_pred.tif
    - Vectorized mosaic will be saved as mosaic_pred.shp

    NOTE:adjust min and max area values based on target oil slick size.
    """
    if mosaic_tiles and vectorize_output:
        assert mosaic_classify_softmax, 'preds must be classified for shp export'

    img_dir = os.path.dirname(tif_img_path)
    tiles_folder = os.path.join(img_dir, 'tiles')
    split_s1_geotiff_gdal(
        in_tif_path=tif_img_path,
        out_folder_path=tiles_folder,
        out_pixel_size=20,
        out_tile_size=512,
        overlap=256,
        landmask_path=landmask_path,
        in_nodata_value=0
    )

    all_tiles = glob(os.path.join(tiles_folder, '*.tif'))
    ll = len(all_tiles)
    print(f'Running mosaic inference script for {ll} tiles.')

    preds_folder = os.path.join(img_dir, 'preds')
    if predict:
        print(f'{datetime.now()}: Model inference on {ll} tiles...')
        predict_pytorch(
            model=model,
            tile_paths=all_tiles,
            transform=transform,
            predict_prob_or_class='class',
            return_dice_iou=False,
            save_out_preds=True,
            out_pred_path=preds_folder,
            batch_size=batch_size
        )
    
    if mosaic_tiles:
        all_preds_img = glob(os.path.join(preds_folder, '*.tif'))
        if not all_preds_img:
            print(f'Warning: no preds found for {tif_img_path}. Exiting.')
            return #if for some reason there are no preds, dont mosaic etc.

        print(f'{datetime.now()}: mosaicking {tif_img_path}: {len(all_preds_img)} geotiff preds')
        
        if predict_prob_or_class == 'class':
            method = 'max' ##TODO: Check other possibilities.
            out_dtype = 'int'

        elif predict_prob_or_class == 'probs':
            method = 'mean'
            if mosaic_classify_softmax:
                out_dtype = 'int'
            else:
                out_dtype = 'float'
        else:
            raise ValueError(predict_prob_or_class)

        mosaic_path = os.path.join(img_dir, 'mosaic_pred.tif')

        mosaic_with_rasterio(
            in_tile_paths=all_preds_img,
            out_mosaic_path=mosaic_path,
            method=method,
            padding=padding,
            out_mosaic_dtype=out_dtype,
            mosaic_classify_softmax=True,
            softmax_threshold=0.5
        )

        del all_preds_img  # free list memory
        gc.collect() # avoid memory building

    if vectorize_output:
        print(f'vectorizing {vector_format} mosaic pred for {tif_img_path}')

        out_shp_path = os.path.join(img_dir, f'mosaic_pred.{vector_format}')
        raster_to_polygon(
            in_raster_path=mosaic_path,
            out_folder=out_shp_path,
            min_area_m2=min_area_m2,
            max_area_m2=max_area_m2,
            out_format=vector_format,
            class_values=1,
            postprocessing=True
        )









