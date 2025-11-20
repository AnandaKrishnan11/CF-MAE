import os
import glob
import json
import random
import rasterio
from skimage.measure import shannon_entropy
from skimage import io
from skimage.io import imread
from rasterio.mask import mask
from rasterio.windows import Window
from rasterio.warp import reproject, calculate_default_transform, Resampling
from rasterio.io import MemoryFile
from shapely.geometry import box
import numpy as np 
import geopandas as gpd



class GEO_PreProcessing:

    def __init__(self):

        self.global_counter = 0  # Track counter across multiple calls

    def open(self, path, no_channels=[1,2,3], transpose=True): 
        """Open an image and optionally select channels""" 
        src = rasterio.open(path) 
        img = src.read(no_channels) 
        if transpose: 
            img = img.transpose(1, 2, 0)
            return src, img
        else:
             return src,img

    def findOverlap(self,in_pre,in_post,out_pre,out_post,no_channels=[1,2,3]):
        
        '''
        Find the overlapping area an save both raster within intersection extent

        Args:
        in_Pre(str) : Absolute filepath to the pre image file
        in_post(str) : Absolute filepath to the post image file
        out_pre(str) : Absolute filepath to the Save cropped pre image file
        out_post(str) : Absolute filepath to the Save cropped post image file
        no_channels(list(int)): how many channels do we need
        
        '''
        _,pre_src = self.open(in_pre,no_channels)
        pre_bounds = box(*pre_src.bounds)
        crs = pre_src.crs
 
        _,post_src = self.open(in_post,no_channels)
        post_bounds = box(*post_src.bounds)

        common_area = pre_bounds.intersection(post_bounds)
        assert not common_area.is_empty, f'No Overlapping areas in the given images {os.path.basename(in_pre)} and {os.path.basename(in_post)}'
        # Converts Shapely geometry to GeoJSON format
        common_geojson = [json.loads(gpd.GeoSeries([common_area], crs=crs).to_json())['features'][0]['geometry']]
        pre_clip, pre_transform = mask(pre_src, common_geojson, crop=True)
        pre_meta = pre_src.meta.copy()
        
        pre_meta.update({
            "height": pre_clip.shape[1],
            "width": pre_clip.shape[2],
            "count": pre_clip.shape[0],
            "transform": pre_transform
            })

        post_clip, post_transform = mask(post_src, common_geojson, crop=True)
        post_meta = post_src.meta.copy()
        post_meta.update({
            "height": post_clip.shape[1],
            "width": post_clip.shape[2],
            "count": post_clip.shape[0],
            "transform": post_transform
            })
        
        try:
            with rasterio.open(out_pre,'w',**pre_meta) as pre_dest:
                                pre_dest.write(pre_clip)
            with rasterio.open(out_post,'w',**post_meta) as post_dest:
                                post_dest.write(post_clip)
        except ValueError:
              print('Invalid filepath!')

    def align_to_ref(self, ref_src, src_src):
        """
        Align src raster to ref raster grid (CRS, resolution, extent).
        Returns aligned array and its updated transform.
        """
        # Create output array matching reference dimensions
        aligned_data = np.empty(
            (src_src.count, ref_src.height, ref_src.width), 
            dtype=src_src.dtypes[0]
        )
        # Reproject each band
        for i in range(src_src.count):
            reproject(
                source=rasterio.band(src_src, i + 1),
                destination=aligned_data[i],
                src_transform=src_src.transform,
                src_crs=src_src.crs,
                dst_transform=ref_src.transform,
                dst_crs=ref_src.crs,
                resampling=Resampling.bilinear
            )
        print("[ALIGN DONE] Aligned image to reference grid (in-memory).")
        return aligned_data, ref_src.transform

    def makeChips(self, in_pre, in_post, out_pre, out_post, no_channels=[1, 2, 3], chip_size=224, stride=224, file_prefix="", reset_counter=False):
        """
        Creates chips from pre and post images, automatically aligning or cropping if needed.
        Returns chips that are spatially aligned.
        
        Args:
            file_prefix: Optional prefix for output filenames to avoid conflicts
            reset_counter: If True, reset global counter to 0
        """
        os.makedirs(out_pre, exist_ok=True)
        os.makedirs(out_post, exist_ok=True)     
        # Reset counter if needed
        if reset_counter:
            self.global_counter = 0   
        counter = self.global_counter

        with rasterio.open(in_pre) as pre_src, rasterio.open(in_post) as post_src:
            # Check if alignment is needed
            needs_alignment = (
                #pre_src.crs != post_src.crs or
               # pre_src.transform != post_src.transform or
                pre_src.height != post_src.height or
                pre_src.width != post_src.width
            )
            if needs_alignment:
                print("[MISMATCH] Images need alignment. Aligning post to pre...")              
                aligned_post_data, aligned_transform = self.align_to_ref(pre_src, post_src)              
                # Create in-memory raster for aligned post
                memfile = MemoryFile()
                aligned_post_src = memfile.open(
                    driver="GTiff",
                    height=pre_src.height,
                    width=pre_src.width,
                    count=post_src.count,
                    dtype=post_src.dtypes[0],
                    crs=pre_src.crs,
                    transform=pre_src.transform
                )
                aligned_post_src.write(aligned_post_data)                
                print(f"[INFO] Starting chipping from aligned images ({pre_src.height}x{pre_src.width})...")
                
                for row in range(0, pre_src.height - chip_size + 1, stride):
                    for col in range(0, pre_src.width - chip_size + 1, stride):
                        window = Window(col, row, chip_size, chip_size)                       
                        pre_chip = pre_src.read(no_channels, window=window)
                        post_chip = aligned_post_src.read(no_channels, window=window)                       
                        chip_transform = rasterio.windows.transform(window, pre_src.transform)                       
                        chip_meta = {
                            "driver": "GTiff",
                            "height": chip_size,
                            "width": chip_size,
                            "count": len(no_channels),
                            "dtype": pre_src.dtypes[0],
                            "crs": pre_src.crs,
                            "transform": chip_transform
                        }                       
                        pre_filename = os.path.join(out_pre, f"{file_prefix}pre_{counter}.tif")
                        post_filename = os.path.join(out_post, f"{file_prefix}post_{counter}.tif")
                        
                        with rasterio.open(pre_filename, "w", **chip_meta) as pre_dest:
                            pre_dest.write(pre_chip)
                        with rasterio.open(post_filename, "w", **chip_meta) as post_dest:
                            post_dest.write(post_chip)
                        counter += 1         
                aligned_post_src.close()
                memfile.close()       
            else:
                # Images already aligned
                print("[INFO] Images are already aligned. Proceeding with chipping...")
                
                for row in range(0, pre_src.height - chip_size + 1, stride):
                    for col in range(0, pre_src.width - chip_size + 1, stride):
                        window = Window(col, row, chip_size, chip_size)
                        pre_chip = pre_src.read(no_channels, window=window)
                        post_chip = post_src.read(no_channels, window=window)
                        chip_transform = rasterio.windows.transform(window, pre_src.transform)                    
                        chip_meta = {
                            "driver": "GTiff",
                            "height": chip_size,
                            "width": chip_size,
                            "count": len(no_channels),
                            "dtype": pre_src.dtypes[0],
                            "crs": pre_src.crs,
                            "transform": chip_transform
                        }                   
                        pre_filename = os.path.join(out_pre, f"{file_prefix}pre_{counter}.tif")
                        post_filename = os.path.join(out_post, f"{file_prefix}post_{counter}.tif")
                        
                        with rasterio.open(pre_filename, "w", **chip_meta) as pre_dest:
                            pre_dest.write(pre_chip)                    
                        with rasterio.open(post_filename, "w", **chip_meta) as post_dest:
                            post_dest.write(post_chip)                  
                        counter += 1

            print(f"[DONE] Created {counter - self.global_counter} chips from this image pair.")
            print(f"[TOTAL] {counter} total chips created so far.")           
            self.global_counter = counter
            
    def Mean_Std(self, FOLDER_PATH, no_channels):
            '''
            Computing the Mean and Std for the whole dataset

            Args:
                FOLDER_PATH(str): Provide the directory consisting of images
                no_channels(list): List of channel indices to read
            
            Returns:
                tuple: ((pre_mean, pre_std), (post_mean, post_std)) as numpy arrays, 
                    or ((None, None), (None, None)) if no valid images
            '''

            try:
                pre_list, post_list = [], []

                pre_mean = np.zeros(len(no_channels))
                pre_std = np.zeros(len(no_channels))

                post_mean = np.zeros(len(no_channels))
                post_std = np.zeros(len(no_channels))


                for folds in os.listdir(FOLDER_PATH):
                    if folds == 'SEN4FLOOD_A' or folds == 'SEN4FLOOD_B':
                        print(f'Collecting Images from-----{folds}')
                        pre_collection = os.path.join(FOLDER_PATH, folds, 'pre')
                        post_collection = os.path.join(FOLDER_PATH, folds, 'post')

                        pre_list.extend(sorted(glob.glob(os.path.join(pre_collection, '*.*'))))
                        post_list.extend(sorted(glob.glob(os.path.join(post_collection, '*.*'))))
                        
                    else:
                        continue
                        
                print(f'[INFO] Found {len(pre_list)} pre-images and {len(post_list)} post-images')

                assert len(pre_list) == len(post_list), 'MISMATCH IN NUMBER OF FILES'

                skipped_count = 0
                processed_count = 0

                for preimg_name, postimg_name in tqdm(zip(pre_list, post_list),total=len(pre_list)):
                    preimg = imread(preimg_name)
                    postimg = imread(postimg_name)

                    has_invalid_pre = np.any(~np.isfinite(preimg))
                    has_invalid_post = np.any(~np.isfinite(postimg))

                    if has_invalid_pre or has_invalid_post:
                        skipped_count += 1
                        print(f'\n[WARNING] Skipping image pair {skipped_count} due to invalid values:')
                        
                        if has_invalid_pre:
                            print(f'  PRE-IMAGE:  {preimg_name}')
                            print(f'    - NaN count: {np.isnan(preimg).sum()}')
                            print(f'    - Inf count: {np.isinf(preimg).sum()}')
                        
                        if has_invalid_post:
                            print(f'  POST-IMAGE: {postimg_name}')
                            print(f'    - NaN count: {np.isnan(postimg).sum()}')
                            print(f'    - Inf count: {np.isinf(postimg).sum()}')
                        
                        continue 
                    
                    pre_mean += np.mean(preimg, axis=(0, 1))      
                    pre_std += np.std(preimg, axis=(0, 1)) 

                    post_mean += np.mean(postimg, axis=(0, 1))      
                    post_std += np.std(postimg, axis=(0, 1)) 

                    processed_count += 1

                print(f'\n[INFO] Processed {processed_count} valid image pairs')
                print(f'[INFO] Skipped {skipped_count} image pairs with invalid values')

                # Calculate statistics
                pre_main_mean = pre_mean / processed_count
                pre_main_std = pre_std  / processed_count

                post_main_mean = post_mean / processed_count
                post_main_std = post_std / processed_count

                print(f'\n[PRE_MEAN]-----{pre_main_mean}')
                print(f'[PRE_STD]-----{pre_main_std}')

                print(f'[POST_MEAN]-----{post_main_mean}')
                print(f'[POST_STD]-----{post_main_std}')

                return None

            except Exception as e:
                print(f'[ERROR] Exception occurred: {str(e)}')
                import traceback
                traceback.print_exc()
                return None

    def split_sar_ratio(self,img):

        counter = self.global_counter

        with rasterio.open(img) as src:
            img = src.read().transpose(1, 2, 0)
            pre_meta = src.meta.copy()
            post_meta = src.meta.copy()
            #It is a ( band image were the first two band in VV and VH OF PRE EVENT and third and fourth is VV and VH OF POST EVENT)
            pre = img[:,:,:2]
            post = img[:,:,2:4]

            pre_vv = pre[:,:,0]
            pre_vh = pre[:,:,1]
            eps = 1e-6
            pre_ratio = pre_vv / (pre_vh + eps)
            pre_img = np.stack((pre_vv,pre_vh,pre_ratio),axis=-1).astype(np.float32)

            post_vv = post[:,:,0]
            post_vh = post[:,:,1]
            post_ratio = post_vv / (post_vh + eps)
            post_img = np.stack((post_vv,post_vh,post_ratio),axis=-1).astype(np.float32)

            pre_meta.update({
                'count':pre_img.shape[-1]
            })
            post_meta.update({
                'count':post_img.shape[-1]        
            })
            pre_img = pre_img.transpose(2,0,1)
            post_img = post_img.transpose(2,0,1)

            with rasterio.open(os.path.join(r'C:\Users\b1114674\Desktop\thesis\urbanfloodsar_check\pre',f'pre_img{counter}.tif'),mode='w',**pre_meta) as pre_dest:
                pre_dest.write(pre_img)
            with rasterio.open(os.path.join(r'C:\Users\b1114674\Desktop\thesis\urbanfloodsar_check\post',f'post_img{counter}.tif'),mode='w',**post_meta) as post_dest:
                post_dest.write(post_img)

    def image_entropy (self,FOLDER_PATH):

        pre_list, post_list = [], []

        # Collect all image paths (tif, png, jpg)
        for folds in os.listdir(FOLDER_PATH):
            if folds == 'SEN4FLOOD_A' or folds == 'SEN4FLOOD_B':
                pre_collection = os.path.join(FOLDER_PATH, folds, 'pre')
                post_collection = os.path.join(FOLDER_PATH, folds, 'post')

                pre_list.extend(sorted(glob.glob(os.path.join(pre_collection, '*.*'))))
                post_list.extend(sorted(glob.glob(os.path.join(post_collection, '*.*'))))
                print(f'The images from {folds} folder are collected')       
            else:
                continue            
        # Shuffle and select subset
        random.shuffle(pre_list)
        random.shuffle(post_list)
        new_pre_list = pre_list[:3000]
        new_post_list = post_list[:3000]

        pre_entropies = []
        post_entropies = []

        # Compute Shannon entropy for pre images
        for img_path in new_pre_list:
            img = io.imread(img_path)         
            # Handle grayscale or alpha images safely
            if img.ndim == 2:
                pre_entropies.append(shannon_entropy(img))
            else:
                # Take first 3 channels only if available (ignore alpha)
                channels = min(img.shape[2], 3)
                for i in range(channels):
                    pre_entropies.append(shannon_entropy(img[:, :, i]))

        # Compute Shannon entropy for post images
        for img_path in new_post_list:
            img = io.imread(img_path)   
            if img.ndim == 2:
                post_entropies.append(shannon_entropy(img))
            else:
                channels = min(img.shape[2], 3)
                for i in range(channels):
                    post_entropies.append(shannon_entropy(img[:, :, i]))
        print(f'[ENTROPY OF {len(new_pre_list)} PRE IMAGES]  --> {np.mean(pre_entropies):.4f}')
        print(f'[ENTROPY OF {len(new_post_list)} POST IMAGES] --> {np.mean(post_entropies):.4f}')