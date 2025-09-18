import os
import json
import glob
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from sklearn.cluster import KMeans
from datetime import datetime

# Optional cloud detection - simple heuristic (NOT production quality)
def simple_cloud_mask(band_blue, band_red, band_nir, threshold=0.18):
    """Create a crude cloud mask using brightness + low NIR test.
    Parameters:
        band_blue, band_red, band_nir: arrays scaled 0-1
        threshold: brightness threshold
    Returns mask (True = cloud)"""
    brightness = (band_blue + band_red + band_nir) / 3.0
    mask = (brightness > threshold) & (band_nir < 0.25)
    return mask


def read_band(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile
    # Scale Sentinel-2 style 0-10000 to 0-1 (heuristic)
    if data.max() > 1.1:
        data /= 10000.0
    return data, profile


def compute_indices(bands):
    """Compute common spectral indices. bands dict expects keys B02,B03,B04,B08,B11,B12 if available."""
    idx = {}
    B02 = bands.get('B02'); B03 = bands.get('B03'); B04 = bands.get('B04'); B08 = bands.get('B08'); B11 = bands.get('B11'); B12 = bands.get('B12')

    # NDVI
    if B08 is not None and B04 is not None:
        ndvi = (B08 - B04) / (B08 + B04 + 1e-6)
        idx['NDVI'] = ndvi
    # NDWI (McFeeters) using Green & NIR
    if B03 is not None and B08 is not None:
        ndwi = (B03 - B08) / (B03 + B08 + 1e-6)
        idx['NDWI'] = ndwi
    # MNDWI using Green & SWIR1
    if B03 is not None and B11 is not None:
        mndwi = (B03 - B11) / (B03 + B11 + 1e-6)
        idx['MNDWI'] = mndwi
    # SAVI (L=0.5)
    if B08 is not None and B04 is not None:
        savi = ((B08 - B04) / (B08 + B04 + 0.5)) * (1.0 + 0.5)
        idx['SAVI'] = savi
    # EVI (simplified) 2.5*(NIR-Red)/(NIR+6*Red-7.5*Blue+1)
    if B08 is not None and B04 is not None and B02 is not None:
        evi = 2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1.0 + 1e-6)
        idx['EVI'] = evi
    # LST proxy: inverse of SWIR absorption (very naive)
    if B11 is not None:
        lst_proxy = B11  # placeholder
        idx['LST_PROXY'] = lst_proxy
    return idx


def find_band_files(root_dir):
    patterns = {
        'B02': '*B02*.tif',
        'B03': '*B03*.tif',
        'B04': '*B04*.tif',
        'B08': '*B08*.tif',
        'B11': '*B11*.tif',
        'B12': '*B12*.tif'
    }
    band_files = {}
    for b, pat in patterns.items():
        matches = glob.glob(os.path.join(root_dir, pat))
        if matches:
            band_files[b] = matches[0]  # first match
    return band_files


def stack_features(indices):
    # order features consistently
    keys = sorted(indices.keys())
    stack = np.dstack([indices[k] for k in keys])
    return stack, keys


def kmeans_cluster(feature_stack, n_clusters=5, random_state=42):
    h, w, c = feature_stack.shape
    X = feature_stack.reshape(-1, c)
    # Remove NaNs
    mask_valid = ~np.isnan(X).any(axis=1)
    X_valid = X[mask_valid]
    km = KMeans(n_clusters=n_clusters, n_init='auto', random_state=random_state)
    km.fit(X_valid)
    labels = np.full(X.shape[0], -1, dtype=np.int16)
    labels[mask_valid] = km.labels_
    return labels.reshape(h, w), km


def save_raster(array, profile, out_path):
    prof = profile.copy()
    prof.update({'count': 1, 'dtype': 'float32', 'compress': 'lzw'})
    with rasterio.open(out_path, 'w', **prof) as dst:
        dst.write(array.astype(np.float32), 1)
    return out_path


def process_area(input_dir='bio sop/data/sample', output_dir='bio sop/data/derived', region_name='pilani'):
    os.makedirs(output_dir, exist_ok=True)
    band_files = find_band_files(input_dir)
    if not band_files:
        raise FileNotFoundError(f'No band files found under {input_dir}')

    # Read bands
    bands = {}
    reference_profile = None
    for b, path in band_files.items():
        data, profile = read_band(path)
        if reference_profile is None:
            reference_profile = profile
        # Resample if shapes differ
        if bands:
            ref_shape = next(iter(bands.values())).shape
            if data.shape != ref_shape:
                # simple nearest resize
                data = resize_array(data, ref_shape)
        bands[b] = data

    indices = compute_indices(bands)

    # Cloud mask if enough bands
    cloud_mask = None
    if 'B02' in bands and 'B04' in bands and 'B08' in bands:
        cloud_mask = simple_cloud_mask(bands['B02'], bands['B04'], bands['B08'])
        indices['CLOUD_MASK'] = cloud_mask.astype(np.float32)

    # Save indices
    saved_indices = {}
    for name, arr in indices.items():
        out_file = os.path.join(output_dir, f'{region_name}_{name}.tif')
        save_raster(arr, reference_profile, out_file)
        saved_indices[name] = out_file

    # Feature stack (exclude mask from clustering)
    cluster_input = {k:v for k,v in indices.items() if k not in ['CLOUD_MASK']}
    stack, keys = stack_features(cluster_input)
    labels, km = kmeans_cluster(stack, n_clusters=5)
    cluster_path = os.path.join(output_dir, f'{region_name}_CLUSTERS.tif')
    save_raster(labels.astype(np.float32), reference_profile, cluster_path)

    # Quick PNG previews
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # NDVI preview
        if 'NDVI' in indices:
            plt.figure(figsize=(4,4)); plt.imshow(indices['NDVI'], cmap='RdYlGn'); plt.title('NDVI'); plt.axis('off')
            plt.savefig(os.path.join(output_dir, f'{region_name}_NDVI.png'), dpi=120, bbox_inches='tight'); plt.close()
        # Cluster preview
        plt.figure(figsize=(4,4)); plt.imshow(labels, cmap='tab20'); plt.title('Clusters'); plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'{region_name}_CLUSTERS.png'), dpi=120, bbox_inches='tight'); plt.close()
    except Exception:
        pass

    summary = {
        'region': region_name,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'bands_found': list(band_files.keys()),
        'indices_computed': list(indices.keys()),
        'cluster_feature_order': keys,
        'cluster_centers': km.cluster_centers_.tolist(),
        'files': {
            'bands': band_files,
            'indices': saved_indices,
            'clusters': cluster_path
        },
        'images_processed': 1,
        'estimated_high_risk_areas': int((labels == labels.max()).sum()),
        'cloud_coverage_pct': float(cloud_mask.mean()*100) if cloud_mask is not None else None
    }

    with open(os.path.join(output_dir, 'processing_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def resize_array(data, target_shape):
    """Nearest neighbor resize using rasterio's reproject for correctness."""
    import rasterio.warp
    dst = np.empty(target_shape, dtype=data.dtype)
    transform_src = Affine.identity()
    transform_dst = Affine.identity()
    rasterio.warp.reproject(
        data,
        dst,
        src_transform=transform_src,
        src_crs='EPSG:4326',
        dst_transform=transform_dst,
        dst_crs='EPSG:4326',
        resampling=Resampling.nearest
    )
    return dst

if __name__ == '__main__':
    s = process_area()
    print('Processing complete. Summary:')
    print(json.dumps(s, indent=2)[:800])
