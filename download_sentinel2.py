"""
Sentinel-2 Data Acquisition Script
==================================

Script to download and preprocess Sentinel-2 imagery from Copernicus Data Space
for mosquito habitat prediction.

Usage:
    python download_sentinel2.py --region west_africa --start_date 2023-01-01 --end_date 2023-12-31
"""

import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import zipfile
import rasterio
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentinelDownloader:
    """
    Download Sentinel-2 data from Copernicus Data Space Ecosystem.
    """
    
    def __init__(self, username=None, password=None):
        self.base_url = "https://catalogue.dataspace.copernicus.eu/odata/v1"
        self.download_url = "https://zipper.dataspace.copernicus.eu/odata/v1"
        self.username = username
        self.password = password
        self.access_token = None
        
        # Region definitions
        self.regions = {
            'west_africa': {
                'bounds': (-18.0, 4.0, 15.0, 25.0),  # (min_lon, min_lat, max_lon, max_lat)
                'description': 'West Africa - Mali, Burkina Faso, Ghana, Nigeria region'
            },
            'east_africa': {
                'bounds': (28.0, -12.0, 52.0, 18.0),
                'description': 'East Africa - Kenya, Tanzania, Uganda, Ethiopia region'
            },
            'southeast_asia': {
                'bounds': (90.0, -10.0, 140.0, 28.0),
                'description': 'Southeast Asia - Thailand, Vietnam, Malaysia, Indonesia region'
            },
            'test_region': {
                'bounds': (0.0, 0.0, 5.0, 5.0),
                'description': 'Small test region for development'
            },
            'pilani': {
                # Approximate bounding box around Pilani, Rajasthan, India (BITS Pilani campus + surroundings)
                # Adjust if you need a larger/smaller area. Format: (min_lon, min_lat, max_lon, max_lat)
                'bounds': (75.53, 28.32, 75.64, 28.41),
                'description': 'Pilani, Rajasthan, India (BITS Pilani area)'
            }
        }
    
    def authenticate(self):
        """
        Authenticate with Copernicus Data Space and get access token.
        """
        if not self.username or not self.password:
            logger.warning("No credentials provided. Using public access (limited functionality)")
            return None
        
        auth_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        
        data = {
            'grant_type': 'password',
            'username': self.username,
            'password': self.password,
            'client_id': 'cdse-public'
        }
        
        try:
            response = requests.post(auth_url, data=data)
            response.raise_for_status()
            
            self.access_token = response.json()['access_token']
            logger.info("Successfully authenticated with Copernicus Data Space")
            return self.access_token
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Authentication failed: {e}")
            return None
    
    def search_products(self, region, start_date, end_date, cloud_cover_max=20, custom_bbox=None):
        """
        Search for Sentinel-2 products in the specified region and time range.
        
        Parameters:
        -----------
        region : str
            Region name from predefined regions
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        cloud_cover_max : int
            Maximum cloud cover percentage
            
        Returns:
        --------
        list: List of product information dictionaries
        """
        
        if custom_bbox is not None:
            if len(custom_bbox) != 4:
                raise ValueError('custom_bbox must be a 4-tuple (min_lon, min_lat, max_lon, max_lat)')
            bounds = tuple(custom_bbox)
            logger.info(f"Using custom bounding box: {bounds}")
        else:
            if region not in self.regions:
                raise ValueError(f"Region '{region}' not found. Available: {list(self.regions.keys())}")
            bounds = self.regions[region]['bounds']
        
        # Create bounding box string for API
        bbox = f"POLYGON(({bounds[0]} {bounds[1]},{bounds[2]} {bounds[1]},{bounds[2]} {bounds[3]},{bounds[0]} {bounds[3]},{bounds[0]} {bounds[1]}))"
        
        # Build search query
        search_params = {
            '$filter': f"""
                Collection/Name eq 'SENTINEL-2' and
                OData.CSC.Intersects(area=geography'SRID=4326;{bbox}') and
                ContentDate/Start ge {start_date}T00:00:00.000Z and
                ContentDate/Start le {end_date}T23:59:59.999Z and
                Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {cloud_cover_max})
            """.replace('\n', '').replace('  ', ' '),
            '$orderby': 'ContentDate/Start asc',
            '$top': 1000
        }
        
        logger.info(f"Searching for Sentinel-2 products in {region}")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Max cloud cover: {cloud_cover_max}%")
        
        try:
            response = requests.get(f"{self.base_url}/Products", params=search_params)
            response.raise_for_status()
            
            results = response.json()
            products = results.get('value', [])
            
            logger.info(f"Found {len(products)} products")
            
            # Extract relevant information
            product_list = []
            for product in products:
                product_info = {
                    'id': product['Id'],
                    'name': product['Name'],
                    'date': product['ContentDate']['Start'][:10],
                    'cloud_cover': self._extract_cloud_cover(product),
                    'size_mb': product.get('ContentLength', 0) / (1024 * 1024),
                    'download_url': f"{self.download_url}/Products({product['Id']})/$value"
                }
                product_list.append(product_info)
            
            return product_list
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _extract_cloud_cover(self, product):
        """Extract cloud cover percentage from product attributes."""
        attributes = product.get('Attributes', [])
        for attr in attributes:
            if attr.get('Name') == 'cloudCover':
                return attr.get('Value', 0)
        return 0
    
    def download_product(self, product_info, download_dir):
        """
        Download a single Sentinel-2 product.
        
        Parameters:
        -----------
        product_info : dict
            Product information from search results
        download_dir : str
            Directory to save downloaded files
            
        Returns:
        --------
        str: Path to downloaded file
        """
        
        download_dir = Path(download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{product_info['name']}.zip"
        filepath = download_dir / filename
        
        # Skip if already downloaded
        if filepath.exists():
            logger.info(f"File already exists: {filename}")
            return str(filepath)
        
        logger.info(f"Downloading: {filename}")
        logger.info(f"Size: {product_info['size_mb']:.1f} MB")
        
        headers = {}
        if self.access_token:
            headers['Authorization'] = f'Bearer {self.access_token}'
        
        try:
            response = requests.get(product_info['download_url'], headers=headers, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded: {filename}")
            return str(filepath)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Download failed for {filename}: {e}")
            return None
    
    def extract_and_process(self, zip_path, output_dir, target_bands=None):
        """
        Extract and process Sentinel-2 zip file.
        
        Parameters:
        -----------
        zip_path : str
            Path to downloaded zip file
        output_dir : str
            Directory for processed outputs
        target_bands : list
            List of bands to extract (e.g., ['B02', 'B03', 'B04', 'B08'])
        """
        
        if target_bands is None:
            target_bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']  # Blue, Green, Red, NIR, SWIR1, SWIR2
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        product_name = Path(zip_path).stem
        product_dir = output_dir / product_name
        
        logger.info(f"Extracting and processing: {product_name}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract specific bands
                extracted_bands = {}
                
                for file_info in zip_ref.filelist:
                    filename = file_info.filename
                    
                    # Check if this is a band file we want
                    for band in target_bands:
                        if f"_{band}_" in filename and filename.endswith('.jp2'):
                            band_dir = product_dir / band
                            band_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Extract the band file
                            zip_ref.extract(file_info, band_dir)
                            extracted_file = band_dir / filename
                            
                            # Process the band (convert to GeoTIFF, resample if needed)
                            processed_file = self._process_band(extracted_file, band_dir, band)
                            extracted_bands[band] = processed_file
                            
                            # Clean up original JP2 file
                            try:
                                extracted_file.unlink()
                            except:
                                pass
                
                logger.info(f"Processed {len(extracted_bands)} bands")
                return extracted_bands
                
        except Exception as e:
            logger.error(f"Processing failed for {product_name}: {e}")
            return {}
    
    def _process_band(self, jp2_path, output_dir, band_name):
        """
        Process individual band file (convert JP2 to GeoTIFF, resample to 10m).
        """
        
        output_file = output_dir / f"{band_name}.tif"
        
        try:
            with rasterio.open(jp2_path) as src:
                # Read band data
                data = src.read(1)
                profile = src.profile.copy()
                
                # Update profile for GeoTIFF
                profile.update({
                    'driver': 'GTiff',
                    'compress': 'lzw',
                    'dtype': 'uint16'
                })
                
                # Resample to 10m if needed (some bands are 20m)
                if src.res[0] != 10.0:
                    # Calculate new dimensions for 10m resolution
                    new_width = int(src.width * src.res[0] / 10.0)
                    new_height = int(src.height * src.res[1] / 10.0)
                    
                    # Update transform
                    transform = src.transform * src.transform.scale(
                        (src.width / new_width),
                        (src.height / new_height)
                    )
                    
                    # Resample data
                    data_resampled = np.empty((new_height, new_width), dtype=data.dtype)
                    
                    reproject(
                        data,
                        data_resampled,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=src.crs,
                        resampling=Resampling.bilinear
                    )
                    
                    profile.update({
                        'height': new_height,
                        'width': new_width,
                        'transform': transform
                    })
                    
                    data = data_resampled
                
                # Write processed band
                with rasterio.open(output_file, 'w', **profile) as dst:
                    dst.write(data, 1)
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Band processing failed for {band_name}: {e}")
            return None

def create_sample_dataset():
    """
    Create a sample dataset for testing when real Sentinel-2 data is not available.
    """
    
    logger.info("Creating sample synthetic dataset...")
    
    # Create sample directory structure
    sample_dir = Path("data/sample")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic spectral data
    np.random.seed(42)
    
    # Image dimensions (small for testing)
    height, width = 100, 100
    
    # Simulate different land cover types
    # Water bodies (low in all bands except blue/green)
    water_mask = np.random.random((height, width)) < 0.1
    
    # Vegetation (high NIR, moderate red/green)
    vegetation_mask = np.random.random((height, width)) < 0.4
    
    # Urban/bare soil (moderate in all bands)
    urban_mask = ~(water_mask | vegetation_mask)
    
    bands_data = {}
    band_info = {
        'B02': {'name': 'Blue', 'base_value': 0.1},
        'B03': {'name': 'Green', 'base_value': 0.12},
        'B04': {'name': 'Red', 'base_value': 0.08},
        'B08': {'name': 'NIR', 'base_value': 0.15},
        'B11': {'name': 'SWIR1', 'base_value': 0.2},
        'B12': {'name': 'SWIR2', 'base_value': 0.15}
    }
    
    # Create coordinate system (example: small area in Ghana)
    from rasterio.transform import from_bounds
    bounds = (-2.0, 6.0, -1.0, 7.0)  # Small area in Ghana
    transform = from_bounds(*bounds, width, height)
    
    crs = 'EPSG:4326'
    
    for band_id, info in band_info.items():
        # Generate synthetic band data
        data = np.full((height, width), info['base_value'], dtype=np.float32)
        
        # Add noise
        data += np.random.normal(0, 0.02, (height, width))
        
        # Modify based on land cover
        if band_id == 'B08':  # NIR - high for vegetation
            data[vegetation_mask] *= 3.0
            data[water_mask] *= 0.1
        elif band_id in ['B02', 'B03']:  # Blue/Green - higher for water
            data[water_mask] *= 1.5
            data[vegetation_mask] *= 0.8
        elif band_id == 'B04':  # Red - absorbed by vegetation
            data[vegetation_mask] *= 0.5
            data[water_mask] *= 0.3
        
        # Convert to typical Sentinel-2 scale (0-10000)
        data = (data * 10000).astype(np.uint16)
        data = np.clip(data, 0, 10000)
        
        # Save as GeoTIFF
        profile = {
            'driver': 'GTiff',
            'dtype': 'uint16',
            'nodata': None,
            'width': width,
            'height': height,
            'count': 1,
            'crs': crs,
            'transform': transform,
            'compress': 'lzw'
        }
        
        band_file = sample_dir / f"sample_{band_id}.tif"
        with rasterio.open(band_file, 'w', **profile) as dst:
            dst.write(data, 1)
        
        bands_data[band_id] = str(band_file)
    
    # Create metadata file
    metadata = {
        'product_name': 'SAMPLE_SENTINEL2_GHANA_20230601',
        'date': '2023-06-01',
        'region': 'test_region',
        'cloud_cover': 5.0,
        'bounds': bounds,
        'crs': crs,
        'bands': bands_data,
        'description': 'Synthetic Sentinel-2 data for testing mosquito habitat prediction pipeline'
    }
    
    with open(sample_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Sample dataset created in: {sample_dir}")
    logger.info(f"Bands available: {list(bands_data.keys())}")
    
    return metadata

def main():
    """
    Main function for command-line usage.
    """
    
    parser = argparse.ArgumentParser(description='Download Sentinel-2 data for mosquito habitat prediction')
    parser.add_argument('--region', choices=['west_africa', 'east_africa', 'southeast_asia', 'test_region', 'pilani'], 
                       default='pilani', help='Target region')
    parser.add_argument('--custom_bbox', nargs=4, type=float, metavar=('MIN_LON','MIN_LAT','MAX_LON','MAX_LAT'),
                       help='Override region with custom bounding box (WGS84)')
    parser.add_argument('--start_date', default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', default='2023-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--cloud_cover_max', type=int, default=20, help='Maximum cloud cover percentage')
    parser.add_argument('--output_dir', default='data', help='Output directory')
    parser.add_argument('--username', help='Copernicus Data Space username')
    parser.add_argument('--password', help='Copernicus Data Space password')
    parser.add_argument('--sample_only', action='store_true', help='Create sample dataset only')
    parser.add_argument('--max_products', type=int, default=10, help='Maximum number of products to download')
    
    args = parser.parse_args()
    
    if args.sample_only:
        create_sample_dataset()
        return
    
    # Initialize downloader
    downloader = SentinelDownloader(args.username, args.password)
    
    # Authenticate if credentials provided
    if args.username and args.password:
        downloader.authenticate()
    else:
        logger.warning("No credentials provided. Will create sample dataset instead.")
        create_sample_dataset()
        return
    
    # Search for products
    products = downloader.search_products(
        region=args.region,
        start_date=args.start_date,
        end_date=args.end_date,
        cloud_cover_max=args.cloud_cover_max,
        custom_bbox=args.custom_bbox
    )
    
    if not products:
        logger.error("No products found. Creating sample dataset instead.")
        create_sample_dataset()
        return
    
    # Limit number of products
    products = products[:args.max_products]
    
    # Create output directories
    download_dir = Path(args.output_dir) / 'raw'
    processed_dir = Path(args.output_dir) / 'processed'
    
    download_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Download and process products
    successful_downloads = []
    
    for i, product in enumerate(products, 1):
        logger.info(f"Processing product {i}/{len(products)}: {product['name']}")
        
        # Download
        zip_path = downloader.download_product(product, download_dir)
        
        if zip_path:
            # Extract and process
            processed_bands = downloader.extract_and_process(zip_path, processed_dir)
            
            if processed_bands:
                product['processed_bands'] = processed_bands
                successful_downloads.append(product)
    
    # Save download summary
    summary = {
        'region': args.region,
        'date_range': f"{args.start_date} to {args.end_date}",
        'total_products_found': len(products),
        'successful_downloads': len(successful_downloads),
        'products': successful_downloads
    }
    
    summary_file = Path(args.output_dir) / 'download_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Download complete. Summary saved to: {summary_file}")
    logger.info(f"Successfully processed {len(successful_downloads)} products")

if __name__ == "__main__":
    main()
