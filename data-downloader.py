import os
from azure.storage.blob import BlobServiceClient
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time

def get_connection_string():
    """Get Azure Storage connection string from environment variables"""
    connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    if not connection_string:
        print("Error: AZURE_STORAGE_CONNECTION_STRING not found in environment variables")
        print("Make sure it's set in your .bashrc file")
        sys.exit(1)
    return connection_string

def download_single_blob(blob_info, container_client, download_path, progress_lock, progress_counter):
    """Download a single blob"""
    blob_name, blob_size = blob_info
    try:
        # Create local file path
        local_file_path = os.path.join(download_path, blob_name)
        
        # Create local directory structure
        local_dir = os.path.dirname(local_file_path)
        os.makedirs(local_dir, exist_ok=True)
        
        # Download the blob
        blob_client = container_client.get_blob_client(blob_name)
        
        with open(local_file_path, "wb") as download_file:
            blob_data = blob_client.download_blob()
            download_file.write(blob_data.readall())
        
        # Update progress counter thread-safely
        with progress_lock:
            progress_counter[0] += 1
            print(f"[{progress_counter[0]}/{progress_counter[1]}] Downloaded: {blob_name}")
        
        return True
    except Exception as e:
        print(f"Error downloading {blob_name}: {str(e)}")
        return False

def download_folders_by_prefix(prefix, container_name="deepstream-data", download_path="./downloads", max_workers=10):
    """Download all folders from Azure Storage that start with the given prefix using concurrent threads"""
    try:
        # Initialize the BlobServiceClient
        connection_string = get_connection_string()
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        # Create download directory if it doesn't exist
        os.makedirs(download_path, exist_ok=True)
        
        print("Listing blobs...")
        # List all blobs with the given prefix and collect blob info
        blob_list = container_client.list_blobs(name_starts_with=prefix)
        blob_info_list = [(blob.name, blob.size) for blob in blob_list]
        
        if not blob_info_list:
            print(f"No files found with prefix '{prefix}' in container '{container_name}'")
            return
        
        total_files = len(blob_info_list)
        total_size = sum(size for _, size in blob_info_list)
        print(f"Found {total_files} files ({total_size / 1024 / 1024:.2f} MB)")
        
        # Progress tracking
        progress_lock = Lock()
        progress_counter = [0, total_files]  # [current, total]
        
        start_time = time.time()
        
        # Download files concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(download_single_blob, blob_info, container_client, download_path, progress_lock, progress_counter)
                for blob_info in blob_info_list
            ]
            
            # Wait for all downloads to complete
            successful_downloads = 0
            for future in as_completed(futures):
                if future.result():
                    successful_downloads += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nDownload completed!")
        print(f"Successfully downloaded {successful_downloads}/{total_files} files")
        print(f"Total time: {duration:.2f} seconds")
        print(f"Average speed: {(total_size / 1024 / 1024) / duration:.2f} MB/s")
        print(f"Files saved to: {download_path}")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        sys.exit(1)

def main():
    """Main function to get user input and start download"""
    print("Azure Storage Folder Downloader (Optimized)")
    print("=" * 45)
    
    # Get prefix from user
    prefix = input("Enter the folder prefix (e.g., 06-06): ").strip()
    
    if not prefix:
        print("Error: Prefix cannot be empty")
        sys.exit(1)
    
    # Ask for number of concurrent downloads
    try:
        max_workers = input("Number of concurrent downloads (default 10, max 20): ").strip()
        max_workers = int(max_workers) if max_workers else 10
        max_workers = min(max_workers, 20)  # Limit to prevent overwhelming
    except ValueError:
        max_workers = 10
    
    print(f"Searching for folders starting with '{prefix}' using {max_workers} concurrent downloads...")
    download_folders_by_prefix(prefix, max_workers=max_workers)

if __name__ == "__main__":
    main()
