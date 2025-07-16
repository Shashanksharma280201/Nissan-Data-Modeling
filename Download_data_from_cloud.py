import os
from azure.storage.blob import BlobServiceClient
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
from azure.core.exceptions import AzureError

def get_connection_string():
    """Get Azure Storage connection string from environment variables"""
    connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    if not connection_string:
        print("Error: AZURE_STORAGE_CONNECTION_STRING not found in environment variables")
        print("Make sure it's set in your .bashrc file")
        sys.exit(1)
    return connection_string

def download_single_blob(blob_info, container_client, download_path, progress_lock, progress_counter, max_retries=3):
    """Download a single blob with retry logic"""
    blob_name, blob_size = blob_info
    
    for attempt in range(max_retries):
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
            if attempt < max_retries - 1:
                print(f"Retry {attempt + 1} for {blob_name}: {str(e)}")
                time.sleep(1)  # Brief pause before retry
            else:
                print(f"Failed to download {blob_name} after {max_retries} attempts: {str(e)}")
                return False
    
    return False

def get_available_folders(container_client, timestamp_prefix, max_retries=3):
    """Get list of available timestamp folders that start with the prefix and contain F2 data"""
    for attempt in range(max_retries):
        try:
            print(f"Scanning for timestamp folders starting with '{timestamp_prefix}' that contain F2 data...")
            if attempt > 0:
                print(f"Retry attempt {attempt + 1}/{max_retries}...")
                time.sleep(2)  # Wait before retry
            
            # Use pagination to handle large result sets more reliably
            folders_with_f2 = set()
            continuation_token = None
            
            while True:
                try:
                    # List blobs in smaller chunks
                    blob_iter = container_client.list_blobs(
                        name_starts_with=timestamp_prefix,
                        results_per_page=1000,  # Process in chunks
                        include=['metadata']
                    )
                    
                    for blob in blob_iter:
                        # Split path and check if it contains F2
                        path_parts = blob.name.split('/')
                        if len(path_parts) >= 2 and path_parts[1] == 'F2':
                            timestamp_folder = path_parts[0]  # First part is the full timestamp folder
                            # Only include if it starts with our prefix
                            if timestamp_folder.startswith(timestamp_prefix):
                                folders_with_f2.add(timestamp_folder)
                    
                    break  # Successfully completed
                    
                except Exception as chunk_error:
                    print(f"Error during blob listing: {str(chunk_error)}")
                    if attempt == max_retries - 1:
                        raise
                    break  # Try the whole operation again
            
            return sorted(list(folders_with_f2))
            
        except AzureError as e:
            print(f"Azure error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                print("Failed to scan folders after multiple retries")
                raise
        except Exception as e:
            print(f"Error scanning folders on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise
    
    return []

def download_folders_by_prefix(timestamp_prefix, selected_folders, container_name="deepstream-data", download_path="./downloads", max_workers=10):
    """Download F2 folders from selected timestamp folders"""
    try:
        # Initialize the BlobServiceClient
        connection_string = get_connection_string()
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        # Create download directory if it doesn't exist
        os.makedirs(download_path, exist_ok=True)
        
        print("Listing F2 blobs...")
        blob_info_list = []
        
        # For each selected folder, get F2 blobs with retry logic
        for folder in selected_folders:
            f2_prefix = f"{folder}/F2/"
            print(f"Scanning {f2_prefix}...")
            
            # Retry logic for listing blobs in each folder
            for attempt in range(3):
                try:
                    blob_list = container_client.list_blobs(
                        name_starts_with=f2_prefix,
                        results_per_page=1000  # Process in smaller chunks
                    )
                    folder_blobs = [(blob.name, blob.size) for blob in blob_list]
                    blob_info_list.extend(folder_blobs)
                    break  # Success, exit retry loop
                except Exception as e:
                    if attempt < 2:
                        print(f"Retry scanning {f2_prefix} (attempt {attempt + 2}/3)...")
                        time.sleep(1)
                    else:
                        print(f"Failed to scan {f2_prefix}: {str(e)}")
                        continue
        
        if not blob_info_list:
            print(f"There is no data in F2 of timestamp folders starting with '{timestamp_prefix}'")
            return
        
        total_files = len(blob_info_list)
        total_size = sum(size for _, size in blob_info_list)
        print(f"Found {total_files} F2 files ({total_size / 1024 / 1024:.2f} MB)")
        
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
        print(f"Successfully downloaded {successful_downloads}/{total_files} F2 files")
        print(f"Total time: {duration:.2f} seconds")
        print(f"Average speed: {(total_size / 1024 / 1024) / duration:.2f} MB/s")
        print(f"Files saved to: {download_path}")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        sys.exit(1)

def main():
    """Main function to get user input and start download"""
    print("Azure Storage F2 Data Downloader (Enhanced)")
    print("=" * 50)
    
    # Initialize the BlobServiceClient to scan available data types
    try:
        connection_string = get_connection_string()
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client("deepstream-data")
    except Exception as e:
        print(f"Error connecting to Azure Storage: {str(e)}")
        sys.exit(1)
    
    # Get timestamp prefix from user
    timestamp_prefix = input("Enter the timestamp prefix (format: mm-dd): ").strip()
    
    if not timestamp_prefix:
        print("Error: Timestamp prefix cannot be empty")
        sys.exit(1)
    
    # Get available timestamp folders that start with the prefix and have F2 data
    available_folders = get_available_folders(container_client, timestamp_prefix)
    
    if not available_folders:
        print(f"There is no data in F2 of timestamp folders starting with '{timestamp_prefix}'")
        sys.exit(1)
    
    print(f"\nFound {len(available_folders)} timestamp folders starting with '{timestamp_prefix}' that contains the data:")
    for i, folder in enumerate(available_folders, 1):
        print(f"{i}. {folder}")
    
    # Ask user about download preference
    download_choice = input(f"\nDo you want to download ALL {len(available_folders)} folders or specify a number? (all/number): ").strip().lower()
    
    selected_folders = []
    
    if download_choice == "all":
        selected_folders = available_folders
        print(f"Selected all {len(selected_folders)} folders")
    else:
        try:
            if download_choice == "number":
                num_folders = input("How many folders do you want to download? ").strip()
                num_folders = int(num_folders)
            else:
                # Try to parse the input as a number
                num_folders = int(download_choice)
            
            if num_folders <= 0:
                print("Error: Number of folders must be positive")
                sys.exit(1)
            
            if num_folders > len(available_folders):
                print(f"Error: Requested {num_folders} folders but only {len(available_folders)} available")
                sys.exit(1)
            
            selected_folders = available_folders[:num_folders]
            print(f"Selected first {num_folders} folders:")
            for folder in selected_folders:
                print(f"  - {folder}")
            
        except ValueError:
            print("Error: Invalid input. Please enter 'all' or a valid number")
            sys.exit(1)
    
    # Set optimal concurrent downloads automatically (reduced for stability)
    max_workers = 10  # Reduced for better network stability
    
    print(f"\nStarting download of F2 data from timestamp folders starting with '{timestamp_prefix}'...")
    
    download_folders_by_prefix(timestamp_prefix, selected_folders, max_workers=max_workers)

if __name__ == "__main__":
    main()