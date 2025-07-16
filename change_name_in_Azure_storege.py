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

def find_all_folders_with_f1(container_client, limit=20):
    """Find first 20 folders that contain F1 subdirectories"""
    try:
        print(f"Scanning for first {limit} folders with F1 data...")
        
        # Get all blobs and extract folder structure
        blob_list = container_client.list_blobs()
        folders_with_f1 = set()
        
        for blob in blob_list:
            path_parts = blob.name.split('/')
            # Check if path has at least main_folder/F1/file structure
            if len(path_parts) >= 3 and path_parts[1] == 'F1':
                main_folder = path_parts[0]
                folders_with_f1.add(main_folder)
                
                # Stop once we reach the limit
                if len(folders_with_f1) >= limit:
                    break
        
        return sorted(list(folders_with_f1))[:limit]  # Ensure we don't exceed limit
        
    except Exception as e:
        print(f"Error scanning for F1 folders: {str(e)}")
        return []

def check_floMobility123_F1_exists(container_client, main_folder_name):
    """Check if floMobility123_F1 folder already exists"""
    try:
        floMobility123_F1_prefix = f"{main_folder_name}/floMobility123_F1/"
        floMobility123_F1_blobs = list(container_client.list_blobs(name_starts_with=floMobility123_F1_prefix, results_per_page=1))
        return len(floMobility123_F1_blobs) > 0
    except Exception as e:
        print(f"Error checking floMobility123_F1 folder: {str(e)}")
        return False

def duplicate_single_blob(blob_info, container_client, main_folder_name, progress_lock, progress_counter, max_retries=3):
    """Duplicate a single blob from F1 to floMobility123_F1"""
    source_blob_name, blob_size = blob_info
    
    # Create destination blob name by replacing F1 with floMobility123_F1
    dest_blob_name = source_blob_name.replace(f"{main_folder_name}/F1/", f"{main_folder_name}/floMobility123_F1/")
    
    for attempt in range(max_retries):
        try:
            # Get source and destination blob clients
            source_blob_client = container_client.get_blob_client(source_blob_name)
            dest_blob_client = container_client.get_blob_client(dest_blob_name)
            
            # Copy the blob
            copy_source = source_blob_client.url
            dest_blob_client.start_copy_from_url(copy_source)
            
            # Wait for copy to complete
            copy_properties = dest_blob_client.get_blob_properties()
            while copy_properties.copy.status == 'pending':
                time.sleep(0.1)
                copy_properties = dest_blob_client.get_blob_properties()
            
            if copy_properties.copy.status == 'success':
                with progress_lock:
                    progress_counter[0] += 1
                    print(f"  [{progress_counter[0]}/{progress_counter[1]}] Duplicated: {os.path.basename(source_blob_name)}")
                return True
            else:
                raise Exception(f"Copy failed with status: {copy_properties.copy.status}")
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1} for {os.path.basename(source_blob_name)}: {str(e)}")
                time.sleep(1)
            else:
                print(f"  Failed to duplicate {os.path.basename(source_blob_name)} after {max_retries} attempts: {str(e)}")
                return False
    
    return False

def delete_single_blob(blob_name, container_client, progress_lock, progress_counter, max_retries=3):
    """Delete a single blob from F1 folder"""
    for attempt in range(max_retries):
        try:
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.delete_blob()
            
            with progress_lock:
                progress_counter[0] += 1
                print(f"  [{progress_counter[0]}/{progress_counter[1]}] Deleted: {os.path.basename(blob_name)}")
            
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1} for deleting {os.path.basename(blob_name)}: {str(e)}")
                time.sleep(1)
            else:
                print(f"  Failed to delete {os.path.basename(blob_name)} after {max_retries} attempts: {str(e)}")
                return False
    
    return False

def process_single_folder(main_folder_name, container_client, max_workers=10):
    """Process a single folder: duplicate F1 to floMobility123_F1 and delete original F1"""
    try:
        print(f"\n{'='*60}")
        print(f"Processing folder: {main_folder_name}")
        print(f"{'='*60}")
        
        # Check if floMobility123_F1 already exists
        if check_floMobility123_F1_exists(container_client, main_folder_name):
            print(f"⚠️  floMobility123_F1 already exists in '{main_folder_name}' - skipping...")
            return True
        
        # Get list of all F1 blobs
        print(f"Getting list of F1 files...")
        f1_prefix = f"{main_folder_name}/F1/"
        blob_list = container_client.list_blobs(name_starts_with=f1_prefix)
        blob_info_list = [(blob.name, blob.size) for blob in blob_list]
        
        if not blob_info_list:
            print(f"No files found in F1 folder of '{main_folder_name}' - skipping...")
            return True
        
        total_files = len(blob_info_list)
        total_size = sum(size for _, size in blob_info_list)
        print(f"Found {total_files} files in F1 folder ({total_size / 1024 / 1024:.2f} MB)")
        
        # Step 1: Duplicate F1 to floMobility123_F1
        print(f"\nStep 1: Duplicating F1 → floMobility123_F1...")
        progress_lock = Lock()
        progress_counter = [0, total_files]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(duplicate_single_blob, blob_info, container_client, main_folder_name, progress_lock, progress_counter)
                for blob_info in blob_info_list
            ]
            
            successful_duplications = 0
            for future in as_completed(futures):
                if future.result():
                    successful_duplications += 1
        
        print(f"Duplication result: {successful_duplications}/{total_files} files")
        
        if successful_duplications != total_files:
            print(f"❌ Not all files were duplicated successfully. Skipping deletion of original F1.")
            return False
        
        # Step 2: Delete original F1 folder
        print(f"\nStep 2: Deleting original F1 folder...")
        progress_counter = [0, total_files]  # Reset counter
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(delete_single_blob, blob_name, container_client, progress_lock, progress_counter)
                for blob_name, _ in blob_info_list
            ]
            
            successful_deletions = 0
            for future in as_completed(futures):
                if future.result():
                    successful_deletions += 1
        
        print(f"Deletion result: {successful_deletions}/{total_files} files")
        
        success = successful_deletions == total_files
        if success:
            print(f"✅ SUCCESS: F1 → floMobility123_F1 completed for '{main_folder_name}'")
        else:
            print(f"❌ PARTIAL FAILURE: Some files could not be deleted in '{main_folder_name}'")
        
        return success
        
    except Exception as e:
        print(f"❌ Error processing folder '{main_folder_name}': {str(e)}")
        return False

def process_all_folders_with_f1(container_name="deepstream-data", max_workers=10):
    """Find first 20 folders with F1 and process them"""
    try:
        # Initialize the BlobServiceClient
        connection_string = get_connection_string()
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        # Find first 20 folders with F1
        folders_with_f1 = find_all_folders_with_f1(container_client, limit=20)
        
        if not folders_with_f1:
            print("No folders with F1 data found")
            return False
        
        print(f"\nFound {len(folders_with_f1)} folders with F1 data (showing first 20):")
        for i, folder in enumerate(folders_with_f1, 1):
            print(f"  {i}. {folder}")
        
        # Auto-process all found folders (up to 20)
        print(f"\nThis will automatically process all {len(folders_with_f1)} folders:")
        print("• Duplicate each F1 folder as floMobility123_F1")
        print("• Delete the original F1 folders")
        
        confirm = input(f"\nProceed with processing these {len(folders_with_f1)} folders? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("Operation cancelled")
            return False
        
        # Process each folder
        start_time = time.time()
        successful_folders = 0
        
        for i, folder_name in enumerate(folders_with_f1, 1):
            print(f"\n🔄 Processing folder {i}/{len(folders_with_f1)}: {folder_name}")
            success = process_single_folder(folder_name, container_client, max_workers)
            if success:
                successful_folders += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Final summary
        print(f"\n" + "="*80)
        print(f"BATCH OPERATION COMPLETED")
        print(f"="*80)
        print(f"Total folders processed: {len(folders_with_f1)}")
        print(f"Successful: {successful_folders}")
        print(f"Failed: {len(folders_with_f1) - successful_folders}")
        print(f"Total time: {duration:.2f} seconds")
        print(f"Average time per folder: {duration/len(folders_with_f1):.2f} seconds")
        
        if successful_folders == len(folders_with_f1):
            print(f"🎉 ALL FOLDERS PROCESSED SUCCESSFULLY!")
        else:
            print(f"⚠️  {len(folders_with_f1) - successful_folders} folders had issues")
        
        return successful_folders == len(folders_with_f1)
        
    except Exception as e:
        print(f"Error in batch operation: {str(e)}")
        return False

def main():
    """Main function to start the batch operation"""
    print("Azure Storage Batch F1 → floMobility123_F1 Processor")
    print("=" * 60)
    print("This script will:")
    print("1. Find the first 20 folders that contain F1 subdirectories")
    print("2. For each of these 20 folders:")
    print("   • Duplicate F1 as floMobility123_F1")
    print("   • Delete the original F1 folder")
    print("=" * 60)
    
    # Start the batch operation
    success = process_all_folders_with_f1()
    
    if success:
        print(f"\n🎉 BATCH OPERATION COMPLETED SUCCESSFULLY!")
    else:
        print(f"\n❌ BATCH OPERATION COMPLETED WITH ISSUES")
        sys.exit(1)

if __name__ == "__main__":
    main()