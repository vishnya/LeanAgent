# import argparse
# import os
# from googleapiclient.http import MediaFileUpload
# from Google import create_service

# def main():
#     parser = argparse.ArgumentParser(description="Upload files from a folder to Google Drive")
#     parser.add_argument("folder_path", help="Path to the folder containing files to upload")
#     parser.add_argument("prefix", help="Prefix of files to upload")
#     parser.add_argument("--parent_folder_id", default='1FWcM-J5xfZ5Vg6xsr7IquVJbOXDo39SR', help="Parent folder ID in Google Drive")
#     args = parser.parse_args()

#     CLIENT_SECRET_FILE = 'client_secret.json'
#     API_NAME = 'drive'
#     API_VERSION = 'v3'
#     SCOPES = ['https://www.googleapis.com/auth/drive']

#     service = create_service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

#     for filename in os.listdir(args.folder_path):
#         if filename.startswith(args.prefix):
#             file_path = os.path.join(args.folder_path, filename)
            
#             file_metadata = {
#                 'name': filename,
#                 'parents': [args.parent_folder_id]
#             }

#             media_content = MediaFileUpload(file_path, resumable=True)

#             file = service.files().create(
#                 body=file_metadata,
#                 media_body=media_content
#             ).execute()

#             print(f"Uploaded {filename} - File ID: {file.get('id')}")

# if __name__ == "__main__":
#     main()


# import argparse
# from googleapiclient.http import MediaFileUpload
# from Google import create_service

# def main():
#     parser = argparse.ArgumentParser(description="Upload a file to Google Drive")
#     parser.add_argument("file_path", help="Path to the file you want to upload")
#     parser.add_argument("file_name", help="Name for the file in Google Drive")
#     args = parser.parse_args()

#     CLIENT_SECRET_FILE = 'client_secret.json'
#     API_NAME = 'drive'
#     API_VERSION = 'v3'
#     SCOPES = ['https://www.googleapis.com/auth/drive']

#     service = create_service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

#     file_metadata = {
#         'name': args.file_name,
#         'parents': ['1FWcM-J5xfZ5Vg6xsr7IquVJbOXDo39SR']
#     }

#     media_content = MediaFileUpload(args.file_path, resumable=True)

#     file = service.files().create(
#         body=file_metadata,
#         media_body=media_content
#     ).execute()

#     print(f"File ID: {file.get('id')}")

# if __name__ == "__main__":
#     main()

# import argparse
# import os
# from googleapiclient.http import MediaFileUpload
# from Google import create_service

# def create_folder(service, folder_name, parent_id):
#     """Create a folder in Google Drive"""
#     file_metadata = {
#         'name': folder_name,
#         'mimeType': 'application/vnd.google-apps.folder',
#         'parents': [parent_id]
#     }
    
#     file = service.files().create(
#         body=file_metadata,
#         fields='id'
#     ).execute()
    
#     return file.get('id')

# def upload_file(service, file_path, file_name, parent_id):
#     """Upload a single file to Google Drive"""
#     file_metadata = {
#         'name': file_name,
#         'parents': [parent_id]
#     }

#     media_content = MediaFileUpload(file_path, resumable=True)

#     file = service.files().create(
#         body=file_metadata,
#         media_body=media_content,
#         fields='id'
#     ).execute()

#     return file.get('id')

# def upload_folder(service, folder_path, parent_id):
#     """Recursively upload a folder and its contents to Google Drive"""
#     folder_name = os.path.basename(folder_path)
#     print(f"Creating folder: {folder_name}")
    
#     # Create the folder in Google Drive
#     folder_id = create_folder(service, folder_name, parent_id)
    
#     # Iterate through all items in the folder
#     for item in os.listdir(folder_path):
#         item_path = os.path.join(folder_path, item)
        
#         if os.path.isfile(item_path):
#             # Upload file
#             print(f"Uploading file: {item}")
#             upload_file(service, item_path, item, folder_id)
#         elif os.path.isdir(item_path):
#             # Recursively upload subfolder
#             upload_folder(service, item_path, folder_id)

# def main():
#     parser = argparse.ArgumentParser(description="Upload files or folders to Google Drive")
#     parser.add_argument("path", help="Path to the file or folder you want to upload")
#     parser.add_argument("--name", help="Optional: Name for the file/folder in Google Drive (defaults to original name)")
#     args = parser.parse_args()

#     CLIENT_SECRET_FILE = 'client_secret.json'
#     API_NAME = 'drive'
#     API_VERSION = 'v3'
#     SCOPES = ['https://www.googleapis.com/auth/drive']
    
#     # Parent folder ID in Google Drive where files will be uploaded
#     PARENT_FOLDER_ID = '1FWcM-J5xfZ5Vg6xsr7IquVJbOXDo39SR'

#     service = create_service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

#     # Determine if the path is a file or folder
#     if os.path.isfile(args.path):
#         file_name = args.name if args.name else os.path.basename(args.path)
#         print(f"Uploading file: {file_name}")
#         file_id = upload_file(service, args.path, file_name, PARENT_FOLDER_ID)
#         print(f"File uploaded successfully. ID: {file_id}")
    
#     elif os.path.isdir(args.path):
#         print(f"Uploading folder: {os.path.basename(args.path)}")
#         upload_folder(service, args.path, PARENT_FOLDER_ID)
#         print("Folder and its contents uploaded successfully")
    
#     else:
#         print(f"Error: {args.path} does not exist")

# if __name__ == "__main__":
#     main()

import argparse
import os
import json
import time
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from Google import create_service
from socket import timeout as SocketTimeout
from typing import Optional
import sys

class UploadTracker:
    def __init__(self, track_file=".upload_progress.json"):
        self.track_file = track_file
        self.uploaded_files = self._load_progress()
        self.current_operation = self._load_current_operation()
    
    def _load_progress(self):
        if os.path.exists(self.track_file):
            try:
                with open(self.track_file, 'r') as f:
                    data = json.load(f)
                    return data.get('files', {})
            except json.JSONDecodeError:
                return {}
        return {}
    
    def _load_current_operation(self):
        if os.path.exists(self.track_file):
            try:
                with open(self.track_file, 'r') as f:
                    data = json.load(f)
                    return data.get('current_operation', None)
            except json.JSONDecodeError:
                return None
        return None
    
    def save_progress(self, current_path: Optional[str] = None):
        """Save current upload progress and operation state"""
        with open(self.track_file, 'w') as f:
            json.dump({
                'files': self.uploaded_files,
                'current_operation': current_path or self.current_operation
            }, f, indent=2)
    
    def mark_uploaded(self, file_path: str, file_id: str):
        self.uploaded_files[os.path.abspath(file_path)] = file_id
        self.save_progress()
    
    def is_uploaded(self, file_path: str) -> bool:
        return os.path.abspath(file_path) in self.uploaded_files
    
    def get_file_id(self, file_path: str) -> Optional[str]:
        return self.uploaded_files.get(os.path.abspath(file_path))
    
    def set_current_operation(self, path: str):
        """Set the current operation being processed"""
        self.current_operation = path
        self.save_progress()
    
    def clear_current_operation(self):
        """Clear the current operation after successful completion"""
        self.current_operation = None
        self.save_progress()

def create_folder_with_retries(service, folder_name: str, parent_id: str, max_retries: int = 3) -> str:
    """Create a folder with automatic retry on timeout"""
    for attempt in range(max_retries):
        try:
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_id]
            }
            
            file = service.files().create(
                body=file_metadata,
                fields='id'
            ).execute()
            
            return file.get('id')
        except (HttpError, SocketTimeout) as e:
            if attempt == max_retries - 1:
                raise
            print(f"Timeout creating folder {folder_name}, retrying... ({attempt + 1}/{max_retries})")
            time.sleep(2 ** attempt)  # Exponential backoff

def upload_file_with_retries(service, file_path: str, file_name: str, parent_id: str, 
                           tracker: UploadTracker, max_retries: int = 3) -> Optional[str]:
    """Upload a file with automatic retry on timeout"""
    if tracker.is_uploaded(file_path):
        print(f"Skipping {file_name} - already uploaded")
        return tracker.get_file_id(file_path)
    
    for attempt in range(max_retries):
        try:
            tracker.set_current_operation(file_path)
            
            file_metadata = {
                'name': file_name,
                'parents': [parent_id]
            }

            media_content = MediaFileUpload(file_path, resumable=True)

            file = service.files().create(
                body=file_metadata,
                media_body=media_content,
                fields='id'
            ).execute()

            file_id = file.get('id')
            tracker.mark_uploaded(file_path, file_id)
            tracker.clear_current_operation()
            return file_id
            
        except (HttpError, SocketTimeout) as e:
            if attempt == max_retries - 1:
                print(f"Failed to upload {file_name} after {max_retries} attempts")
                raise
            print(f"Timeout uploading {file_name}, retrying... ({attempt + 1}/{max_retries})")
            time.sleep(2 ** attempt)  # Exponential backoff

def process_directory(service, folder_path: str, parent_id: str, tracker: UploadTracker) -> None:
    """Process a directory and its contents with timeout handling"""
    folder_name = os.path.basename(folder_path)
    print(f"\nProcessing folder: {folder_name}")
    
    # Create folder in Google Drive
    folder_id = create_folder_with_retries(service, folder_name, parent_id)
    
    # Get list of all items to process
    items = sorted(os.listdir(folder_path))  # Sort for consistent ordering
    total_items = len(items)
    
    # Find starting point if resuming
    start_index = 0
    if tracker.current_operation:
        try:
            current_item = os.path.basename(tracker.current_operation)
            if current_item in items:
                start_index = items.index(current_item)
                print(f"Resuming from {current_item}")
        except ValueError:
            pass

    # Process items
    for i, item in enumerate(items[start_index:], start=start_index):
        item_path = os.path.join(folder_path, item)
        print(f"\nProcessing {i + 1}/{total_items}: {item}")
        
        try:
            if os.path.isfile(item_path):
                if item.startswith('.') or item.startswith('~'):
                    continue
                
                upload_file_with_retries(service, item_path, item, folder_id, tracker)
                print(f"Successfully uploaded: {item}")
                
            elif os.path.isdir(item_path):
                process_directory(service, item_path, folder_id, tracker)
                
        except Exception as e:
            print(f"Error processing {item}: {str(e)}")
            # Continue with next item instead of breaking
            continue

def main():
    parser = argparse.ArgumentParser(description="Upload files or folders to Google Drive with automatic retry")
    parser.add_argument("path", help="Path to the file or folder you want to upload")
    parser.add_argument("--name", help="Optional: Name for the file/folder in Google Drive (defaults to original name)")
    parser.add_argument("--progress-file", default=".upload_progress.json", 
                       help="File to store upload progress (default: .upload_progress.json)")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum number of retries per operation (default: 3)")
    args = parser.parse_args()

    CLIENT_SECRET_FILE = 'client_secret.json'
    API_NAME = 'drive'
    API_VERSION = 'v3'
    SCOPES = ['https://www.googleapis.com/auth/drive']
    PARENT_FOLDER_ID = '1FWcM-J5xfZ5Vg6xsr7IquVJbOXDo39SR'

    tracker = UploadTracker(args.progress_file)
    
    while True:  # Main retry loop
        try:
            service = create_service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)
            
            if os.path.isfile(args.path):
                file_name = args.name if args.name else os.path.basename(args.path)
                if not tracker.is_uploaded(args.path):
                    file_id = upload_file_with_retries(service, args.path, file_name, 
                                                     PARENT_FOLDER_ID, tracker, args.max_retries)
                    print(f"File uploaded successfully. ID: {file_id}")
                else:
                    print(f"File already uploaded. ID: {tracker.get_file_id(args.path)}")
            
            elif os.path.isdir(args.path):
                process_directory(service, args.path, PARENT_FOLDER_ID, tracker)
                print("\nFolder processing completed!")
            
            else:
                print(f"Error: {args.path} does not exist")
            
            # If we get here, everything completed successfully
            tracker.clear_current_operation()
            break
            
        except (HttpError, SocketTimeout) as e:
            print(f"\nConnection timeout or error occurred: {str(e)}")
            print("Waiting 30 seconds before retrying...")
            time.sleep(30)  # Wait before retrying the entire operation
            continue
            
        except KeyboardInterrupt:
            print("\nOperation interrupted by user. Progress saved.")
            sys.exit(1)
            
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            print("Progress saved. Please check the error and retry.")
            sys.exit(1)

if __name__ == "__main__":
    main()