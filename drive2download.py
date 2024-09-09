from googleapiclient.http import MediaIoBaseDownload
from Google import create_service
import io

CLIENT_SECRET_FILE = 'client_secret.json'
API_NAME = 'drive'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

service = create_service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

def download_file(file_id, output_path):
    try:
        # Get the file metadata
        file_metadata = service.files().get(fileId=file_id).execute()
        file_name = file_metadata['name']

        # Create a BytesIO object for the file content
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        
        # Create a downloader object and download the file
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")

        # Save the file content to the specified output path
        fh.seek(0)
        with open(output_path, 'wb') as f:
            f.write(fh.read())
        
        print(f"File '{file_name}' downloaded successfully to '{output_path}'.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage
file_id = '1eC7MiPrymsgaPs9X95t4s5RXpqFp5yvd'  # Replace with your file ID
output_path = 'leanprover-community-mathlib4-2b29e73438e240a427bcecc7c0fe19306beb1310.7z'

download_file(file_id, output_path)