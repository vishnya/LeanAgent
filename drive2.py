import argparse
from googleapiclient.http import MediaFileUpload
from Google import create_service

def main():
    parser = argparse.ArgumentParser(description="Upload a file to Google Drive")
    parser.add_argument("file_path", help="Path to the file you want to upload")
    parser.add_argument("file_name", help="Name for the file in Google Drive")
    args = parser.parse_args()

    CLIENT_SECRET_FILE = 'client_secret.json'
    API_NAME = 'drive'
    API_VERSION = 'v3'
    SCOPES = ['https://www.googleapis.com/auth/drive']

    service = create_service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

    file_metadata = {
        'name': args.file_name,
        'parents': ['1FWcM-J5xfZ5Vg6xsr7IquVJbOXDo39SR']
    }

    media_content = MediaFileUpload(args.file_path, resumable=True)

    file = service.files().create(
        body=file_metadata,
        media_body=media_content
    ).execute()

    print(f"File ID: {file.get('id')}")

if __name__ == "__main__":
    main()