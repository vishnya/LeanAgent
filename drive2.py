from googleapiclient.http import MediaFileUpload
from Google import create_service

CLIENT_SECRET_FILE = 'client_secret.json'
API_NAME = 'drive'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/drive']

service = create_service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

file_metadata = {
    'name': 'lecopivo-SciLean-22d53b2f4e3db2a172e71da6eb9c916e62655744.7z',
    'parents': ['1FWcM-J5xfZ5Vg6xsr7IquVJbOXDo39SR']
}

media_content = MediaFileUpload('lecopivo-SciLean-22d53b2f4e3db2a172e71da6eb9c916e62655744.7z', mimetype='application/zip')

file = service.files().create(
    body=file_metadata,
    media_body=media_content
).execute()

print(file)