from mega import Mega
import os
import shutil

logs_path = 'lightning_logs'
zip_name = 'lightning_logs.zip'

email = os.environ.get('MEGA_EMAIL')
password = os.environ.get('MEGA_PASSWORD')

print("Compressing lightning_logs folder...")
shutil.make_archive('lightning_logs', 'zip', logs_path)

mega = Mega()

try:
    print("Logging in to Mega...")
    m = mega.login(email, password)
    print("Uploading zip file...")
    file = m.upload(zip_name)
    link = m.get_upload_link(file)
    print(f"File uploaded successfully. Download link: {link}")
except Exception as e:
    print(f"An error occurred: {str(e)}")