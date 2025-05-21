from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import requests

class GooglePhotosService:
    @staticmethod
    def get_google_api_credentials(email):
        url = "https://api.weaiu.com/google-api/photo_service"
        params = {"email": email}

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                credentials_info = data["credentials"]
                return credentials_info
            else:
                return None
        else:
            return None

    @staticmethod
    def get_google_photos_service(credentials_info):
        creds = Credentials(
            token=credentials_info["token"],
            refresh_token=credentials_info["refresh_token"],
            token_uri=credentials_info["token_uri"],
            client_id=credentials_info["client_id"],
            client_secret=credentials_info["client_secret"],
            scopes=credentials_info["scopes"]
        )

        service = build('photoslibrary', 'v1', credentials=creds, static_discovery=False)
        return service

    @staticmethod
    def get_photos(service):
        try:
            # Get a list of recent photos
            results = service.mediaItems().list(pageSize=5).execute()

            # Extract data from each photo
            photos = []
            for item in results['mediaItems']:
                photo_id = item['id']
                filename = item.get('filename', 'N/A')  # Might not be available for all photos
                create_time = item['baseUrl']  # Base URL for downloading (requires further processing)

                # Append photo data to list
                photos.append({
                    "photo_id": photo_id,
                    "filename": filename,
                    "create_time": create_time,
                })

            return photos

        except Exception as e:
            # Handle any exceptions
            print(f"Error fetching recent photos: {e}")
            return []

    @staticmethod
    def fetch_all_photos(service):
        photos = []
        nextPageToken = None
        while True:
            results = service.mediaItems().list(pageSize=100, pageToken=nextPageToken).execute()
            photos.extend(results['mediaItems'])
            nextPageToken = results.get('nextPageToken')
            if not nextPageToken:
                break
        return photos

    @staticmethod
    def match_image(uploaded_image_filename, google_photos):
        for photo in google_photos:
            if photo['filename'] == uploaded_image_filename:
                return photo
        return None