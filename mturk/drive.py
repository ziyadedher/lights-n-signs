from __future__ import print_function
from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools

# If modifying these scopes, delete the file token.json.
SCOPES = 'https://www.googleapis.com/auth/drive.metadata.readonly'
FOLDER_ID = "15EXCJs9VF6t7L6X9077m9TYvAQNobDqP" # 10mph
# 1X0bQyX2uE85uKlyhXBxnU-H8zWJapN_5 15mph
# 1D3Su5K8AnYQ1NTnKGtLfw85J1C0zCDFL 20mph
# 1ttD8RJvVAblATLa1Lt4agLWuys0S00w3 Right turn only (words)

def get_image_urls(n):
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    store = file.Storage('token.json')
    creds = store.get()
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets('credentials.json', SCOPES)
        creds = tools.run_flow(flow, store)
    service = build('drive', 'v3', http=creds.authorize(Http()))

    # Call the Drive v3 API
    next_page_token = ''
    image_urls = []

    while len(image_urls) < n:
        print(len(image_urls), next_page_token)
        results = service.files().list(
            pageSize=50, pageToken=next_page_token, q="'{}' in parents".format(FOLDER_ID), fields="nextPageToken, files(id, name, webContentLink)").execute()
        next_page_token = results.get('nextPageToken', '')
        if next_page_token == '':
            print("Could not find enough images")
            return []
        items = results.get('files', [])

        if not items:
            print('No files found.')
        else:
            for item in items:
                image_urls.append((item['webContentLink'], item['id']))

    return image_urls

if __name__ == '__main__':
    get_image_urls(5)
