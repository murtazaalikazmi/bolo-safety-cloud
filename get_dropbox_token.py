"""
Run this ONCE to get a permanent Dropbox refresh token.
Usage: python get_dropbox_token.py

You need your App Key and App Secret from:
  dropbox.com/developers/apps → your app → Settings tab
"""

from dropbox import DropboxOAuth2FlowNoRedirect

APP_KEY    = input("Paste your App Key here: ").strip()
APP_SECRET = input("Paste your App Secret here: ").strip()

auth_flow     = DropboxOAuth2FlowNoRedirect(APP_KEY, APP_SECRET, token_access_type="offline")
authorize_url = auth_flow.start()

print("\n" + "="*60)
print("STEP 1: Open this URL in your browser:")
print(authorize_url)
print("="*60)
print("STEP 2: Click 'Allow' on the Dropbox page.")
print("STEP 3: Copy the authorisation code shown.")
print("="*60 + "\n")

auth_code    = input("Paste the authorisation code here: ").strip()
oauth_result = auth_flow.finish(auth_code)

print("\n" + "="*60)
print("SUCCESS! Copy these 3 values into your secrets.toml:")
print("="*60)
print(f"DROPBOX_APP_KEY    = \"{APP_KEY}\"")
print(f"DROPBOX_APP_SECRET = \"{APP_SECRET}\"")
print(f"DROPBOX_REFRESH_TOKEN = \"{oauth_result.refresh_token}\"")
print("="*60)
print("The refresh token never expires.")
