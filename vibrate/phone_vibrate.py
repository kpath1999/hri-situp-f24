from pyicloud import PyiCloudService
import keyring

# Your Apple ID
apple_id = "kausarpatheryaa@gmail.com"

# Get an app-specific password from keyring
app_specific_password = keyring.get_password("pyicloud", apple_id)

if not app_specific_password:
    print("App-specific password not found in keyring.")
    app_specific_password = input("Enter your app-specific password (visible): ")
    
    # Optionally save the app-specific password in keyring
    save = input("Do you want to save this app-specific password in keyring? (y/n): ").lower()
    if save == "y":
        keyring.set_password("pyicloud", apple_id, app_specific_password)
else:
    update = input("Do you want to update the app-specific password in keyring? (y/n): ").lower()
    if update == "y":
        app_specific_password = input("Enter your new app-specific password (visible): ")
        keyring.set_password("pyicloud", apple_id, app_specific_password)
        
# Initialize the PyiCloudService with the app-specific password
try:
    api = PyiCloudService(apple_id, keyring.get_password("pyicloud", apple_id))
    
    if api.requires_2fa:
        print("Two-factor authentication required.")
        code = input("Enter the code you received on your devices: ")
        result = api.validate_2fa_code(code)
        print("2FA validation result:", result)
    
    if not result:
        print("Failed to verify 2FA code.")
    else:
        print("2FA verified successfully.")

    print("Successfully logged in to iCloud.")
    
    # Now you can use the API
    devices = api.devices
    for device in devices:
        print(device)

except Exception as e:
    print("Failed to log in to iCloud.")
    print(e)