# immanager
A image manager powered with AI.

![readme.png](readme.png)

## Features
- Upload images or specify a path to a local folder with images
- Supports Samba (e.g. `smb://192.168.x.x/`)
- Search images with a prompt text
- Cluster images into groups

## Setup
Install the dependencies:
```bash
pip install -r requirements.txt
```
Run the app:
```bash
python gui.py
```
Open the gradio interface provided by the IP and port in the terminal.

## Samba support
- Create a `.env` file specifying your Samba credentials to the server you will later enter the path, like:
```env
SMB_USERNAME=<username>
SMB_PASSWORD=<password>
SMB_DOMAIN=WORKGROUP # or your domain
CLIENT_NAME=<client_name>
```

You can obtain CLIENT_NAME by running `hostname` in your terminal.
