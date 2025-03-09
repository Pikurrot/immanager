# gui.py
import gradio as gr
import os
from PIL import Image
import base64
from io import BytesIO
from smb.SMBConnection import SMBConnection
from src.image_embeddings import get_image_embeddings, get_text_embedding
from src.search import search_similar_images
from src.clustering import cluster_images
from dotenv import load_dotenv
load_dotenv()

# --- Helper Functions ---

def load_env_settings():
    """
    Load SMB connection settings from .env.
    Returns a dict with keys: username, password, domain, client_name.
    """
    return {
        "username": os.getenv("SMB_USERNAME"),
        "password": os.getenv("SMB_PASSWORD"),
        "domain": os.getenv("SMB_DOMAIN", ""),
        "client_name": os.getenv("CLIENT_NAME")
    }

def parse_smb_url(url):
    """
    Parse an SMB URL of the form:
      smb://server/share/optional/path
    Returns (server, share, relative_path)
    """
    if not url.startswith("smb://"):
        raise ValueError("Not an SMB URL")
    trimmed = url[6:]  # Remove "smb://"
    parts = trimmed.split('/')
    if len(parts) < 2:
        raise ValueError("SMB URL must contain at least server and share")
    server = parts[0]
    share = parts[1]
    rel_path = "/" + "/".join(parts[2:]) if len(parts) > 2 and parts[2] != "" else "/"
    return server, share, rel_path

def smb_walk(conn, share, top):
    """
    Recursively walk through an SMB share directory using an SMBConnection instance.
    Yields tuples of (current_path, subdirectories, files) where current_path is relative to the share.
    """
    try:
        entries = conn.listPath(share, top)
    except Exception as e:
        print(f"Error listing directory {top}: {e}")
        return

    dirs = []
    files = []
    for entry in entries:
        if entry.filename in [".", ".."]:
            continue
        if entry.isDirectory:
            dirs.append(entry.filename)
        else:
            files.append(entry.filename)
    yield top, dirs, files
    for d in dirs:
        new_top = os.path.join(top, d)
        yield from smb_walk(conn, share, new_top)

def pil_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

# --- Main Function to Load Images ---
def load_images_combined(files, folder_path):
    """
    Load images from either a local folder or an SMB folder.
    - If 'files' (list of file paths) is provided, load those.
    - If 'folder_path' is provided and is an SMB URL (starts with "smb://"),
      recursively load all image files within that share.
    - If 'folder_path' is a local directory, use os.walk.
    Returns a tuple (status message, state) where state is a dict with:
       - "images": dict mapping file paths to PIL Image objects
       - "embeddings": computed embeddings (via get_image_embeddings)
    """
    images = {}
    # Option 1: Load from file uploader if files are provided.
    if files:
        for file in files:
            try:
                images[file] = Image.open(file).convert("RGB")
            except Exception as e:
                print("Error loading file:", file, e)
    # Option 2: Load from folder path.
    elif folder_path:
        if folder_path.startswith("smb://"):
            try:
                server, share, rel_path = parse_smb_url(folder_path)
            except Exception as e:
                return f"Invalid SMB URL: {e}", None

            env = load_env_settings()
            remote_name = server  # Use the server name from the URL
            server_ip = server    # Assuming DNS resolution works

            # Create an SMBConnection instance
            conn = SMBConnection(
                env["username"],
                env["password"],
                env["client_name"],
                remote_name,
                env["domain"],
                use_ntlm_v2=True
            )
            # Connect using port 445 (SMB2/3)
            if not conn.connect(server_ip, 445):
                return "Connection failed.", None

            # Walk the SMB share starting at rel_path
            for root, dirs, files_in_dir in smb_walk(conn, share, rel_path):
                for filename in files_in_dir:
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        full_path = os.path.join(root, filename)
                        try:
                            # Use retrieveFile to read file contents into a BytesIO buffer.
                            file_buffer = BytesIO()
                            conn.retrieveFile(share, full_path, file_buffer)
                            file_buffer.seek(0)
                            img = Image.open(file_buffer).convert("RGB")
                            images[full_path] = img
                        except Exception as e:
                            print("Error loading SMB file:", full_path, e)
            conn.close()
        elif os.path.isdir(folder_path):
            # Local folder
            for root, _, files_in_dir in os.walk(folder_path):
                for filename in files_in_dir:
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        full_path = os.path.join(root, filename)
                        try:
                            images[full_path] = Image.open(full_path).convert("RGB")
                        except Exception as e:
                            print("Error loading local file:", full_path, e)
        else:
            return "Invalid folder path provided.", None
    else:
        return "No images provided", None

    if not images:
        return "No images loaded", None

    embeddings = get_image_embeddings(images)
    state = {"images": images, "embeddings": embeddings}
    return "Images loaded successfully", state

def search_images(prompt, state):
    if state is None:
        return []
    text_emb = get_text_embedding(prompt)
    results = search_similar_images(text_emb, state["embeddings"], top_k=5)
    result_images = [state["images"][filename] for filename, _ in results]
    return result_images

def cluster_images_gui(state, num_clusters):
    if state is None:
        return "No images loaded."
    clusters = cluster_images(state["embeddings"], num_clusters=num_clusters)
    html_output = ""
    for cluster_id, filenames in clusters.items():
        html_output += f"<h3>Cluster {cluster_id}</h3>"
        html_output += "<div style='display:flex; flex-wrap: wrap;'>"
        for filename in filenames:
            img = state["images"][filename]
            img_base64 = pil_to_base64(img)
            html_output += f"<img src='{img_base64}' style='width: 100px; margin: 5px;'>"
        html_output += "</div>"
    return html_output

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# Immanager")
    
    with gr.Tab("Load Images"):
        gr.Markdown("Choose either to upload images or enter a folder path.\n\nFor SMB connection, use a URL like: `smb://ds912.local/images/`")
        file_input = gr.File(label="Select images (simulate folder)", file_count="multiple", type="filepath")
        folder_input = gr.Textbox(label="Or enter folder path", placeholder="/path/to/your/images or smb://server/share/path")
        load_button = gr.Button("Load Images")
        load_status = gr.Textbox(label="Status")
        state_images = gr.State()  # To store loaded images and embeddings

    with gr.Tab("Search"):
        prompt_input = gr.Textbox(label="Enter text prompt")
        search_button = gr.Button("Search Images")
        search_gallery = gr.Gallery(label="Search Results", columns=3)
    
    with gr.Tab("Clustering"):
        cluster_slider = gr.Slider(minimum=2, maximum=10, step=1, label="Number of Clusters", value=5)
        cluster_button = gr.Button("Cluster Images")
        cluster_output = gr.HTML(label="Clusters")
    
    load_button.click(fn=load_images_combined, inputs=[file_input, folder_input], outputs=[load_status, state_images])
    search_button.click(fn=search_images, inputs=[prompt_input, state_images], outputs=search_gallery)
    cluster_button.click(fn=cluster_images_gui, inputs=[state_images, cluster_slider], outputs=cluster_output)

if __name__ == "__main__":
    demo.launch()
