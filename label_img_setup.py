import os
import subprocess
from pathlib import Path

def setup_labeling_environment():
    """
    Setup the labeling environment and directory structure
    """
    # Create directory structure
    directories = [
        'raw_images',
        'dataset/images/train',
        'dataset/images/val',
        'dataset/labels/train',
        'dataset/labels/val'
    ]
    
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
    
    # Create classes.txt file
    with open('classes.txt', 'w') as f:
        f.write('logo\n')  # Add more classes if needed

def install_labelimg():
    """
    Install LabelImg tool
    """
    try:
        subprocess.run(['pip', 'install', 'labelImg'])
        print("LabelImg installed successfully!")
    except Exception as e:
        print(f"Error installing LabelImg: {e}")
        print("Try installing manually with: pip install labelImg")

def prepare_images_for_labeling():
    """
    Prepare images for labeling by ensuring correct format and size
    """
    import cv2
    
    raw_dir = Path('raw_images')
    train_dir = Path('dataset/images/train')
    
    # Process each image in raw_images directory
    for img_path in raw_dir.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Read image
            img = cv2.imread(str(img_path))
            
            # Resize if too large (optional)
            max_size = 1280
            height, width = img.shape[:2]
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height))
            
            # Save processed image
            new_path = train_dir / img_path.name
            cv2.imwrite(str(new_path), img)

def start_labeling():
    """
    Launch LabelImg with correct configuration
    """
    try:
        # Configure LabelImg for YOLO format
        cmd = [
            'labelImg',
            'dataset/images/train',  # Image directory
            'classes.txt',           # Class file
            'dataset/labels/train'   # Output directory
        ]
        
        print("Starting LabelImg...")
        print("\nInstrucciones de uso:")
        print("1. Presiona 'w' para crear un nuevo bounding box")
        print("2. Haz clic y arrastra para dibujar el box alrededor del logo")
        print("3. Selecciona la clase 'logo' en el popup")
        print("4. Presiona 'd' para ir a la siguiente imagen")
        print("5. Presiona 's' para guardar")
        print("6. Presiona 'q' para salir cuando termines\n")
        
        subprocess.run(cmd)
    except Exception as e:
        print(f"Error launching LabelImg: {e}")
        print("Make sure LabelImg is installed correctly")

if __name__ == "__main__":
    # Setup environment
    setup_labeling_environment()
    
    # Install LabelImg
    install_labelimg()
    
    # Prepare images
    prepare_images_for_labeling()
    
    # Start labeling
    start_labeling()