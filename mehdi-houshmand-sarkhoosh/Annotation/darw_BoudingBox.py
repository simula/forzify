from PIL import Image, ImageDraw, ImageFont

# Function to convert YOLO format to bounding box
def yolo_to_bbox(img_width, img_height, box):
    x_center, y_center, width, height = box
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x_min = int(x_center - (width / 2))
    y_min = int(y_center - (height / 2))
    x_max = int(x_center + (width / 2))
    y_max = int(y_center + (height / 2))
    return x_min, y_min, x_max, y_max

# Load the image
image_path = '/content/100000000018.jpg'  # Change this to the path of your image
image = Image.open(image_path)
img_width, img_height = image.size

# Load the YOLO annotations
annotations_path = '/content/folder3/100000000018.txt'  # Change this to the path of your YOLO annotations
with open(annotations_path, 'r') as file:
    annotations = file.readlines()

# Create a draw object
draw = ImageDraw.Draw(image)

# Optionally add a font
try:
    # Use a truetype font
    font = ImageFont.truetype("arial.ttf", 15)  # You can choose a font and size that's appropriate for your image
except IOError:
    # If the true type font can't be loaded, it will fall back to a default font
    font = ImageFont.load_default()

# Draw each bounding box
for annotation in annotations:
    # Each line in the YOLO annotation file is: class x_center y_center width height
    class_id, x_center, y_center, width, height = map(float, annotation.split())
    box = yolo_to_bbox(img_width, img_height, (x_center, y_center, width, height))
    draw.rectangle(box, outline='red', width=2)  # Change 'red' and width if you want a different color or line width
    
    # Draw the text
    text = f"Class {int(class_id)}"
    # Get the size of the text
    text_size = draw.textsize(text, font=font)
    # Set up the text background
    text_background = (255, 255, 255)  # White background for text
    # Draw the background rectangle for text
    draw.rectangle([box[0], box[1] - text_size[1], box[0] + text_size[0], box[1]], fill=text_background)
    # Draw the text
    draw.text((box[0], box[1] - text_size[1]), text, fill='black', font=font)

# Save or display the image
image.show()  # To display the image
image.save('annotated_image1.jpg')  # To save the image to a file
