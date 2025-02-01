import pygame
import sys
from PIL import Image
import pytesseract
import easyocr

# Initialize Pygame
pygame.init()

# Screen setup
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Evidence Board - Drag & Zoom")

# Colors
BACKGROUND_COLOR = (20, 26, 48)
DEFAULT_CARD_COLOR = (180, 220, 255)
DEFAULT_TEXT_COLOR = (0, 0, 0)

# Camera system
camera_offset = [0, 0]
zoom_min = 0.1
zoom_max = 5.0
zoom_level = 1.0
zoom_max_heading = 5.0
dragging = False
last_mouse_pos = (0, 0)

global bufferTextMultiLine
global bufferTextLine
bufferTextMultiLine = []
bufferTextLine = []

class TextMultiLine: 
    def __init__(self, multi_line_string, world_pos, base_size = 36, parent=None):
        self.strings = multi_line_string.split("\n")
        self.textLines = []
        self.world_pos = world_pos
        self.base_size = base_size
        lines = multi_line_string.split("\n")
        for i in range(len(lines)) :
            self.textLines.append(TextLine(lines[i], world_pos=(self.world_pos[0], self.world_pos[1] + (i * (self.base_size * 1.0))), base_size=base_size, parent=self, dynamic=False))
        self.parent = parent

        global bufferTextMultiLine
        bufferTextMultiLine.append(self)
        
    def update_opacity(self, zoom):
        for it in self.textLines :
            it.alpha = max(50, min(255, int(255 * zoom)))
            it.surface.set_alpha(it.alpha)
        
    def draw(self, screen, camera_offset, zoom):
        for it in self.textLines :
            it.draw(screen, camera_offset, zoom)

class TextLine:
    def __init__(self, text, world_pos, base_size=36, parent=None, dynamic=False):
        self.text = text
        self.dynamic = dynamic
        self.world_pos = world_pos
        self.base_size = base_size
        self.alpha = 255  # Start fully opaque
        self.font = pygame.font.Font(None, self.base_size)
        self.surface = self.font.render(self.text, True, DEFAULT_TEXT_COLOR).convert_alpha()
        self.parent = parent

        global bufferTextLine
        bufferTextLine.append(self)
        
    def update_opacity(self, zoom):
        # Calculate opacity based on zoom level (50-255 range)
        self.alpha = max(50, min(255, int(255 * zoom)))
        self.surface.set_alpha(self.alpha)

    def draw(self, screen, camera_offset, zoom) :
        if self.dynamic  == True :
            self.draw_dynamic(screen=screen, camera_offset=camera_offset, zoom=zoom)
        else :
            self.draw_static(screen=screen, camera_offset=camera_offset, zoom=zoom)
        
    def draw_dynamic(self, screen, camera_offset, zoom):
        print(self.text, " : dynamic draw")
        screen_pos = (
            (self.world_pos[0] - camera_offset[0]) * zoom,
            (self.world_pos[1] - camera_offset[1]) * zoom
        )
        screen.blit(self.surface, screen_pos)

    def draw_static(self, screen, camera_offset, zoom):
        print(self.text, " : static draw")
        # Scale the text surface instead of re-rendering it
        scaled_surface = pygame.transform.scale(
            self.surface,
            (int(self.surface.get_width() * zoom), int(self.surface.get_height() * zoom))
        )
        scaled_surface.set_alpha(self.alpha)  # Maintain transparency

        # Compute screen position
        screen_pos = (
            (self.world_pos[0] - camera_offset[0]) * zoom,
            (self.world_pos[1] - camera_offset[1]) * zoom
        )

        # Draw the scaled text
        screen.blit(scaled_surface, screen_pos)

#class CardNode:
#    def __init__(self, title, world_pos, base_size=36, parent=None):
#        global CARD_TITLE_SIZE
#        self.title = TextLine(title, world_pos, CARD_TITLE_SIZE, parent, True)
#        self.world_pos = world_pos
#        self.base_size = base_size
#
#        self.parent = parent
#
#        global bufferTextLine
#        bufferTextLine.append(self)
#
#    def draw(self, screen, camera_offset, zoom) :
#        print("na")



pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
test_img = Image.open("ocr_test.jpg")
text_tesseract = pytesseract.image_to_string(test_img)

reader = easyocr.Reader(['en'])
result = reader.readtext("ocr_test.jpg")
text_easyocr = ""
# Print extracted text and append it to text_easyocr
for (bbox, text, confidence) in result:
    # Append the extracted text, bounding box, and confidence to text_easyocr
    text_easyocr += text + '\n'

TextLine("Card Node", (150, 80), 72, None, True)
TextMultiLine(text_tesseract, (150, 120), 36, None)
TextMultiLine(text_easyocr, (400, 120), 36, None)

def update_text_opacity_global():
    for text_obj in bufferTextLine:
        text_obj.update_opacity(zoom_level)

# Initial opacity setup
update_text_opacity_global()





# CONTROLS
def handle_user_input(event) :
    global camera_offset
    global zoom_min
    global zoom_max
    global zoom_level
    global zoom_max_heading
    global dragging
    global last_mouse_pos

    # Mouse drag handling
    if event.type == pygame.MOUSEBUTTONDOWN:
        if event.button == 1:  # Left click
            dragging = True
            last_mouse_pos = pygame.mouse.get_pos()
    elif event.type == pygame.MOUSEBUTTONUP:
        if event.button == 1:
            dragging = False
    elif event.type == pygame.MOUSEMOTION and dragging:
        mouse_pos = pygame.mouse.get_pos()
        dx = (mouse_pos[0] - last_mouse_pos[0]) / zoom_level
        dy = (mouse_pos[1] - last_mouse_pos[1]) / zoom_level
        camera_offset[0] -= dx
        camera_offset[1] -= dy
        last_mouse_pos = mouse_pos
        
    # Mouse wheel zoom
    elif event.type == pygame.MOUSEWHEEL:
        zoom_direction = event.y
        mouse_screen = pygame.mouse.get_pos()
        mouse_world_before = (
            mouse_screen[0]/zoom_level + camera_offset[0],
            mouse_screen[1]/zoom_level + camera_offset[1]
        )
        # Adjust zoom
        zoom_level *= 1.1 ** zoom_direction
        zoom_level = max(0.1, min(zoom_level, 10.0))
        # Maintain mouse position
        mouse_world_after = (
            mouse_screen[0]/zoom_level + camera_offset[0],
            mouse_screen[1]/zoom_level + camera_offset[1]
        )
        camera_offset[0] += mouse_world_before[0] - mouse_world_after[0]
        camera_offset[1] += mouse_world_before[1] - mouse_world_after[1]
        # Update all text opacities
        update_text_opacity_global()


# Main loop
running = True
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        else :
            handle_user_input(event)

    # Clear screen
    screen.fill(BACKGROUND_COLOR)

    # Draw orange rectangle (world coordinates)
    rect_pos = ((100 - camera_offset[0]) * zoom_level, (100 - camera_offset[1]) * zoom_level)
    rect_size = (SCREEN_WIDTH * zoom_level, SCREEN_HEIGHT * zoom_level)
    pygame.draw.rect(screen, DEFAULT_CARD_COLOR, (*rect_pos, *rect_size), border_radius=10)

    # Draw all text objects
    for it in bufferTextLine:
        it.draw(screen, camera_offset, zoom_level)
        
    pygame.display.flip()

pygame.quit()
sys.exit()